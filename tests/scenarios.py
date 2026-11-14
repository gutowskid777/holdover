"""Scenario test: whole calls, start to finish, against the real Bonsai model.
   python tests/scenarios.py [-v] [--only substring] [--url http://127.0.0.1:8080/v1]
Each scenario in scenarios.yaml lists the caller's lines and what must be true afterwards:
  protocols: [a, b]   protocols entered, in order (prefix match on the order actually entered)
  visit: [p.node]     steps that must be reached          never: [p.node]  steps that must NOT be reached
  final: p.node       where the call ends up              says: ["text"]   text the dispatcher must say
  never_says: [..]    text the dispatcher must never say  none: true       no protocol may be entered
Always checked: no crash, and the dispatcher never says the same line three turns in a row."""
import argparse, pathlib, sys, time, yaml
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain"))
from engine import load_protocols, validate, Session
from llm import Bonsai

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
ap.add_argument("--only")
ap.add_argument("-v", action="store_true")
a = ap.parse_args()

P = load_protocols(ROOT / "protocols")
assert not validate(P), validate(P)
llm = Bonsai(a.url)
scen = yaml.safe_load((ROOT / "tests" / "scenarios.yaml").read_text())
if a.only:
    scen = [s for s in scen if a.only in s["name"]]

def full(x, default=None):
    return x if "." in x else f"{default}.{x}"

passed, fails, t0, calls0 = 0, [], time.time(), len(llm.calls)
for s in scen:
    ses = Session(P, llm)
    turns, problems = [("", ses.start())], []
    try:
        for line in s["lines"]:
            turns.append((line, ses.hear(line)))
    except Exception as e:
        problems.append(f"CRASH {type(e).__name__}: {e}")
    protos = [v.split(".")[0] for v in ses.visited]
    order = [p for i, p in enumerate(protos) if i == 0 or protos[i - 1] != p]
    said = [x for _, o in turns for x in o["say"]]
    alltext = " ".join(said)
    first = s.get("protocols", [None])[0] if s.get("protocols") else None
    if s.get("none") and order:
        problems.append(f"entered {order} but should have entered nothing")
    if s.get("protocols") and order[:len(s["protocols"])] != s["protocols"]:
        problems.append(f"protocols {order}, wanted {s['protocols']}")
    for v in s.get("visit", []):
        if full(v, first) not in ses.visited:
            problems.append(f"never reached {full(v, first)}")
    for v in s.get("never", []):
        if full(v, first) in ses.visited:
            problems.append(f"reached forbidden {full(v, first)}")
    if s.get("first_in") and (not order or order[0] not in s["first_in"]):
        problems.append(f"first protocol {order[:1]}, wanted one of {s['first_in']}")
    for v in s.get("never_protocols", []):
        if v in protos:
            problems.append(f"entered forbidden protocol {v}")
    if s.get("final"):
        end = f"{ses.proto['id']}.{ses.node['id']}" if ses.node else "none"
        if end != full(s["final"], first):
            problems.append(f"ended at {end}, wanted {full(s['final'], first)}")
    for v in s.get("says", []):
        if v.lower() not in alltext.lower():
            problems.append(f"never said: {v}")
    for v in s.get("never_says", []):
        if v.lower() in alltext.lower():
            problems.append(f"said forbidden: {v}")
    lasts = [" | ".join(o["say"]) for _, o in turns if o["say"]]
    for i in range(2, len(lasts)):
        if lasts[i] == lasts[i - 1] == lasts[i - 2]:
            problems.append(f"same line three turns in a row: {lasts[i][:60]}")
            break
    if problems:
        fails.append(s["name"])
    else:
        passed += 1
    if a.v or problems:
        print(f"\n=== {s['name']}  {'FAIL' if problems else 'ok'}")
        for line, o in turns:
            if line:
                print(f"   CALLER: {line}")
            print(f"   DISPATCH [{o['protocol']}.{o['node']}]: {' | '.join(o['say']) or '(silent)'}")
        for p in problems:
            print(f"   !! {p}")
print(f"\nSCENARIOS {passed}/{len(scen)} passed in {time.time() - t0:.0f}s, {len(llm.calls) - calls0} model calls")
if fails:
    print("failed:", ", ".join(fails))
