"""Backtest the Bonsai ears against tests/cases.yaml.
   python tests/backtest.py [--url http://127.0.0.1:8080/v1] [--only triage|interpret] [-v]"""
import argparse, pathlib, sys, time, yaml
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain"))
from engine import load_protocols, options, Session, question_of, UNSURE, CANT, ASKING
from llm import Bonsai

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
ap.add_argument("--only")
ap.add_argument("-v", action="store_true")
ap.add_argument("--heldout", action="store_true", help="score the held-out phrases instead of the tuning set")
a = ap.parse_args()

P = load_protocols(ROOT / "protocols")
cases = yaml.safe_load((ROOT / "tests" / "cases.yaml").read_text())
llm = Bonsai(a.url)
print("model:", llm._model().split("/")[-1])
total = wrong = 0
groups = ("triage", "interpret") if not a.heldout else ("heldout_triage", "heldout_interpret")
for name in groups:
    if a.only and a.only not in name:
        continue
    ok, misses, t0, fast = 0, [], time.time(), 0
    for c in cases[name]:
        if name.endswith("triage"):
            text, want = c
            got = llm.triage(text, P)
            where = "triage"
        else:
            where, text, want = c
            pid, nid = where.split(".")
            n = P[pid]["nodes"][nid]
            kind = n["kind"]
            opts = {k: m for k, (m, _) in options(n).items()}
            if kind == "do":
                opts = {"done": "the caller did it, is doing it, or is ready for the next step"}
                if n.get("cant"):
                    opts["cant"] = "the caller cannot do it or does not have what is needed"
            want = {True: "yes", False: "no"}.get(want, want)
            # same order of rules as Session.hear
            if kind != "do" and len(text.split()) <= 8 and UNSURE.search(text):
                got = "unsure"
            elif kind == "do" and "cant" in opts and CANT.search(text) and not ASKING.search(text):
                got = "cant"
            else:
                got = Session._fast(text, kind, opts) or Session._keywords(text, n)
            if got:
                fast += 1
            else:
                got = llm.interpret(question_of(n) if kind != "do" else n[kind], opts, text)
                if got == "other" and n.get("steady"):
                    got = "question"
        if got == want:
            ok += 1
        else:
            misses.append((where, text, want, got, getattr(llm, "last_reason", "")))
    n_cases = len(cases[name]); dt = time.time() - t0
    total += n_cases; wrong += len(misses)
    print(f"\n{name}: {ok}/{n_cases} correct  ({dt / n_cases:.2f}s per case, {fast} answered by rules with no model)")
    for m in misses:
        print(f"  MISS {m[0]}: \"{m[1]}\"  wanted {m[2]}, got {m[3]}  [{m[4][:90]!r}]")
print(f"\nTOTAL {total - wrong}/{total}")
