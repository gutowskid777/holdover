"""Conversation mode. The caller just talks. Bonsai holds a real conversation, and the protocol
trees are its checked reference: every instruction it gives must come from the protocol on the table.
Only the two seconds-matter protocols (not breathing, choking) stay fully scripted, with the CPR beat."""
from __future__ import annotations
import re, time
from engine import Session, options, keyword_hit, sentences, REPORT, CAMERA, FILLER

SCRIPTED = {"unresponsive", "choking"}       # seconds matter: fixed words, rules first, metronome

STYLE = (
    "You are Holdover, a calm first-aid dispatcher on a voice call. You run fully offline on this laptop.\n"
    "HOW TO TALK\n"
    "- This is a spoken conversation. Reply with 1 to 3 short sentences. Make the first sentence very short.\n"
    "- Talk like a steady, kind person, not a form. Ask open questions, one at a time. Never demand a yes or no.\n"
    "- Use everything the caller has already told you. Never ask for something they already said.\n"
    "- If the caller is the hurt person, say 'you'. If it is someone else, say he, she, or the name the caller "
    "uses. Never say 'they', 'them', 'it' or 'the patient' for a person.\n"
    "- Plain words only. No lists, no symbols, no brackets except the camera tag below.\n"
)

GROUNDED = (
    "WHAT YOU MAY SAY\n"
    "- Every instruction and every medical statement must come from the CHECKED PROTOCOL below. You may reword "
    "it to fit the conversation, but do not add treatments, doses, causes or reassurance that are not in it.\n"
    "- From the whole conversation, work out which branch of the protocol applies. Skip steps and questions the "
    "caller has already covered. Give the next one or two steps, then stop and listen.\n"
    "- If a warning sign in the protocol fits what the caller said, say so plainly and give that step first.\n"
    "- If the caller asks something the protocol does not cover, say you do not have a checked answer for that.\n"
    "- If the protocol has a CAMERA step and seeing it would help, ask the caller to hold it up to the camera "
    "and end your reply with the tag [LOOK]. Only do this once.\n"
)

UNGROUNDED = (
    "WHAT YOU MAY SAY\n"
    "- No checked protocol is on the table yet. If the caller has not described a problem, ask what happened, "
    "openly and briefly.\n"
    "- If they describe something, give only basic, widely accepted first aid, say it is general knowledge and "
    "not one of your checked protocols, and name the warning signs that mean a doctor or emergency care.\n"
    "- Never diagnose. If one missing detail would change the advice, ask for it instead of advising.\n"
    "- If seeing the injury would help, ask the caller to hold it up to the camera and end your reply with [LOOK].\n"
)


def flatten(p) -> str:
    """The tree as a numbered flowchart the model can read: its checked reference text."""
    ids = {nid: f"S{i + 1}" for i, nid in enumerate(p["nodes"])}

    def to(t):
        return f"switch to the {t[1:]} protocol" if t.startswith("@") else ids.get(t, t)

    out = [f"CHECKED PROTOCOL: {p['title']}. Start at {ids[p['start']]}."]
    for nid, n in p["nodes"].items():
        k = n["kind"]
        if k == "ask":
            br = "; ".join(f"if {key if key in ('yes', 'no') else m}: {to(g)}" for key, (m, g) in options(n).items())
            if n.get("unsure"):
                br += f"; if unknown: {to(n['unsure'])}"
            out.append(f"{ids[nid]} FIND OUT: {n['ask']}  ({br})")
        elif k == "look":
            br = "; ".join(f"if {m}: {to(g)}" for _, (m, g) in options(n).items())
            out.append(f"{ids[nid]} CAMERA: {n.get('question', n['look'])}  ({br})")
        elif k == "do":
            nxt = n.get("next") or n.get("then")
            extra = f"; if they cannot: {to(n['cant'])}" if n.get("cant") else ""
            out.append(f"{ids[nid]} STEP: {n['do']}  (then {to(nxt)}{extra})")
        else:
            out.append(f"{ids[nid]} FINISH: {n['end']}")
    if p.get("facts"):
        out.append("FACTS YOU MAY USE: " + " ".join(p["facts"]))
    return "\n".join(out)


class Conversation:
    def __init__(self, protocols, llm):
        self.protocols, self.llm = protocols, llm
        self.t0 = time.time()
        self.turns = []          # (who, text): the whole call
        self.log = []            # timeline for the handoff note
        self.proto = None
        self.node = None         # only set while a scripted protocol is running (server reads .node)
        self.drill = None        # the scripted Session, while CPR or choking is running
        self.scene = None        # what the idle camera last noticed: context only, never spoken unasked
        self.looked = False
        self.flat = {}

    # ---------- shared helpers
    def _log(self, kind, text):
        self.log.append({"t": round(time.time() - self.t0), "kind": kind, "text": text})

    def _out(self, say, **extra):
        return {"say": say, "protocol": self.proto["id"] if self.proto else None,
                "title": self.proto["title"] if self.proto else None, "node": None, "kind": "talk",
                "step": " ".join(say), "expects": "speech", "choices": {}, "timer": None, "metronome": None,
                "log": self.log, "open": self.proto is None, **extra}

    def start(self):
        hello = "Holdover here. I work with no signal. Tell me what happened."
        self.turns.append(("dispatch", hello))
        return self._out([hello])

    # ---------- routing
    def _alarm(self, text):
        for p in self.protocols.values():
            if p["id"] in SCRIPTED and any(re.search(a, text, re.I) for a in p.get("alarms", [])):
                return p["id"]
        return None

    def _route(self, text):
        hits = [pid for pid, p in self.protocols.items() if p.get("triggers") and keyword_hit(text, p["triggers"])]
        if self.proto and (not hits or self.proto["id"] in hits):
            return self.proto["id"]                      # stay on the topic we are on
        if len(hits) == 1:
            return hits[0]                               # plain keyword: no model needed
        pool = {p: self.protocols[p] for p in hits} if len(hits) > 1 else self.protocols
        said = " ".join(t for w, t in self.turns[-4:] if w == "caller")
        picks = self.llm.triage_all(said, pool)
        if picks and picks[0] != "general":
            return picks[0]
        return self.proto["id"] if self.proto else None

    def _start_drill(self, pid, text):
        """Seconds matter: hand the call to the scripted engine, sharing this call's clock and timeline."""
        s = Session(self.protocols, self.llm)
        s.t0, s.log = self.t0, self.log
        say = []
        s._log("caller", text)
        s._enter("@" + pid, say, text)
        self.drill, self.proto, self.node = s, s.proto, s.node
        out = s._out(say)
        for line in say:
            self.turns.append(("dispatch", line))
        return out

    # ---------- the turn
    def hear(self, text: str):
        text = text.strip()
        if not text or FILLER.match(text):
            return self._out([])
        self.turns.append(("caller", text))

        if self.drill:
            out = self.drill.hear(text)
            self.proto, self.node = self.drill.proto, self.drill.node
            for line in out["say"]:
                self.turns.append(("dispatch", line))
            if self.drill.node and self.drill.node["kind"] == "end" and not self.drill.pending:
                self.drill, self.node = None, None       # back to conversation, same topic
            return out

        self._log("caller", text)
        if self.log and REPORT.search(text) and len(text.split()) <= 8:
            return self._out(["Here is the handoff note."], report=True)
        pid = self._alarm(text)
        if pid:
            return self._start_drill(pid, text)
        if CAMERA.search(text):
            return self._out(["Okay. Hold it up to the camera and keep still."], capture="describe")

        pid = self._route(text)
        if pid in SCRIPTED:
            return self._start_drill(pid, text)
        if pid and (not self.proto or self.proto["id"] != pid):
            self.proto = self.protocols[pid]
            self.looked = False
            self._log("protocol", self.proto["title"])
        return self._reply()

    def see(self, image_b64: str, mode: str = "look"):
        if self.drill:
            out = self.drill.see(image_b64, mode)
            self.proto, self.node = self.drill.proto, self.drill.node
            return out
        desc = self.llm.describe(image_b64)
        self._log("camera", desc)
        self.turns.append(("caller", f"(I held it up to the camera. The camera shows: {desc})"))
        return self._reply()

    def _reply(self):
        if self.proto:
            pid = self.proto["id"]
            if pid not in self.flat:
                self.flat[pid] = flatten(self.proto)
            system = STYLE + GROUNDED + "\n" + self.flat[pid]
        else:
            system = STYLE + UNGROUNDED
        if self.scene:
            system += (f"\n\nBACKGROUND: the laptop camera currently shows: {self.scene}. Do not mention the camera "
                       "or what it shows unless the caller asks or it changes the advice.")
        if self.looked:
            system += "\nYou have already looked through the camera once. Do not use [LOOK] again."
        raw = self.llm.converse(system, self.turns[-14:])
        look = "[LOOK]" in raw.upper() and not self.looked
        clean = re.sub(r"\[\s*look\s*\]", "", raw, flags=re.I).replace("*", "").strip()
        parts = [s for s in sentences(clean) if s][:4]
        reply = " ".join(parts)
        self.turns.append(("dispatch", reply))
        self._log("holdover", reply)
        if look:
            self.looked = True
            return self._out(parts, capture="describe")
        return self._out(parts)

    # ---------- handoff
    def report(self):
        mm = lambda s: f"{s // 60:02d}:{s % 60:02d}"
        names = {"caller": "CALLER", "holdover": "HOLDOVER", "protocol": "CHECKED PROTOCOL", "camera": "CAMERA"}
        lines = [f"[{mm(e['t'])}] {names.get(e['kind'], e['kind'].upper())}: {e['text']}" for e in self.log]
        return {"started": time.strftime("%Y-%m-%d %H:%M", time.localtime(self.t0)),
                "duration": mm(round(time.time() - self.t0)),
                "protocols": [e["text"] for e in self.log if e["kind"] == "protocol"],
                "timeline": lines, "summary": self.llm.summary(lines) if lines else ""}
