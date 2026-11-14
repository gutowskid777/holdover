"""Protocol tree engine. Every word of medical advice comes from the YAML trees in
../protocols. The language model never writes treatment: it only decides which branch
the caller is on, answers side questions from the protocol's own facts, and reads photos."""
from __future__ import annotations
import pathlib, re, time, yaml

YES = re.compile(r"^(yes|yeah|yea|yep|yup|ya|correct|affirmative|uh huh|definitely|absolutely|i think so|it is|he is|she is|they are|i do|we do|i am|he does|she does|sure)\b", re.I)
NO = re.compile(r"^(no|nope|nah|negative|not really|not yet|nothing|he isn'?t|she isn'?t|it isn'?t|they aren'?t|i don'?t think so|he doesn'?t|she doesn'?t|none)\b", re.I)
DONE = re.compile(r"^(ok|okay|done|all done|it'?s done|did it|got it|ready|next|alright|all right|finished|it'?s on|it'?s in|i did|we did|go on|continue|go ahead)\b", re.I)
REPEAT = re.compile(r"^\W*(what|huh|sorry|pardon|excuse me|say again|again)\W*$|\b(repeat|say (that|it) again|what did you say|come again|one more time|what was that|didn'?t (hear|catch))\b", re.I)
REPORT = re.compile(r"\b(report|handoff|hand off|summary for (the )?(medics|paramedics|doctor))\b", re.I)
CAMERA = re.compile(r"\b(look at (this|it|that|him|her|my|the)|can you see|take a look|see this|check (this|it) out|i'?ll show you|let me show you)\b", re.I)
UNSURE = re.compile(r"\b(i (don'?t|do not|really don'?t) know|not sure|no idea|can'?t (really )?tell|cannot tell|hard to (tell|say)|dunno|unsure|no clue)\b", re.I)
CANT = re.compile(r"\b(i can'?t|i cannot|we can'?t|we cannot|don'?t have|do not have|there'?s no|there is no|there isn'?t|we have no|i have no|haven'?t got|not able|unable)\b", re.I)
WAIT = re.compile(r"^\W*(wait|hold on|hang on|one (sec|second|minute|moment)|just a (sec|second|minute|moment)|give me a (sec|second|minute|moment))(\W+(wait|hold on|hang on))*\W*$", re.I)
FILLER = re.compile(r"^\W*((um+|uh+|er+|erm+|hm+|ah+|oh|like|so|well)\W*)+$", re.I)
BACK = re.compile(r"\b(go back|back up|previous (step|question)|last (step|question)|the (step|question) before)\b", re.I)
ASKING = re.compile(r"\?|^\W*(what|how|where|which|why|should|can i|can we|do i|do we|is it|does it)\b", re.I)
NEG = re.compile(r"\b(not|no|isn'?t|aren'?t|wasn'?t|never|stopped|without|still no|hasn'?t|doesn'?t|don'?t|can'?t)\b", re.I)

# Questions about what happens AFTER an action can never be known in advance, so skip-ahead leaves them alone.
LATER = re.compile(r"\b(slowing|stopped|stopping|now|still|yet|better|worse|came out|come out|clear|staying|working|improv\w*|any change|after)\b", re.I)

GREETING = "Emergency line. I work with no signal. What is happening?"
ASK_WHAT = "Tell me what happened. Who is hurt, and how?"
STILL_HERE = "I am still here. Tell me if anything changes."
PLAY_SAFE = "We will play it safe."
FIRST = "Understood. That comes first."
NOTED = "Noted. We will get to that next."
NEXT_PROBLEM = "Now the other problem."
HERE = "Okay. I am here."
NO_FACT = "I don't have that. Stay on this step."
DONE_OR_CANT = "Say done, or say you cannot."
WHICH = "Say the injury in a few words. Like bleeding, a burn, or not breathing."


def load_protocols(folder) -> dict:
    out = {}
    for f in sorted(pathlib.Path(folder).glob("*.yaml"), key=lambda f: (f.stem == "general", f.stem)):
        p = yaml.safe_load(f.read_text())
        for nid, n in p["nodes"].items():
            for flag, word in ((True, "yes"), (False, "no")):  # YAML reads bare yes/no as booleans
                if flag in n:
                    n[word] = n.pop(flag)
            n["id"] = nid
            n["kind"] = next(k for k in ("ask", "do", "look", "end") if k in n)
        out[p["id"]] = p
    return out


def validate(protocols: dict) -> list[str]:
    """Every goto must land on a real node, and every question needs a play-it-safe path."""
    bad = []
    for pid, p in protocols.items():
        nodes = p["nodes"]
        if p["start"] not in nodes:
            bad.append(f"{pid}: start '{p['start']}' missing")
        for e in p.get("entries", []):
            if e["goto"] not in nodes:
                bad.append(f"{pid}: entry goto '{e['goto']}' missing")
        for nid, n in nodes.items():
            for target in targets(n):
                if target.startswith("@"):
                    if target[1:] not in protocols:
                        bad.append(f"{pid}.{nid}: unknown protocol {target}")
                elif target not in nodes:
                    bad.append(f"{pid}.{nid}: unknown node '{target}'")
            if n["kind"] in ("ask", "look") and not options(n):
                bad.append(f"{pid}.{nid}: no options")
            if n["kind"] in ("ask", "look") and not n.get("unsure"):
                bad.append(f"{pid}.{nid}: question has no 'unsure' path")
            if n["kind"] == "do" and not (n.get("next") or n.get("then")):
                bad.append(f"{pid}.{nid}: do-step has no next/then")
    return bad


def options(n) -> dict:
    """key -> (meaning, goto)"""
    if "choices" in n:
        return {k: (v["means"], v["goto"]) for k, v in n["choices"].items()}
    o = {}
    if "yes" in n:
        o["yes"] = ("Yes: the reply says, or clearly describes, that what the dispatcher asked about is true", n["yes"])
    if "no" in n:
        o["no"] = ("No: the reply says it is not true, or describes something milder, smaller, or the opposite", n["no"])
    return o


def targets(n):
    t = [g for _, g in options(n).values()]
    t += [n[k] for k in ("next", "then", "cant", "unsure") if n.get(k)]
    if n.get("worse"):
        t.append(n["worse"]["goto"])
    return t


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def question_of(n):
    """The short form of a question step: its last sentence that ends in a question mark."""
    if n.get("question"):
        return n["question"]
    qs = [s for s in sentences(n[n["kind"]]) if s.endswith("?")]
    return qs[-1] if qs else n[n["kind"]]


def keyword_hit(text, words):
    """True if one of the words is said and not negated just before ('not breathing' is not 'breathing')."""
    for w in words:
        for m in re.finditer(rf"\b{re.escape(w)}s?\b", text, re.I):
            before = " ".join(text[:m.start()].split()[-3:])
            if not NEG.search(before):
                return True
    return False


class Session:
    def __init__(self, protocols: dict, llm):
        self.protocols, self.llm = protocols, llm
        self.t0 = time.time()
        self.proto = None
        self.node = None
        self.log = []          # {t, kind, text}
        self.misses = 0
        self.transcript = []   # everything said, both sides
        self.visited = []      # "protocol.node" in order, including chained steps
        self.history = []      # (protocol id, node id) of steps the caller answered, for "go back"
        self.pending = []      # ("start", pid) or ("resume", pid, node id): problems waiting their turn
        self.triage_text = ""
        self.idle_said = False
        self.known = {}        # (protocol id, node id) -> answer the caller already gave, before being asked
        self.skipped = 0

    # ---------- helpers
    def _log(self, kind, text):
        self.log.append({"t": round(time.time() - self.t0), "kind": kind, "text": text})

    def _out(self, say, **extra):
        say = [s for s in say if s]
        for s in say:
            self.transcript.append({"t": round(time.time() - self.t0), "who": "dispatch", "text": s})
        n = self.node or {}
        return {
            "say": say,
            "protocol": self.proto["id"] if self.proto else None,
            "title": self.proto["title"] if self.proto else None,
            "node": n.get("id"),
            "kind": n.get("kind", "triage"),
            "step": n.get(n.get("kind", ""), None),
            "expects": {"look": "camera", "do": "ack", "ask": "answer"}.get(n.get("kind"), "speech"),
            "choices": {k: m for k, (m, _) in options(n).items()} if n else {},
            "timer": n.get("timer"),
            "metronome": n.get("metronome"),
            "log": self.log,
            **extra,
        }

    def _enter(self, target, say, text=""):
        """Move to a node (or '@protocol'), speak it, and chain through 'then' steps."""
        for _ in range(16):
            if target.startswith("@"):
                self.proto = self.protocols[target[1:]]
                self._log("protocol", self.proto["title"])
                target = self.proto["start"]
                for e in self.proto.get("entries", []):   # plain words can pick the starting branch
                    if text and keyword_hit(text, e["words"]):
                        target = e["goto"]
                        break
                self._prefill(text)
            self.node = self.proto["nodes"][target]
            n, kind = self.node, self.node["kind"]
            self.visited.append(f"{self.proto['id']}.{n['id']}")
            got = self.known.pop((self.proto["id"], n["id"]), None)
            if kind == "ask" and got in options(n) and not n.get("noskip"):
                # The caller already told us. Do not make them answer it again: walk on.
                self._log("skip", f"{question_of(n)} -> {got if got in ('yes', 'no') else options(n)[got][0]} (already said, not asked)")
                self.history.append((self.proto["id"], n["id"]))
                self.skipped += 1
                target = options(n)[got][1]
                continue
            say.append(n[kind])
            if n.get("log") and (kind != "do" or n.get("then")):
                self._log("action", n["log"])
            if kind == "do" and n.get("then"):
                target = n["then"]
                continue
            if kind == "end":
                self._log("end", f"{self.proto['title']} steps complete")
                if self.pending:
                    say.pop()                      # skip the sign-off, there is more to do
                    item = self.pending.pop(0)
                    self.proto = self.protocols[item[1]]
                    if item[0] == "resume":
                        say.append(f"Now back to {self.proto['short']}.")
                        target = item[2]
                    else:
                        say.append(NEXT_PROBLEM)
                        target = "@" + item[1]
                    continue
            return
        raise RuntimeError("protocol loops without waiting for the caller")

    def _switch(self, pid, say, text):
        """A more urgent problem interrupts. Remember where we were so we can come back."""
        if self.node and self.node["kind"] != "end":
            self.pending.insert(0, ("resume", self.proto["id"], self.node["id"]))
        self.pending = [p for p in self.pending if not (p[0] == "start" and p[1] == pid)]
        self._log("caller", f"New problem reported: {text}")
        say.append(FIRST)
        self.misses = 0
        self._enter("@" + pid, say, text)

    def _prefill(self, text):
        """Skip-ahead: one model call reads what the caller said and answers the questions still to come."""
        if not text or len(text.split()) < 6 or not hasattr(self.llm, "prefill"):
            return
        pid, nodes = self.proto["id"], self.proto["nodes"]
        seen = {v.split(".", 1)[1] for v in self.visited if v.startswith(pid + ".")}
        # Only the next few questions on the road ahead, nearest first.
        start = self.node["id"] if self.node and self.node["id"] in nodes and self.proto["nodes"].get(self.node["id"]) is self.node else self.proto["start"]
        order, queue, done = [], [start], set()
        while queue and len(order) < 6:
            nid = queue.pop(0)
            if nid in done or nid.startswith("@") or nid not in nodes:
                continue
            done.add(nid)
            n = nodes[nid]
            if n["kind"] == "ask" and nid not in seen and not LATER.search(question_of(n)) \
                    and not (n.get("steady") or n.get("quiet") or n.get("metronome") or n.get("noskip")):
                order.append(nid)
            queue += targets(n)
        items = [(nid, question_of(nodes[nid]), {k: m for k, (m, _) in options(nodes[nid]).items()}) for nid in order]
        try:
            for nid, key in self.llm.prefill(items, text).items():
                self.known[(pid, nid)] = key
        except Exception:
            pass

    def _remind(self, n):
        """Short form of the current step, for after a side answer or an unclear reply."""
        kind = n["kind"]
        if kind in ("ask", "look"):
            return question_of(n)
        return sentences(n[kind])[0]

    # ---------- entry points
    def start(self):
        return self._out([GREETING])

    def hear(self, text: str):
        text = text.strip()
        self.transcript.append({"t": round(time.time() - self.t0), "who": "caller", "text": text})
        say = []
        if not text or FILLER.match(text):
            return self._out([])
        if self.log and REPORT.search(text) and len(text.split()) <= 8:
            return self._out(["Here is the handoff note."], report=True)
        if WAIT.match(text):
            return self._out([HERE])
        if self.node and REPEAT.search(text):
            return self._out([self.node[self.node["kind"]]])
        if self.node and BACK.search(text) and len(text.split()) <= 8:
            if self.history:
                pid, nid = self.history.pop()
                self.proto, self.node = self.protocols[pid], self.protocols[pid]["nodes"][nid]
                self.misses = 0
                self._log("caller", "Went back one step")
            return self._out([self.node[self.node["kind"]]])
        if self.node and CAMERA.search(text) and self.node["kind"] != "look":
            return self._out(["Hold it up to the camera and keep still."], capture="describe")

        # 1. No protocol yet, or the last one ended: work out what the emergency is.
        if self.proto is None or self.node["kind"] == "end":
            ended = self.proto is not None
            self.triage_text = text if ended else (self.triage_text + " " + text).strip()
            pids = self._triage(self.triage_text)
            if not pids:
                self.misses += 1
                if ended:
                    said, self.idle_said = self.idle_said, True
                    return self._out([] if said else [STILL_HERE])
                if self.misses == 1:
                    return self._out([ASK_WHAT])
                return self._out([WHICH])
            self.misses, self.idle_said = 0, False
            self._log("caller", self.triage_text)
            self.pending += [("start", p) for p in pids[1:]]
            self._enter("@" + pids[0], say, self.triage_text)
            self.triage_text = ""
            return self._out(say)

        n, kind = self.node, self.node["kind"]
        # 2. Alarm phrases jump straight to a more urgent protocol. No model needed.
        for p in self.protocols.values():
            if p["id"] != self.proto["id"] and p.get("priority", 5) <= self.proto.get("priority", 5) \
                    and any(re.search(a, text, re.I) for a in p.get("alarms", [])):
                self._switch(p["id"], say, text)
                return self._out(say)

        w = n.get("worse")            # a mild branch that just turned serious
        if w and re.search(w["pattern"], text, re.I) and not NEG.search(" ".join(text.split()[:3])):
            self._log("caller", f"Got worse: {text}")
            self.history.append((self.proto["id"], n["id"]))
            self._enter(w["goto"], say)
            return self._out(say)

        opts = options(n)
        if kind == "do":
            opts = {"done": ("the caller did it, is doing it, or is ready for the next step", n.get("next") or n.get("then"))}
            if n.get("cant"):
                opts["cant"] = ("the caller cannot do it or does not have what is needed", n["cant"])

        # 3. What does the reply mean? Rules first, the model only when rules cannot tell.
        short = len(text.split()) <= 8
        if kind != "do" and short and UNSURE.search(text):
            key = "unsure"
        elif kind == "do" and "cant" in opts and CANT.search(text) and not ASKING.search(text):
            key = "cant"
        else:
            key = self._fast(text, kind, opts) or self._keywords(text, n) \
                or self.llm.interpret(question_of(n) if kind != "do" else n[kind], {k: m for k, (m, _) in opts.items()}, text)

        if key in opts:
            self.misses = 0
            self.history.append((self.proto["id"], n["id"]))
            if len(text.split()) >= 12:
                self._prefill(text)
            if kind == "do":
                if key == "done" and n.get("log"):
                    self._log("action", n["log"])
            elif not n.get("quiet"):
                self._log("answer", f"{question_of(n)} -> {key if key in ('yes', 'no') else opts[key][0]}")
            self._enter(opts[key][1], say)
            return self._out(say)

        if key == "other" and n.get("steady"):
            key = "question"          # mid-CPR nothing gets queued; alarms already had their chance
        if key == "other":
            pids = self._triage(text)
            pid = pids[0] if pids else None
            if pid and pid != self.proto["id"] and pid != "general":
                if self.protocols[pid].get("priority", 5) <= self.proto.get("priority", 5):
                    self._switch(pid, say, text)
                    return self._out(say)
                if ("start", pid) not in self.pending:
                    self.pending.append(("start", pid))
                    self._log("caller", f"Also reported: {text}")
                return self._out([NOTED, self._remind(n)])
            key = "question"

        if key == "question":
            ans = self.llm.answer(text, self.proto.get("facts", []), n[kind])
            if ans != NO_FACT:
                self._log("question", f"{text} -> {ans}")
                return self._out([ans, self._remind(n)])
            if self.misses == 0 and not n.get("steady"):
                self.misses = 1
                return self._out([NO_FACT, self._remind(n)])
            key = "unclear"

        # 4. "I don't know", or two unclear replies in a row: take the safe path.
        self.misses += 1
        if n.get("steady"):           # mid-CPR: no questions, no menus, just keep them going
            self.misses = 0
            self._enter(n["unsure"], say)
            return self._out(say)
        if kind != "do" and (key == "unsure" or self.misses >= 2) and n.get("unsure"):
            self.misses = 0
            self.history.append((self.proto["id"], n["id"]))
            self._log("answer", f"{question_of(n)} -> not sure, treated as the worse case")
            say.append(PLAY_SAFE)
            self._enter(n["unsure"], say)
            return self._out(say)
        if kind == "do":
            if self.misses >= 2:      # never loop: move on and say so in the log
                self.misses = 0
                self.history.append((self.proto["id"], n["id"]))
                self._log("action", f"{n.get('log') or sentences(n['do'])[0]} (not confirmed)")
                self._enter(opts["done"][1], say)
                return self._out(say)
            return self._out([sentences(n["do"])[0], DONE_OR_CANT])
        if set(opts) == {"yes", "no"}:
            return self._out(["Yes or no. " + question_of(n)])
        if kind == "look":
            return self._out([n[kind]], capture="look")
        return self._out([n.get("pick") or "Pick one: " + ", or ".join(m for m, _ in opts.values()) + "."])

    def see(self, image_b64: str, mode: str = "look"):
        """A camera frame arrived. mode='look' answers the current look-node; 'describe' is caller-initiated."""
        n = self.node
        if mode == "look" and n and n["kind"] == "look":
            opts = options(n)
            key = self.llm.look(image_b64, n.get("question", n["look"]), {k: m for k, (m, _) in opts.items()})
            if key in opts:
                self._log("camera", f"{n.get('question', n['look'])} -> {opts[key][0]}")
                self.history.append((self.proto["id"], n["id"]))
                say = [f"I see it: {opts[key][0]}."]
                self._enter(opts[key][1], say)
                return self._out(say)
            return self._out(["I can't make it out. Tell me instead. " + n.get("question", n["look"])])
        desc = self.llm.describe(image_b64)
        self._log("camera", desc)
        return self._out([f"I see: {desc}", n[n["kind"]] if n else "Tell me what is happening."])

    def _triage(self, text):
        """Protocol ids, most urgent first. Works with models that only offer triage()."""
        if hasattr(self.llm, "triage_all"):
            return self.llm.triage_all(text, self.protocols)
        pid = self.llm.triage(text, self.protocols)
        return [pid] if pid else []

    @staticmethod
    def _fast(text, kind, opts):
        """Plain yes/no/done never needs the model. Right-sized inference."""
        words = len(text.split())
        if words > 12 or (words > 5 and re.search(r"\b(but|wait|actually|except|or)\b|\?", text, re.I)):
            return None
        rest = " ".join(text.split()[1:])
        if kind in ("ask", "look") and "yes" in opts and YES.search(text) \
                and not re.search(r"\b(no|not|nope|nah|never|i mean)\b|n'?t\b", rest, re.I):
            return "yes"
        if kind in ("ask", "look") and "no" in opts and NO.search(text) \
                and not re.search(r"\b(yes|yeah|yep|i mean)\b", rest, re.I):
            return "no"
        if kind == "do" and DONE.search(text):
            return "done"
        return None

    @staticmethod
    def _keywords(text, n):
        """Choices can list plain words. If exactly one choice matches, no model call is needed."""
        hits = [k for k, v in (n.get("choices") or {}).items() if keyword_hit(text, v.get("words", []))]
        hits += [k for k in ("yes", "no") if k in n and keyword_hit(text, n.get(k + "_words", []))]
        return hits[0] if len(hits) == 1 else None

    def report(self):
        mm = lambda s: f"{s // 60:02d}:{s % 60:02d}"
        lines = [f"[{mm(e['t'])}] {e['kind'].upper()}: {e['text']}" for e in self.log]
        return {
            "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(self.t0)),
            "duration": mm(round(time.time() - self.t0)),
            "protocols": [e["text"] for e in self.log if e["kind"] == "protocol"],
            "timeline": lines,
            "summary": self.llm.summary(lines) if lines else "",
        }


def fixed_lines(protocols):
    """Every sentence the dispatcher can say that is known ahead of time. Feeds app/speech/lines.txt."""
    L = []
    for p in protocols.values():
        for n in p["nodes"].values():
            kind = n["kind"]
            L.append(n[kind])
            if kind in ("ask", "look"):
                o = options(n)
                L.append(question_of(n))
                if set(o) == {"yes", "no"}:
                    L.append("Yes or no. " + question_of(n))
                else:
                    L.append(n.get("pick") or "Pick one: " + ", or ".join(m for m, _ in o.values()) + ".")
            if kind == "look":
                L.append("I can't make it out. Tell me instead. " + n.get("question", n["look"]))
                L += [f"I see it: {m}." for m, _ in options(n).values()]
        L += p.get("facts", [])
        L.append(f"Now back to {p['short']}.")
    L += [WHICH, GREETING, ASK_WHAT, STILL_HERE, PLAY_SAFE, FIRST, NOTED, NEXT_PROBLEM, HERE, NO_FACT, DONE_OR_CANT,
          "Hold it up to the camera and keep still.",
          "Here is the handoff note.", "Time is up.", "Tell me what is happening."]
    return list(dict.fromkeys(L))
