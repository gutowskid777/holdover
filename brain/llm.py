"""Bonsai wrapper. Talks to the local mlx-vlm server (OpenAI-compatible) on localhost.
Every call is a tiny classification or a one-line answer. No call ever writes treatment."""
from __future__ import annotations
import os, re, time, httpx

URL = os.environ.get("BONSAI_URL", "http://127.0.0.1:8080/v1")
SAMPLING = {"temperature": 0.2, "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}


class Bonsai:
    def __init__(self, url: str = URL):
        self.url = url
        self.http = httpx.Client(timeout=60)
        self.model = None
        self.calls = []  # (name, seconds) for the on-screen "local compute" meter

    def _model(self):
        if not self.model:
            data = self.http.get(self.url + "/models").json()["data"]
            loaded = [m for m in data if m.get("loaded")] or data
            self.model = loaded[0]["id"]
        return self.model

    def chat(self, name, system, user, max_tokens=12, image_b64=None, **over):
        content = user
        if image_b64:
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}}]
        body = {"model": self._model(), "max_tokens": max_tokens, **SAMPLING, **over,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        t = time.time()
        r = self.http.post(self.url + "/chat/completions", json=body)
        r.raise_for_status()
        out = (r.json()["choices"][0]["message"]["content"] or "").strip()
        self.calls.append((name, round(time.time() - t, 2)))
        return out

    # ---------- the five jobs

    def triage_all(self, text, protocols):
        """Protocol ids that match, most life-threatening first (at most two). Empty = no emergency described yet."""
        ids = list(protocols)
        menu = "\n".join(f"{i + 1}. {p['title']}: {p['when']}" for i, p in enumerate(protocols.values()))
        out = self.chat("triage",
            "You route calls on an emergency first-aid line. Pick the protocol that matches what the caller describes. "
            "Usually there is ONE. Only if the caller clearly describes two separate problems, give both numbers, "
            "the most life-threatening first, like: 3, 5. "
            "Answer 0 if the caller has not described an injury or illness yet (a greeting, a test, a joke, a question "
            "about you), or if it is too vague to tell what kind of injury it is. Answer with numbers only.",
            f"Protocols:\n{menu}\n0. No injury or illness described yet, or too vague to tell\n\n"
            f"Caller: \"{text}\"\n\nNumber:", max_tokens=6)
        picks = []
        for m in re.findall(r"\d+", out):
            i = int(m)
            if 1 <= i <= len(ids) and ids[i - 1] not in picks:
                picks.append(ids[i - 1])
        picks = picks[:2]
        if len(picks) == 2 and "general" in picks:   # the catch-all never rides along with a real protocol
            picks = [x for x in picks if x != "general"]
        picks.sort(key=lambda x: protocols[x].get("priority", 5))
        return picks

    def triage(self, text, protocols):
        picks = self.triage_all(text, protocols)
        return picks[0] if picks else None

    def open(self, text, turns=()):
        """Open mode: no checked protocol covers this, so Bonsai talks it through from general first-aid
        knowledge. Always labelled as such on screen and in the handoff note."""
        system = ("You are Holdover, a calm first-aid dispatcher that runs fully offline. No checked protocol covers "
                  "this topic, so you are answering from general first-aid knowledge. Rules: at most 3 short "
                  "sentences in plain spoken words. Give only basic, widely accepted first aid. Never diagnose: "
                  "you may say what it could be, as a possibility. Always say which warning signs mean they need a "
                  "doctor or emergency care. If one missing detail would change the advice, ask ONE short open question "
                  "instead of advising. If the caller is the hurt person say 'you'; for someone else say he, she or "
                  "the name the caller uses, never 'they' or 'them'. No lists, no symbols, no abbreviations.")
        msgs = [{"role": "system", "content": system}]
        for who, said in list(turns)[-6:]:
            msgs.append({"role": "user" if who == "caller" else "assistant", "content": said})
        msgs.append({"role": "user", "content": text})
        body = {"model": self._model(), "max_tokens": 110, **SAMPLING, "temperature": 0.4, "messages": msgs}
        t = time.time()
        r = self.http.post(self.url + "/chat/completions", json=body)
        r.raise_for_status()
        out = (r.json()["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")
        self.calls.append(("open", round(time.time() - t, 2)))
        parts = re.split(r"(?<=[.!?])\s+", out)
        return " ".join(parts[:3]).replace("*", "")

    def voice(self, turns, step, unclear=False):
        """Say the next checked step like a person would, fitted to what the caller just said.
        The step text is the only medical content allowed in the reply."""
        convo = "\n".join(("Caller: " if w == "caller" else "Holdover: ") + t for w, t in turns[-6:])
        system = (
            "You are Holdover, a calm first-aid dispatcher on a voice call. You get the conversation so far and the "
            "CHECKED STEP that must be said next. Say that step in your own spoken words so it fits the conversation.\n"
            "Rules:\n"
            "- Open with a few words that reflect what the caller just told you (never 'I heard you said').\n"
            "- Keep every instruction and warning from the checked step. Add no other medical advice or reassurance.\n"
            "- A question becomes ONE open, natural question. Never say 'yes or no'. Never ask what they already told you.\n"
            "- The caller who is hurt is 'you'. Someone else is he, she, or a name. Never 'they', 'them', 'the patient'.\n"
            "- One to three short sentences. Plain spoken words.\n\n"
            "Example 1\nCaller: my buddy cut his leg with a hatchet theres blood everywhere\n"
            "CHECKED STEP: Press hard on the wound right now. Use a cloth, a shirt, or your bare hand. Say done when you are pressing.\n"
            "Reply: Okay, a hatchet cut to his leg. Press hard on the wound right now with a cloth, a shirt, or your bare hand. Tell me when you are pressing.\n\n"
            "Example 2\nCaller: i got into a fight last week and my arm is bruised\n"
            "CHECKED STEP: Are they on blood thinners, or do they bruise or bleed very easily?\n"
            "Reply: Got it, a week old bruise from a fight. Are you on any blood thinners, or do you tend to bruise or bleed easily?\n\n"
            "Example 3\nCaller: she grabbed the hot pan\nHoldover: How big is the burn?\nCaller: its her whole palm\n"
            "CHECKED STEP: Show me the burn. Hold it up to the camera.\n"
            "Reply: Her whole palm, okay. Hold her hand up to the camera so I can see the burn.")
        note = "\n(The caller's last answer did not settle this step. Ask it again in a different, simpler way.)" if unclear else ""
        user = f"{convo}\nCHECKED STEP: {step}{note}\nReply:"
        out = self.chat("voice", system, user, max_tokens=90, temperature=0.4, presence_penalty=0.2)
        return out.replace("\n", " ").replace("*", "").strip().strip('"')

    def converse(self, system, turns, max_tokens=110):
        """One spoken reply in a real conversation. `system` carries the style rules and the checked protocol."""
        msgs = [{"role": "system", "content": system}]
        for who, said in turns:
            role = "user" if who == "caller" else "assistant"
            if msgs[-1]["role"] == role:                 # the chat template wants strict turns
                msgs[-1]["content"] += " " + said
            else:
                msgs.append({"role": role, "content": said})
        if msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": "(the caller is waiting)"})
        body = {"model": self._model(), "max_tokens": max_tokens, **SAMPLING, "temperature": 0.4,
                "presence_penalty": 0.3, "messages": msgs}
        t = time.time()
        r = self.http.post(self.url + "/chat/completions", json=body)
        r.raise_for_status()
        j = r.json()
        out = (j["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")
        self.calls.append(("talk", round(time.time() - t, 2)))
        self.last_usage = j.get("usage", {})
        return out

    def prefill(self, items, text):
        """Skip-ahead. items = [(id, question, {label: meaning})]. Returns {id: label} ONLY for questions
        the caller has already clearly answered in what they said. Everything else stays unasked-and-unknown."""
        if not items:
            return {}
        qs = []
        for i, (_, q, opts) in enumerate(items, 1):
            if set(opts) == {"yes", "no"}:
                labels = "YES or NO"
            else:
                labels = " / ".join(f"{k.upper()} ({m})" for k, m in opts.items())
            qs.append(f'{i}. "{q}"  [{labels}]')
        out = self.chat("skip-ahead",
            "You are the ears of an emergency first-aid line. The caller has described what is happening. Below are "
            "questions the dispatcher would ask next, each with its allowed labels in brackets. List ONLY the "
            "questions the caller has ALREADY answered in what they said. One per line, in the form "
            "'number: LABEL', using exactly one of that question's labels in capitals, nothing else. "
            "Example: if the caller said 'its gushing and nothing is in the wound', and question 1 is "
            "'Is the blood pouring?' and question 2 is 'Is something stuck in the wound?', you reply:\n1: YES\n2: NO\n"
            "Use what the caller actually said. A clear implication counts: 'I got into a fight' answers 'did it come "
            "from a hit' with YES. If the caller is talking about their own injury ('my arm', 'I fell'), then the hurt "
            "person is awake, talking and breathing. Do not invent facts. Leave out questions the caller did not answer. "
            "If none were answered, reply NONE.",
            f"Caller said: \"{text}\"\n\nQuestions:\n" + "\n".join(qs) + "\n\nAnswered:",
            max_tokens=6 * len(items) + 4, temperature=0.0)
        found = {}
        for line in out.splitlines():
            m = re.match(r"\s*(\d+)\s*[:.)]\s*(.+)", line)
            if not m:
                continue
            i = int(m.group(1)) - 1
            if not 0 <= i < len(items):
                continue
            nid, _, opts = items[i]
            tail = m.group(2).upper()
            if "UNKNOWN" in tail:
                continue
            hits = [k for k in opts if re.search(rf"\b{re.escape(k.upper())}\b", tail)]
            if len(hits) == 1:
                found[nid] = hits[0]
        return found

    def interpret(self, question, opts, text):
        keys = list(opts)
        menu = "\n".join(f"{k.upper()} = {opts[k]}" for k in keys)
        out = self.chat("interpret",
            "You are the ears of an emergency first-aid line. The dispatcher just said something to the caller. "
            "Decide what the caller's reply means. Callers are panicked and answer indirectly: they describe what "
            "they see instead of saying yes or no. If the reply answers the dispatcher, even indirectly, pick that "
            "meaning. But never guess or assume: if the reply is about something else and does not tell you the "
            "answer, the label is UNCLEAR. Write one short reason (under 15 words), then on a new line 'Answer:' and ONE label.",
            f"Dispatcher said: \"{question}\"\n\nLabels:\n{menu}\n"
            "QUESTION = the caller asks a question, asks what something means, or asks what to use or do\n"
            "NEW = the reply does not answer the dispatcher and instead reports a different, worse emergency "
            "that the dispatcher did not ask about\n"
            "UNSURE = the caller says they do not know or cannot tell\n"
            "UNCLEAR = none of the above\n\n"
            f"Caller replied: \"{text}\"\n\nReason:", max_tokens=50)
        self.last_reason = out
        tail = out.split("Answer:")[-1] if "Answer:" in out else out.strip().split("\n")[-1]
        tail = tail.upper()
        for k in sorted(keys, key=len, reverse=True):
            if re.search(rf"\b{re.escape(k.upper())}\b", tail):
                return k
        if "QUESTION" in tail:
            return "question"
        if "NEW" in tail:
            return "other"
        if "UNSURE" in tail:
            return "unsure"
        return "unclear"

    def answer(self, question, facts, step):
        if not facts:
            return "I don't have that. Stay on this step."
        notes = "\n".join("- " + f for f in facts)
        out = self.chat("answer",
            "You are an emergency first-aid dispatcher. Answer the caller's question in ONE short sentence, "
            "using ONLY the notes below. Never add advice that is not in the notes. "
            "If the notes do not cover it, reply exactly: NOT COVERED",
            f"Notes:\n{notes}\n\nCurrent step: \"{step}\"\n\nCaller asks: \"{question}\"\n\nAnswer:", max_tokens=50)
        out = out.split("\n")[0].strip().strip('"')
        if not out or "NOT COVERED" in out.upper():
            return "I don't have that. Stay on this step."
        return out

    def look(self, image_b64, question, opts):
        keys = list(opts)
        letters = "ABCDEFGH"
        menu = "\n".join(f"{letters[i]}. {opts[k]}" for i, k in enumerate(keys))
        out = self.chat("look",
            "You are looking at a photo taken by someone giving first aid. Answer the question about the photo "
            "with ONE letter only. If the photo does not show it clearly, answer U.",
            f"Question: {question}\n{menu}\nU. Cannot tell from this photo\n\nLetter:", max_tokens=3, image_b64=image_b64)
        m = re.search(r"[A-HU]", out.upper())
        c = m.group() if m else "U"
        if c == "U":
            return "unclear"
        i = letters.index(c)
        return keys[i] if i < len(keys) else "unclear"

    def describe(self, image_b64):
        out = self.chat("describe",
            "You are looking at a photo taken during a first-aid emergency. In ONE short sentence, say only what is "
            "visible (body part, wound, swelling, color, objects). Give no advice and no diagnosis.",
            "What is visible?", max_tokens=40, image_b64=image_b64)
        return out.split("\n")[0].strip()

    def summary(self, lines):
        keep = [l for l in lines if "QUESTION:" not in l]
        out = self.chat("summary",
            "You write the handoff note a first responder reads on arrival. Using ONLY the timeline, write 2 short "
            "sentences: what happened, and what first aid was given. State nothing that is not in the timeline. "
            "No durations, no outcomes, no advice, no guesses about age or sex.",
            "Timeline:\n" + "\n".join(keep) + "\n\nHandoff note:", max_tokens=90, temperature=0.0)
        return out.strip()


class Scripted:
    """Stand-in for tests of the tree itself: no model, answers come from a list."""
    def __init__(self, triage=None, answers=()):
        self._triage, self._answers, self.calls = triage, list(answers), []

    def triage(self, text, protocols):
        return self._triage

    def interpret(self, question, opts, text):
        return self._answers.pop(0) if self._answers else "unclear"

    def answer(self, *a):
        return "I don't have that. Stay on this step."

    def look(self, *a):
        return "unclear"

    def describe(self, *a):
        return "nothing"

    def summary(self, lines):
        return ""
