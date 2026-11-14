"""Holdover brain: one local web server that ties the tree engine, Bonsai, the speech
helper and the screen together. Nothing here talks to the internet."""
import asyncio, json, os, pathlib, re, sys, time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain"))
from engine import load_protocols, validate, Session
from llm import Bonsai

PROTOCOLS = load_protocols(ROOT / "protocols")
problems = validate(PROTOCOLS)
assert not problems, problems
llm = Bonsai()
app = FastAPI()

ADVICE = re.compile(r"\b(press|apply|put|give|take|call|raise|elevate|wrap|tie|cool|pour|remove|lay|sit|roll|push|use|try|should|need to|make sure|don'?t|do not|will be (ok|fine|alright)|going to be)\b", re.I)
PLAIN = re.compile(r"^(yes|yeah|yep|no|nope|nah|ok|okay|done|got it|ready|next)\b", re.I)


def acknowledge(text):
    """One short human line so the caller knows they were heard. Never advice: anything that
    smells like an instruction is thrown away and the scripted step speaks alone."""
    try:
        out = llm.chat("ack",
            "You are a calm, steady emergency dispatcher on a call. Reply with ONE short sentence, at most 9 words, "
            "that shows you heard the caller, naming the key detail they said. Sound like a person, not a form. "
            "Never start with 'I hear' or 'I understand'. Vary your wording. "
            "Examples of the style: 'Okay. A hatchet cut to the leg. I am with you.' / 'Got it, a snake bite on the ankle.' / "
            "'Okay, boiling water on your hand. Stay with me.' "
            "No advice. No instructions. No questions. No promises about how it will turn out.",
            f"Caller: \"{text}\"\n\nYour one short sentence:", max_tokens=22, temperature=0.5)
    except Exception:
        return None
    out = out.strip().strip('"\'').split("\n")[0]
    if not out or "?" in out or len(out.split()) > 13 or ADVICE.search(out.replace("Stay with me", "")):
        return None
    return out if out[-1] in ".!" else out + "."


def glance(image_b64):
    """Idle look through the camera. Returns a short phrase, or None when there is nothing to note."""
    try:
        out = llm.chat("glance",
            "You are the eyes of an emergency first-aid line, looking through a laptop camera. "
            "If you can CLEARLY see an injury or emergency (blood, an open wound, a burn, heavy swelling, a bent limb, "
            "a person lying motionless, a bite mark, a bandage or tourniquet, an injector pen), describe it in at most "
            "8 words. If it is just a person, a room, or you are not sure, reply exactly: NONE",
            "What do you see?", max_tokens=18, image_b64=image_b64, temperature=0.0)
    except Exception:
        return None
    out = out.strip().split("\n")[0].strip('". ')
    if not out or "NONE" in out.upper() or len(out.split()) > 12:
        return None
    return out


NOISE = re.compile(r"^\W*(thank you|thanks for watching|you|bye|okay\.?|\[.*\]|\(.*\))?\W*$", re.I)


class Hub:
    def __init__(self):
        self.session = Session(PROTOCOLS, llm)
        self.uis: set[WebSocket] = set()
        self.speech: WebSocket | None = None
        self.speech_info = {}
        self.lock = asyncio.Lock()
        self.speak_id = 0
        self.last_node = None
        self.started = False
        self.turns = self.model_turns = 0
        self.clock = {}          # seconds from call start: arrest recognized, first compression
        self.voices = set()
        self.scene = []          # last few things the idle camera noticed
        self.glancing = False
        self.scene_said = False

    async def ui(self, msg):
        dead = []
        for ws in self.uis:
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.uis.discard(ws)

    async def speak(self, text):
        self.speak_id += 1
        if self.speech:
            try:
                await self.speech.send_text(json.dumps({"type": "speak", "text": text, "id": self.speak_id}))
            except Exception:
                self.speech = None

    async def emit(self, out):
        capture = out.pop("capture", None)
        if not capture and out["kind"] == "look" and out["node"] != self.last_node:
            capture = "look"
        self.last_node = out["node"]
        calls = llm.calls[-6:]
        now = round(time.time() - self.session.t0)
        if out["protocol"] == "unresponsive":
            self.clock.setdefault("recognized", now)
        if out.get("metronome"):
            self.clock.setdefault("recognized", now)
            self.clock.setdefault("compression", now)
        out["clock"] = self.clock
        out["turns"], out["model_turns"] = self.turns, self.model_turns
        await self.ui({"type": "state", **out, "capture": capture, "meter": calls,
                       "model": (llm.model or "").split("/")[-1], "speech": self.speech_info if self.speech else None})
        for s in out["say"]:
            await self.speak(s)
        if out.get("report"):
            await self.report()

    async def start(self):
        async with self.lock:
            if self.speech:
                await self.speech.send_text(json.dumps({"type": "stop"}))
            self.session = Session(PROTOCOLS, llm)
            self.last_node = None
            self.started = True
            self.scene, self.scene_said = [], False
            self.turns = self.model_turns = 0
            self.clock, self.voices = {}, set()
            await self.emit(self.session.start())

    async def hear(self, text, speaker=None):
        if not self.started or NOISE.match(text or ""):
            return
        await self.ui({"type": "heard", "text": text, "speaker": speaker})
        async with self.lock:
            t = time.time()
            n_calls = len(llm.calls)
            if speaker is not None and speaker not in self.voices:
                self.voices.add(speaker)
                if len(self.voices) == 2:
                    self.session._log("voice", "A second voice joined the call (Argmax speaker labels)")
            node = self.session.node or {}
            want_ack = (len(text.split()) >= 6 and not text.rstrip().endswith("?") and not PLAIN.match(text.strip())
                        and not node.get("metronome"))
            ack_task = asyncio.create_task(asyncio.to_thread(acknowledge, text)) if want_ack else None
            hear_task = asyncio.create_task(asyncio.to_thread(self.session.hear, text))
            if ack_task:
                ack = await ack_task
                if ack and not hear_task.done():      # fill the thinking gap with a human line
                    await self.ui({"type": "note", "text": ack, "ack": True})
                    await self.speak(ack)
            out = await hear_task
            out["think_ms"] = round((time.time() - t) * 1000)
            self.turns += 1
            if any(c[0] in ("triage", "interpret", "answer") for c in llm.calls[n_calls:]):
                self.model_turns += 1
            await self.emit(out)

    async def ambient(self, image):
        """Spare compute: while nobody is talking, look through the camera and remember what is there."""
        if self.glancing or self.lock.locked() or not self.started:
            return
        self.glancing = True
        try:
            t = time.time()
            seen = await asyncio.to_thread(glance, image)
            ms = round((time.time() - t) * 1000)
            prev = self.scene[-1] if self.scene else None
            self.scene = (self.scene + [seen])[-3:]
            await self.ui({"type": "scene", "text": seen, "ms": ms})
            if seen and prev and not self.lock.locked():
                if not any(e["kind"] == "camera" and e["text"].startswith("Noticed") for e in self.session.log[-3:]):
                    self.session._log("camera", f"Noticed on camera: {seen}")
                if self.session.proto is None and not self.scene_said:
                    self.scene_said = True
                    line = f"I can see {seen[0].lower() + seen[1:]}. Tell me what happened."
                    await self.ui({"type": "note", "text": line})
                    await self.speak(line)
        finally:
            self.glancing = False

    async def see(self, image, mode):
        async with self.lock:
            t = time.time()
            out = await asyncio.to_thread(self.session.see, image, mode or "look")
            out["think_ms"] = round((time.time() - t) * 1000)
            await self.emit(out)

    async def report(self):
        rep = await asyncio.to_thread(self.session.report)
        folder = ROOT / "reports"
        folder.mkdir(exist_ok=True)
        name = time.strftime("handoff-%Y%m%d-%H%M%S.md")
        md = [f"# Handoff note", f"Started {rep['started']} · on the line {rep['duration']}", "",
              rep["summary"], "", "## Timeline", *[f"- {l}" for l in rep["timeline"]], "",
              "_Generated on-device by Holdover. First aid guidance only, not a diagnosis._"]
        (folder / name).write_text("\n".join(md))
        await self.ui({"type": "report", **rep, "file": f"reports/{name}"})


hub = Hub()


@app.on_event("startup")
async def warm():
    def go():
        try:
            llm.chat("warmup", "Reply OK.", "OK?", max_tokens=2)
        except Exception as e:
            print("Bonsai is not up yet:", e)
    asyncio.get_event_loop().run_in_executor(None, go)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/protocols")
def protocols():
    return [{"id": p["id"], "title": p["title"], "steps": len(p["nodes"])} for p in PROTOCOLS.values()]


@app.websocket("/ui")
async def ui_ws(ws: WebSocket):
    await ws.accept()
    hub.uis.add(ws)
    await ws.send_text(json.dumps({"type": "hello", "speech": hub.speech_info if hub.speech else None,
                                   "protocols": [p["title"] for p in PROTOCOLS.values()]}))
    try:
        while True:
            m = json.loads(await ws.receive_text())
            if m["type"] == "start":
                asyncio.create_task(hub.start())
            elif m["type"] == "text":
                asyncio.create_task(hub.hear(m["text"]))
            elif m["type"] == "frame" and m.get("mode") == "ambient":
                asyncio.create_task(hub.ambient(m["image"]))
            elif m["type"] == "frame":
                asyncio.create_task(hub.see(m["image"], m.get("mode")))
            elif m["type"] == "timer":
                await hub.ui({"type": "note", "text": "Time is up."})
                await hub.speak("Time is up.")
            elif m["type"] == "report":
                asyncio.create_task(hub.report())
            elif m["type"] == "stop" and hub.speech:
                await hub.speech.send_text(json.dumps({"type": "stop"}))
    except WebSocketDisconnect:
        hub.uis.discard(ws)


@app.websocket("/speech")
async def speech_ws(ws: WebSocket):
    await ws.accept()
    hub.speech = ws
    try:
        while True:
            m = json.loads(await ws.receive_text())
            t = m.get("type")
            if t == "ready":
                hub.speech_info = {k: m.get(k) for k in ("stt", "tts", "pro")}
                await hub.ui({"type": "speech", "speech": hub.speech_info})
            elif t == "utterance":
                asyncio.create_task(hub.hear(m.get("text", ""), m.get("speaker")))
            elif t in ("hypothesis", "level", "speaking", "spoken"):
                await hub.ui(m)
    except WebSocketDisconnect:
        pass
    finally:
        if hub.speech is ws:
            hub.speech = None
            await hub.ui({"type": "speech", "speech": None})
