# Holdover

**The emergency line that works when there is no signal.**

You are hiking, fishing or camping. Someone is hurt. There are no bars. Holdover is a 911-style dispatcher
that runs entirely on a Mac with WiFi off. You talk, it talks back, hands-free. It asks short blunt questions,
walks you through first aid one step at a time, looks at the injury through the camera, switches protocols
the moment things get worse, and writes a handoff note for the paramedics when help arrives.

Built at the Cornell Local AI Hackathon (theme: Recreate) by Dylan Gutowski and Rithik.

> First aid guidance only. Not a medical device, not a diagnosis, and not a replacement for emergency services.

## What it recreates

A 911 dispatcher. Real dispatchers do not improvise: they follow scripted protocols and their skill is hearing a
panicked caller and picking the right branch. That is exactly the job a small local model can do well.

## How it uses Bonsai (PrismML) and Argmax

| Job | What does it | On device |
|---|---|---|
| Hear the caller, live | **Argmax WhisperKit** | yes |
| Speak every instruction | **Argmax TTSKit** (Qwen3-TTS) | yes |
| Tell the caller from a second voice | **Argmax Pro SDK** (SpeakerKit + Sortformer, real time) | yes |
| Work out which emergency this is | **Bonsai 2 9B** (ternary, 1.58 bits per weight) | yes |
| Understand messy, indirect answers | **Bonsai 2 9B** | yes |
| Look at the injury on camera | **Bonsai 2 9B** vision | yes |
| Watch the camera while nobody is talking | **Bonsai 2 9B** vision, on spare compute | yes |
| Answer side questions | **Bonsai 2 9B**, only from that protocol's own notes | yes |
| Write the handoff note | **Bonsai 2 9B**, only from the logged timeline | yes |

**The model never writes treatment.** Every word of medical advice is pre-written in plain-text protocol trees
(`protocols/*.yaml`, 10 protocols, 155 steps). Bonsai only steers. A plain "yes", "no" or "done" never touches
the model at all, and alarm phrases like "he's not breathing" jump straight to CPR by rule. The screen shows
what share of replies needed no model: right-sized inference, measured.

## Run it

Needs an Apple Silicon Mac, Python 3.10+, Swift (Command Line Tools are enough) and an Argmax key for speaker labels.

```bash
./setup.sh      # once, online: Python env, Bonsai 9B, Argmax voice helper, voice clips
./run.sh        # every time, WiFi off: starts Bonsai, the brain, the voice, and opens the call screen
```

Put `ARGMAX_API_KEY=ax_...` in a `.env` file next to `run.sh` (never committed). Without it everything works
except speaker labels.

## How it is built

```
web/index.html      the call screen (camera, captions, current step, timer, CPR beat, timeline)
brain/server.py     local server on 127.0.0.1:8770, ties everything together
brain/engine.py     the protocol tree engine: branching, alarms, queueing two problems, fail-safe paths
brain/llm.py        Bonsai wrapper (triage, interpret, answer-from-notes, look, summary)
protocols/*.yaml    the first aid trees, editable in plain text
speech/             Swift helper: Argmax ears, voice and speaker labels over a local WebSocket
tests/              backtest of panicked phrases, 66 full scripted calls, held-out phrases
```

## Tested

- 66 of 66 full scripted calls (mild and severe, off-script replies, two problems at once, things we have no protocol for).
- 123 of 123 tuning phrases; 95 of 106 on held-out phrases never used for tuning. Most misses come back
  "unclear", which re-asks once and then takes the worse-case branch.

```bash
.venv/bin/python tests/scenarios.py
.venv/bin/python tests/backtest.py --heldout
```
