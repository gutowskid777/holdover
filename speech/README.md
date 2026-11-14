# speechd: the app's ears and voice (Argmax, on device)

A small Swift program. It listens on the microphone, turns speech into text, speaks the
dispatcher's lines, and labels who is talking. It connects to the brain over a local WebSocket.
Nothing leaves the Mac.

| Job | Argmax piece | Model |
|---|---|---|
| Speech to text | WhisperKit | `openai_whisper-small.en` |
| Text to speech | TTSKit | Qwen3-TTS 0.6B, voice `ryan` |
| Live speaker labels | Pro SDK: SpeakerKitPro + Sortformer | `nemotron-3-diarization` |

## Build

One-time (already done on this Mac): log in to Argmax's package registry.

```bash
swift package-registry set --global --scope argmaxinc https://api.argmaxinc.com/v1/sdk
swift package-registry login https://api.argmaxinc.com/v1/sdk/login --token <axst_ token>
```

Then:

```bash
cd app/speech
swift build -c release        # Command Line Tools are enough, no Xcode
```

## Run

```bash
.build/release/speechd                                  # live: mic + voice, connects to ws://127.0.0.1:8770/speech
.build/release/speechd say "Press hard." --out a.wav    # text to speech into a file
.build/release/speechd transcribe a.wav                 # speech to text from a file
.build/release/speechd prerender lines.txt              # speak every line once into the cache (checked by listening back)
.build/release/speechd diarize two-people.wav           # Pro speaker labels on a file
.build/release/speechd --feed two-people.wav            # live mode with a file standing in for the mic
```

Options: `--url ws://...`, `--stt <model>`, `--voice ryan|aiden|serena|...`, `--rate 1.12`,
`--no-pro`, `--system-voice`, `--no-tts-cache`, `--models <dir>`.

The Pro runtime key is read from the `ARGMAX_API_KEY` environment variable, or from the project's
private `.env` file two folders up. It is never printed. Without a key, everything still works
except speaker labels.

## WebSocket messages

speechd is the client. It reconnects every second until the brain is up.

Sent by speechd:

```json
{"type":"ready","stt":"openai_whisper-small.en","tts":"qwen3-tts-0.6b/ryan","pro":true}
{"type":"hypothesis","text":"he fell off the"}          // live partial, about twice a second
{"type":"utterance","text":"He fell off the ledge.","speaker":0}   // one finished turn; speaker is null without Pro
{"type":"speaking","on":true}
{"type":"spoken","id":3,"cut":false}                    // cut = it was stopped early
{"type":"level","rms":0.4}                              // 10 per second, for a waveform
```

Received by speechd:

```json
{"type":"speak","text":"Press hard on the wound.","id":3}   // queued in order
{"type":"stop"}                                             // cut speech now and clear the queue
```

## How it behaves

- A turn ends after 0.7 seconds of quiet. The text arrives about 0.2 seconds later.
- While it is speaking, and for 0.3 seconds after, the microphone is ignored, so it never hears itself.
- Each sentence is cached as a wav under `models/tts-cache/<voice>/`. A cached sentence plays
  instantly. A new sentence is generated live (first sound in about 0.1 seconds) and then cached.
- `prerender` listens back to every clip with the speech-to-text model and retries up to 5 times,
  because the voice model sometimes garbles a line. Run it on every fixed dispatcher line.

## Offline

All models live in `app/speech/models/` (git-ignored, about 1.7 GB). When a model is on disk the
loader is pointed straight at it and makes no network call. The Pro license is checked online once,
then cached on this Mac. Checked with the network blocked: `say`, `transcribe` and `diarize` all pass.
