// speechd: Argmax speech-to-text (WhisperKit) and text-to-speech (TTSKit), fully on device.
//
//   speechd                      live mode: mic -> text, text -> voice, over a WebSocket to the brain
//   speechd say "text" --out f.wav     text-to-speech into a wav file (no microphone needed)
//   speechd transcribe f.wav           speech-to-text from a file (no microphone needed)
//   speechd prerender lines.txt        speak every line once into the cache so live playback is instant
//
// Options: --url ws://127.0.0.1:8770/speech  --stt <whisper model>  --voice ryan  --models <dir>
//          --no-tts-cache  --system-voice (macOS built-in voice instead of Argmax TTS)
//          --rate 1.12 (playback speed of cached lines)  --no-pro (skip Argmax Pro speaker labels)
//   speechd --feed f.wav               live mode with a wav file standing in for the microphone (testing)
//   speechd diarize f.wav              run a file through the Argmax Pro live speaker labeler

import Argmax
import AVFoundation
import CoreML
import CryptoKit
import Foundation

// MARK: - Arguments

struct Args {
    var command = "live"
    var positional: [String] = []
    var opts: [String: String] = [:]
    var flags: Set<String> = []

    init(_ argv: [String]) {
        var rest = Array(argv.dropFirst())
        if let first = rest.first, ["say", "transcribe", "prerender", "diarize", "live"].contains(first) {
            command = first
            rest.removeFirst()
        }
        let boolFlags: Set<String> = ["--no-tts-cache", "--system-voice", "--verbose", "--offline-check", "--no-pro"]
        var i = 0
        while i < rest.count {
            let a = rest[i]
            if boolFlags.contains(a) {
                flags.insert(a)
            } else if a.hasPrefix("--"), i + 1 < rest.count {
                opts[a] = rest[i + 1]
                i += 1
            } else {
                positional.append(a)
            }
            i += 1
        }
    }
}

let args = Args(CommandLine.arguments)
setvbuf(stdout, nil, _IOLBF, 0)

func log(_ s: String) {
    FileHandle.standardError.write(("[speechd] " + s + "\n").data(using: .utf8)!)
}

// MARK: - Paths and model choices

/// Models live next to the package (app/speech/models), which is git-ignored.
let modelsDir: URL = {
    if let m = args.opts["--models"] { return URL(fileURLWithPath: m) }
    if let m = ProcessInfo.processInfo.environment["SPEECHD_MODELS"] { return URL(fileURLWithPath: m) }
    // executable is .build/<triple>/release/speechd -> walk up to the package folder
    var url = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    for _ in 0..<6 {
        url.deleteLastPathComponent()
        if FileManager.default.fileExists(atPath: url.appendingPathComponent("Package.swift").path) {
            return url.appendingPathComponent("models")
        }
    }
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("models")
}()

let sttModel = args.opts["--stt"] ?? ProcessInfo.processInfo.environment["SPEECHD_STT"] ?? "openai_whisper-small.en"
let voiceName = args.opts["--voice"] ?? ProcessInfo.processInfo.environment["SPEECHD_VOICE"] ?? "ryan"
let useSystemVoice = args.flags.contains("--system-voice")
let useCache = !args.flags.contains("--no-tts-cache")
/// Playback speed for cached lines. The voice model speaks slowly; 1.12 sounds brisk but clear.
let playRate = Float(args.opts["--rate"] ?? ProcessInfo.processInfo.environment["SPEECHD_RATE"] ?? "1.12") ?? 1.12
let cacheDir = modelsDir.appendingPathComponent("tts-cache/\(voiceName)")
// Point downloads at a dead address to prove nothing needs the network once models are on disk.
let deadEndpoint: String? = args.flags.contains("--offline-check") ? "http://127.0.0.1:9" : nil

/// The Argmax runtime key. Comes from the environment, or from the project's private .env file
/// (two folders above this package). It is never printed or written anywhere.
let argmaxKey: String? = {
    if args.flags.contains("--no-pro") { return nil }
    if let k = ProcessInfo.processInfo.environment["ARGMAX_API_KEY"], k.hasPrefix("ax_") { return k }
    let envFile = modelsDir.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent(".env")
    guard let text = try? String(contentsOf: envFile, encoding: .utf8) else { return nil }
    for line in text.split(separator: "\n") where line.hasPrefix("ARGMAX_API_KEY=") {
        let k = line.dropFirst("ARGMAX_API_KEY=".count).trimmingCharacters(in: CharacterSet(charactersIn: " \"'\r"))
        if k.hasPrefix("ax_") { return k }
    }
    return nil
}()

/// Split dispatcher text into sentences. Each sentence is spoken and cached on its own.
func sentences(_ text: String) -> [String] {
    var out: [String] = []
    var cur = ""
    let chars = Array(text)
    for (i, c) in chars.enumerated() {
        cur.append(c)
        let endOfSentence = ".!?".contains(c) && (i + 1 == chars.count || chars[i + 1] == " ")
        if endOfSentence, cur.trimmingCharacters(in: .whitespaces).count >= 14 {
            out.append(cur.trimmingCharacters(in: .whitespaces)); cur = ""
        }
    }
    let tail = cur.trimmingCharacters(in: .whitespaces)
    if !tail.isEmpty {
        if tail.count < 14, let last = out.popLast() { out.append(last + " " + tail) } else { out.append(tail) }
    }
    return out
}

/// How much of the reference sentence shows up in what the speech-to-text heard (0...1).
func overlap(reference: String, heard: String) -> Double {
    func words(_ s: String) -> [String] {
        s.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 2 && Int($0) == nil }
    }
    let numbers: Set<String> = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "fifteen", "twenty", "thirty"]
    let ref = words(reference).filter { !numbers.contains($0) }
    guard !ref.isEmpty else { return 1 }
    let got = Set(words(heard))
    return Double(ref.filter { got.contains($0) }.count) / Double(ref.count)
}

/// A folder under models/ if it is already on disk. When it is, the loader is told to use it and
/// never touch the network (this is what makes WiFi-off work).
func local(_ relative: String) -> URL? {
    let url = modelsDir.appendingPathComponent(relative)
    return FileManager.default.fileExists(atPath: url.path) ? url : nil
}

// MARK: - Speech to text

final class Ears {
    let whisper: WhisperKit
    let options: DecodingOptions

    init() async throws {
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        let t = Date()
        let onDisk = local("models/argmaxinc/whisperkit-coreml/\(sttModel)")
        let config = WhisperKitConfig(
            model: sttModel,
            downloadBase: modelsDir,
            modelEndpoint: deadEndpoint,
            modelFolder: onDisk?.path,
            verbose: args.flags.contains("--verbose"),
            logLevel: args.flags.contains("--verbose") ? .debug : .error,
            prewarm: false,
            load: true,
            download: onDisk == nil
        )
        whisper = try await WhisperKit(config)
        options = DecodingOptions(
            task: .transcribe,
            language: "en",
            temperature: 0,
            temperatureFallbackCount: 1,
            usePrefillPrompt: true,
            skipSpecialTokens: true,
            withoutTimestamps: true,
            wordTimestamps: false,
            suppressBlank: true
        )
        log(String(format: "speech-to-text ready: %@ (%.1fs)", sttModel, Date().timeIntervalSince(t)))
    }

    func transcribe(_ samples: [Float]) async -> String {
        guard samples.count > 1600 else { return "" }
        do {
            let results = try await whisper.transcribe(audioArray: samples, decodeOptions: options)
            return clean(results.map(\.text).joined(separator: " "))
        } catch {
            log("transcribe failed: \(error)")
            return ""
        }
    }

    /// Whisper invents these on silence or noise. Never pass them on as speech.
    static let junk: Set<String> = ["", "you", "thank you", "thanks for watching", "thank you for watching", "bye", ".", "so", "okay"]

    func clean(_ raw: String) -> String {
        var s = raw.replacingOccurrences(of: #"\[[^\]]*\]|\([^\)]*\)|<\|[^|]*\|>"#, with: "", options: .regularExpression)
        s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return s
    }

    func isJunk(_ s: String) -> Bool {
        let k = s.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: " .!?,"))
        return k.isEmpty || (Ears.junk.contains(k) && k != "okay")
    }
}

// MARK: - Text to speech

final class Voice {
    var tts: TTSKit?
    let synth = AVSpeechSynthesizer()
    var player: AVAudioPlayer?
    private var stopped = false
    private let lock = NSLock()

    init() async throws {
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        guard !useSystemVoice else {
            log("text-to-speech: macOS system voice")
            return
        }
        let t = Date()
        let onDisk = local("models/argmaxinc/ttskit-coreml/qwen3_tts/code_decoder/12hz-0.6b-customvoice") != nil
            ? local("models/argmaxinc/ttskit-coreml") : nil
        let config = TTSKitConfig(
            model: .qwen3TTS_0_6b,
            modelFolder: onDisk,
            downloadBase: modelsDir,
            tokenizerFolder: local("tokenizers/Qwen3-0.6B"),
            modelEndpoint: deadEndpoint ?? Qwen3TTSConstants.defaultEndpoint,
            verbose: args.flags.contains("--verbose")
        )
        config.download = onDisk == nil
        config.load = true
        tts = try await TTSKit(config)
        log(String(format: "text-to-speech ready: Qwen3-TTS 0.6B, voice %@ (%.1fs)", voiceName, Date().timeIntervalSince(t)))
    }

    var modelName: String { useSystemVoice ? "macos-system-voice" : "qwen3-tts-0.6b/\(voiceName)" }

    var speaker: Qwen3Speaker { Qwen3Speaker(rawValue: voiceName) ?? .ryan }

    func cacheFile(_ text: String) -> URL {
        let digest = SHA256.hash(data: Data(text.utf8)).prefix(12).map { String(format: "%02x", $0) }.joined()
        return cacheDir.appendingPathComponent(digest + ".wav")
    }

    /// Generate audio for a line (no playback). Returns samples, sample rate and seconds to first audio.
    func render(_ text: String) async throws -> (samples: [Float], rate: Int, firstAudio: Double, total: Double) {
        guard let tts else { throw NSError(domain: "speechd", code: 1, userInfo: [NSLocalizedDescriptionKey: "Argmax TTS is not loaded"]) }
        let t = Date()
        let opts = GenerationOptions(concurrentWorkerCount: 1)
        let r = try await tts.generate(text: text, speaker: speaker, language: .english, options: opts)
        return (r.audio, r.sampleRate, r.timings.timeToFirstBuffer, Date().timeIntervalSince(t))
    }

    func save(_ raw: [Float], rate: Int, to url: URL) async throws {
        // Keep only 0.15s of the silence the voice model leaves at the end of each sentence.
        var samples = raw
        if let lastLoud = samples.lastIndex(where: { abs($0) > 0.01 }) {
            samples = Array(samples.prefix(min(samples.count, lastLoud + Int(Double(rate) * 0.15))))
        }
        let name = url.deletingPathExtension().lastPathComponent
        try await AudioOutput.saveAudio(samples, toFolder: url.deletingLastPathComponent(), filename: name, sampleRate: rate, format: .wav)
    }

    /// Speak one line through the default output. Returns when it has finished (or was stopped).
    func speak(_ text: String) async {
        setStopped(false)
        if useSystemVoice {
            await systemSpeak(text)
            return
        }
        for sentence in sentences(text) {
            if isStopped { return }
            await speakSentence(sentence)
        }
    }

    private func setStopped(_ v: Bool) { lock.lock(); stopped = v; lock.unlock() }

    private func speakSentence(_ text: String) async {
        let file = cacheFile(text)
        if useCache, FileManager.default.fileExists(atPath: file.path) {
            await playFile(file)
            return
        }
        guard let tts else { await systemSpeak(text); return }
        do {
            let opts = GenerationOptions(concurrentWorkerCount: 1)
            // Returning false from the callback stops generation as soon as stop() is called.
            let r = try await tts.play(text: text, speaker: speaker, language: .english, options: opts) { [weak self] _ in
                (self?.isStopped ?? false) ? false : nil
            }
            if useCache, !isStopped, !r.audio.isEmpty {
                try? await save(r.audio, rate: r.sampleRate, to: file)
            }
        } catch {
            if !isStopped { log("argmax tts failed (\(error)); using system voice for this line"); await systemSpeak(text) }
        }
    }

    var isStopped: Bool { lock.lock(); defer { lock.unlock() }; return stopped }

    func stop() {
        setStopped(true)
        player?.stop()
        synth.stopSpeaking(at: .immediate)
        if let tts { Task { await tts.audioOutput.stopPlayback(waitForCompletion: false) } }
    }

    private func playFile(_ url: URL) async {
        do {
            let p = try AVAudioPlayer(contentsOf: url)
            p.enableRate = true
            p.rate = playRate
            player = p
            p.play()
            while p.isPlaying, !isStopped { try? await Task.sleep(nanoseconds: 30_000_000) }
            p.stop()
        } catch {
            log("could not play cached audio: \(error)")
        }
    }

    private func systemSpeak(_ text: String) async {
        let u = AVSpeechUtterance(string: text)
        u.rate = 0.5
        synth.speak(u)
        try? await Task.sleep(nanoseconds: 100_000_000)
        while synth.isSpeaking, !isStopped { try? await Task.sleep(nanoseconds: 30_000_000) }
    }
}

// MARK: - Argmax Pro: live speaker labels (Sortformer)

/// Turns on only when an `ax_` runtime key is present. The key is checked online once; after that the
/// license is cached on this Mac and works offline. If anything here fails, the app carries on
/// with the open-source path and no speaker labels.
final class Speakers {
    let kit: SpeakerKitPro
    let session: DiarizeStreamSession
    private let input: AsyncThrowingStream<[Float], Error>.Continuation
    private let lock = NSLock()
    private var segments: [(start: Float, end: Float, id: Int)] = []
    private(set) var fed = 0            // samples fed so far (the session's clock)
    private(set) var speakerCount = 0

    static func activate(key: String) async -> Bool {
        await ArgmaxSDK.with(ArgmaxConfig(apiKey: key))
        let info = await ArgmaxSDK.licenseInfo()
        let exp = info.licenseTokenExpiresAt.map { ISO8601DateFormatter().string(from: $0) } ?? "unknown"
        log("argmax pro license: proAccess=\(info.proAccess) expires=\(exp)")
        return info.proAccess
    }

    init() async throws {
        let t = Date()
        // Try the copy on disk first (several possible roots), then fall back to downloading.
        let roots = ["models/argmaxinc/speakerkit-pro", "models/argmaxinc/speakerkit-pro/sortformer/nemotron-3-diarization/684_74MB",
                     "models/argmaxinc/speakerkit-pro/sortformer"].compactMap { local($0)?.path }
        var loaded: SpeakerKitPro?
        for root in roots where loaded == nil {
            let config = SortformerConfig(downloadBase: modelsDir.path, modelFolder: root, verbose: false, logLevel: .error,
                                          download: false, load: true, streamingConfig: .realtime)
            loaded = try? await SpeakerKitPro(config)
        }
        if loaded == nil {
            let config = SortformerConfig(downloadBase: modelsDir.path, verbose: false, logLevel: .error,
                                          download: true, load: true, streamingConfig: .realtime)
            loaded = try await SpeakerKitPro(config)
        }
        kit = loaded!
        session = try await kit.makeDiarizeStreamSession(config: SortformerDiarizationOptions(sortformerMode: .realtime))
        let (stream, continuation) = AsyncThrowingStream<[Float], Error>.makeStream()
        input = continuation
        await session.start(audioInputStream: stream)
        log(String(format: "argmax pro speaker labels ready: Sortformer realtime (%.1fs)", Date().timeIntervalSince(t)))
        Task { [weak self] in
            guard let self else { return }
            do {
                for try await update in self.session.results { self.absorb(update) }
            } catch {
                log("speaker labels stopped: \(error)")
            }
        }
    }

    func feed(_ samples: [Float]) {
        lock.lock(); fed += samples.count; lock.unlock()
        input.yield(samples)
    }

    func finish() { input.finish() }

    var now: Float { lock.lock(); defer { lock.unlock() }; return Float(fed) / 16000 }

    private func absorb(_ update: DiarizationResultUpdate) {
        let fresh = update.speakerSegments.compactMap { seg -> (start: Float, end: Float, id: Int)? in
            guard let id = seg.speaker.speakerIds.first else { return nil }
            return (seg.startTime, seg.endTime, id)
        }
        lock.lock(); defer { lock.unlock() }
        speakerCount = max(speakerCount, update.speakerCount)
        guard let earliest = fresh.map(\.start).min() else { return }
        // An update restates everything from its earliest segment on; replace what we held there.
        segments.removeAll { $0.end > earliest }
        segments.append(contentsOf: fresh)
        if let latest = segments.map(\.end).max() { segments.removeAll { $0.end < latest - 120 } }
    }

    var allSegments: [(start: Float, end: Float, id: Int)] {
        lock.lock(); defer { lock.unlock() }
        return segments.sorted { $0.start < $1.start }
    }

    /// Who spoke most between two points on the session clock (seconds). nil if not known yet.
    func speaker(from start: Float, to end: Float) -> Int? {
        var total: [Int: Float] = [:]
        for seg in allSegments {
            let o = min(seg.end, end) - max(seg.start, start)
            if o > 0 { total[seg.id, default: 0] += o }
        }
        return total.max { $0.value < $1.value }?.key
    }
}

// MARK: - WebSocket link to the brain

final class Link: NSObject, URLSessionWebSocketDelegate {
    let url: URL
    var task: URLSessionWebSocketTask?
    var connected = false
    var onMessage: (([String: Any]) -> Void)?
    var onConnect: (() -> Void)?
    lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)

    init(url: URL) { self.url = url }

    func start() { connect() }

    private func connect() {
        let t = session.webSocketTask(with: url)
        task = t
        t.resume()
        receive(t)
    }

    private func retry() {
        connected = false
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) { [weak self] in self?.connect() }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        connected = true
        log("connected to \(url.absoluteString)")
        onConnect?()
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        if webSocketTask === task { log("brain closed the connection; retrying"); retry() }
    }

    func urlSession(_ session: URLSession, task t: URLSessionTask, didCompleteWithError error: Error?) {
        if t === task, error != nil { retry() }
    }

    private func receive(_ t: URLSessionWebSocketTask) {
        t.receive { [weak self] result in
            guard let self, t === self.task else { return }
            switch result {
            case .failure:
                return  // didCompleteWithError handles the retry
            case .success(let msg):
                var data: Data?
                if case .string(let s) = msg { data = s.data(using: .utf8) }
                if case .data(let d) = msg { data = d }
                if let data, let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    self.onMessage?(obj)
                }
                self.receive(t)
            }
        }
    }

    func send(_ obj: [String: Any]) {
        guard connected, let task,
              let data = try? JSONSerialization.data(withJSONObject: obj),
              let s = String(data: data, encoding: .utf8) else { return }
        task.send(.string(s)) { _ in }
    }
}

// MARK: - Live mode

/// Mic samples arrive on an audio thread; a single loop decides when a turn starts and ends
/// and runs every transcription one at a time.
final class Live {
    let ears: Ears
    let voice: Voice
    let link: Link
    var speakers: Speakers?
    private var turnStart: Float = 0        // session-clock time the current turn began

    private let lock = NSLock()
    private var buffer: [Float] = []        // current turn (16 kHz mono)
    private var preroll: [Float] = []       // last 300 ms before speech started
    private var inSpeech = false
    private var quiet: Double = 0           // seconds of silence inside a turn
    private var voiced: Double = 0          // seconds of speech in the turn
    private var noise: Float = 0.004        // running noise floor (rms)
    private var levelPeak: Float = 0
    private var turnDone = false

    private var speakQueue: [(id: Int, text: String)] = []
    private var speaking = false
    private var muteUntil = Date.distantPast

    let endSilence = 0.7
    let maxTurn = 28.0

    init(ears: Ears, voice: Voice, link: Link) {
        self.ears = ears; self.voice = voice; self.link = link
    }

    func run() async throws {
        let feedFile = args.opts["--feed"]   // test mode: a wav file stands in for the microphone
        var allowed = feedFile != nil
        if !allowed { allowed = await AudioProcessor.requestRecordPermission() }
        guard allowed else {
            log("""
            NO MICROPHONE ACCESS. Open System Settings > Privacy & Security > Microphone and turn it on \
            for the app that launched this (Terminal, iTerm, or your editor), then run again.
            """)
            exit(2)
        }
        link.onConnect = { [weak self] in
            guard let self else { return }
            self.link.send(["type": "ready", "stt": sttModel, "tts": self.voice.modelName, "pro": self.speakers != nil])
        }
        link.onMessage = { [weak self] msg in self?.handle(msg) }
        link.start()

        if let feedFile {
            let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: feedFile)
            Task {
                while !self.link.connected { try? await Task.sleep(nanoseconds: 100_000_000) }
                for start in stride(from: 0, to: samples.count, by: 1600) {
                    self.ingest(Array(samples[start..<min(start + 1600, samples.count)]))
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                for _ in 0..<30 { self.ingest([Float](repeating: 0, count: 1600)); try? await Task.sleep(nanoseconds: 100_000_000) }
                log("feed finished")
            }
        } else {
            try ears.whisper.audioProcessor.startRecordingLive(inputDeviceID: nil) { [weak self] samples in
                self?.ingest(samples)
            }
        }
        log("listening. Ctrl+C to quit.")
        Task { await self.speakLoop() }

        var lastHypothesis = Date.distantPast
        var lastLevel = Date.distantPast
        var lastPurge = Date()
        while true {
            try? await Task.sleep(nanoseconds: 50_000_000)
            let now = Date()

            if now.timeIntervalSince(lastLevel) > 0.1 {
                lastLevel = now
                lock.lock(); let peak = levelPeak; levelPeak = 0; lock.unlock()
                link.send(["type": "level", "rms": Double(min(1, peak * 6))])
            }
            if feedFile == nil, now.timeIntervalSince(lastPurge) > 20 {
                lastPurge = now
                ears.whisper.audioProcessor.purgeAudioSamples(keepingLast: 0)
            }

            lock.lock()
            let done = turnDone
            let talking = inSpeech
            let snapshot = (done || talking) ? buffer : []
            let began = turnStart
            if done { buffer = []; turnDone = false; inSpeech = false; quiet = 0; voiced = 0 }
            lock.unlock()

            if done {
                let ended = speakers?.now ?? 0
                let text = await ears.transcribe(snapshot)
                if !ears.isJunk(text) {
                    var who: Int? = speakers?.speaker(from: began, to: ended)
                    if who == nil, let speakers {   // the labeler runs about a second behind; give it a moment
                        for _ in 0..<8 where who == nil {
                            try? await Task.sleep(nanoseconds: 50_000_000)
                            who = speakers.speaker(from: began, to: ended)
                        }
                    }
                    log("heard\(who.map { " [speaker \($0)]" } ?? ""): \(text)")
                    link.send(["type": "utterance", "text": text, "speaker": who.map { $0 as Any } ?? NSNull()])
                } else {
                    link.send(["type": "hypothesis", "text": ""])
                }
                lastHypothesis = Date()
            } else if talking, now.timeIntervalSince(lastHypothesis) > 0.45, snapshot.count > 8000 {
                let text = await ears.transcribe(snapshot)
                lastHypothesis = Date()
                if !ears.isJunk(text) { link.send(["type": "hypothesis", "text": text]) }
            }
        }
    }

    private func ingest(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        var sum: Float = 0
        for s in samples { sum += s * s }
        let rms = (sum / Float(samples.count)).squareRoot()
        let dur = Double(samples.count) / 16000.0

        lock.lock(); defer { lock.unlock() }
        if speaking || Date() < muteUntil {       // half duplex: never hear our own voice
            buffer = []; preroll = []; inSpeech = false; quiet = 0; voiced = 0; turnDone = false
            return
        }
        speakers?.feed(samples)
        levelPeak = max(levelPeak, rms)
        let threshold = max(0.012, noise * 3.5)
        let isVoice = rms > threshold
        if !isVoice { noise = noise * 0.95 + rms * 0.05 }

        if !inSpeech {
            preroll.append(contentsOf: samples)
            if preroll.count > 4800 { preroll.removeFirst(preroll.count - 4800) }
            if isVoice {
                inSpeech = true; quiet = 0; voiced = dur
                turnStart = (speakers?.now ?? 0) - Float(preroll.count) / 16000
                buffer = preroll; preroll = []
            }
            return
        }
        guard !turnDone else { return }
        buffer.append(contentsOf: samples)
        if isVoice { quiet = 0; voiced += dur } else { quiet += dur }
        let length = Double(buffer.count) / 16000.0
        if quiet >= endSilence || length >= maxTurn {
            if voiced < 0.15 {                     // a click or a cough, not a turn
                buffer = []; inSpeech = false; quiet = 0; voiced = 0
            } else {
                turnDone = true
            }
        }
    }

    private func handle(_ msg: [String: Any]) {
        switch msg["type"] as? String {
        case "speak":
            guard let text = msg["text"] as? String, !text.isEmpty else { return }
            let id = (msg["id"] as? Int) ?? 0
            lock.lock(); speakQueue.append((id, text)); lock.unlock()
        case "stop":
            lock.lock(); speakQueue.removeAll(); lock.unlock()
            voice.stop()
        default:
            break
        }
    }

    private func speakLoop() async {
        while true {
            lock.lock()
            let next = speakQueue.isEmpty ? nil : speakQueue.removeFirst()
            if next != nil, !speaking { speaking = true; lock.unlock(); link.send(["type": "speaking", "on": true]) } else { lock.unlock() }
            guard let next else {
                lock.lock()
                let was = speaking
                if was { speaking = false; muteUntil = Date().addingTimeInterval(0.3) }
                lock.unlock()
                if was { link.send(["type": "speaking", "on": false]) }
                try? await Task.sleep(nanoseconds: 20_000_000)
                continue
            }
            await voice.speak(next.text)
            link.send(["type": "spoken", "id": next.id, "cut": voice.isStopped])
        }
    }
}

// MARK: - Commands

func runSay() async throws {
    guard let text = args.positional.first else { log("usage: speechd say \"text\" --out file.wav"); exit(64) }
    let out = URL(fileURLWithPath: args.opts["--out"] ?? "say.wav")
    let voice = try await Voice()
    let r = try await voice.render(text)
    try await voice.save(r.samples, rate: r.rate, to: out)
    let audio = Double(r.samples.count) / Double(r.rate)
    print(String(format: "wrote %@  audio %.2fs  first-audio %.2fs  total %.2fs  speed %.1fx real time",
                 out.path, audio, r.firstAudio, r.total, audio / max(r.total, 0.001)))
}

func runTranscribe() async throws {
    guard let path = args.positional.first else { log("usage: speechd transcribe file.wav"); exit(64) }
    let ears = try await Ears()
    let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: path)
    let audio = Double(samples.count) / 16000.0
    _ = await ears.transcribe(Array(samples.prefix(16000)))  // warm up so the timing below is the steady state
    let t = Date()
    let text = await ears.transcribe(samples)
    let took = Date().timeIntervalSince(t)
    print(text)
    print(String(format: "audio %.2fs  transcribed in %.2fs  real-time factor %.3f", audio, took, took / max(audio, 0.001)))
}

func runPrerender() async throws {
    guard let path = args.positional.first else { log("usage: speechd prerender lines.txt"); exit(64) }
    let lines = try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    var seen = Set<String>()
    let all = lines.flatMap { sentences($0) }.filter { seen.insert($0).inserted }
    async let e = Ears()
    async let v = Voice()
    let (ears, voice) = try await (e, v)
    var made = 0
    var weak: [String] = []
    for (i, line) in all.enumerated() {
        let file = voice.cacheFile(line)
        if FileManager.default.fileExists(atPath: file.path) { continue }
        // The voice model sometimes garbles a line. Speak it, listen back, and keep the best of up to 5 takes.
        var best: (score: Double, samples: [Float], rate: Int)? = nil
        var tries = 0
        while tries < 5 {
            tries += 1
            let r = try await voice.render(line)
            let mono16 = resample(r.samples, from: r.rate, to: 16000)
            let heard = await ears.transcribe(mono16)
            let seconds = Double(r.samples.count) / Double(r.rate)
            let tooLong = seconds > Double(line.count) * 0.13 + 2.0   // drawn-out or spelled-out speech
            let score = overlap(reference: line, heard: heard) - (tooLong ? 0.3 : 0)
            if best == nil || score > best!.score { best = (score, r.samples, r.rate) }
            if score >= 0.9 { break }
        }
        guard let best else { continue }
        try await voice.save(best.samples, rate: best.rate, to: file)
        made += 1
        if best.score < 0.75 { weak.append(line) }
        log("\(i + 1)/\(all.count) takes=\(tries) match=\(String(format: "%.2f", best.score))  \(line.prefix(70))")
    }
    print("prerendered \(made) new sentences (\(all.count) total) into \(cacheDir.path)")
    if !weak.isEmpty {
        print("WEAK (under 0.75 match, reword or re-run after moving the file out of the cache):")
        weak.forEach { print("  - \($0)") }
    }
}

/// Plain linear resample, good enough for a listen-back check.
func resample(_ x: [Float], from: Int, to: Int) -> [Float] {
    guard from != to, !x.isEmpty else { return x }
    let n = Int(Double(x.count) * Double(to) / Double(from))
    var out = [Float](repeating: 0, count: n)
    let step = Double(from) / Double(to)
    for i in 0..<n {
        let pos = Double(i) * step
        let j = min(Int(pos), x.count - 1), k = min(j + 1, x.count - 1)
        let f = Float(pos - Double(j))
        out[i] = x[j] * (1 - f) + x[k] * f
    }
    return out
}

func runLive() async throws {
    let url = URL(string: args.opts["--url"] ?? ProcessInfo.processInfo.environment["SPEECHD_URL"] ?? "ws://127.0.0.1:8770/speech")!
    async let e = Ears()
    async let v = Voice()
    let (ears, voice) = try await (e, v)
    let live = Live(ears: ears, voice: voice, link: Link(url: url))
    live.speakers = await makeSpeakers()
    try await live.run()
}

/// Argmax Pro speaker labels, or nil (with the reason logged) so the caller falls back to open source only.
func makeSpeakers() async -> Speakers? {
    guard let key = argmaxKey else { log("no ax_ key: running open-source path, no speaker labels"); return nil }
    guard await Speakers.activate(key: key) else { log("argmax pro license not active: open-source path only"); return nil }
    do { return try await Speakers() } catch { log("argmax pro speaker labels failed to start (\(error)): open-source path only"); return nil }
}

/// Test mode: push a file through the live speaker labeler in real-time-sized chunks.
func runDiarize() async throws {
    guard let path = args.positional.first else { log("usage: speechd diarize file.wav"); exit(64) }
    guard let speakers = await makeSpeakers() else { exit(3) }
    let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: path)
    let t = Date()
    for start in stride(from: 0, to: samples.count, by: 1600) {
        speakers.feed(Array(samples[start..<min(start + 1600, samples.count)]))
        try? await Task.sleep(nanoseconds: 5_000_000)
    }
    speakers.feed([Float](repeating: 0, count: 32000))   // flush the tail
    try? await Task.sleep(nanoseconds: 2_500_000_000)
    for seg in speakers.allSegments {
        print(String(format: "%6.2f - %6.2f  speaker %d", seg.start, seg.end, seg.id))
    }
    print(String(format: "audio %.1fs  speakers found %d  wall %.1fs", Double(samples.count) / 16000, speakers.speakerCount, Date().timeIntervalSince(t)))
}

do {
    switch args.command {
    case "say": try await runSay()
    case "transcribe": try await runTranscribe()
    case "prerender": try await runPrerender()
    case "diarize": try await runDiarize()
    default: try await runLive()
    }
} catch {
    log("fatal: \(error)")
    exit(1)
}
