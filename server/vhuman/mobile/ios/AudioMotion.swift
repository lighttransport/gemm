import AVFoundation
import Foundation

@MainActor final class AudioMotion {
    struct Pose {
        let sample: Int64
        let expression: [Float]
        let rotations: [Float]
        let translation: [Float]
    }
    private struct Packet: Decodable {
        let type: String
        let epoch: Int64?
        let schema: String?
        let geometry_sha256: String?
        let sequence: Int64?
        let sample_start: Int64?
        let sample_position: Int64?
        let sample_rate: Int?
        let samples: Int64?
        let pcm: Data?
        let expression: [Float]?
        let rotations: [Float]?
        let translation: [Float]?
        let message: String?
    }
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
    private var socket: URLSessionWebSocketTask?
    private var receiver: Task<Void, Never>?
    private var epoch: Int64 = -1
    private var sequence: Int64 = 0
    private var accepted: Int64 = 0
    private var lastMotion: Int64 = -1
    private var frames: [Pose] = []
    private var ended = false
    private var waitingBegin = true
    private var ready = false
    private var identity = ""
    private var reference = [Float](repeating: 0, count: 383)
    var onStatus: (String) -> Void = { _ in }

    init() {
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
    }

    func connect(_ url: URL, package: AvatarPackage) throws {
        disconnect()
        guard ["ws", "wss"].contains(url.scheme), url.host != nil else { throw AvatarError.invalid("Invalid WebSocket address") }
        let audio = AVAudioSession.sharedInstance()
        try audio.setCategory(.playback, mode: .spokenAudio)
        try audio.setActive(true)
        try engine.start()
        identity = package.geometryHash
        reference = package.reference
        let ws = URLSession.shared.webSocketTask(with: url)
        ws.maximumMessageSize = 128 * 1024
        socket = ws; ws.resume()
        receiver = Task { [weak self] in
            do {
                try await ws.send(.string(String(data: JSONSerialization.data(withJSONObject: [
                    "type": "hello", "schema": "vhuman.mobile_stream.v1", "package_sha256": package.packageHash
                ]), encoding: .utf8)!))
                while !Task.isCancelled {
                    let message = try await ws.receive()
                    let data: Data
                    switch message {
                    case .data(let bytes): data = bytes
                    case .string(let text): data = Data(text.utf8)
                    @unknown default: throw AvatarError.invalid("Unsupported server message")
                    }
                    guard !Task.isCancelled else { return }
                    try self?.receive(data)
                }
            } catch {
                guard !Task.isCancelled else { return }
                self?.disconnect(); self?.onStatus(error.localizedDescription)
            }
        }
    }

    func command(text: String? = nil, language: String = "en") async throws {
        guard ready, let socket else { throw AvatarError.invalid("Connect to the speech server first") }
        if let text, text.isEmpty || text.count > 4096 { throw AvatarError.invalid("Enter 1–4096 characters") }
        player.stop(); frames.removeAll(); waitingBegin = true
        let object: [String: Any] = text.map { ["type": "speak", "text": $0, "language": language] } ?? ["type": "cancel"]
        try await socket.send(.string(String(data: JSONSerialization.data(withJSONObject: object), encoding: .utf8)!))
    }

    func disconnect() {
        receiver?.cancel(); receiver = nil
        socket?.cancel(with: .goingAway, reason: nil); socket = nil
        player.stop(); engine.stop(); frames.removeAll()
        epoch = -1; ready = false; waitingBegin = true
        accepted = 0; sequence = 0
    }

    private func receive(_ data: Data) throws {
        guard data.count <= 128 * 1024 else { throw AvatarError.invalid("Oversized server packet") }
        let p = try JSONDecoder().decode(Packet.self, from: data)
        if p.type == "ready" {
            guard !ready, p.schema == "vhuman.mobile_stream.v1", p.geometry_sha256 == identity else { throw AvatarError.invalid("Server identity mismatch") }
            ready = true; onStatus("Connected"); return
        }
        guard ready, let incoming = p.epoch, incoming >= 0 else { throw AvatarError.invalid("Missing stream handshake") }
        if p.type == "begin" {
            guard incoming > epoch else { throw AvatarError.invalid("Stream epoch did not advance") }
            player.stop(); frames.removeAll(); accepted = 0; sequence = 0; lastMotion = -1
            epoch = incoming; ended = false; waitingBegin = false; return
        }
        if incoming < epoch || waitingBegin { return }
        guard incoming == epoch else { throw AvatarError.invalid("Unexpected stream epoch") }
        switch p.type {
        case "audio":
            guard !ended, p.sequence == sequence, p.sample_start == accepted, p.sample_rate == 24000,
                let bytes = p.pcm, !bytes.isEmpty, bytes.count % 4 == 0, bytes.count <= 4800 * 4,
                accepted - position < 48000 else { throw AvatarError.invalid("Invalid or excessive audio data") }
            let count = bytes.count / 4
            var samples = [Float](repeating: 0, count: count)
            _ = samples.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
            guard samples.allSatisfy({ $0.isFinite && abs($0) <= 1 }) else { throw AvatarError.invalid("Invalid PCM samples") }
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count))!
            buffer.frameLength = buffer.frameCapacity
            samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: count) }
            player.scheduleBuffer(buffer, at: AVAudioTime(sampleTime: accepted, atRate: 24000))
            accepted += Int64(count); sequence += 1
            if !player.isPlaying && accepted >= 3840 { player.play() }
        case "motion":
            guard !ended, let sample = p.sample_position, sample >= 0, sample > lastMotion,
                frames.count < 256, let x = p.expression, let r = p.rotations, let t = p.translation,
                x.count == 383, r.count == 12, t.count == 3,
                x.allSatisfy({ $0.isFinite && abs($0) <= 3 }),
                r.allSatisfy({ $0.isFinite && abs($0) <= 3.15 }),
                t.allSatisfy({ $0.isFinite && abs($0) <= 10 }) else { throw AvatarError.invalid("Invalid motion data") }
            if lastMotion < 0 && sample != 0 { throw AvatarError.invalid("Motion must start at sample zero") }
            frames.append(Pose(sample: sample, expression: x, rotations: r, translation: t)); lastMotion = sample
        case "end":
            guard !ended, p.samples == accepted else { throw AvatarError.invalid("Audio length mismatch") }
            ended = true
            if !player.isPlaying && accepted > 0 { player.play() }
        case "error": throw AvatarError.invalid(p.message ?? "Speech server failed")
        default: throw AvatarError.invalid("Unknown server packet")
        }
    }

    // Device audio timeline, corrected for output presentation latency. Wall time
    // never advances animation. Route changes stop the stream via the app shell.
    var position: Int64 {
        guard player.isPlaying, let node = player.lastRenderTime,
            let time = player.playerTime(forNodeTime: node) else { return 0 }
        let latency = player.outputPresentationLatency
        return max(0, min(accepted, Int64((Double(time.sampleTime) / time.sampleRate - latency) * 24000)))
    }

    func currentPose() -> Pose? {
        guard !waitingBegin else { return nil }
        let sample = position
        if ended && sample >= accepted {
            return Pose(sample: sample, expression: reference,
                rotations: [Float](repeating: 0, count: 12), translation: [Float](repeating: 0, count: 3))
        }
        if player.isPlaying, !ended, accepted > 0, sample >= accepted {
            disconnect(); onStatus("Audio underrun; reconnect to restart the utterance"); return nil
        }
        while frames.count > 2 && frames[1].sample <= sample { frames.removeFirst() }
        guard let a = frames.first else { return nil }
        guard frames.count > 1, sample > a.sample else { return a }
        let b = frames[1]
        let blend = min(1, max(0, Float(sample - a.sample) / Float(b.sample - a.sample)))
        func mix(_ a: [Float], _ b: [Float]) -> [Float] { zip(a, b).map { pair in pair.0 + blend * (pair.1 - pair.0) } }
        return Pose(sample: sample, expression: mix(a.expression, b.expression),
            rotations: mix(a.rotations, b.rotations), translation: mix(a.translation, b.translation))
    }
}
