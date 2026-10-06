import UIKit
import AVFoundation
import QuartzCore

final class MetalView: UIView {
    override class var layerClass: AnyClass { CAMetalLayer.self }
    var metalLayer: CAMetalLayer { layer as! CAMetalLayer }
}

@main @MainActor final class App: UIResponder, UIApplicationDelegate {
    var window: UIWindow?
    func application(_ application: UIApplication, didFinishLaunchingWithOptions options: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        window = UIWindow(frame: UIScreen.main.bounds)
        window?.rootViewController = AvatarController()
        window?.makeKeyAndVisible(); return true
    }
}

@MainActor final class AvatarController: UIViewController {
    private let surface = MetalView()
    private let renderer = VHRenderer()
    private let speech = AudioMotion()
    private let status = UILabel()
    private let endpoint = UITextField()
    private let utterance = UITextField()
    private let language = UISegmentedControl(items: ["English", "日本語"])
    private var package: AvatarPackage?
    private var display: CADisplayLink?
    private var loaded = false
    private var frameTimes: [Double] = []
    private var lastReport = CACurrentMediaTime()

    override func viewDidLoad() {
        super.viewDidLoad(); view.backgroundColor = .black
        surface.translatesAutoresizingMaskIntoConstraints = false; view.addSubview(surface)
        status.textColor = .white; status.numberOfLines = 3; status.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        endpoint.placeholder = "wss://speech.example/ws"; endpoint.autocapitalizationType = .none
        endpoint.keyboardType = .URL; endpoint.borderStyle = .roundedRect
        utterance.placeholder = "What should the avatar say?"; utterance.borderStyle = .roundedRect
        language.selectedSegmentIndex = 0
        let load = UIButton(type: .system); load.setTitle("Load avatar", for: .normal); load.addTarget(self, action: #selector(loadAvatar), for: .touchUpInside)
        let connect = UIButton(type: .system); connect.setTitle("Connect", for: .normal); connect.addTarget(self, action: #selector(connectServer), for: .touchUpInside)
        let speak = UIButton(type: .system); speak.setTitle("Speak", for: .normal); speak.addTarget(self, action: #selector(speakText), for: .touchUpInside)
        let stop = UIButton(type: .system); stop.setTitle("Stop", for: .normal); stop.addTarget(self, action: #selector(stopSpeech), for: .touchUpInside)
        let buttons = UIStackView(arrangedSubviews: [load, connect, speak, stop]); buttons.distribution = .fillEqually
        let panel = UIStackView(arrangedSubviews: [status, endpoint, utterance, language, buttons]); panel.axis = .vertical; panel.spacing = 8
        panel.translatesAutoresizingMaskIntoConstraints = false; view.addSubview(panel)
        NSLayoutConstraint.activate([
            surface.topAnchor.constraint(equalTo: view.topAnchor), surface.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            surface.leadingAnchor.constraint(equalTo: view.leadingAnchor), surface.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            panel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            panel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            panel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12)
        ])
        status.text = "Copy an exported avatar folder into Documents/avatar using Finder, then Load avatar."
        speech.onStatus = { [weak self] message in self?.status.text = message }
        for name in [UIApplication.willResignActiveNotification, AVAudioSession.interruptionNotification,
            AVAudioSession.routeChangeNotification] {
            NotificationCenter.default.addObserver(self, selector: #selector(suspend), name: name, object: nil)
        }
        NotificationCenter.default.addObserver(self, selector: #selector(resume), name: UIApplication.didBecomeActiveNotification, object: nil)
        display = CADisplayLink(target: self, selector: #selector(tick))
        display?.preferredFramesPerSecond = 30; display?.add(to: .main, forMode: .common)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        let ratio = min(720 / max(1, surface.bounds.width), 1280 / max(1, surface.bounds.height))
        surface.metalLayer.drawableSize = CGSize(width: max(1, floor(surface.bounds.width * ratio)), height: max(1, floor(surface.bounds.height * ratio)))
        if loaded { do { try renderer.resizeWidth(UInt32(surface.metalLayer.drawableSize.width), height: UInt32(surface.metalLayer.drawableSize.height)) } catch { fail(error) } }
    }

    @objc private func loadAvatar() {
        speech.disconnect(); renderer.close(); loaded = false; package = nil
        do {
            let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("avatar")
            let p = try AvatarPackage(directory: directory)
            try renderer.openDirectory(directory.path, layer: surface.metalLayer,
                width: UInt32(surface.metalLayer.drawableSize.width), height: UInt32(surface.metalLayer.drawableSize.height))
            try renderer.pose(p.reference.bytes, rotations: [Float](repeating: 0, count: 12).bytes, translation: [Float](repeating: 0, count: 3).bytes)
            package = p; loaded = true; status.text = "Avatar loaded • 30 FPS target"
        } catch { fail(error) }
    }
    @objc private func connectServer() {
        do {
            guard let package, let url = URL(string: endpoint.text ?? "") else { throw AvatarError.invalid("Load an avatar and enter the server address") }
            try speech.connect(url, package: package)
        } catch { status.text = error.localizedDescription }
    }
    @objc private func speakText() {
        Task { do { try await speech.command(text: utterance.text ?? "", language: language.selectedSegmentIndex == 1 ? "ja" : "en") }
            catch { status.text = error.localizedDescription } }
        view.endEditing(true)
    }
    @objc private func stopSpeech() {
        if loaded, let package {
            do { try renderer.pose(package.reference.bytes, rotations: [Float](repeating: 0, count: 12).bytes,
                translation: [Float](repeating: 0, count: 3).bytes) } catch { fail(error) }
        }
        Task { do { try await speech.command() } catch { status.text = error.localizedDescription } }
    }
    @objc private func suspend() { display?.isPaused = true; speech.disconnect() }
    @objc private func resume() { display?.isPaused = false }
    @objc private func tick() {
        guard loaded else { return }
        let started = CACurrentMediaTime()
        do {
            if let p = speech.currentPose() { try renderer.pose(p.expression.bytes, rotations: p.rotations.bytes, translation: p.translation.bytes) }
            try renderer.draw()
        } catch { fail(error); return }
        frameTimes.append((CACurrentMediaTime() - started) * 1000)
        if started - lastReport >= 5 {
            let sorted = frameTimes.sorted(); let p95 = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
            // Submission wall time is not GPU completion time or achieved FPS.
            NSLog("vhuman submit_p95_ms=%.3f thermal=%ld audio_sample=%lld", p95, ProcessInfo.processInfo.thermalState.rawValue, speech.position)
            frameTimes.removeAll(keepingCapacity: true); lastReport = started
        }
    }
    private func fail(_ error: Error) { loaded = false; speech.disconnect(); renderer.close(); status.text = error.localizedDescription }
}
