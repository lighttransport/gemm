import argparse
import json
from pathlib import Path
import subprocess
import time
import wave
import numpy as np
from .src.avatar.bundle import GaussianAvatar, bind
from .src.audio.ring import PcmRing, build
from .src.pipeline.protocol import AudioChunk, MotionFrame
from .src.pipeline.session import Session

WORK = Path("tmp/vhuman-realtime")


def doctor():
    from .src.avatar.cuda_runtime import CudaRuntime
    import importlib.util
    report = {'cuda_available': False, 'renderer': 'native-cuda',
              'torch_installed': importlib.util.find_spec('torch') is not None,
              'gsplat_installed': importlib.util.find_spec('gsplat') is not None}
    try:
        runtime = CudaRuntime()
    except RuntimeError as error:
        report['cuda_error'] = str(error)
    else:
        report['cuda_available'] = True
        runtime.close()
    return report


def load_audio(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, 24000, 2):
            raise ValueError("replay requires mono PCM16 WAV at 24kHz")
        return np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768


def replay(args):
    from .src.animation.timeline import retarget
    pcm = load_audio(args.audio)
    from .src.animation.performance import load_performance
    names, positions, values = load_performance(args.motion)
    ranges = [[-1 if n in ("headYaw", "headPitch", "headRoll") else 0, 1] for n in names]
    frames = list(zip(positions, values))
    if any(b[0] <= a[0] for a, b in zip(frames, frames[1:])): raise ValueError("nonmonotonic performance")
    from .src.audio.device import AudioDevice, build_device
    library = build_device(WORK / "native/libaudio.so") if args.sink == "device" else build(WORK / "native/libpcm.so")
    ring = PcmRing(library)
    session = Session(ring, names, ranges)
    renderer = rig = device = shared = handle = None
    vertices = None
    try:
        if args.avatar:
            from .src.avatar.rig import RigAvatar
            from .src.renderer.gaussian import GaussianRenderer
            from .src.renderer.camera import for_avatar
            rig = RigAvatar(args.rig, WORK / "native")
            avatar = GaussianAvatar.load(args.avatar, rig.triangles)
            if tuple(avatar.metadata["control_names"]) != rig.names: raise ValueError("avatar control order differs")
            renderer = GaussianRenderer(avatar, rig.triangles)
            if args.rig_backend == "cuda":
                from .src.avatar.native_gpu import NativeSharedGPU
                shared = NativeSharedGPU(Path(args.rig) / "rig_deformer.safetensors", WORK / "native", runtime=renderer.runtime)
            view, intrinsics = for_avatar(avatar, rig.rest)
        if args.sink == "device": device = AudioDevice(ring, library)
        head, index, ticks = 0, 0, 0
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as log:
            while head < len(pcm) or ring.depth or (device and session.clock.position() < len(pcm)):
                # Decode/read-ahead bounded at 400ms; producer stalls rather than dropping speech.
                while head < len(pcm) and ring.depth < session.config.high_water_samples:
                    end = min(head + 480, len(pcm))
                    if ring.depth + end - head > session.config.high_water_samples: break
                    while index < len(frames) and frames[index][0] < end:
                        position, values = frames[index]
                        session.motion.push(MotionFrame(0, position, values)); index += 1
                    session.ingest(AudioChunk(0, session.sequence, head, pcm[head:end])); head = end
                if device and ticks == 0: device.start()
                if device:
                    device.update(session.clock)
                    position = session.clock.position()
                    controls = session.motion.sample(position)
                else:
                    _, controls = session.pull(min(400, len(pcm)-session.consumed), ticks * 1_000_000_000 // 60)
                    position = session.clock.position()
                if renderer:
                    start = time.monotonic_ns()
                    mapped = retarget(controls, names, rig.names)
                    vertices = shared.submit(mapped) if shared else rig.deform(mapped)
                    handle = renderer.render(vertices, view, intrinsics, controls=mapped, sample_position=position)
                    handle.ready.synchronize()
                    session.metrics.add("rig_plus_render_wall_ms", (time.monotonic_ns()-start)/1e6)
                log.write(json.dumps({"sample_position": position, "controls": controls.tolist()}) + "\n")
                ticks += 1
                if device: time.sleep(1/60)
        return {"samples": len(pcm), "ticks": ticks, "metrics": session.metrics.report(), "output": str(output)}
    finally:
        from .src.pipeline.cleanup import close_resources
        vertices = handle = None
        close_resources(device, shared, renderer, rig, ring)


def render(args):
    from .src.avatar.rig import RigAvatar
    from .src.renderer.gaussian import GaussianRenderer
    from .src.renderer.camera import for_avatar
    from PIL import Image
    rig = RigAvatar(args.rig, WORK / "native")
    shared = renderer = handle = begin = end = None
    vertices = None
    try:
        avatar = GaussianAvatar.load(args.avatar, rig.triangles)
        renderer = GaussianRenderer(avatar, rig.triangles)
        view, intrinsics = for_avatar(avatar, rig.rest, (args.res, args.res))
        if args.rig_backend == "cuda":
            from .src.avatar.native_gpu import NativeSharedGPU
            shared = NativeSharedGPU(Path(args.rig) / "rig_deformer.safetensors", WORK / "native", runtime=renderer.runtime)
        zero = np.zeros(len(rig.names), np.float32)
        vertices = shared.submit(zero) if shared else rig.deform(zero)
        for _ in range(5): renderer.render(vertices, view, intrinsics, (args.res, args.res)).ready.synchronize()
        renderer.reset_memory_stats()
        start = time.monotonic_ns()
        begin = renderer.runtime.record()
        for index in range(args.frames):
            controls = zero.copy()
            if args.animate:
                controls[rig.names.index("jawOpen")] = .3 + .3 * np.sin(index * .1)
                vertices = shared.submit(controls) if shared else rig.deform(controls)
            handle = renderer.render(vertices, view, intrinsics, (args.res, args.res), controls)
        end = renderer.runtime.record(); end.synchronize()
        elapsed = (time.monotonic_ns()-start)/1e9
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        png = handle.pixels(straight_alpha=True)
        Image.fromarray(png, "RGBA").save(output)
        return dict(purpose=avatar.metadata["purpose"], trained=avatar.metadata["trained"], frames=args.frames,
                    animated=args.animate, rig_backend=args.rig_backend,
                    gaussians=len(avatar.arrays["triangle"]), fps_wall=args.frames/elapsed,
                    gpu_ms_per_frame=begin.elapsed_time(end)/args.frames,
                    **renderer.memory_stats(), output=str(output))
    finally:
        from .src.pipeline.cleanup import close_resources
        vertices = handle = None
        close_resources(begin, end, shared, renderer, rig)


def main():
    parser = argparse.ArgumentParser(description="vhuman neural avatar PoC")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    p = commands.add_parser("generate-identity")
    p.add_argument('--identity-backend', choices=('native', 'torch-reference'), default='native')
    p.add_argument('--native-assets', help='Hashed native FLUX.2 component manifest')
    p.add_argument('--identity-runner', help='Repository FLUX.2 executable')
    p.add_argument('--identity-device', type=int, default=0)
    p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cache", default=str(WORK / "cache/hf")); p.add_argument("--expressions", action="store_true")
    p.add_argument("--resume", action="store_true")
    p = commands.add_parser("export-neutral")
    p.add_argument("--head", required=True); p.add_argument("--identity", required=True); p.add_argument("--output", required=True)
    p = commands.add_parser("bind")
    p.add_argument("--rig", required=True); p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=50000)
    p = commands.add_parser("render")
    p.add_argument("--rig", required=True); p.add_argument("--avatar", required=True)
    p.add_argument("--output", default=str(WORK / "render.png"))
    p.add_argument("--frames", type=int, default=100); p.add_argument("--res", type=int, default=512)
    p.add_argument("--rig-backend", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--animate", action="store_true")
    p = commands.add_parser("replay")
    p.add_argument("--audio", required=True); p.add_argument("--motion", required=True)
    p.add_argument("--sink", choices=["offline", "device"], default="offline")
    p.add_argument("--avatar"); p.add_argument("--rig")
    p.add_argument("--rig-backend", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--output", default=str(WORK / "replay.jsonl"))
    p = commands.add_parser("live")
    for name in ("rig", "avatar", "adapter", "model", "revision", "text"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--runner", default=str(WORK / "speech/qwen3_tts_cuda"))
    p.add_argument("--sink", choices=["offline", "device"], default="device")
    p.add_argument("--max-frames", type=int, default=256)
    p.add_argument("--display", action="store_true"); p.add_argument("--video")
    p.add_argument("--virtual-camera", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--motion-threads", type=int, default=1)
    p.add_argument("--tts-threads", type=int, default=8)
    p.add_argument("--resident", action="store_true")
    p.add_argument("--text-feed", choices=["incremental", "full"], default="incremental")
    p = commands.add_parser("evaluate-appearance")
    p.add_argument("--manifest", required=True); p.add_argument("--avatar", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("collect-motion")
    for name in ("rig", "model", "runner", "aligner", "align-model", "sentences", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--text-feed", choices=["incremental", "full"], default="incremental")
    p = commands.add_parser("evaluate-motion")
    p.add_argument("--reference-parity", action="store_true", help="Offline Torch reference comparison")
    for name in ("manifest", "checkpoint", "output"): p.add_argument("--"+name, required=True)
    p = commands.add_parser("stress-tts")
    for name in ("model", "runner", "output"): p.add_argument("--"+name, required=True)
    p.add_argument("--requests", type=int, default=100); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--text-feed", choices=["incremental", "full"], default="incremental")
    p = commands.add_parser("audit-references")
    for name in ("head", "identity", "output"): p.add_argument("--"+name, required=True)
    for name in ("fit-appearance", "train-motion"):
        p = commands.add_parser(name); p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
        if name == "fit-appearance":
            p.add_argument("--count", type=int, default=50000); p.add_argument("--steps", type=int, default=1000)
        else:
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "doctor": result = doctor()
    elif args.command == "generate-identity":
        from .src.avatar.generate import generate
        result = generate(args.output, args.cache, args.seed, args.expressions, args.resume,
                          backend=args.identity_backend, native_assets=args.native_assets,
                          runner=args.identity_runner, device=args.identity_device)
    elif args.command == "export-neutral":
        from .src.avatar.corpus import export_neutral
        result = export_neutral(args.head, args.identity, args.output, WORK / "native")
    elif args.command == "bind":
        from .src.avatar.rig import RigAvatar
        rig = RigAvatar(args.rig, WORK / "native")
        try:
            avatar = bind(rig.rest, rig.triangles, rig.names, args.count)
            avatar.arrays["component"] = rig.components[avatar.arrays["triangle"]]
            avatar.save(args.output)
            result = dict(output=args.output, purpose="diagnostic", trained=False)
        finally: rig.close()
    elif args.command == "render":
        if args.frames < 1 or args.res < 16: parser.error("positive frame count/resolution required")
        result = render(args)
    elif args.command == "replay":
        if bool(args.avatar) != bool(args.rig): parser.error("--avatar and --rig are paired")
        result = replay(args)
    elif args.command == "live":
        from .src.pipeline.live import run
        result = run(args, WORK)
    elif args.command == "fit-appearance":
        from .src.avatar.train import fit
        result = fit(args.manifest, args.output, args.count, args.steps)
    elif args.command == "evaluate-appearance":
        from .src.benchmark.appearance import evaluate
        result = evaluate(args.manifest, args.avatar, args.output)
    elif args.command == "collect-motion":
        from .src.animation.collect import collect
        result = collect(args.rig, args.model, args.runner, args.aligner, args.align_model, args.sentences, args.output, WORK, args.threads, args.text_feed)
    elif args.command == "evaluate-motion":
        from .src.benchmark.motion import evaluate
        result = evaluate(args.manifest, args.checkpoint, args.output, reference_parity=args.reference_parity)
    elif args.command == "stress-tts":
        from .src.benchmark.tts import stress
        result = stress(args.model, args.runner, args.output, WORK/"tts-soak", args.requests, args.threads, args.text_feed)
    elif args.command == "audit-references":
        from .src.benchmark.references import audit
        result = audit(args.head, args.identity, args.output, WORK)
    else:
        from .src.animation.train import train
        result = train(args.manifest, args.output, args.epochs, threads=args.threads)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
