"""Incremental TTS + causal motion + native rig + Gaussian renderer loop."""
import json
from pathlib import Path
import queue
import time
import numpy as np
from ..audio.ring import PcmRing, build
from ..audio.device import AudioDevice, build_device
from ..animation.causal import MotionAdapter
from ..animation.timeline import retarget
from ..avatar.bundle import GaussianAvatar
from ..avatar.rig import RigAvatar
from ..avatar.native_gpu import NativeSharedGPU
from ..renderer.gaussian import GaussianRenderer
from ..renderer.camera import for_avatar
from ..tts.native import NativeTTS
from .session import Session
from .cleanup import close_resources


def validate_appearance(avatar, diagnostic=False):
    if not diagnostic:
        if avatar.metadata.get("purpose") != "production":
            raise ValueError("diagnostic appearance requires --diagnostic")
        if avatar.metadata.get("trained") is not True:
            raise ValueError("untrained appearance requires --diagnostic")


def run(args, work):
    import torch
    if args.motion_threads < 1: raise ValueError("motion threads must be positive")
    torch.set_num_threads(args.motion_threads)
    work = Path(work)
    rig = RigAvatar(args.rig, work / "native")
    shared = ring = device = source = output = sampler = None
    vertices = None
    try:
        adapter = MotionAdapter(args.adapter, args.revision, allow_diagnostic=args.diagnostic)
        if adapter.text_feed != args.text_feed: raise ValueError("motion adapter/TTS text-feed mismatch; recapture and retrain")
        avatar = GaussianAvatar.load(args.avatar, rig.triangles)
        validate_appearance(avatar, args.diagnostic)
        if tuple(avatar.metadata["control_names"]) != rig.names: raise ValueError("avatar/rig control mismatch")
        renderer = GaussianRenderer(avatar, rig.triangles)
        from ..ui.output import FrameOutput
        output = FrameOutput(display=args.display, video=args.video, virtual_camera=args.virtual_camera)
        shared = NativeSharedGPU(Path(args.rig) / "rig_deformer.safetensors", work / "native")
        library = build_device(work / "native/libaudio.so") if args.sink == "device" else build(work / "native/libpcm.so")
        ring = PcmRing(library)
        session = Session(ring, rig.names, rig.ranges)
        if args.sink == "device": device = AudioDevice(ring, library)
        view, intrinsics = [torch.tensor(x, device="cuda") for x in for_avatar(avatar, rig.rest)]
        # Resolve CUDA library/kernel initialization before starting audio playback.
        for _ in range(3):
            neutral = np.zeros(len(rig.names), np.float32)
            vertices = shared.submit(neutral)
            handle = renderer.render(vertices, view, intrinsics, controls=torch.tensor(neutral, device="cuda"))
            handle.ready.synchronize()
        if args.display or args.video or args.virtual_camera: output.prepare(handle)
        if args.resident:
            from ..tts.resident import ResidentTTS
            source = ResidentTTS(args.runner, args.model, args.revision, work / "live", max_frames=args.max_frames, threads=args.tts_threads, text_feed=args.text_feed)
            source.submit(args.text, 0)
        else:
            source = NativeTTS(args.runner, args.model, args.revision, args.text, work / "live", max_frames=args.max_frames, threads=args.tts_threads, text_feed=args.text_feed)
        from ..benchmark.gpu import GpuSampler
        sampler = GpuSampler(session.metrics, source.process.pid)
        started, ticks, pending, feature_end = False, 0, None, 0
        playout = None
        first_render_ns = previous_render_ns = None
        deadline = time.monotonic()
        while True:
            sampler.poll()
            source.check()
            # Features may precede PCM. Do not overwrite motion which has not played.
            while len(session.motion.frames) + 8 <= session.motion.capacity:
                try: features = source.features.get_nowait()
                except queue.Empty: break
                with session.metrics.wall("motion_inference_wall_ms"):
                    motion = adapter.push(features)
                for frame in motion:
                    mapped = retarget(frame.controls, adapter.model.names, rig.names)
                    from .protocol import MotionFrame
                    session.motion.push(MotionFrame(frame.epoch, frame.sample_position, mapped, frame.confidence))
                feature_end = features.sample_start + 1920
            if pending is None:
                try: pending = source.audio.get_nowait()
                except queue.Empty: pass
            if pending is not None and pending.sample_start + len(pending.pcm) <= feature_end and ring.depth + len(pending.pcm) <= session.config.high_water_samples:
                session.ingest(pending); pending = None
            if not started and (ring.depth >= session.config.startup_samples or (source.done_audio.is_set() and ring.depth)):
                started = True
                if device: device.start()
                else:
                    from ..audio.offline import OfflinePlayout
                    playout = OfflinePlayout(time.monotonic_ns())
            if started:
                if device:
                    device.update(session.clock)
                    position = session.clock.position(time.monotonic_ns())
                    controls = session.motion.sample(position)
                else:
                    now = time.monotonic_ns()
                    due = playout.due(now)
                    # A renderer stall can span multiple ring capacities. Account
                    # for every elapsed audio sample, including inserted silence.
                    while due:
                        count = min(due, ring.capacity)
                        pcm, _ = session.pull(count, now)
                        output.audio(pcm)
                        due -= count
                    position = session.clock.position()
                    controls = session.motion.sample(position)
                with session.metrics.wall("rig_submit_wall_ms"):
                    vertices = shared.submit(controls)
                with session.metrics.wall("render_completion_wall_ms"):
                    handle = renderer.render(vertices, view, intrinsics, controls=torch.tensor(controls, device="cuda"), sample_position=position)
                    handle.ready.synchronize()
                # Initial display/record path intentionally synchronizes; optimization is a separate measurement.
                with session.metrics.wall("compositing_wall_ms"):
                    if not output.write(handle, session.clock.submitted if not device else None): break
                rendered_ns = time.monotonic_ns()
                if first_render_ns is None: first_render_ns = rendered_ns
                if previous_render_ns is not None:
                    session.metrics.add("presentation_interval_ms", (rendered_ns-previous_render_ns)/1e6)
                previous_render_ns = rendered_ns
                ticks += 1
                session.metrics.add("audio_buffer_ms", ring.depth / 24)
                session.metrics.add("torch_allocated_mib", torch.cuda.memory_allocated()/2**20)
            if source.done_audio.is_set() and source.done_features.is_set() and source.audio.empty() and source.features.empty() and pending is None and ring.depth == 0:
                if not device or session.clock.position() >= session.accepted: break
            # Drop expired frame slots; never accelerate audio to catch up frames.
            now = time.monotonic()
            deadline += (max(1, int((now-deadline)*60)+1)) / 60
            time.sleep(max(0, deadline-time.monotonic()))
        if source.done_audio.is_set() and not args.resident: source.process.wait(timeout=10); source.check()
        if source.first_audio_ns:
            session.metrics.add("tts_warm_time_to_first_audio_ms" if args.resident else "tts_cold_time_to_first_audio_ms", (source.first_audio_ns-source.started_ns)/1e6)
        if source.startup.first_speech_ns:
            session.metrics.add("tts_first_speech_chunk_heuristic_ms", (source.startup.first_speech_ns-source.started_ns)/1e6)
            session.metrics.add("tts_leading_low_energy_ms", source.startup.leading_low_energy_samples/24)
        session.metrics.add("fps_elapsed", ticks / max(1e-9, (time.monotonic_ns()-source.started_ns)/1e9))
        if ticks > 1:
            session.metrics.add("fps_presentation", (ticks-1) / ((previous_render_ns-first_render_ns)/1e9))
        return {"ticks": ticks, "audio_samples": session.accepted, "metrics": session.metrics.report(),
                "sink": args.sink, "purpose": avatar.metadata["purpose"]}
    finally:
        vertices = None
        close_resources(source, sampler, device, output, shared, ring, rig)
