"""Persistent TTS soak test with bounded draining, timing and process memory."""
import json
from pathlib import Path
import queue
import time
import numpy as np
from ..tts.resident import ResidentTTS
from ..avatar.provenance import sha256


def stress(model, runner, output, work, requests=100, threads=4, text_feed="incremental"):
    if requests < 2: raise ValueError("at least two requests required")
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    worker = None
    timings, memory, durations, rms_values, speech_timings = [], [], [], [], []
    try:
        worker = ResidentTTS(runner, model, sha256(Path(model)/"model.safetensors"), work,
                             max_frames=256, threads=threads, text_feed=text_feed)
        pid = worker.process.pid
        for epoch in range(requests):
            worker.submit(("こんにちは。", "ありがとうございます。", "今日はいい天気ですね。")[epoch%3], epoch)
            counts = [0, 0]; energy = 0.
            deadline = time.monotonic()+30
            while True:
                worker.check()
                for index, destination in enumerate((worker.audio, worker.features)):
                    while True:
                        try: record = destination.get_nowait()
                        except queue.Empty: break
                        if record.epoch != epoch or record.sample_start != counts[index]*1920:
                            raise ValueError("stale or noncontiguous soak-test record")
                        counts[index] += 1
                        if index == 0: energy += float(np.square(record.pcm.astype(np.float64)).sum())
                if worker.done_audio.is_set() and worker.done_features.is_set() and worker.audio.empty() and worker.features.empty(): break
                if time.monotonic() > deadline: raise TimeoutError("soak request exceeded30s")
                time.sleep(.001)
            if not counts[0] or counts[0] != counts[1]: raise ValueError("soak PCM/feature mismatch")
            rms = float(np.sqrt(energy/(counts[0]*1920)))
            if rms < .003: raise ValueError("soak request produced nearly silent audio")
            timings.append((worker.first_audio_ns-worker.started_ns)/1e6)
            if worker.startup.first_speech_ns is None: raise ValueError("no speech-bearing chunk observed")
            speech_timings.append((worker.startup.first_speech_ns-worker.started_ns)/1e6)
            durations.append(counts[0]*.08); rms_values.append(rms)
            memory.append(next(p.usedGpuMemory/2**20 for p in nv.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid == pid))
            if epoch%10 == 0: print(json.dumps(dict(completed=epoch+1, first_pcm_ms=timings[-1], vram_mib=memory[-1])), flush=True)
    finally:
        if worker: worker.close()
        nv.nvmlShutdown()
    result = dict(format="vhuman.tts_soak.v1", requests=requests, threads=threads, text_feed=text_feed,
        first_pcm_ms=dict(p50=float(np.percentile(timings,50)),p95=float(np.percentile(timings,95)),p99=float(np.percentile(timings,99)),maximum=max(timings)),
        first_speech_chunk_heuristic_ms=dict(p50=float(np.percentile(speech_timings,50)),p95=float(np.percentile(speech_timings,95)),maximum=max(speech_timings)),
        process_vram_mib=dict(first=memory[0],last=memory[-1],minimum=min(memory),maximum=max(memory),growth=memory[-1]-memory[0]),
        audio_seconds=sum(durations), minimum_utterance_rms=min(rms_values), worker_pid=pid,
        limitations="post-request VRAM samples, not allocation peaks; first PCM is not first audible phoneme; external contention is not controlled by this tool")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    return result
