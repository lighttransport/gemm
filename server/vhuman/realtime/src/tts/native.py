"""Streaming native Qwen subprocess adapter; model residency is not yet a daemon.

Two independent pipes prevent hidden-state/audio framing ambiguity. Bound queues
apply backpressure on the model worker; cancellation kills and joins it, then
the owner can stop the device and reset the session epoch.
"""
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from .pcm import read_pcm
from .features import read_features
from .startup import StartupMeter


class NativeTTS:
    def __init__(self, runner, model, revision, text, work, epoch=0, speaker="Ono_Anna", max_frames=256, threads=8, text_feed="incremental", *, backend="cuda", language="Japanese"):
        if text_feed not in ("incremental", "full"): raise ValueError("invalid text feed")
        if backend not in ("cpu", "cuda", "rocm"): raise ValueError("invalid native TTS backend")
        if language not in ("Japanese", "English"): raise ValueError("unsupported mobile speech language")
        from .identity import verify_model
        verify_model(model, revision)
        self.audio, self.features = queue.Queue(4), queue.Queue(4)
        self.errors = queue.SimpleQueue()
        self.done_audio = threading.Event(); self.done_features = threading.Event()
        self.stop = threading.Event()
        self.started_ns = time.monotonic_ns(); self.first_audio_ns = None
        self.startup = StartupMeter()
        work = Path(work); work.mkdir(parents=True, exist_ok=True)
        reader, writer = os.pipe()
        self.stderr = (work / "tts.stderr.log").open("w")
        try:
            self.process = subprocess.Popen([str(Path(runner).resolve()), "--backend", backend, "--model", str(model),
                "--text", text, "--speaker", speaker, "--language", language, "--streaming" if text_feed == "incremental" else "--non-streaming", "--max-frames", str(max_frames),
                "--threads", str(threads),
                "--pcm-out", "-", "--features-out", f"/proc/self/fd/{writer}", "--out", str(work / "tts.wav")],
                stdout=subprocess.PIPE, stderr=self.stderr, pass_fds=(writer,), bufsize=0)
        except Exception:
            os.close(reader); self.stderr.close(); raise
        finally: os.close(writer)
        self.feature_pipe = os.fdopen(reader, "rb", buffering=0)
        def pump(stream, records, dest, done, is_audio=False):
            try:
                for record in records:
                    if is_audio:
                        received = time.monotonic_ns()
                        if self.first_audio_ns is None: self.first_audio_ns = received
                        self.startup.observe(record, received)
                    while not self.stop.is_set():
                        try: dest.put(record, timeout=.05); break
                        except queue.Full: pass
                    if self.stop.is_set(): break
            except Exception as error:
                if not self.stop.is_set(): self.errors.put(error)
            finally:
                stream.close(); done.set()
        self.threads = [threading.Thread(target=pump, args=(self.process.stdout, read_pcm(self.process.stdout, epoch), self.audio, self.done_audio, True)),
                        threading.Thread(target=pump, args=(self.feature_pipe, read_features(self.feature_pipe, revision, epoch), self.features, self.done_features))]
        for thread in self.threads: thread.start()

    def check(self):
        if not self.errors.empty(): raise self.errors.get()
        code = self.process.poll()
        if code is not None and code != 0: raise RuntimeError(f"native TTS exited {code}; see {self.stderr.name}")

    def close(self):
        self.stop.set()
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        for thread in self.threads: thread.join(timeout=2)
        if any(t.is_alive() for t in self.threads): raise RuntimeError("native TTS readers failed to stop")
        self.stderr.close()
