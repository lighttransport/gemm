"""Persistent native worker with serialized requests and frame-boundary cancel."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from .features import read_features
from .pcm import read_exact, read_pcm
from .startup import StartupMeter


class ResidentTTS:
    def __init__(self, runner, model, revision, work, speaker="Ono_Anna", max_frames=256, threads=8, text_feed="incremental"):
        if text_feed not in ("incremental", "full"): raise ValueError("invalid text feed")
        from .identity import verify_model
        verify_model(model, revision)
        self.revision = revision
        self.errors = queue.SimpleQueue()
        self.audio = queue.Queue(4); self.features = queue.Queue(4)
        self.done_audio = threading.Event(); self.done_features = threading.Event()
        self.done_audio.set(); self.done_features.set()
        self.discard = threading.Event(); self.threads = []
        self.started_ns = self.first_audio_ns = None
        self.startup = StartupMeter()
        self.epoch = -1
        work = Path(work); work.mkdir(parents=True, exist_ok=True)
        self.stderr = (work / "resident.stderr.log").open("w")
        reader, writer = os.pipe()
        try:
            self.process = subprocess.Popen([str(Path(runner).resolve()), "--backend", "cuda", "--model", str(model),
                "--serve-stdin", "--streaming" if text_feed == "incremental" else "--non-streaming", "--speaker", speaker, "--language", "Japanese",
                "--threads", str(threads), "--max-frames", str(max_frames), "--pcm-out", "-", "--features-out", f"/proc/self/fd/{writer}"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, pass_fds=(writer,), bufsize=0)
        except Exception:
            os.close(reader); self.stderr.close(); raise
        finally: os.close(writer)
        self.feature_pipe = os.fdopen(reader, "rb", buffering=0)
        try:
            if read_exact(self.process.stdout, 8) != b"VHTTSRDY": raise RuntimeError("invalid native ready marker")
        except Exception:
            self.close(); raise

    def _pump(self, records, destination, done, audio=False):
        try:
            for record in records:
                if audio:
                    received = time.monotonic_ns()
                    if self.first_audio_ns is None: self.first_audio_ns = received
                    self.startup.observe(record, received)
                while not self.discard.is_set():
                    try: destination.put(record, timeout=.05); break
                    except queue.Full: pass
        except Exception as error:
            if self.process.poll() is None: self.errors.put(error)
        finally: done.set()

    def submit(self, text, epoch):
        from ..pipeline.protocol import integer
        integer(epoch, "epoch")
        self.check()
        if epoch <= self.epoch: raise ValueError("utterance epochs must increase")
        if not self.done_audio.is_set() or not self.done_features.is_set() or not self.audio.empty() or not self.features.empty():
            raise BufferError("finish/drain or cancel the previous utterance before submit")
        if not isinstance(text, str) or not text or len(text.encode("utf-8")) > 4096 or "\0" in text:
            raise ValueError("text must be nonempty UTF8 within4096 bytes")
        for thread in self.threads: thread.join()
        self.epoch = epoch; self.discard.clear()
        self.done_audio.clear(); self.done_features.clear()
        self.started_ns = time.monotonic_ns(); self.first_audio_ns = None
        self.startup = StartupMeter()
        self.threads = [threading.Thread(target=self._pump, args=(read_pcm(self.process.stdout, epoch, True), self.audio, self.done_audio, True)),
                        threading.Thread(target=self._pump, args=(read_features(self.feature_pipe, self.revision, epoch, True), self.features, self.done_features))]
        for thread in self.threads: thread.start()
        request = memoryview((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
        while request:
            written = self.process.stdin.write(request)
            if not written: raise RuntimeError("native request pipe closed")
            request = request[written:]

    def check(self):
        if not self.errors.empty(): raise self.errors.get()
        code = self.process.poll()
        if code is not None: raise RuntimeError(f"resident TTS exited {code}; see {self.stderr.name}")

    def cancel(self):
        self.discard.set()
        if not self.done_audio.is_set() or not self.done_features.is_set():
            self.process.send_signal(signal.SIGUSR1)
        for thread in self.threads: thread.join(timeout=3)
        if any(t.is_alive() for t in self.threads):
            self.close(); raise RuntimeError("cancel timed out; worker terminated")
        for destination in (self.audio, self.features):
            while not destination.empty(): destination.get_nowait()
        self.check()

    def close(self):
        self.discard.set()
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        for thread in self.threads: thread.join(timeout=2)
        self.feature_pipe.close(); self.process.stdout.close(); self.process.stdin.close(); self.stderr.close()
