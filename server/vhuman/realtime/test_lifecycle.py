"""Inference gates and failure-path cleanup without GPU/model dependencies."""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from .src.pipeline.cleanup import close_resources
from .src.pipeline.live import validate_appearance
from .src.tts.resident import ResidentTTS
from .src.ui.output import FrameOutput


class LifecycleTests(unittest.TestCase):
    def test_shared_renderer_rejects_rocm_torch_before_build(self):
        import torch
        from .src.avatar.native_gpu import NativeSharedGPU
        with patch.object(torch.version, "hip", "test-rocm"), \
             patch("server.vhuman.realtime.src.avatar.native_gpu.build_gpu_library") as build:
            with self.assertRaisesRegex(RuntimeError, "requires CUDA PyTorch"):
                NativeSharedGPU("unused", "unused")
            build.assert_not_called()

    def test_shared_renderer_explicitly_builds_cuda(self):
        import torch
        from .. import gpu
        from .src.avatar.native_gpu import NativeSharedGPU
        stream = SimpleNamespace(device=SimpleNamespace(index=0), cuda_stream=7)
        with gpu.execution("rocm"), patch.object(torch.version, "hip", None), \
             patch.object(torch.cuda, "current_stream", return_value=stream), \
             patch("server.vhuman.realtime.src.avatar.native_gpu.build_gpu_library",
                   side_effect=RuntimeError("stop before loading library")) as build:
            with self.assertRaisesRegex(RuntimeError, "stop before loading library"):
                NativeSharedGPU("unused", "test-build")
            build.assert_called_once_with("test-build", backend="cuda")

    def test_untrained_production_appearance_requires_diagnostic(self):
        avatar = SimpleNamespace(metadata={"purpose": "production", "trained": False})
        with self.assertRaisesRegex(ValueError, "untrained appearance"):
            validate_appearance(avatar)
        validate_appearance(avatar, diagnostic=True)
        avatar.metadata["trained"] = True
        validate_appearance(avatar)
        avatar.metadata["purpose"] = "diagnostic"
        with self.assertRaisesRegex(ValueError, "diagnostic appearance"):
            validate_appearance(avatar)

    def test_cleanup_attempts_every_resource(self):
        first, second, third = Mock(), Mock(), Mock()
        first.close.side_effect = RuntimeError("recording failed")
        second.close.side_effect = ValueError("camera failed")
        with self.assertRaisesRegex(RuntimeError, "recording failed") as caught:
            close_resources(first, second, third)
        for resource in (first, second, third): resource.close.assert_called_once()
        self.assertIn("camera failed", caught.exception.__notes__[0])

    def test_cleanup_preserves_original_error(self):
        resource = Mock()
        resource.close.side_effect = RuntimeError("cleanup failed")
        original = ValueError("inference failed")
        with self.assertRaises(ValueError) as caught:
            try:
                raise original
            finally:
                close_resources(resource)
        self.assertIs(caught.exception, original)
        self.assertIn("cleanup failed", original.__notes__[0])

    def test_recording_failure_still_closes_camera_and_window(self):
        output = FrameOutput()
        output.encoder, output.camera, output.pygame = Mock(), Mock(), Mock()
        output.encoder.wait.return_value = 1
        output.encoder.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "ffmpeg recording failed"):
            output.close()
        output.camera.close.assert_called_once()
        output.pygame.quit.assert_called_once()

    def test_encoder_timeout_kills_process_and_closes_camera(self):
        import subprocess
        output = FrameOutput()
        output.encoder, output.camera = Mock(), Mock()
        output.encoder.wait.side_effect = [subprocess.TimeoutExpired("ffmpeg", 20), -9]
        output.encoder.poll.return_value = None
        with self.assertRaises(subprocess.TimeoutExpired): output.close()
        output.encoder.kill.assert_called_once()
        output.camera.close.assert_called_once()

    def test_cancel_waits_for_request_start_ack(self):
        import queue
        worker = ResidentTTS.__new__(ResidentTTS)
        worker.audio, worker.features = queue.Queue(), queue.Queue()
        worker.discard, worker.request_started = threading.Event(), threading.Event()
        worker.done_audio, worker.done_features = threading.Event(), threading.Event()
        worker.threads = []
        worker.check = lambda: None
        worker.process = Mock()
        worker.process.send_signal.side_effect = lambda _: self.assertTrue(worker.request_started.is_set())
        thread = threading.Thread(target=worker.cancel)
        thread.start()
        try:
            self.assertTrue(worker.discard.wait(1))
            worker.process.send_signal.assert_not_called()
            worker.request_started.set()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            worker.process.send_signal.assert_called_once()
        finally:
            worker.request_started.set(); thread.join(timeout=1)
