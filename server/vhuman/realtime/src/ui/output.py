"""Explicit CPU transfer/compositing stage; optional display, recording and camera."""
from pathlib import Path
import subprocess
import time
import wave
import numpy as np


class FrameOutput:
    def __init__(self, size=(512, 512), fps=60, display=False, video=None, virtual_camera=False):
        self.size, self.fps = size, fps
        self.pygame = self.window = self.encoder = self.camera = None
        self.encoder_origin = None
        self.encoder_index = -1
        self.last_pixels = None
        self.video_path = self.silent_path = self.audio_path = self.audio_file = None
        if display:
            import pygame
            self.pygame = pygame
            pygame.init(); self.window = pygame.display.set_mode(size)
            pygame.display.set_caption("vhuman neural avatar")
        if video:
            Path(video).parent.mkdir(parents=True, exist_ok=True)
            self.video_path = Path(video)
            self.silent_path = self.video_path.with_name(self.video_path.stem + ".video-only.mp4")
            self.audio_path = self.video_path.with_name(self.video_path.stem + ".playout.wav")
            self.encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{size[0]}x{size[1]}",
                "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.silent_path)], stdin=subprocess.PIPE)
        if virtual_camera:
            import pyvirtualcam
            self.camera = pyvirtualcam.Camera(width=size[0], height=size[1], fps=fps)

    def prepare(self, handle):
        """Warm conversion kernels before playback, or produce a composited RGB8 frame."""
        handle.ready.synchronize()
        import torch
        rgba = handle.rgba
        # Premultiplied renderer RGB over a fixed grey background, then sRGB.
        linear = (rgba[..., :3] + (1 - rgba[..., 3:4]) * .18).clamp(0, 1)
        rgb = torch.where(linear <= .0031308, linear * 12.92, 1.055 * linear.pow(1/2.4) - .055)
        # Transfer RGB8, not RGBA float32; keep color conversion on the GPU.
        pixels = rgb.mul(255).round().to(torch.uint8).contiguous().cpu().numpy()
        return pixels

    def audio(self, pcm):
        """Record actual headless playout, including jitter/underrun silence."""
        if not self.encoder or not len(pcm): return
        pcm = np.asarray(pcm)
        if pcm.ndim != 1 or not np.isfinite(pcm).all(): raise ValueError("invalid recording PCM")
        if self.audio_file is None:
            self.audio_file = wave.open(str(self.audio_path), "wb")
            self.audio_file.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        self.audio_file.writeframesraw((np.clip(pcm, -1, 1) * 32767).round().astype("<i2").tobytes())

    def write(self, handle, playout_sample=None):
        if not any((self.window, self.encoder, self.camera)): return True
        pixels = self.prepare(handle)
        if self.pygame:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT: return False
            self.window.blit(self.pygame.surfarray.make_surface(pixels.transpose(1, 0, 2)), (0, 0))
            self.pygame.display.flip()
        if self.encoder:
            if playout_sample is None:
                now = time.monotonic_ns()
                if self.encoder_origin is None: self.encoder_origin = now
                index = (now - self.encoder_origin) * self.fps // 1_000_000_000
            else:
                from ..pipeline.protocol import integer
                integer(playout_sample, "playout_sample")
                index = playout_sample * self.fps // 24000
            # Rawvideo has fixed FPS: duplicate previous pixels on missed deadlines
            # and drop extra renders in the same slot, preserving elapsed duration.
            while self.encoder_index < index - 1:
                self.encoder.stdin.write(self.last_pixels if self.last_pixels is not None else pixels.tobytes())
                self.encoder_index += 1
            if index > self.encoder_index:
                self.last_pixels = pixels.tobytes()
                self.encoder.stdin.write(self.last_pixels)
                self.encoder_index = index
        if self.camera: self.camera.send(pixels)
        return True

    def close(self):
        if self.audio_file: self.audio_file.close()
        if self.encoder:
            self.encoder.stdin.close()
            if self.encoder.wait(timeout=20): raise RuntimeError("ffmpeg recording failed")
            if self.audio_file:
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                    "-i", str(self.silent_path), "-i", str(self.audio_path),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    str(self.video_path)], check=True)
                self.silent_path.unlink()
            else: self.silent_path.replace(self.video_path)
        if self.camera: self.camera.close()
        if self.pygame: self.pygame.quit()
