"""The procedural eye/idle motion layer: determinism, statistics, VOR, parity with the JS port."""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from .rig import motion_layer as ML

ROOT = Path(__file__).resolve().parents[2]


class MotionLayerTest(unittest.TestCase):
    def test_deterministic_and_query_independent(self):
        a = ML.track(10.0, 30, seed=3)
        b = ML.track(10.0, 30, seed=3)
        self.assertEqual(a, b)
        layer = ML.MotionLayer(3)
        coarse = [layer.sample(t) for t in np.arange(0, 10.0, 1 / 7.0)]
        dense = ML.MotionLayer(3)
        for t in np.arange(0, 10.0, 1 / 7.0):
            self.assertEqual(dense.sample(t)["eyeBlinkLeft"], coarse[int(round(t * 7))]["eyeBlinkLeft"])
        self.assertNotEqual(ML.track(5.0, 30, seed=4)[40], ML.track(5.0, 30, seed=5)[40])
        with self.assertRaises(ValueError):
            layer.sample(1.0)

    def test_blink_and_saccade_statistics(self):
        frames = ML.track(120.0, 60, seed=11)
        blink = np.array([f["eyeBlinkLeft"] for f in frames])
        starts = np.flatnonzero((blink[1:] > 0) & (blink[:-1] == 0))
        rate = len(starts) / 2.0                                   # per minute
        self.assertTrue(12 <= rate <= 22, rate)
        closed = (blink > 0.98).sum() / 60.0
        self.assertLess(closed, 0.02 * 120)                        # eyes closed about 1% of the time
        gaze = np.array([f["_gaze_deg"] for f in frames])
        speed = np.abs(np.diff(gaze[:, 0])) * 60                  # deg/s
        self.assertGreater(speed.max(), 100.0)                     # saccades are fast
        self.assertLess(np.quantile(speed, 0.9), 40.0)             # ... and rare; fixations are still
        self.assertLess(np.abs(gaze).max(), 25.0)
        for f in frames:
            for k, v in f.items():
                if k != "_gaze_deg":
                    self.assertTrue(0.0 <= v <= 1.0 or k.startswith("head"), (k, v))
                    self.assertTrue(np.isfinite(v))

    def test_vor_convergence_and_pauses(self):
        quiet = ML.MotionLayer(2)
        turned = ML.MotionLayer(2)
        a = quiet.sample(1.0, focus_distance=10.0)
        b = turned.sample(1.0, head_yaw=0.5, focus_distance=10.0)        # the head turns 20 deg left
        self.assertGreater(b["eyeLookInLeft"] + b["eyeLookOutRight"], a["eyeLookInLeft"] + a["eyeLookOutRight"])
        near = ML.MotionLayer(2).sample(1.0, focus_distance=0.25)
        far = ML.MotionLayer(2).sample(1.0, focus_distance=10.0)
        self.assertGreater(near["eyeLookInLeft"], far["eyeLookInLeft"])
        self.assertGreater(near["eyeLookInRight"], far["eyeLookInRight"])
        self.assertGreater(near["pupilNarrow"], far["pupilNarrow"])
        bright = ML.MotionLayer(2).sample(1.0, light=1.0)["pupilNarrow"]
        dark = ML.MotionLayer(2).sample(1.0, light=0.0)["pupilNarrow"]
        self.assertGreater(bright, dark)
        # a pause right after a blink-free stretch triggers a blink soon after
        layer = ML.MotionLayer(9)
        t = 2.0
        while layer.sample(t, pauses=())["eyeBlinkLeft"] > 0 or t < layer.next_blink - 1.5:
            t += 1 / 30
        blinked = any(layer.sample(t + k / 30, pauses=(t + 0.01,))["eyeBlinkLeft"] > 0 for k in range(1, 12))
        self.assertTrue(blinked or layer.cfg["blink_at_pause_p"] < 1)   # probabilistic; see the parity fixture

    @unittest.skipIf(shutil.which("node") is None, "node is not installed")
    def test_js_port_matches(self):
        script = """
        import * as ML from '%s';
        const layer = new ML.MotionLayer(5);
        const out = [];
        for (let i = 0; i <= 300; i++) {
          const t = i / 30;
          out.push(layer.sample(t, { headYaw: 0.2 * ML.valueNoise(t * 0.5, 99), headPitch: 0.1, focusDistance: 0.5,
                                     speaking: i > 100, light: 0.7, pauses: [3.0, 6.5] }));
        }
        console.log(JSON.stringify(out));
        """ % (ROOT / "web" / "vhuman_motion_layer.js").as_uri()
        tmp = ROOT / "tmp" / "motion_layer_parity.mjs"
        tmp.write_text(script)
        try:
            js = json.loads(subprocess.run(["node", str(tmp)], capture_output=True, text=True, check=True).stdout)
        finally:
            tmp.unlink(missing_ok=True)
        layer = ML.MotionLayer(5)
        for i, frame in enumerate(js):
            t = i / 30
            py = layer.sample(t, head_yaw=0.2 * ML.value_noise(t * 0.5, 99), head_pitch=0.1, focus_distance=0.5,
                              speaking=i > 100, light=0.7, pauses=(3.0, 6.5))
            for k, v in py.items():
                if k == "_gaze_deg":
                    continue
                self.assertAlmostEqual(v, frame[k], places=9, msg=(i, k))


    @unittest.skipIf(shutil.which("cc") is None and shutil.which("gcc") is None, "no C compiler")
    def test_c_port_matches(self):
        build = subprocess.run(["make", "-C", str(ROOT / "cpu" / "vhuman"), "test_motion_layer"],
                               capture_output=True, text=True)
        if build.returncode != 0:
            self.skipTest("cannot build the native motion layer: " + build.stderr[-500:])
        c = json.loads(subprocess.run([str(ROOT / "cpu" / "vhuman" / "test_motion_layer")],
                                      capture_output=True, text=True, check=True).stdout)
        layer = ML.MotionLayer(5)
        for i, frame in enumerate(c):
            t = i / 30
            py = layer.sample(t, head_yaw=0.2 * ML.value_noise(t * 0.5, 99), head_pitch=0.1, focus_distance=0.5,
                              speaking=i > 100, light=0.7, pauses=(3.0, 6.5))
            for k, v in frame.items():
                self.assertAlmostEqual(py[k], v, places=9, msg=(i, k))


    def test_native_library_binding(self):
        build = subprocess.run(["make", "-C", str(ROOT / "cpu" / "vhuman"), "libvhuman_motion_layer.so"],
                               capture_output=True, text=True)
        if build.returncode != 0:
            self.skipTest("cannot build the native motion layer library: " + build.stderr[-500:])
        from .realtime.src.animation.idle_layer import NativeMotionLayer
        native = NativeMotionLayer(5)
        layer = ML.MotionLayer(5)
        for i in range(0, 301, 3):
            t = i / 30
            kw = dict(head_yaw=0.2 * ML.value_noise(t * 0.5, 99), head_pitch=0.1, focus_distance=0.5,
                      speaking=i > 100, light=0.7, pauses=(3.0, 6.5))
            a, b = native.sample(t, **kw), layer.sample(t, **kw)
            for k, v in b.items():
                if k != "_gaze_deg":
                    self.assertAlmostEqual(a[k], v, places=9, msg=(i, k))
        names = ["jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookUpLeft", "headYaw", "pupilNarrow"]
        merged = native.apply(np.array([0.5, 0.9, 0.0, 0.0, 0.1, 0.0], np.float32), names, 10.5)
        self.assertEqual(merged[0], 0.5)
        self.assertGreaterEqual(merged[1], 0.9)                       # blink keeps the larger value
        self.assertNotEqual(merged[4], 0.1)                           # head micro-motion added
        with self.assertRaises(ValueError):
            native.sample(1.0)

        # Inserted silence holds speech position, but procedural motion must advance.
        from .realtime.src.audio.clock import SampleClock
        clock = SampleClock()
        clock.submit(24000, 0)
        clock.observe(24000, 24000, 1_000_000_000)
        native.reset()
        controls = np.zeros(len(names), np.float32)
        with patch('server.vhuman.realtime.src.animation.idle_layer.time.monotonic_ns',
                   side_effect=[1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000]):
            first = native.apply_realtime(controls, names)
            clock.submit(24000)
            clock.observe(48000, 24000, 2_000_000_000)
            self.assertEqual(clock.position(), 24000)
            silent = native.apply_realtime(controls, names)
            self.assertFalse(np.array_equal(first, silent))
            clock.reset(1)
            resumed = native.apply_realtime(controls, names)
            self.assertTrue(np.isfinite(resumed).all())
            native.reset()
            np.testing.assert_array_equal(native.apply_realtime(controls, names), first)


if __name__ == "__main__":
    unittest.main()
