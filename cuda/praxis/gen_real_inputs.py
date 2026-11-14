"""Deterministic synthetic DROID-like policy inputs for real_praxis runs (no model needed).
  ~/.venv/bin/python -I cuda/praxis/gen_real_inputs.py --out tmp/praxis_real/in [--prompt "..."] [--frames 25]
Writes ids.npy / ids_u.npy (Qwen3.5 tokenizer, no chat template, blank caption " " for the unguided branch), ext.npy / wrist.npy
(uint8 [T][720][1280][3]) and pose8.npy (float32 [K'][8] xyz, quat wxyz, grip)."""
import argparse, os, numpy as np
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--prompt", default="pick up the red cup and put it in the sink")
ap.add_argument("--frames", type=int, default=25); ap.add_argument("--qwen", default=PRAXIS_ROOT + "/qwen3.5-2b"); ap.add_argument("--K", type=int, default=2)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(a.qwen)
for name, txt in (("ids", a.prompt), ("ids_u", " ")):
    ids = tk(txt, add_special_tokens=True)["input_ids"]
    np.save(f"{a.out}/{name}.npy", np.asarray(ids, np.int64)); print(name, len(ids))
H, W = 720, 1280
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def view(seed):
    rng = np.random.RandomState(seed); fr = []
    base = np.stack([60 + 100 * xx / W, 80 + 90 * yy / H, 120 + 40 * np.sin(xx / 97.0 + seed) * np.cos(yy / 71.0)], -1)
    for t in range(a.frames):
        f = base.copy()
        cx, cy = 300 + 8 * t + 40 * seed, 360 + 3 * t
        m = (np.abs(xx - cx) < 90) & (np.abs(yy - cy) < 70); f[m] = (200, 40, 40)
        m2 = (xx - (900 - 5 * t)) ** 2 + (yy - 500) ** 2 < 60 ** 2; f[m2] = (230, 230, 60)
        f += rng.randn(H, W, 3).astype(np.float32) * 3
        fr.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(fr)
np.save(f"{a.out}/ext.npy", view(1)); np.save(f"{a.out}/wrist.npy", view(2))
p = np.zeros((a.K, 8), np.float32)
for k in range(a.K): p[k] = [0.50 + 0.01 * k, 0.0 + 0.005 * k, 0.30, 0.0, 1.0, 0.0, 0.0, 0.4]
np.save(f"{a.out}/pose8.npy", p)
print("wrote", a.out)
