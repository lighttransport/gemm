#!/usr/bin/env python3
"""Dump Qwen3.5-2B text-encoder reference (hidden_states[-3], pre-final-norm).

Uses plain transformers (no fla kernels needed; fp32 compute by default for an
accurate oracle). Output: <out>/ids.npy, layer_NN.npy (per-layer output, fp32),
cond.npy (= hidden_states[-3], the praxis conditioning).
"""
import argparse, os, numpy as np, torch
from transformers import AutoModel, AutoProcessor
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=PRAXIS_ROOT + "/qwen3.5-2b")
ap.add_argument("--prompt", default="pick up the red cup and place it on the plate")
ap.add_argument("--prompt-file", help="read the prompt text from this file (overrides --prompt)")
ap.add_argument("--ids", help="use token ids from this .npy (int32/int64) instead of tokenizing")
ap.add_argument("--pad-to", type=int, default=1, help="right-pad ids with <|endoftext|> to a multiple of N (outputs are trimmed to the real length in the C runner; here only for oracle of padded run)")
ap.add_argument("--out", default="tmp/praxis_ref/text")
ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.prompt_file:
    a.prompt = open(a.prompt_file, encoding="utf-8").read().rstrip("\n")
if a.ids:
    ids = torch.from_numpy(np.load(a.ids).astype(np.int64).reshape(1, -1)).to(a.device)
else:
    proc = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    b = proc(text=[a.prompt], padding=False, truncation=False, add_special_tokens=True, return_tensors="pt")
    ids = b["input_ids"].to(a.device)
if a.pad_to > 1 and ids.shape[1] % a.pad_to:
    npad = (-ids.shape[1]) % a.pad_to
    ids = torch.cat([ids, torch.full((1, npad), 151643, dtype=ids.dtype, device=ids.device)], 1)
# NOTE: with transformers>=5 AutoModel (Qwen3_5Model) expects "language_model.*" keys while the checkpoint
# stores "model.language_model.*" -> every weight is silently re-initialised (LOAD REPORT: MISSING).
# Load the full conditional-generation class and use its text tower instead.
from transformers import AutoModelForImageTextToText
full = AutoModelForImageTextToText.from_pretrained(a.model, local_files_only=True, dtype=getattr(torch, a.dtype))
m = full.model.language_model.to(a.device).eval()
del full
with torch.no_grad():
    o = m(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
hs = o.hidden_states  # 25 entries: embed + 24 layers (last is post-final-norm)
np.save(f"{a.out}/ids.npy", ids.cpu().numpy())
for i, h in enumerate(hs):
    np.save(f"{a.out}/layer_{i:02d}.npy", h[0].float().cpu().numpy())
np.save(f"{a.out}/cond.npy", hs[-3][0].float().cpu().numpy())
print("tokens", ids.shape, "n_hidden", len(hs), "cond", tuple(hs[-3].shape))
