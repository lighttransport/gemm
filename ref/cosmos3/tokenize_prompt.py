#!/usr/bin/env python3
"""Standalone token-id helper for the CUDA pipeline (no model/GPU needed): reproduces Cosmos3OmniPipeline.tokenize_prompt for action mode.
  ~/.venv/bin/python -I ref/cosmos3/tokenize_prompt.py --prompt "pick up the red cup and place it on the plate" --out tmp/cosmos3_ref [--canvas-h 192 --canvas-w 320 --chunk 32 --fps 15]
Writes input_ids_cond.npy / input_ids_uncond.npy (int64) = ids the C driver reads. --check compares with an existing gen_ref.py export."""
import argparse, json, os, sys, numpy as np
from transformers import AutoTokenizer
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/disk1/praxis/cosmos3-edge-droid"); ap.add_argument("--out", default="tmp/cosmos3_ref")
ap.add_argument("--prompt", default="pick up the red cup and place it on the plate")
ap.add_argument("--canvas-h", type=int, default=192); ap.add_argument("--canvas-w", type=int, default=320)
ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--view-point", default="concat_view"); ap.add_argument("--check", action="store_true")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(f"{a.model}/text_tokenizer")
special = [tok.eos_token_id, tok.convert_tokens_to_ids("<|vision_start|>")]
def enc(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, add_vision_id=False, return_dict=True).input_ids
    return np.array(list(ids) + special, dtype=np.int64)       # default_use_system_prompt=false
cond = enc(Cosmos3OmniPipeline._build_action_json_prompt(a.prompt, view_point=a.view_point, num_frames=a.chunk + 1, fps=a.fps, height=a.canvas_h, width=a.canvas_w))
unc = enc("")
os.makedirs(a.out, exist_ok=True)
if a.check:
    c0, u0 = np.load(f"{a.out}/input_ids_cond.npy"), np.load(f"{a.out}/input_ids_uncond.npy")
    print("cond ids equal:", np.array_equal(cond, c0), len(cond), "| uncond ids equal:", np.array_equal(unc, u0), len(unc)); sys.exit(0 if np.array_equal(cond, c0) and np.array_equal(unc, u0) else 1)
np.save(f"{a.out}/input_ids_cond.npy", cond); np.save(f"{a.out}/input_ids_uncond.npy", unc); print("wrote", len(cond), len(unc))
