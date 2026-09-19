#!/usr/bin/env python3
"""Generate the real-checkpoint DSpark golden consumed by validate_dspark.

The semantic sources are the checkpoint-shipped DFlash/DSpark modules plus
the official SGLang wrapper behavior pinned in the output metadata.  The LM
head is evaluated as W4A16, matching the A64FX execution contract.
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import load_file

SGLANG_REVISION = "3a64faa1f22a86abd37a759c84267d929e820d5b"


def load_checkpoint_modules(draft_dir: pathlib.Path):
    names = ["specforge", "specforge.modeling", "specforge.modeling.draft"]
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    spec = importlib.util.spec_from_file_location(
        "specforge.modeling.draft.dflash", draft_dir / "dflash.py"
    )
    dflash = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dflash
    spec.loader.exec_module(dflash)
    spec = importlib.util.spec_from_file_location("checkpoint_dspark", draft_dir / "dspark.py")
    dspark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dspark)
    return dspark


def deterministic_taps(tokens=2):
    state = 1
    taps = []
    for _ in range(5):
        values = []
        for _ in range(tokens * 5120):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            values.append((((state >> 16) & 0xFFFF) - 32768) / 32768.0 * 0.02)
        taps.append(torch.tensor(values, dtype=torch.float32).view(tokens, 5120))
    return taps


def fp8_e4m3(raw):
    raw = raw.to(torch.int32)
    sign = torch.where((raw & 0x80) != 0, -1.0, 1.0)
    exp = (raw >> 3) & 15
    mant = raw & 7
    normal = torch.ldexp(1.0 + mant.float() * 0.125, exp - 7)
    subnormal = torch.ldexp(mant.float(), torch.full_like(exp, -9))
    return sign * torch.where(exp == 0, subnormal, normal)


FP4 = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])


def w4a16_logits(hidden, target_dir: pathlib.Path, chunk=256):
    index = json.loads((target_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shard = target_dir / index["lm_head.weight"]
    outputs = []
    with safe_open(shard, framework="pt", device="cpu") as f:
        codes = f.get_tensor("lm_head.weight").to(torch.uint8)
        scales = f.get_tensor("lm_head.weight_scale").view(torch.uint8)
        global_scale = float(f.get_tensor("lm_head.weight_scale_2"))
        for begin in range(0, codes.shape[0], chunk):
            q = codes[begin:begin+chunk]
            lo, hi = q & 15, q >> 4
            unpacked = torch.stack((lo, hi), dim=-1).reshape(q.shape[0], -1)
            scale = fp8_e4m3(scales[begin:begin+chunk]).repeat_interleave(16, dim=1)
            weight = FP4[unpacked.long()] * scale * global_scale
            outputs.append(hidden @ weight.t())
    return torch.cat(outputs, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("draft_dir", type=pathlib.Path)
    ap.add_argument("target_dir", type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    args = ap.parse_args()
    mod = load_checkpoint_modules(args.draft_dir)
    config = mod.DSparkConfig.from_json_file(args.draft_dir / "config.json")
    model = mod.DSparkDraftModel(config).float().eval()
    model.load_state_dict(load_file(args.draft_dir / "model.safetensors"), strict=True)
    taps = deterministic_taps()
    target_hidden = torch.cat(taps, dim=1).unsqueeze(0)
    anchor, mask = 42, int(config.mask_token_id)
    index = json.loads((args.target_dir / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.target_dir / index["model.language_model.embed_tokens.weight"], framework="pt", device="cpu") as f:
        emb = f.get_tensor("model.language_model.embed_tokens.weight")
        noise = emb[torch.tensor([anchor] + [mask] * 6)].float().unsqueeze(0)
    with torch.inference_mode():
        hidden = model(position_ids=torch.arange(9).unsqueeze(0), noise_embedding=noise,
                       target_hidden=target_hidden).squeeze(0).float()
        base = w4a16_logits(hidden, args.target_dir)
        prev, ids, confidence, selected = anchor, [], [], []
        for row in range(7):
            latent = model.markov_head.markov_w1.weight[prev].float()
            logits = base[row] + model.markov_head.markov_w2.weight.float() @ latent
            token = int(torch.argmax(logits))
            feature = torch.cat((hidden[row], latent))
            conf = torch.sigmoid(model.confidence_head.proj(feature)).item()
            ids.append(token); confidence.append(conf); selected.append(float(logits[token])); prev = token
    result = {"schema": 1, "sglang_revision": SGLANG_REVISION,
              "transformers_version": __import__("transformers").__version__,
              "precision": "BF16 weights promoted to FP32; NVFP4 W4A16",
              "context_tokens": 2, "tap_seed": 1, "anchor_token": anchor,
              "token_ids": ids, "confidence": confidence, "selected_logits": selected}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
