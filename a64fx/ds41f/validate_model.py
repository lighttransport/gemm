#!/usr/bin/env python3
"""Validate the local DeepSeek-V4.1-Flash checkpoint contract.

This intentionally reads only safetensors headers.  It never maps or loads a
large tensor, which makes it safe to run on a login node or before staging.
"""
import argparse
import json
import os
import struct
import sys

EXPECTED = {
    "hidden_size": 5120, "num_hidden_layers": 40,
    "num_attention_heads": 64, "num_key_value_heads": 1,
    "head_dim": 512, "q_lora_rank": 1280, "o_lora_rank": 1024,
    "num_experts_per_tok": 6, "n_routed_experts": 384,
    "n_shared_experts": 1, "moe_intermediate_size": 2304,
    "max_position_embeddings": 1048576,
}
REQUIRED = {
    "embed.weight": ("BF16", [129280, 5120]),
    "head.weight": ("BF16", [129280, 5120]),
    "norm.weight": ("BF16", [5120]),
    "layers.0.attn.wq_a.weight": ("F8_E4M3", [1280, 5120]),
    "layers.0.attn.wq_b.weight": ("F8_E4M3", [32768, 1280]),
    "layers.0.attn.wkv.weight": ("F8_E4M3", [512, 5120]),
    "layers.0.ffn.experts.0.w1.weight": ("I8", [2304, 2560]),
    "layers.0.ffn.experts.0.w2.weight": ("I8", [5120, 1152]),
    "layers.0.ffn.shared_experts.w1.weight": ("F8_E4M3", [2304, 5120]),
    "layers.1.engram.embed.weight": ("F8_E4M3", [384006168, 256]),
    "layers.2.attn.compressor.wkv.weight": ("BF16", [512, 5120]),
    "layers.2.attn.indexer.wq_b.weight": ("F8_E4M3", [4096, 1280]),
}

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def fail(msg):
    print("DS41F_VALIDATE FAIL:", msg, file=sys.stderr)
    raise SystemExit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    args = ap.parse_args()
    root = os.path.abspath(args.model_dir)
    try:
        cfg = json.load(open(os.path.join(root, "config.json")))
        idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    except OSError as e:
        fail(str(e))
    text = cfg.get("text_config", cfg)
    for key, value in EXPECTED.items():
        if text.get(key, cfg.get(key)) != value:
            fail(f"config {key}={text.get(key, cfg.get(key))!r}, expected {value!r}")
    ratios = text.get("compress_ratios", cfg.get("compress_ratios"))
    if ratios[:40] != [0, 0] + [2] * 18 + [1] * 20:
        fail(f"unexpected compression ratios: {ratios!r}")
    if text.get("kv_source_layer_ids", cfg.get("kv_source_layer_ids")) != [2, 8, 14, 20]:
        fail("unexpected KV source layers")
    if text.get("index_source_layer_ids", cfg.get("index_source_layer_ids")) != [2, 8, 14, 20, 24, 28, 32, 36]:
        fail("unexpected index source layers")
    wm = idx.get("weight_map", {})
    if len(wm) != 96085 or len(set(wm.values())) != 48:
        fail(f"weight map has {len(wm)} tensors in {len(set(wm.values()))} files")
    headers = {}
    for fn in sorted(set(wm.values())):
        path = os.path.join(root, fn)
        if not os.path.exists(path): fail(f"missing shard {fn}")
        try: headers[fn] = header(path)
        except Exception as e: fail(f"cannot read header {fn}: {e}")
    def tensor(name):
        fn = wm.get(name)
        if fn is None: fail(f"missing tensor {name}")
        try: return headers[fn][name]
        except KeyError: fail(f"tensor {name} absent from shard header")
    for name, (dtype, shape) in REQUIRED.items():
        got = tensor(name)
        if got.get("dtype") != dtype or got.get("shape") != shape:
            fail(f"{name}: {(got.get('dtype'), got.get('shape'))!r}, expected {(dtype, shape)!r}")
    for layer in range(40):
        for expert in (0, 383):
            name = f"layers.{layer}.ffn.experts.{expert}.w1.weight"
            if name not in wm: fail(f"missing {name}")
    print("DS41F_VALIDATE PASS layers=40 experts=384 tensors=96085 shards=48")

if __name__ == "__main__": main()
