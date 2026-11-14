#!/usr/bin/env python3
"""Write a config.json (praxis checkpoint layout, the keys cuda_praxis_runner.h reads) at the INFERRED real
praxis-1-droid dims, for the synthetic full-size model (bench_praxis_synth --synthetic DIR).
Real dims are inferred from praxis-1-oss constructor defaults + README (checkpoint is gated):
dim 3072, 24 Q / 8 KV heads, head_dim 128, dense 11520, 128 experts top-8 (hidden 1280), encoder 14 blocks (0,1 dense),
decoder 6 blocks (0 dense), action expert 16 blocks dim 1024."""
import argparse, json, os
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=PRAXIS_ROOT + "/synth")
ap.add_argument("--dim", type=int, default=3072)
ap.add_argument("--head-dim", type=int, default=128)
ap.add_argument("--kv-heads", type=int, default=8)
ap.add_argument("--dense", type=int, default=11520)
ap.add_argument("--experts", type=int, default=128)
ap.add_argument("--top-k", type=int, default=8)
ap.add_argument("--expert-hidden", type=int, default=1280)
ap.add_argument("--enc-depth", type=int, default=14)
ap.add_argument("--dec-depth", type=int, default=6)
ap.add_argument("--ae-dim", type=int, default=1024)
ap.add_argument("--ae-layers", type=int, default=16)
ap.add_argument("--ae-heads", type=int, default=16)
ap.add_argument("--ae-head-dim", type=int, default=64)
ap.add_argument("--nfut", type=int, default=8)
ap.add_argument("--text-c", type=int, default=2048)
a = ap.parse_args()
TEXT, VIDEO = "TextEncodingTensor", "VideoTensor"
taps = [t for t in (0, 2, 4, 6, 7, 9, 11, 13) if t < a.enc_depth]
cfg = {
  "format_version": 2, "model_type": "praxis", "embodiment": "single_arm",
  "backbone": {
    "layout": "chunk_causal", "dim": a.dim, "timestep_dim": 128, "num_registers": 4, "modulation_dtype": "mixed",
    "dense_ffn_hidden": a.dense,
    "attention": {"num_heads": a.dim // a.head_dim, "num_kv_heads": a.kv_heads, "head_dim": a.head_dim, "flex_block_size": 128},
    "rope": {"effective_dim": 64, "base": 26000000, "efficient_cache_max": 1000000, "ref_placement": "group"},
    "moe": {"num_routed_experts": a.experts, "top_k_routed": a.top_k, "expert_hidden": a.expert_hidden,
            "routed_scaling_factor": 2.5, "dispatch_chunk_size": 4096},
    "encoder": {"depth": a.enc_depth, "dense_blocks": [0, 1], "tap_blocks": taps,
                "input_projections": {TEXT: {"c_in": a.text_c}, VIDEO: {"c_in": 192}}, "positive_zero_residual_add": True},
    "decoder": {"depth": a.dec_depth, "dense_blocks": [0], "modulation_dim": a.dim, "unpatch_factor": [1, 2, 2],
                "input_projections": {VIDEO: {"c_in": 48}}, "positive_zero_residual_add": True},
  },
  "action_expert": {
    "dim": a.ae_dim, "num_layers": a.ae_layers, "num_heads": a.ae_heads, "head_dim": a.ae_head_dim, "trunk_dim": a.dim,
    "action_patch_dim": 40, "proprio_patch_dim": 10, "num_future_per_chunk": a.nfut, "max_chunks": 16, "max_action_tokens": 64,
    "tap_norm": True, "clean_only_conditioning": False, "rope_cross": False, "rope_self": False, "rope_base": 100.0,
    "absolute_positions": True, "action_pos_table": False, "future_chunk_offset": "zero",
    "rope": {"kind": "trunk", "dim": 64, "effective_dim": 64, "base": 26000000, "table_size": 1000000, "order": "before_norm"},
    "global_order": ["register", "anchor", "text"], "attention": "sdpa", "head_plus_zero": True,
  },
  "action_space": {"dim": 10},
  "chunking": {"n_chunks": 4, "latents_per_chunk": 6, "action_chunk": 24},
  "text_encoder": {"uncond": "blank_caption", "blank_caption": " "},
}
os.makedirs(a.out, exist_ok=True)
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
print("wrote", os.path.join(a.out, "config.json"))
