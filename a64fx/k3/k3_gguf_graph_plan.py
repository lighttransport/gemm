#!/usr/bin/env python3
"""Validate the K3 IQ GGUF loader/graph contract without mapping payloads."""
from __future__ import print_function

import argparse
import sys
from pathlib import Path

from k3_gguf_stage import discover

LAYERS = 93
MLA = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
       51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}

KDA = {
    "attn_q.weight": (7168, 12288), "attn_k.weight": (7168, 12288),
    "attn_v.weight": (7168, 12288), "ssm_g.weight": (7168, 12288),
    "ssm_f_a.weight": (7168, 128), "ssm_f_b.weight": (128, 12288),
    "ssm_beta.weight": (7168, 96), "ssm_conv1d_q.weight": (4, 1, 12288),
    "ssm_conv1d_k.weight": (4, 1, 12288), "ssm_conv1d_v.weight": (4, 1, 12288),
    "ssm_a": (96,), "ssm_dt.bias": (12288,), "ssm_norm.weight": (128,),
}
MLA_ROLES = {
    "attn_q_a.weight": (7168, 1536), "attn_q_a_norm.weight": (1536,),
    "attn_q_b.weight": (1536, 18432), "attn_kv_a_mqa.weight": (7168, 576),
    "attn_kv_a_norm.weight": (512,), "attn_k_b.weight": (128, 512, 96),
    "attn_v_b.weight": (512, 128, 96), "attn_gate.weight": (7168, 12288),
}

def expect(names, layer, suffix):
    name = "blk.%d.%s" % (layer, suffix)
    if name not in names:
        raise ValueError("missing %s" % name)
    return names[name]

def check_shape(rec, wanted, name):
    got = tuple(rec["dims"])
    if got != wanted:
        raise ValueError("%s shape=%s expected=%s" % (name, got, wanted))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--layers", default="0,1,3,92",
                    help="comma-separated representative layers (default: 0,1,3,92)")
    args = ap.parse_args()
    _, records, split_count = discover(args.model_dir)
    names = {r["name"]: r for r in records}
    for global_name in ("token_embd.weight", "output.weight", "output_norm.weight"):
        if global_name not in names:
            raise ValueError("missing global tensor %s" % global_name)
    checked = 0
    for layer in [int(x) for x in args.layers.split(",")]:
        if layer < 0 or layer >= LAYERS:
            raise ValueError("layer out of range: %d" % layer)
        for suffix, shape in {
            "attn_norm.weight": (7168,), "ffn_norm.weight": (7168,),
            "attn_res_score.weight": (7168,), "ffn_res_score.weight": (7168,),
        }.items():
            rec = expect(names, layer, suffix)
            check_shape(rec, shape, rec["name"])
            checked += 1
        roles = KDA if layer not in MLA else MLA_ROLES
        for suffix, shape in roles.items():
            rec = expect(names, layer, suffix)
            check_shape(rec, shape, rec["name"])
            checked += 1
        if layer == 0:
            for suffix, shape in {
                "ffn_gate.weight": (7168, 33792),
                "ffn_up.weight": (7168, 33792),
                "ffn_down.weight": (33792, 7168),
            }.items():
                rec = expect(names, layer, suffix)
                check_shape(rec, shape, rec["name"])
                checked += 1
        else:
            for suffix, shape in {
                "exp_probs_b.bias": (896,), "ffn_gate_inp.weight": (7168, 896),
                "ffn_up_exps.weight": (3584, 3072, 896),
                "ffn_gate_exps.weight": (3584, 3072, 896),
                "ffn_down_exps.weight": (3072, 3584, 896),
                "ffn_routed_down.weight": (7168, 3584),
                "ffn_routed_norm.weight": (3584,),
                "ffn_routed_up.weight": (3584, 7168),
                "ffn_gate_shexp.weight": (7168, 6144),
                "ffn_up_shexp.weight": (7168, 6144),
                "ffn_down_shexp.weight": (6144, 7168),
            }.items():
                rec = expect(names, layer, suffix)
                check_shape(rec, shape, rec["name"])
                checked += 1
    print("K3_GGUF_GRAPH_PLAN PASS layers=%s checked=%d tensors=%d split_count=%s mla=%d kda=%d" %
          (args.layers, checked, len(records), split_count, len(MLA), LAYERS - len(MLA)))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("K3_GGUF_GRAPH_PLAN FAIL: %s" % exc, file=sys.stderr)
        sys.exit(1)
