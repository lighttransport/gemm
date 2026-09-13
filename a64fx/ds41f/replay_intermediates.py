#!/usr/bin/env python3
"""Replay bounded A64FX layer dumps against original safetensors with NumPy.

Each operator receives the recorded C input, so upstream rounding differences
do not contaminate its comparison. Attention history is rebuilt from recorded
attention inputs, including the compressed source layers. This is an independent
CPU check, not an official GPU oracle or a full autoregressive comparison.
"""
import argparse
from pathlib import Path
import struct

import numpy as np

from reference_numpy import Reference, bf, hc_post


def read_dump(path):
    records = {}
    with Path(path).open("rb") as f:
        if f.read(8) != b"DS41FD1\0":
            raise ValueError(f"invalid dump header: {path}")
        while True:
            label = f.read(32)
            if not label:
                break
            if len(label) != 32:
                raise ValueError(f"truncated label: {path}")
            raw_count = f.read(8)
            if len(raw_count) != 8:
                raise ValueError(f"truncated count: {path}")
            count, = struct.unpack("<Q", raw_count)
            if not 1 <= count <= 20480:
                raise ValueError(f"invalid record length: {path}")
            data = f.read(count * 4)
            if len(data) != count * 4:
                raise ValueError(f"truncated payload: {path}")
            name = label.split(b"\0", 1)[0].decode("ascii")
            if name in records:
                raise ValueError(f"duplicate record {name}: {path}")
            records[name] = np.frombuffer(data, dtype="<f4")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--dump-prefix", required=True)
    ap.add_argument("--prompt-ids", required=True,
                    help="all input IDs for replay, including generated inputs")
    ap.add_argument("--positions", type=int, default=1)
    ap.add_argument("--layers", default="0,1,2,3,4,5,6,7,8",
                    help="comma-separated layer IDs")
    ap.add_argument("--min-cosine", type=float, default=.999)
    ap.add_argument("--max-relative-rms", type=float, default=.01)
    ap.add_argument("--skip-ffn", action="store_true")
    args = ap.parse_args()
    layers = sorted(set(int(x) for x in args.layers.split(",")))
    tokens = [int(x) for x in Path(args.prompt_ids).read_text().split()]
    if (not layers or min(layers) < 0 or max(layers) >= 40 or
            not 1 <= args.positions <= min(64, len(tokens)) or
            not -1 <= args.min_cosine <= 1 or
            not np.isfinite(args.max_relative_rms) or args.max_relative_rms < 0 or
            any(t < 0 or t >= 129280 for t in tokens)):
        ap.error("invalid layer, position, token or cosine bounds")
    ref = Reference(args.model, args.metadata)
    failures = 0
    checks = 0
    worst = (1., "")

    def compare(name, expected, actual, exact=False):
        nonlocal failures, checks, worst
        expected = np.asarray(expected, np.float64).reshape(-1)
        actual = np.asarray(actual, np.float64).reshape(-1)
        if expected.shape != actual.shape:
            raise ValueError(f"geometry mismatch: {name}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError(f"nonfinite values: {name}")
        error = actual - expected
        denom = np.linalg.norm(expected) * np.linalg.norm(actual)
        same = np.array_equal(expected, actual)
        cosine = float(expected @ actual / denom) if denom else float(same)
        relative = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-30))
        ok = same if exact else (cosine >= args.min_cosine and relative <= args.max_relative_rms)
        checks += 1
        failures += not ok
        if cosine < worst[0]:
            worst = (cosine, name)
        print(f"REPLAY {'PASS' if ok else 'FAIL'} {name} cosine={cosine:.9f} "
              f"rms={np.sqrt(np.mean(error**2)):.7g} "
              f"relative_rms={relative:.7g} "
              f"max_abs={np.max(np.abs(error)):.7g} "
              f"different={np.count_nonzero(error)}/{len(error)}", flush=True)

    # Consumers reuse the preceding source's cache. Rebuild every source up
    # through the last selected layer even when it is not itself compared.
    needed = set(layers) | {l for l in (2, 8, 14, 20, 24, 28, 32, 36) if l <= max(layers)}
    for pos in range(args.positions):
        hashes = ref.hashes(tokens[pos])
        for layer in sorted(needed):
            path = f"{args.dump_prefix}.pos{pos}.layer{layer}.bin"
            d = read_dump(path)
            y = ref.attention(layer, pos, d["attn_input"])
            if layer not in layers:
                continue
            label = f"pos={pos} layer={layer} op="
            base = f"layers.{layer}"
            h = d["residual"].reshape(4, 5120)
            if layer in (1, 14):
                expected = ref.engram(layer, h, hashes[0 if layer == 1 else 1])
                compare(label + "engram", expected, d["engram"])
                h = d["engram"].reshape(4, 5120)
            for name, value in zip(("pre", "post", "comb"), ref.mixes(layer, "attn", h)):
                compare(label + "attn_" + name, value, d["attn_" + name])
            x = bf(np.sum(d["pre_mix"][:, None] * h, axis=0))
            compare(label + "attn_input", ref.norm(base + ".attn_norm", x), d["attn_input"])
            compare(label + "attention", y, d["attn_output"])
            expected = hc_post(d["attn_output"], h, d["attn_post"], d["attn_comb"].reshape(4, 4))
            compare(label + "attn_residual", expected, d["attn_residual"])
            h = d["attn_residual"].reshape(4, 5120)
            for name, value in zip(("pre", "post", "comb"), ref.mixes(layer, "ffn", h)):
                compare(label + "ffn_" + name, value, d["ffn_" + name])
            x = bf(np.sum(d["attn_pre"][:, None] * h, axis=0))
            compare(label + "ffn_input", ref.norm(base + ".ffn_norm", x), d["ffn_input"])
            logits = ref.linear(base + ".ffn.gate", d["ffn_input"], raw=True)
            compare(label + "gate_logits", logits, d["gate_logits"])
            scores = np.sqrt(np.logaddexp(np.float32(0), logits))
            biased = scores + ref.tensor(base + ".ffn.gate.bias")
            ids = np.lexsort((np.arange(384), -biased))[:6]
            compare(label + "route_ids", ids, d["route"][:6], exact=True)
            prob = scores[ids] / (scores[ids].sum() + 1e-20) * 1.5
            compare(label + "route_weights", prob, d["route"][6:])
            if not args.skip_ffn:
                combined = np.zeros(5120, np.float32)
                for expert, prob in zip(d["route"][:6].astype(int), d["route"][6:]):
                    combined += ref.expert(base + f".ffn.experts.{expert}", d["ffn_input"], prob)
                combined += ref.expert(base + ".ffn.shared_experts", d["ffn_input"])
                compare(label + "ffn", bf(combined), d["ffn_output"])
            expected = hc_post(d["ffn_output"], h, d["ffn_post"], d["ffn_comb"].reshape(4, 4))
            compare(label + "output", expected, d["output"])
    print(f"REPLAY_FINISHED checks={checks} failures={failures} "
          f"worst_cosine={worst[0]:.9f} worst={worst[1]}", flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
