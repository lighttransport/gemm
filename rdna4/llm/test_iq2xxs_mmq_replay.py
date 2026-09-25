"""Check a Qwen3.8 IQ2_XXS/IQ2_XS down projection against a D4 replay.

Usage (gguf-py from the reference llama.cpp checkout must be on PYTHONPATH):
    python3 test_iq2xxs_mmq_replay.py MODEL.gguf TRACE_DIRECTORY [--layer 1]

Capture with LLM_DEBUG_LAYERS=1 and LLM_DEBUG_DUMP_DIR, scalar FFN, and
LLM_IQ_MMQ_D4=1 plus LLM_IQ2_XXS_Q81_SCALAR=1 (layer 0) or
LLM_IQ2_XS_Q81_SCALAR=1 (layer 1). This tests the MMQ diagnostic
contract, not MMVQ (whose integer division intentionally differs).
SiLU is reconstructed on CPU, so this is not byte-exact activation staging.
"""
import argparse
from pathlib import Path

import gguf
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--layer", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    reader = gguf.GGUFReader(str(args.model))
    tensor = next(t for t in reader.tensors if t.name == f"blk.{args.layer}.ffn_down.weight")
    if tensor.tensor_type not in (gguf.GGMLQuantizationType.IQ2_XXS,
                                  gguf.GGMLQuantizationType.IQ2_XS):
        raise ValueError("This replay requires an IQ2_XXS or IQ2_XS down tensor")
    gate, up, actual = [
        np.fromfile(args.trace / f"runner-scalar-ffn-{stage}-{args.layer:02}.bin", dtype=np.float32)
        for stage in ("gate", "up", "out")
    ]
    if gate.shape != up.shape or gate.size != 17408 or actual.size != 5120:
        raise ValueError("Missing or incompatible Qwen3.8 captures")
    if not all(np.isfinite(v).all() for v in (gate, up, actual)):
        raise ValueError("Non-finite input/output capture")
    x = (gate / (1 + np.exp(-gate)) * up).reshape(-1, 32)
    scales = np.max(np.abs(x), axis=1) / 127
    qf = np.divide(x, scales[:, None], out=np.zeros_like(x), where=scales[:, None] != 0)
    q = np.copysign(np.floor(np.abs(qf) + .5), qf)  # roundf, not ties-to-even
    quantized_x = (q * scales[:, None]).reshape(-1)
    raw = tensor.data.reshape(actual.size, -1)
    expected = np.empty_like(actual)
    for start in range(0, actual.size, 64):
        weights = gguf.quants.dequantize(raw[start:start + 64], tensor.tensor_type)
        expected[start:start + 64] = weights @ quantized_x
    relative = np.linalg.norm(expected - actual) / np.linalg.norm(expected)
    maximum = np.max(np.abs(expected - actual))
    if not np.isfinite(relative) or relative > 1e-6:
        raise AssertionError(f"MMQ replay failed: rel_l2={relative:.9g}, max_abs={maximum:.9g}")
    print(f"PASS: {tensor.tensor_type.name} layer {args.layer} MMQ down "
          f"rel_l2={relative:.9g}, max_abs={maximum:.9g}")


if __name__ == "__main__":
    main()
