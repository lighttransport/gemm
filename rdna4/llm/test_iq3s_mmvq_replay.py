"""Build an isolated IQ3_S GPU replay against the local llama.cpp vecdot.

Requires numpy and llama.cpp/gguf-py on PYTHONPATH. Build/prepare on the host,
then execute the printed command with AMD GPU access. Output stays in --out.
The same GPU-quantized input feeds both projections, isolating matvec rounding.
"""
import argparse
import ast
import os
from pathlib import Path
import subprocess

import gguf
import numpy as np


def extract(source, name, array=False):
    marker = source.index(name)
    start = source.rfind("\n", 0, marker) + 1
    brace = source.index("{", marker)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end + int(array)] + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("input", type=Path, help="captured F32 projection input")
    p.add_argument("--tensor", default="blk.0.attn_qkv.weight")
    p.add_argument("--llama", type=Path, default=Path("/mnt/nvme02/work/llama.cpp"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hipcc", default="/opt/rocm/core-10.0/bin/hipcc")
    p.add_argument("--random-vectors", type=int, default=0,
                   help="append deterministic random vectors and a zero vector")
    p.add_argument("--rows", type=int, help="test a prefix of rows, including partial warps")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reader = gguf.GGUFReader(str(args.model))
    t = next(t for t in reader.tensors if t.name == args.tensor)
    if t.tensor_type != gguf.GGMLQuantizationType.IQ3_S or len(t.shape) != 2:
        raise ValueError("Expected a two-dimensional IQ3_S tensor")
    cols, rows = map(int, t.shape)
    if args.rows is not None:
        if not 0 < args.rows <= rows:
            raise ValueError("Invalid row count")
        rows = args.rows
    x = np.fromfile(args.input, dtype=np.float32)
    if x.size == 0 or x.size % cols or not np.isfinite(x).all():
        raise ValueError("Expected one or more finite, complete input vectors")
    if args.random_vectors < 0:
        raise ValueError("Invalid random-vector count")
    if args.random_vectors:
        rng = np.random.default_rng(17)
        extra = rng.standard_normal((args.random_vectors, cols)).astype(np.float32)
        x = np.concatenate((x, extra.ravel(), np.zeros(cols, dtype=np.float32)))
    t.data.reshape(int(t.shape[1]), -1)[:rows].tofile(args.out / "weights.bin")
    x.tofile(args.out / "input.bin")
    here = Path(__file__).resolve().parent
    lines = (here / "hip_llm_runner.c").read_text().splitlines()
    source = "".join(ast.literal_eval(line.rstrip(";")) for line in lines
                     if line.startswith('"'))
    pieces = ["typedef unsigned short half_raw;\n"]
    for name in ("round_f16_contract", "half_to_float", "dp4a_hw",
                 "signmask4_dev", "apply_sign4", "iq3s_grid_dev",
                 "quantize_q8_contract", "quantize_q81_batch_32_exact",
                 "matvec_iq3_s_q81_batch"):
        pieces.append(extract(source, name, name.endswith("_dev")))
    original = pieces[-1]
    candidate = original.replace("matvec_iq3_s_q81_batch", "candidate")
    candidate = candidate.replace(
        "dw * ts[qb] * (float)(1 + 2 * ls) * (float)sumi",
        "(dw * ts[qb]) * (float)((1 + 2 * ls) * sumi)")
    candidate = candidate.replace("__shfl_down(sum, o)", "__shfl_xor(sum, o)")
    expression = "sum += (dw * ts[qb]) * (float)((1 + 2 * ls) * sumi);"
    protected = ('float ds; asm volatile("v_mul_f32 %0, %1, %2" : "=v"(ds) : "v"(dw), "v"(ts[qb]));\n'
                 '        float scaled = (float)((1 + 2 * ls) * sumi);\n')
    pieces.append(candidate.replace(expression, protected +
        '        float term; asm volatile("v_mul_f32 %0, %1, %2" : "=v"(term) : "v"(ds), "v"(scaled));\n'
        '        asm volatile("v_add_f32 %0, %1, %2" : "=v"(sum) : "v"(sum), "v"(term));'))
    pieces.append(extract(source, "matvec_iq3_s_q81_mmvq_batch").replace(
        "matvec_iq3_s_q81_mmvq_batch", "candidate_fma"))
    terms = candidate.replace("void candidate(", "void runner_terms(")
    terms = terms.replace(expression, protected +
        '        float term; asm volatile("v_mul_f32 %0, %1, %2" : "=v"(term) : "v"(ds), "v"(scaled));\n'
        "        dst[((size_t)token * n_rows + row) * G + gg] = term;")
    terms = terms.replace("if (lane == 0) dst[(size_t)token * n_rows + row] = sum;", "")
    pieces.append(terms)
    harness = (here / "test_iq3s_mmvq_replay.cu").read_text()
    generated = args.out / "replay.cu"
    declarations = []
    for name in ("quantize_q81_batch_32_exact", "matvec_iq3_s_q81_batch",
                 "candidate", "candidate_fma", "runner_terms"):
        body = extract("\n".join(pieces), name)
        declarations.append(body[:body.index("{")] + ";\n")
    generated.write_text(harness.replace("// RUNNER_KERNELS", "\n".join(declarations)))
    runner_source = args.out / "runner.cu"
    runner_source.write_text('#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n' + "\n".join(pieces))
    env = dict(os.environ, TMPDIR=str(args.out.resolve()))
    subprocess.run([args.hipcc, "-O3", "-ffast-math", "--offload-arch=gfx1201",
                    "-c", str(runner_source), "-o", str(args.out / "runner.o")],
                   env=env, check=True)
    # The local llama.cpp HIP mmvq translation unit uses -O3 without fast-math.
    # Keep it separate from the runner's HIPRTC-equivalent fast-math build.
    cmd = [args.hipcc, "-O3", "--offload-arch=gfx1201", "-DGGML_USE_HIP",
           "-I" + str(args.llama / "ggml/include"),
           "-I" + str(args.llama / "ggml/src"),
           "-I" + str(args.llama / "ggml/src/ggml-cuda"),
           str(generated), "-x", "none", str(args.out / "runner.o"), "-o", str(args.out / "replay")]
    subprocess.run(cmd, env=env, check=True)
    print(f"tensor={t.name} type={t.tensor_type.name} rows={rows} cols={cols} vectors={x.size // cols}")
    print(f"{args.out}/replay {args.out} {rows} {cols} {x.size // cols}")


if __name__ == "__main__":
    main()
