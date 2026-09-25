"""Build the actual runner KV kernels against llama.cpp's HIP Q8_0 quantizer."""
import argparse
import ast
import os
from pathlib import Path
import subprocess


def extract(source, name):
    """Extract a definition, skipping forward declarations and call sites."""
    marker = -1
    while True:
        marker = source.index(name, marker + 1)
        brace = source.find("{", marker)
        semicolon = source.find(";", marker)
        if brace >= 0 and (semicolon < 0 or brace < semicolon):
            break
    start = source.rfind("\n", 0, marker) + 1
    end = brace + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--llama", type=Path, default=Path("/mnt/nvme02/work/llama.cpp"))
    parser.add_argument("--hipcc", default="/opt/rocm/core-10.0/bin/hipcc")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    source = "".join(ast.literal_eval(line.rstrip(";")) for line in
                     (here / "hip_llm_runner.c").read_text().splitlines()
                     if line.startswith('"'))
    names = ("round_f16_contract", "half_to_float", "f32_to_f16_bits",
             "f32_to_half_raw", "q8_recip_contract", "q8_div_contract",
             "kv_cache_store_q8q8_batch", "pack_kv_q8q4_f16",
             "unpack_kv_q8q4_decode_f16")
    pieces = [extract(source, name) for name in names]
    runner = args.out / "runner.cu"
    runner.write_text('#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n'
                      'typedef unsigned short half_raw;\n'
                      + "\n".join(pieces))
    declarations = "\n".join(p[:p.index("{")] + ";" for p in pieces[6:])
    harness = args.out / "test.cu"
    harness.write_text((here / "test_kv_q8q8.cu").read_text().replace(
        "// RUNNER_DECLARATIONS", declarations))
    env = dict(os.environ, TMPDIR=str(args.out.resolve()))
    subprocess.run([args.hipcc, "-O3", "-ffast-math", "--offload-arch=gfx1201",
                    "-c", str(runner), "-o", str(args.out / "runner.o")], check=True, env=env)
    subprocess.run([args.hipcc, "-O3", "--offload-arch=gfx1201", "-DGGML_USE_HIP",
                    "-I" + str(args.llama / "ggml/include"),
                    "-I" + str(args.llama / "ggml/src"),
                    "-I" + str(args.llama / "ggml/src/ggml-cuda"),
                    str(harness), "-x", "none", str(args.out / "runner.o"),
                    "-o", str(args.out / "test")], check=True, env=env)
    print(args.out / "test")


if __name__ == "__main__":
    main()
