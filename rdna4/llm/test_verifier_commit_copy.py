"""Execute production checkpoint-copy kernel bodies on CPU against row copies.

This checks indexing and write coverage, not GPU scheduling or performance.
"""
import ast
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
source = (root / "hip_llm_runner.c").read_text()
start = source.index('"__global__ void copy_state_rows_f32(')
end = source.index('"__global__ void hc_norm_f32(', start)
kernels = "".join(ast.literal_eval(line) for line in source[start:end].splitlines())
program = r'''
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <vector>
#define __global__
struct { unsigned x, y, z; } blockIdx, blockDim, gridDim, threadIdx;
struct alignas(16) uint4 { unsigned x, y, z, w; };
''' + kernels + r'''
struct Rows {
    size_t n, stride;
    std::vector<float> src, dst;
    Rows(size_t count, size_t padding) : n(count), stride(count + padding),
        src(4 * stride), dst(count + 16, -999.0f) {
        for (size_t i = 0; i < src.size(); ++i) src[i] = float(i + 1);
    }
    void check(int row) const {
        for (size_t i = 0; i < n; ++i) assert(dst[i] == src[row * stride + i]);
        for (size_t i = n; i < dst.size(); ++i) assert(dst[i] == -999.0f);
    }
};
int main() {
    size_t cases = 0;
    for (size_t n : {size_t(1), size_t(3), size_t(4), size_t(5),
                     size_t(1023), size_t(1024), size_t(1025), size_t(4099)})
    for (size_t padding = 0; padding < 4; ++padding)
    for (int row = 0; row < 4; ++row)
    for (bool fused : {false, true}) {
        Rows conv(n, padding), rec(n + 2, padding);
        Rows x(n + 4, padding), logits(n + 2050, padding);
        float *cd[] = {conv.dst.data()}, *rd[] = {rec.dst.data()};
        const float *cs[] = {conv.src.data()}, *rs[] = {rec.src.data()};
        size_t largest = fused ? logits.n : rec.n;
        gridDim.x = unsigned(((largest + 3) / 4 + 255) / 256);
        blockDim.x = 256; blockIdx.y = 0;
        for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)
        for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
            if (fused)
                copy_state_rows_commit_f32(cd, cs, conv.n, conv.stride,
                    rd, rs, rec.n, rec.stride, row, 1, x.dst.data(),
                    x.src.data(), x.n, x.stride, logits.dst.data(),
                    logits.src.data(), logits.n, logits.stride);
            else
                copy_state_rows_f32(cd, cs, conv.n, conv.stride,
                    rd, rs, rec.n, rec.stride, row, 1);
        }
        conv.check(row); rec.check(row);
        if (fused) { x.check(row); logits.check(row); }
        ++cases;
    }
    std::printf("Verifier checkpoint copy: %zu cases PASS\n", cases);
}
'''
tmp_root = root / "tmp"
tmp_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="verifier-copy-", dir=tmp_root) as directory:
    path = Path(directory)
    (path / "test.cpp").write_text(program)
    subprocess.run(["c++", "-std=c++17", "-O2", "-fno-strict-aliasing",
                    "-Wall", "-Wextra", "-Werror", str(path / "test.cpp"),
                    "-o", str(path / "test")], check=True,
                   env=os.environ | {"TMPDIR": str(path)})
    subprocess.run([str(path / "test")], check=True)
