"""rdna4 sparse_submanifold_conv3d as a torch extension + a SparseLinear
chunking workaround for an RDNA4 hipBLASLt tall+skinny matmul bug.

The actual TRELLIS-2 tex_slat_decoder garbage on RDNA4 is caused by hipBLASLt
mishandling the trailing `SparseLinear(64 -> 6)` matmul at M~1.5M rows,
returning ±1e18 from clean fp32 input. Chunking the matmul to 64K rows fixes
it deterministically — that is `install_sparse_linear_chunking()`. See
project_trellis2_rocm_bisect_2026_05_08.md for the bisection.

The submanifold_conv3d torch extension is a separately-useful F32 LDS-tiled
spconv (BN=64, BK=32, hash-table neighbor map) wrapping the kernel from
hip_tex_dec_kernels.h. Standalone correctness vs CPU ref: cos=1.0,
deterministic. Use install() to route SparseConv3d.forward through it; not
required for tex_dec correctness.

Public API:
    install_sparse_linear_chunking() — fixes tex_slat_decoder on RDNA4.
    install()                        — also reroutes SparseConv3d.forward.
    submanifold_conv3d(feats, coords, _shape, weight, bias=None,
                       neighbor_cache=None, dilation=(1,1,1)) -> (out, nmap)

submanifold_conv3d constraints: out_C % 64 == 0 and in_C % 32 == 0; dilation
must be (1,1,1); kernel must be 3x3x3.
"""
import os
import torch
from torch.utils.cpp_extension import load_inline


_KERNEL_SRC = r"""
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

/* 20-bit packed (z,y,x) -> uint64. Coords assumed in [0, 2^20). */
__device__ __forceinline__ unsigned long long pack_zyx(int z, int y, int x) {
    return ((unsigned long long)(unsigned int)z << 40) |
           ((unsigned long long)(unsigned int)y << 20) |
            (unsigned long long)(unsigned int)x;
}

__device__ __forceinline__ unsigned int hash64(unsigned long long k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (unsigned int)k;
}

extern "C" __global__ void rdna4_hash_insert(
    unsigned long long *keys, int *vals, int cap_mask,
    const int *coords, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int z = coords[i*4 + 1], y = coords[i*4 + 2], x = coords[i*4 + 3];
    unsigned long long k = (unsigned long long)pack_zyx(z, y, x) + 1ULL;
    unsigned int h = hash64(k) & (unsigned int)cap_mask;
    for (int probe = 0; probe <= cap_mask; probe++) {
        unsigned long long prev = atomicCAS(&keys[h], 0ULL, k);
        if (prev == 0ULL) { vals[h] = i; return; }
        if (prev == k)    { vals[h] = i; return; }
        h = (h + 1) & (unsigned int)cap_mask;
    }
}

__device__ __forceinline__ int hash_lookup(
    const unsigned long long *keys, const int *vals, int cap_mask,
    int z, int y, int x) {
    unsigned long long k = (unsigned long long)pack_zyx(z, y, x) + 1ULL;
    unsigned int h = hash64(k) & (unsigned int)cap_mask;
    for (int probe = 0; probe <= cap_mask; probe++) {
        unsigned long long cur = keys[h];
        if (cur == 0ULL) return -1;
        if (cur == k)    return vals[h];
        h = (h + 1) & (unsigned int)cap_mask;
    }
    return -1;
}

/* Build [N, 27] neighbor map. v=13 (center) -> idx; absent -> 0xFFFFFFFF. */
extern "C" __global__ void rdna4_build_nmap(
    unsigned int *nmap, const int *coords,
    const unsigned long long *keys, const int *vals, int cap_mask, int N) {
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    int v   = threadIdx.x;
    if (idx >= N || v >= 27) return;
    int c1 = coords[idx*4 + 1];
    int c2 = coords[idx*4 + 2];
    int c3 = coords[idx*4 + 3];
    int d1 = (v / 9) - 1;
    int d2 = ((v / 3) % 3) - 1;
    int d3 = (v % 3) - 1;
    int ni;
    if (v == 13) ni = idx;
    else         ni = hash_lookup(keys, vals, cap_mask, c1 + d1, c2 + d2, c3 + d3);
    nmap[(size_t)idx * 27 + v] = (ni < 0) ? 0xFFFFFFFFu : (unsigned int)ni;
}

/* Tiled submanifold conv3d, kernel=3, stride=1, F32, with precomputed nmap.
 * Weight layout: [Co, 27, Ci] where 27 = kd*9 + kh*3 + kw, kd/kh/kw in [0,3).
 * Grid: (N, Co/64). Block: 64 threads. Requires Co % 64 == 0, Ci % 32 == 0. */
extern "C" __global__ void rdna4_spconv_nmap_tiled_f32(
    float *out, const float *feats,
    const unsigned int *nmap,
    const float *weight, const float *bias,
    int in_C, int out_C) {
    const int BN = 64;
    const int BK = 32;
    int vox     = blockIdx.x;
    int oc_base = blockIdx.y * BN;
    int tid     = threadIdx.x;
    int oc      = oc_base + tid;
    __shared__ int   nbr[27];
    __shared__ float smFeat[BK];
    __shared__ float smW[BN][BK];
    if (tid < 27) {
        unsigned int u = nmap[(size_t)vox * 27 + tid];
        nbr[tid] = (u == 0xFFFFFFFFu) ? -1 : (int)u;
    }
    __syncthreads();
    float acc = bias ? bias[oc] : 0.0f;
    for (int k = 0; k < 27; k++) {
        int ni = nbr[k];
        if (ni < 0) continue;
        for (int ic = 0; ic < in_C; ic += BK) {
            if (tid < BK) smFeat[tid] = feats[(size_t)ni * in_C + ic + tid];
            #pragma unroll
            for (int it = 0; it < (BN * BK) / 64; it++) {
                int ix = it * 64 + tid;
                int row = ix / BK;
                int col = ix % BK;
                smW[row][col] = weight[((size_t)(oc_base + row) * 27 + k) * in_C + ic + col];
            }
            __syncthreads();
            float p = 0.0f;
            #pragma unroll
            for (int j = 0; j < BK; j++) p += smW[tid][j] * smFeat[j];
            acc += p;
            __syncthreads();
        }
    }
    if (oc < out_C) out[(size_t)vox * out_C + oc] = acc;
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* Build the [N, 27] uint32 neighbor_map for these coords, using a coord hashmap.
 * Returns: (nmap, ) — capacity hashmap freed at function exit. */
torch::Tensor build_nmap(torch::Tensor coords) {
    TORCH_CHECK(coords.is_cuda() && coords.dtype() == torch::kInt32 && coords.dim() == 2 && coords.size(1) == 4,
                "coords must be cuda int32 [N, 4]");
    coords = coords.contiguous();
    int N = coords.size(0);
    int cap = next_pow2(N * 2);
    if (cap < 64) cap = 64;
    int cap_mask = cap - 1;

    auto opts_u64 = torch::TensorOptions().dtype(torch::kInt64).device(coords.device());
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());
    auto opts_u32 = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());

    auto keys = torch::zeros({cap}, opts_u64);
    auto vals = torch::empty({cap}, opts_i32);
    auto nmap = torch::empty({(int64_t)N, 27}, opts_u32);

    auto stream = c10::hip::getCurrentHIPStream().stream();

    int t = 256, b = (N + t - 1) / t;
    rdna4_hash_insert<<<b, t, 0, stream>>>(
        (unsigned long long*)keys.data_ptr<int64_t>(),
        vals.data_ptr<int>(),
        cap_mask,
        coords.data_ptr<int>(),
        N);

    dim3 nb_grid((N + 7) / 8), nb_block(27, 8);
    rdna4_build_nmap<<<nb_grid, nb_block, 0, stream>>>(
        (unsigned int*)nmap.data_ptr<int>(),
        coords.data_ptr<int>(),
        (const unsigned long long*)keys.data_ptr<int64_t>(),
        vals.data_ptr<int>(),
        cap_mask,
        N);
    return nmap;
}

torch::Tensor submanifold_conv3d_f32(
    torch::Tensor feats,        /* [N, in_C] float32 */
    torch::Tensor weight,       /* [Co, 27, Ci] float32 */
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor nmap          /* [N, 27] uint32 */) {
    TORCH_CHECK(feats.is_cuda() && feats.dtype() == torch::kFloat32);
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kFloat32);
    TORCH_CHECK(nmap.is_cuda() && nmap.dtype() == torch::kInt32);
    feats = feats.contiguous();
    weight = weight.contiguous();
    nmap = nmap.contiguous();
    int N = feats.size(0);
    int in_C = feats.size(1);
    int out_C = weight.size(0);
    TORCH_CHECK(weight.size(1) == 27 && weight.size(2) == in_C, "weight must be [Co, 27, in_C]");
    TORCH_CHECK(nmap.size(0) == N && nmap.size(1) == 27, "nmap must be [N, 27]");
    TORCH_CHECK(out_C % 64 == 0 && in_C % 32 == 0,
                "out_C must be multiple of 64 and in_C multiple of 32; got out_C=",
                out_C, " in_C=", in_C);

    const float *bias_ptr = nullptr;
    torch::Tensor bias_c;
    if (bias_opt.has_value() && bias_opt->defined()) {
        bias_c = bias_opt->to(torch::kFloat32).contiguous();
        TORCH_CHECK(bias_c.numel() == out_C);
        bias_ptr = bias_c.data_ptr<float>();
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(feats.device());
    auto out = torch::empty({(int64_t)N, (int64_t)out_C}, opts);

    auto stream = c10::hip::getCurrentHIPStream().stream();
    dim3 grid(N, out_C / 64), block(64);
    rdna4_spconv_nmap_tiled_f32<<<grid, block, 0, stream>>>(
        out.data_ptr<float>(),
        feats.data_ptr<float>(),
        (const unsigned int*)nmap.data_ptr<int>(),
        weight.data_ptr<float>(),
        bias_ptr,
        in_C, out_C);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_nmap", &build_nmap, "Build [N, 27] neighbor_map from coords");
    m.def("submanifold_conv3d_f32", &submanifold_conv3d_f32,
          "Submanifold conv3d with precomputed nmap (F32)");
}
"""


_ext = None


def _ensure_loaded():
    global _ext
    if _ext is not None:
        return _ext
    build_dir = os.path.expanduser('~/.cache/torch_extensions/spconv_rdna4_ext')
    os.makedirs(build_dir, exist_ok=True)
    _ext = load_inline(
        name='spconv_rdna4_ext',
        cpp_sources='',
        cuda_sources=_KERNEL_SRC,
        verbose=os.environ.get('SPCONV_RDNA4_VERBOSE') == '1',
        with_cuda=True,
        build_directory=build_dir,
    )
    return _ext


def submanifold_conv3d(feats, coords, shape, weight, bias=None, neighbor_cache=None, dilation=(1, 1, 1)):
    """Drop-in replacement for flex_gemm.ops.spconv.sparse_submanifold_conv3d.

    flex_gemm signature: (feats, coords, shape, weight, bias=None, neighbor_cache=None, dilation=(1,1,1)) -> (out, nmap_cache)
    Weight layout from flex_gemm: [Co, Kd, Kh, Kw, Ci] = [Co, 3, 3, 3, Ci].
    Returns (out, nmap_tensor) — nmap_tensor stored as the cache for next call.
    """
    assert tuple(dilation) == (1, 1, 1), f'rdna4 spconv only supports dilation=(1,1,1); got {dilation}'
    Co, Kd, Kh, Kw, Ci = weight.shape
    assert (Kd, Kh, Kw) == (3, 3, 3), f'rdna4 spconv only supports 3x3x3; got {(Kd,Kh,Kw)}'

    ext = _ensure_loaded()

    nmap = neighbor_cache
    if nmap is None or not isinstance(nmap, torch.Tensor):
        nmap = ext.build_nmap(coords.to(torch.int32).contiguous())

    feats_f = feats.to(torch.float32).contiguous()
    weight_f = weight.contiguous().view(Co, 27, Ci).to(torch.float32).contiguous()
    bias_f = bias.to(torch.float32).contiguous() if bias is not None else None

    out = ext.submanifold_conv3d_f32(feats_f, weight_f, bias_f, nmap)
    return out.to(feats.dtype), nmap


def _patched_sparse_linear_forward(self, input):
    """Chunked F.linear to work around RDNA4 hipBLASLt bug on tall+skinny matmuls.

    Bug: rows past M = 2**19 = 524288 are computed wrong (denormal-zero or
    ±1e18) on gfx1201 / ROCm 7.2.2 for fp32 K=64, N=6. Default chunk = 2**18
    stays a factor of 2 under the boundary while halving launch overhead vs
    the original 65536. Override via SPCONV_RDNA4_LINEAR_CHUNK."""
    import torch.nn.functional as F
    x = input.feats
    M = x.shape[0]
    chunk = int(os.environ.get('SPCONV_RDNA4_LINEAR_CHUNK', '262144'))
    if M <= chunk:
        return input.replace(F.linear(x, self.weight, self.bias))
    out_C = self.weight.shape[0]
    out = torch.empty(M, out_C, device=x.device, dtype=x.dtype)
    for s in range(0, M, chunk):
        out[s:s+chunk] = F.linear(x[s:s+chunk].contiguous(), self.weight, self.bias)
    return input.replace(out)


def install_sparse_linear_chunking():
    """Patch SparseLinear.forward to chunk M-dim, working around hipBLASLt bug."""
    from trellis2.modules.sparse.linear import SparseLinear
    SparseLinear.forward = _patched_sparse_linear_forward
    print('[spconv_rdna4_ext] installed: SparseLinear.forward -> chunked F.linear (hipBLASLt workaround)')


def install():
    """Route every SparseConv3d.forward through the rdna4 F32 tiled kernel.

    Note: this is NOT required for tex_slat_decoder correctness on RDNA4 — the
    tex_dec garbage was a hipBLASLt bug in the trailing nn.Linear, fixed by
    install_sparse_linear_chunking(). flex_gemm spconv produces correct output.
    Use this only as a deterministic, fp32 reference path or to drop the
    flex_gemm dependency.
    """
    _ensure_loaded()
    from trellis2.modules.sparse.conv import conv_flex_gemm
    from trellis2.modules.sparse import SparseTensor

    def _patched_forward(self, x: SparseTensor) -> SparseTensor:
        Co, Kd, Kh, Kw, Ci = self.weight.shape
        cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
        cache = x.get_spatial_cache(cache_key)
        out, cache_ = submanifold_conv3d(
            x.feats, x.coords, None, self.weight, self.bias, cache, self.dilation,
        )
        if cache is None:
            x.register_spatial_cache(cache_key, cache_)
        return x.replace(out)

    conv_flex_gemm.sparse_conv3d_forward = _patched_forward
    conv_flex_gemm.sparse_submanifold_conv3d = submanifold_conv3d
    print('[spconv_rdna4_ext] installed: sparse_conv3d_forward -> rdna4 F32 tiled kernel (flex_gemm spconv fully bypassed)')
    install_sparse_linear_chunking()
