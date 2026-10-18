/* Small-M W4A4 GEMM. The adapter uses the same packed rows and 128x4 scale
 * layout as cutlass_fp4.cu. Only gemm_decode_* OMMA instructions are patched
 * to E0M3; no PTX fallback is embedded. Accumulation and split-K reduction
 * stay FP32, with a single BF16 rounding at the output. */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ void mma(float *d, const unsigned *a, const unsigned *b,
                                   unsigned sa, unsigned sb) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}

__device__ __forceinline__ size_t scale_word(int row, int k64, int K) {
    return (size_t)(row / 128) * 128 * (K / 64) + k64 * 128 +
        (row % 32) * 4 + (row % 128) / 32;
}

template<int BYTES>
__device__ __forceinline__ void copy_async(void *dst, const void *src, bool valid) {
    unsigned address = (unsigned)__cvta_generic_to_shared(dst);
    if constexpr (BYTES == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
            :: "r"(address), "l"(src), "r"(valid ? 4 : 0) : "memory");
}

template<int TM, int TN, int TK>
__device__ __forceinline__ void stage(unsigned *data, unsigned *scales,
    const unsigned *A, const unsigned *B, const unsigned *SA, const unsigned *SB,
    int M, int N, int K, int m0, int n0, int k0, int kend) {
    constexpr int PITCH = TK / 8 + 4; // Avoid 8-way shared bank conflicts.
    #pragma unroll
    for (int i = threadIdx.x; i < (TM + TN) * (TK / 32); i += 128) {
        int r = i / (TK / 32), q = i % (TK / 32) * 4;
        bool is_a = r < TM;
        int row = is_a ? m0 + r : n0 + r - TM;
        bool valid = row < (is_a ? M : N) && k0 + q * 8 < kend;
        const unsigned *base = is_a ? A : B;
        const unsigned *src = valid ? base + (size_t)row * (K / 8) + k0 / 8 + q : base;
        copy_async<16>(data + r * PITCH + q,src,valid);
    }
    #pragma unroll
    for (int i = threadIdx.x; i < (TM + TN) * (TK / 64); i += 128) {
        int r = i / (TK / 64), q = i % (TK / 64);
        bool is_a = r < TM;
        int row = is_a ? m0 + r : n0 + r - TM;
        bool valid = row < (is_a ? M : N) && k0 + q * 64 < kend;
        const unsigned *base = is_a ? SA : SB;
        const unsigned *src = valid ? base + scale_word(row,k0 / 64 + q,K) : base;
        copy_async<4>(scales + i,src,valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

#define ARGS const unsigned *A, const unsigned *B, const unsigned *SA, const unsigned *SB, \
    __nv_bfloat16 *C, float *partials, int M, int N, int K, int splits

template<int TM, int TN, int TK>
__device__ __forceinline__ void tiled(ARGS) {
    constexpr int WM = TM == 16 ? 1 : 2, WN = 4 / WM;
    constexpr int MI = TM / WM / 16, NJ = TN / WN / 8, PITCH = TK / 8 + 4;
    __shared__ __align__(16) unsigned data[2][TM + TN][PITCH];
    __shared__ unsigned scales[2][TM + TN][TK / 64];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    int wm = warp / WN, wn = warp % WN;
    int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN;
    int chunk = ((K / 64 + splits - 1) / splits) * 64;
    int start = blockIdx.z * chunk, end = min(K,start + chunk);
    int tiles = max(0,(end - start + TK - 1) / TK);
    float acc[MI][NJ][4] = {};
    if (tiles) stage<TM,TN,TK>(&data[0][0][0],&scales[0][0][0],A,B,SA,SB,M,N,K,m0,n0,start,end);
    for (int kt = 0; kt < tiles; kt++) {
        int cur = kt & 1;
        if (kt + 1 < tiles) {
            stage<TM,TN,TK>(&data[cur ^ 1][0][0],&scales[cur ^ 1][0][0],A,B,SA,SB,
                M,N,K,m0,n0,start + (kt + 1) * TK,end);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TK / 64; kk++) {
            unsigned af[MI][4], bf[NJ][2], sa[MI], sb[NJ];
            #pragma unroll
            for (int i = 0; i < MI; i++) {
                int r = wm * MI * 16 + i * 16 + g, q = kk * 8 + t;
                af[i][0] = data[cur][r][q]; af[i][1] = data[cur][r + 8][q];
                af[i][2] = data[cur][r][q + 4]; af[i][3] = data[cur][r + 8][q + 4];
                sa[i] = t < 2 ? scales[cur][r + t * 8][kk] : 0;
            }
            #pragma unroll
            for (int j = 0; j < NJ; j++) {
                int r = TM + wn * NJ * 8 + j * 8 + g, q = kk * 8 + t;
                bf[j][0] = data[cur][r][q]; bf[j][1] = data[cur][r][q + 4];
                sb[j] = t == 0 ? scales[cur][r][kk] : 0;
            }
            #pragma unroll
            for (int i = 0; i < MI; i++) {
                #pragma unroll
                for (int j = 0; j < NJ; j++) mma(acc[i][j],af[i],bf[j],sa[i],sb[j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < MI; i++) {
        #pragma unroll
        for (int j = 0; j < NJ; j++) {
            int r = m0 + wm * MI * 16 + i * 16 + g;
            int c = n0 + wn * NJ * 8 + j * 8 + t * 2;
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                int row = r + (x / 2) * 8, col = c + x % 2;
                if (row < M && col < N) {
                    size_t index = (size_t)row * N + col;
                    if (splits == 1) C[index] = __float2bfloat16_rn(acc[i][j][x]);
                    else partials[(size_t)blockIdx.z * M * N + index] = acc[i][j][x];
                }
            }
        }
    }
}

#define TILES(X) X(0,16,64,128) X(1,16,128,128) X(2,32,64,128) X(3,32,128,128) \
    X(4,64,64,128) X(5,64,128,128) X(6,16,64,256) X(7,32,64,256) X(8,64,64,256)
#define DECL(I,TM,TN,TK) extern "C" __global__ __launch_bounds__(128) void gemm_decode_##I(ARGS) { \
    tiled<TM,TN,TK>(A,B,SA,SB,C,partials,M,N,K,splits); }
TILES(DECL)
DECL(9,16,128,256)
DECL(10,16,32,512)
#undef DECL

template<int NJ, bool LOCAL = false>
__device__ __forceinline__ void direct(ARGS) {
    __shared__ float local[LOCAL ? 4 * 16 * 8 * NJ : 1];
    int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    int row = blockIdx.y * 16 + g;
    int warp = threadIdx.x / 32;
    int col = (LOCAL ? blockIdx.x : blockIdx.x * 4 + warp) * NJ * 8 + g;
    int partition = LOCAL ? warp : blockIdx.z;
    if constexpr (LOCAL) splits = 4;
    int chunk = (K / 64 + splits - 1) / splits;
    int end = min(K / 64,(partition + 1) * chunk);
    float acc[NJ][4] = {};
    #pragma unroll 2
    for (int kk = partition * chunk; kk < end; kk++) {
        unsigned af[4] = {}, sa = 0;
        if (row < M) {
            af[0] = A[(size_t)row * (K / 8) + kk * 8 + t];
            af[2] = A[(size_t)row * (K / 8) + kk * 8 + t + 4];
        }
        if (row + 8 < M) {
            af[1] = A[(size_t)(row + 8) * (K / 8) + kk * 8 + t];
            af[3] = A[(size_t)(row + 8) * (K / 8) + kk * 8 + t + 4];
        }
        int sr = row + t * 8;
        if (t < 2 && sr < M) sa = SA[scale_word(sr,kk,K)];
        #pragma unroll
        for (int j = 0; j < NJ; j++) {
            int r = col + j * 8;
            unsigned bf[2] = {}, sb = 0;
            if (r < N) {
                bf[0] = B[(size_t)r * (K / 8) + kk * 8 + t];
                bf[1] = B[(size_t)r * (K / 8) + kk * 8 + t + 4];
                if (!t) sb = SB[scale_word(r,kk,K)];
            }
            mma(acc[j],af,bf,sa,sb);
        }
    }
    #pragma unroll
    for (int j = 0; j < NJ; j++) {
        int c = col - g + j * 8 + t * 2;
        #pragma unroll
        for (int x = 0; x < 4; x++) {
            int r = row + x / 2 * 8, n = c + x % 2;
            if (r < M && n < N) {
                size_t i = (size_t)r * N + n;
                if constexpr (LOCAL) local[warp * 16 * 8 * NJ + (r % 16) * 8 * NJ + n % (8 * NJ)] = acc[j][x];
                else if (splits == 1) C[i] = __float2bfloat16_rn(acc[j][x]);
                else partials[(size_t)blockIdx.z * M * N + i] = acc[j][x];
            }
        }
    }
    if constexpr (LOCAL) {
        __syncthreads();
        for (int i = threadIdx.x; i < 16 * 8 * NJ; i += 128) {
            int r = blockIdx.y * 16 + i / (8 * NJ), c = blockIdx.x * 8 * NJ + i % (8 * NJ);
            if (r < M && c < N) {
                float sum = 0;
                #pragma unroll
                for (int w = 0; w < 4; w++) sum += local[w * 16 * 8 * NJ + i];
                C[(size_t)r * N + c] = __float2bfloat16_rn(sum);
            }
        }
    }
}
#define DIRECT(J) extern "C" __global__ void gemm_decode_direct##J(ARGS) { \
    direct<J>(A,B,SA,SB,C,partials,M,N,K,splits); }
DIRECT(1) DIRECT(2) DIRECT(4)
#undef DIRECT
#define LOCAL(J) extern "C" __global__ void gemm_decode_local##J(ARGS) { \
    direct<J,true>(A,B,SA,SB,C,partials,M,N,K,splits); }
LOCAL(1) LOCAL(2) LOCAL(4)
#undef LOCAL

__device__ __forceinline__ unsigned signed_bytes(unsigned x) {
    unsigned mag = x & 0x07070707, mask = ((x >> 3) & 0x01010101) * 255;
    return (mag & ~mask) | (__vsub4(0,mag) & mask);
}
__device__ __forceinline__ float scale_value(unsigned x) {
    unsigned exp = (x >> 3) & 15, mant = x & 7;
    return exp ? __uint_as_float(((exp + 120) << 23) | (mant << 20)) : mant * (1.0f / 512);
}
/* Single-token decode is bandwidth-bound. DP4A avoids computing 15 padded
 * tensor-core rows. Four exact integer dot products per K16 scale group. */
__global__ void gemv_decode_e0m3(const unsigned *A, const unsigned *B,
    const unsigned *SA, const unsigned *SB, __nv_bfloat16 *out, int N, int K) {
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    float sum = 0;
    if (row < N) {
        for (int group = lane; group < K / 16; group += 32) {
            uint2 a = reinterpret_cast<const uint2 *>(A)[group];
            uint2 b = reinterpret_cast<const uint2 *>(B + (size_t)row * (K / 8))[group];
            int dot = __dp4a((int)signed_bytes(a.x),(int)signed_bytes(b.x),0);
            dot = __dp4a((int)signed_bytes(a.x >> 4),(int)signed_bytes(b.x >> 4),dot);
            dot = __dp4a((int)signed_bytes(a.y),(int)signed_bytes(b.y),dot);
            dot = __dp4a((int)signed_bytes(a.y >> 4),(int)signed_bytes(b.y >> 4),dot);
            unsigned sa = (SA[scale_word(0,group / 4,K)] >> (group % 4 * 8)) & 255;
            unsigned sb = (SB[scale_word(row,group / 4,K)] >> (group % 4 * 8)) & 255;
            sum = fmaf((float)dot,scale_value(sa) * scale_value(sb),sum);
        }
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff,sum,offset);
    if (!lane && row < N) out[row] = __float2bfloat16_rn(sum);
}

__global__ void reduce_decode(const float *partials, __nv_bfloat16 *out, size_t count, int splits) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float sum = 0;
        for (int s = 0; s < splits; s++) sum += partials[(size_t)s * count + i];
        out[i] = __float2bfloat16_rn(sum);
    }
}

struct state {
    const unsigned *a, *b, *sa, *sb;
    __nv_bfloat16 *out;
    float *partials = nullptr;
    int index, m, n, k, splits;
};

extern "C" int fp4_kernel_count() { return 60; }
extern "C" const char *fp4_kernel_build() { return "decode padded_shared=16B stages=2 accumulation=FP32 output=BF16 splits=1/2/4/8 local_split=4"; }
static bool e0m3_format = true;
extern "C" void fp4_kernel_set_format(int e0m3) { e0m3_format = e0m3 != 0; }
extern "C" const char *fp4_kernel_name(int index) {
    static const char *names[] = {
#define NAME(I,M,N,K) #M "x" #N "x" #K,
        TILES(NAME)
#undef NAME
    };
    static const char *direct_names[] = {"direct 16x32x64","direct 16x64x64","direct 16x128x64"};
    static const char *local_names[] = {"local split-K4 16x8x64","local split-K4 16x16x64","local split-K4 16x32x64"};
    if (index >= 49 && index < 52) return local_names[index - 49];
    if (index == 36) return "E0M3 DP4A GEMV";
    static thread_local char label[128];
    if (index >= 52 && index < 60)
        std::snprintf(label,sizeof(label),"%s splitK=%d",index < 56 ? "16x128x256" : "16x32x512",1 << ((index - 52) % 4));
    else if (index >= 37 && index < 49)
        std::snprintf(label,sizeof(label),"%s splitK=%d",direct_names[(index - 37) % 3],1 << ((index - 37) / 3));
    else if (index >= 0 && index < 36)
        std::snprintf(label,sizeof(label),"%s splitK=%d",names[index % 9],1 << (index / 9));
    else return "invalid";
    return label;
}
extern "C" void *fp4_kernel_create(int index, int, int, int m, int n, int k,
                                    void *a, void *b, void *sa, void *sb, void *out, int type) {
    if (index < 0 || index >= 60 || type != 14 || m < 1 || n < 1 || k < 64 || k % 64 ||
        (index == 36 && (m != 1 || !e0m3_format))) return nullptr;
    auto *s = new state{static_cast<unsigned *>(a),static_cast<unsigned *>(b),
        static_cast<unsigned *>(sa),static_cast<unsigned *>(sb),static_cast<__nv_bfloat16 *>(out),
        nullptr,index >= 52 ? (index < 56 ? 9 : 10) : index >= 49 ? index :
            index >= 37 ? 100 + (index - 37) % 3 : index == 36 ? 36 : index % 9,m,n,k,
        index >= 52 ? 1 << ((index - 52) % 4) : index >= 49 ? 1 :
            index >= 37 ? 1 << ((index - 37) / 3) : index == 36 ? 1 : 1 << (index / 9)};
    if (s->splits > k / 64) { delete s; return nullptr; }
    if (s->splits > 1 && cudaMalloc(&s->partials,(size_t)m * n * s->splits * sizeof(float)) != cudaSuccess) {
        delete s; return nullptr;
    }
    return s;
}
extern "C" int fp4_kernel_run_stream(void *ptr, void *stream_ptr) {
    auto *s = static_cast<state *>(ptr); auto stream = static_cast<cudaStream_t>(stream_ptr);
    switch (s->index) {
#define RUN_LOCAL(I,J) case I: gemm_decode_local##J<<<dim3((s->n + 8*J - 1)/(8*J),(s->m + 15)/16),128,0,stream>>>( \
    s->a,s->b,s->sa,s->sb,s->out,s->partials,s->m,s->n,s->k,4); break;
        RUN_LOCAL(49,1) RUN_LOCAL(50,2) RUN_LOCAL(51,4)
#undef RUN_LOCAL
#define RUN_DIRECT(I,J) case I: gemm_decode_direct##J<<<dim3((s->n + 32*J - 1)/(32*J),(s->m + 15)/16,s->splits),128,0,stream>>>( \
    s->a,s->b,s->sa,s->sb,s->out,s->partials,s->m,s->n,s->k,s->splits); break;
        RUN_DIRECT(100,1) RUN_DIRECT(101,2) RUN_DIRECT(102,4)
#undef RUN_DIRECT
        case 36: gemv_decode_e0m3<<<(s->n + 3)/4,128,0,stream>>>(s->a,s->b,s->sa,s->sb,s->out,s->n,s->k); break;
#define RUN(I,TM,TN,TK) case I: gemm_decode_##I<<<dim3((s->n + TN - 1)/TN,(s->m + TM - 1)/TM,s->splits),128,0,stream>>>( \
    s->a,s->b,s->sa,s->sb,s->out,s->partials,s->m,s->n,s->k,s->splits); break;
        TILES(RUN)
        RUN(9,16,128,256)
        RUN(10,16,32,512)
#undef RUN
    }
    if (s->splits > 1) reduce_decode<<<((size_t)s->m * s->n + 255)/256,256,0,stream>>>(
        s->partials,s->out,(size_t)s->m * s->n,s->splits);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int fp4_kernel_run(void *ptr) { return fp4_kernel_run_stream(ptr,nullptr); }
extern "C" void fp4_kernel_destroy(void *ptr) {
    auto *s = static_cast<state *>(ptr); if (s->partials) cudaFree(s->partials); delete s;
}
