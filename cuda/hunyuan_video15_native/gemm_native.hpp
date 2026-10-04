#ifndef PIXAL3D_HV15N_GEMM_NATIVE_HPP
#define PIXAL3D_HV15N_GEMM_NATIVE_HPP
namespace hv15n {
// Private 64x128 tile derived from cuda/gemm/cuda_gemm_ptx_kernels.h.
// The repository's m16n8k16 fragment mapping and FP32 accumulation are
// unchanged. Reuse each B fragment for four M tiles on tall video matrices.
inline const char *large_gemm_source = R"CUDA(
extern "C" __global__ void gemm_f16_large(float *Y, const half_raw *X,
                                        const half_raw *W, int M, int N, int K) {
    Y += (size_t)blockIdx.z*M*N;
    X += (size_t)blockIdx.z*M*K;
    W += (size_t)blockIdx.z*N*K;
    extern __shared__ half_raw shared[];
    int row_base = blockIdx.y*64, warp = threadIdx.x>>5;
    int col_base = blockIdx.x*128+warp*32, lane = threadIdx.x&31;
    int group = lane>>2, part = lane&3;
    float d[4][4][4];
    #pragma unroll
    for(int m=0;m<4;m++)
        #pragma unroll
        for(int n=0;n<4;n++)
            #pragma unroll
            for(int j=0;j<4;j++) d[m][n][j]=0.f;
    for(int k=0;k<K;k+=16) {
        #pragma unroll
        for(int j=threadIdx.x*2;j<64*16;j+=256) {
            int r=row_base+j/16,c=j%16;
            *(unsigned int *)(shared+j)=r<M?*(const unsigned int *)(X+(size_t)r*K+k+c):0;
        }
        __syncthreads();
        #pragma unroll
        for(int n=0;n<4;n++) {
            int c=col_base+n*8+group;
            unsigned int b0=0,b1=0;
            if(c<N) {
                const half_raw *p=W+(size_t)c*K+k;
                b0=*(const unsigned int *)(p+part*2);
                b1=*(const unsigned int *)(p+part*2+8);
            }
            #pragma unroll
            for(int m=0;m<4;m++) {
                int r=m*16+group;
                unsigned int a0=*(const unsigned int *)(shared+r*16+part*2);
                unsigned int a1=*(const unsigned int *)(shared+(r+8)*16+part*2);
                unsigned int a2=*(const unsigned int *)(shared+r*16+part*2+8);
                unsigned int a3=*(const unsigned int *)(shared+(r+8)*16+part*2+8);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                    : "=f"(d[m][n][0]),"=f"(d[m][n][1]),"=f"(d[m][n][2]),"=f"(d[m][n][3])
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                      "f"(d[m][n][0]),"f"(d[m][n][1]),"f"(d[m][n][2]),"f"(d[m][n][3]));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<4;m++) {
        int r0=row_base+m*16+group,r1=r0+8;
        #pragma unroll
        for(int n=0;n<4;n++) {
            int c0=col_base+n*8+part*2,c1=c0+1;
            if(r0<M&&c0<N)Y[(size_t)r0*N+c0]=d[m][n][0];
            if(r0<M&&c1<N)Y[(size_t)r0*N+c1]=d[m][n][1];
            if(r1<M&&c0<N)Y[(size_t)r1*N+c0]=d[m][n][2];
            if(r1<M&&c1<N)Y[(size_t)r1*N+c1]=d[m][n][3];
        }
    }
}
)CUDA";
// Reuse the repository v7 MMA pipeline with THWC convolution gathers.
inline std::string implicit_conv_source(const std::string &base) {
    std::string source = base;
    auto replace = [&](const std::string &from, const std::string &to) {
        size_t position = 0, count = 0;
        while ((position = source.find(from, position)) != std::string::npos) {
            source.replace(position, from.size(), to);
            position += to.size(); count++;
        }
        require(count > 0, "repository convolution source signature changed");
    };
    replace("gemm_f16_v7", "conv_f16_implicit");
    replace("int M, int N, int K)", "int M, int N, int K, int T, int H, int Wd, int C, int KT, int KH, int KW, int replicate, const float *conv_bias, const float *conv_res)");
    // Fused bias epilogue: the same single FP32 add as the separate bias pass.
    for (const char *part : {"Y[(size_t)yr0 * N + yc0] = d0[nt];", "Y[(size_t)yr0 * N + yc1] = d1[nt];",
                             "Y[(size_t)yr1 * N + yc0] = d2[nt];", "Y[(size_t)yr1 * N + yc1] = d3[nt];"}) {
        std::string from = part, column = from.substr(from.find("+ ") + 2, 3),
                    value = from.substr(from.find("= ") + 2, 6);
        // Fused epilogue: bias (single FP32 add), then the optional residual (skip + y),
        // matching the separate bias and residual passes exactly.
        std::string index = from.substr(from.find('[') + 1, from.find(']') - from.find('[') - 1);
        replace(from, "{ float v_ = conv_bias ? " + value + " + conv_bias[" + column + "] : " + value +
                          "; Y[" + index + "] = conv_res ? conv_res[" + index + "] + v_ : v_; }");
    }
    replace("if (cta_m >= M) return;", "if (cta_m >= M || cta_n >= N) return;");
    for (const std::string &offset : {std::string("0"), std::string("next_k")}) {
        replace("const half_raw *src = &X[(size_t)g_row_a * K + " + offset + " + col_a];",
                "long long offset = conv_offset(g_row_a, " + offset + " + col_a, T,H,Wd,C,KT,KH,KW,replicate);\n"
                "            const half_raw *src = X + (offset < 0 ? 0 : offset);\n"
                "            int valid_bytes = offset < 0 ? 0 : 16;");
    }
    replace("\"r\"(dA), \"l\"(src));", "\"r\"(dA), \"l\"(src), \"r\"(valid_bytes));");
    // Only A gathers need zero-fill; B retains its original aligned copies.
    size_t pos = 0;
    while ((pos = source.find("int valid_bytes", pos)) != std::string::npos) {
        auto instruction = source.find("cp.async.cg.shared.global [%0], [%1], 16;", pos);
        require(instruction != std::string::npos, "convolution gather instruction");
        source.replace(instruction, std::string("cp.async.cg.shared.global [%0], [%1], 16;").size(), "cp.async.cg.shared.global [%0], [%1], 16, %2;");
        pos = instruction + 45;
    }
    return R"CUDA(
// Exact quotient of non-negative n < 2^24 by d >= 1 without an integer divide:
// a float estimate corrected by at most one step.
__device__ __forceinline__ int conv_div(int n, int d) {
    int q=__float2int_rz(__int2float_rn(n)*__frcp_rn(__int2float_rn(d)));
    if(q*d>n)q--; else if((q+1)*d<=n)q++;
    return q;
}
__device__ __forceinline__ long long conv_offset(int r, int k, int T,int H,int W,int C,int KT,int KH,int KW,int replicate) {
    // Same offsets as the divide-based form: C is a power of two for every VAE layer
    // (shift/mask), 3x3x3 taps divide by literals, and row splits use conv_div.
    int channel,point;
    if((C&(C-1))==0){int s=__ffs(C)-1;channel=k&(C-1);point=k>>s;}else{channel=k%C;point=k/C;}
    int dx,dy,dt;
    if(KW==3&&KH==3){dx=point%3-1;dy=(point/3)%3-1;dt=point/9-(replicate?KT-1:0);}
    else{dx=point%KW-KW/2;dy=(point/KW)%KH-KH/2;dt=point/(KW*KH)-(replicate?KT-1:0);}
    int rw=conv_div(r,W),plane=conv_div(rw,H);
    int x=r-rw*W+dx,y=rw-plane*H+dy,t=plane+dt;
    if(replicate){x=max(0,min(W-1,x));y=max(0,min(H-1,y));t=max(0,min(T-1,t));}
    if(x<0||x>=W||y<0||y>=H||t<0||t>=T)return -1;
    return ((long long)t*H*W+y*W+x)*C+channel;
}
// Implicit-GEMM convolution v2: 128 x BN CTA tile, K-step BK (32/64) with an XOR-swizzled
// shared layout, 8 warps (2 along M x 4 along N, warp tile 64 x BN/4) and a STAGES-deep
// cp.async pipeline. Rows are output pixels (THWC gather), columns output channels,
// K = taps*C. Each output accumulates its k16 MMA chunks in ascending order exactly as
// conv_f16_implicit, so results are bit-identical across variants.
template <int BN, int BK, int STAGES>
__device__ __forceinline__ void conv_v2_body(float *Y, const half_raw *X, const half_raw *W, int M, int N,
        int K, int T, int H, int Wd, int C, int KT, int KH, int KW, int replicate,
        const float *conv_bias, const float *conv_res) {
    extern __shared__ __align__(16) half_raw smem_c2[];
    constexpr int NT = BN / 32, CH = BK / 8, RA = 128 * CH / 256, RB = BN * CH / 256 > 0 ? (BN * CH + 255) / 256 : 1;
    constexpr int STAGE = (128 + BN) * BK;
    int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, gid = lane >> 2, tid4 = lane & 3;
    // M tiles iterate fastest: consecutive CTAs share one weight tile in L2 (the gathered
    // input neighborhood is small), avoiding repeated DRAM reads of large weight panels.
    int cta_m = blockIdx.x * 128, cta_n = blockIdx.y * BN;
    if (cta_m >= M || cta_n >= N) return;
    int wm = (wid >> 2) * 64, wn = (wid & 3) * (BN / 4);
    float acc[4][NT][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < NT; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    // Loader: chunk i = tid + j*256 -> row i / CH, 16-byte chunk i % CH (constant per thread).
    const int lchunk = tid % CH, lrow0 = tid / CH, rstep = 256 / CH;
    int rx[RA], ry[RA], rt[RA], rok[RA];
#pragma unroll
    for (int j = 0; j < RA; j++) {
        int r = cta_m + lrow0 + j * rstep;
        rok[j] = r < M;
        int rr = rok[j] ? r : 0, rw = conv_div(rr, Wd), plane = conv_div(rw, H);
        rx[j] = rr - rw * Wd; ry[j] = rw - plane * H; rt[j] = plane;
    }
    const bool pow2 = (C & (C - 1)) == 0, k333 = KW == 3 && KH == 3;
    const int cshift = __ffs(C) - 1;
    auto swz = [](int row, int chunk) { return row * BK + ((chunk ^ (row & (CH - 1))) << 3); };
    auto load_stage = [&](int stage, int kbase) {
        half_raw *sA = smem_c2 + stage * STAGE, *sB = sA + 128 * BK;
        int k = kbase + lchunk * 8, channel, point;
        if (pow2) { channel = k & (C - 1); point = k >> cshift; } else { channel = k % C; point = k / C; }
        int dx, dy, dt;
        if (k333) { dx = point % 3 - 1; dy = (point / 3) % 3 - 1; dt = point / 9 - (replicate ? KT - 1 : 0); }
        else { dx = point % KW - KW / 2; dy = (point / KW) % KH - KH / 2; dt = point / (KW * KH) - (replicate ? KT - 1 : 0); }
#pragma unroll
        for (int j = 0; j < RA; j++) {
            int row = lrow0 + j * rstep;
            int x = rx[j] + dx, y = ry[j] + dy, t = rt[j] + dt;
            if (replicate) { x = max(0, min(Wd - 1, x)); y = max(0, min(H - 1, y)); t = max(0, min(T - 1, t)); }
            bool ok = rok[j] && x >= 0 && x < Wd && y >= 0 && y < H && t >= 0 && t < T;
            long long offset = ok ? ((long long)t * H * Wd + (long long)y * Wd + x) * C + channel : 0;
            unsigned dst = __cvta_generic_to_shared(&sA[swz(row, lchunk)]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(X + offset), "r"(ok ? 16 : 0));
        }
#pragma unroll
        for (int j = 0; j < RB; j++) {
            int row = lrow0 + j * rstep;
            if (row < BN) {
                int n = cta_n + row;
                unsigned dst = __cvta_generic_to_shared(&sB[swz(row, lchunk)]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst),
                             "l"(W + (size_t)(n < N ? n : 0) * K + k), "r"(n < N ? 16 : 0));
            }
        }
    };
    int num_k = K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < num_k) load_stage(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }
    for (int ki = 0; ki < num_k; ki++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();
        if (ki + STAGES - 1 < num_k) load_stage((ki + STAGES - 1) % STAGES, (ki + STAGES - 1) * BK);
        asm volatile("cp.async.commit_group;\n");
        const half_raw *sA = smem_c2 + (ki % STAGES) * STAGE, *sB = sA + 128 * BK;
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            int chunk = kk * 2 + (lane >> 4);
            unsigned a[4][4], b[(NT + 1) / 2][4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int row = wm + mt * 16 + (lane & 15);
                unsigned p = __cvta_generic_to_shared(&sA[swz(row, chunk)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[mt][0]), "=r"(a[mt][1]), "=r"(a[mt][2]), "=r"(a[mt][3]) : "r"(p));
            }
#pragma unroll
            for (int nb = 0; nb < (NT + 1) / 2; nb++) {
                int row = wn + nb * 16 + (lane & 15);
                unsigned p = __cvta_generic_to_shared(&sB[swz(row, chunk)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[nb][0]), "=r"(b[nb][1]), "=r"(b[nb][2]), "=r"(b[nb][3]) : "r"(p));
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    unsigned b0 = (nt & 1) ? b[nt >> 1][1] : b[nt >> 1][0], b1 = (nt & 1) ? b[nt >> 1][3] : b[nt >> 1][2];
                    float *d = acc[mt][nt];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                                 : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]), "r"(b0), "r"(b1));
                }
        }
    }
    asm volatile("cp.async.wait_group 0;\n");
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = cta_m + wm + mt * 16 + gid + (e >> 1) * 8, col = cta_n + wn + nt * 8 + tid4 * 2 + (e & 1);
                if (row < M && col < N) {
                    size_t idx = (size_t)row * N + col;
                    float v_ = conv_bias ? acc[mt][nt][e] + conv_bias[col] : acc[mt][nt][e];
                    Y[idx] = conv_res ? conv_res[idx] + v_ : v_;
                }
            }
}
#define H15_CONV_V2(NAME, BN, BK, STAGES, MINB)                                                    \
    extern "C" __global__ __launch_bounds__(256, MINB) void NAME(float *Y, const half_raw *X,      \
        const half_raw *W, int M, int N, int K, int T, int H, int Wd, int C, int KT, int KH, int KW, \
        int replicate, const float *conv_bias, const float *conv_res) {                             \
        conv_v2_body<BN, BK, STAGES>(Y, X, W, M, N, K, T, H, Wd, C, KT, KH, KW, replicate,           \
                                     conv_bias, conv_res);                                          \
    }
// Measured on the HunyuanVideo VAE (RTX 5060 Ti): 128x128, BK=32, 3 stages, M-fastest is
// 1-4% faster per layer than conv_f16_implicit; 128x256 and BK=64 variants were slower.
H15_CONV_V2(conv_f16_implicit_v2, 128, 32, 3, 1)
// Few-output-channel convolution (decoder conv_out, N=3): weights in shared memory as
// FP32; each half-warp covers one output row with 16-byte (8-channel) input loads.
extern "C" __global__ __launch_bounds__(256) void conv_small_n(float *Y, const half_raw *X,
        const half_raw *Wt, const float *bias, int M, int N, int K, int T, int H, int Wd, int C,
        int KT, int KH, int KW, int replicate) {
    extern __shared__ float wsm[]; // [N][K]
    const unsigned short *wraw = reinterpret_cast<const unsigned short *>(Wt);
    // Bank-conflict-free layout: within each 128-channel chunk, channel lane*8+j is
    // stored at j*16+lane so a half-warp reads consecutive words for fixed j.
    for (int i = threadIdx.x; i < N * K; i += blockDim.x) {
        int c = i % 128, rest = i - c;
        float wv; asm("cvt.f32.f16 %0,%1;" : "=f"(wv) : "h"(wraw[i]));
        wsm[rest + (c & 7) * 16 + (c >> 3)] = wv;
    }
    __syncthreads();
    int lane = threadIdx.x & 15, group = (blockIdx.x * blockDim.x + threadIdx.x) >> 4;
    int groups = gridDim.x * blockDim.x >> 4, volume = KT * KH * KW;
    for (int row = group; row < M; row += groups) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        // Row coordinates once per row; per-tap offsets follow conv_offset's rules.
        int rw = conv_div(row, Wd), plane = conv_div(rw, H);
        int x0 = row - rw * Wd, y0 = rw - plane * H, t0 = plane;
        if (volume == 27 && C == 128) {
            // Issue all 27 tap loads before the arithmetic (memory-level parallelism).
            uint4 taps[27];
#pragma unroll
            for (int point = 0; point < 27; point++) {
                int dx = point % 3 - 1, dy = (point / 3) % 3 - 1, dt = point / 9 - (replicate ? KT - 1 : 0);
                int x = x0 + dx, y = y0 + dy, t = t0 + dt;
                if (replicate) { x = max(0, min(Wd - 1, x)); y = max(0, min(H - 1, y)); t = max(0, min(T - 1, t)); }
                bool inside = x >= 0 && x < Wd && y >= 0 && y < H && t >= 0 && t < T;
                long long base = ((long long)t * H * Wd + y * Wd + x) * 128 + lane * 8;
                taps[point] = inside ? *reinterpret_cast<const uint4 *>(reinterpret_cast<const unsigned short *>(X) + base)
                                     : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int point = 0; point < 27; point++) {
                const unsigned short *h = reinterpret_cast<const unsigned short *>(&taps[point]);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    float xv; asm("cvt.f32.f16 %0,%1;" : "=f"(xv) : "h"(h[j]));
                    _Pragma("unroll") for (int o = 0; o < 4; o++) if (o < N)
                        acc[o] = fmaf(xv, wsm[o * K + point * 128 + j * 16 + lane], acc[o]);
                }
            }
        } else
        for (int point = 0; point < volume; point++) {
            int dx = point % KW - KW / 2, dy = (point / KW) % KH - KH / 2,
                dt = point / (KW * KH) - (replicate ? KT - 1 : 0);
            int x = x0 + dx, y = y0 + dy, t = t0 + dt;
            if (replicate) { x = max(0, min(Wd - 1, x)); y = max(0, min(H - 1, y)); t = max(0, min(T - 1, t)); }
            if (x < 0 || x >= Wd || y < 0 || y >= H || t < 0 || t >= T) continue;
            long long base = ((long long)t * H * Wd + y * Wd + x) * C;
            for (int c = lane * 8; c < C; c += 128) {
                uint4 packed = *reinterpret_cast<const uint4 *>(reinterpret_cast<const unsigned short *>(X) + base + c);
                const unsigned short *h = reinterpret_cast<const unsigned short *>(&packed);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    float xv; asm("cvt.f32.f16 %0,%1;" : "=f"(xv) : "h"(h[j]));
                    _Pragma("unroll") for (int o = 0; o < 4; o++) if (o < N)
                        acc[o] = fmaf(xv, wsm[o * K + point * C + (c - lane * 8) + j * 16 + lane], acc[o]);
                }
            }
        }
        _Pragma("unroll") for (int o = 0; o < 4; o++) if (o < N) {
            float v = acc[o];
            for (int d = 8; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d, 16);
            if (lane == 0) Y[(size_t)row * N + o] = bias ? v + bias[o] : v;
        }
    }
}
)CUDA" + source;
}
}
#endif
