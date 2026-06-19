// Flat-layout FA2 kernels for Gemma4 LLM runner
// Compiled as AOT cubin, loaded at runtime
// Supports [S, H*D] flat KV cache layout with GQA
//
// Two specializations:
//   fa2_attn_f      : head_dim=256, BC=32  (SWA path, used at d=256)
//   fa2_attn_d512_f : head_dim=512, BC=16  (full-attn path)
//
// Both use Q@K^T -> softmax -> P@V with the F16 tensor-core pipeline. The
// d=512 variant reduces BC (key block size) from 32 to 16 because 4*32*(512+8)*2
// = 133120 B of shared memory exceeds the 99 KB optin ceiling on sm_120; with
// BC=16 the per-block smem drops to 4*16*520*2 = 66560 B, well within the
// limit. The smaller BC means more grid tiles for the same K length but the
// F16 tensor-core pipeline is still compute-bound (the extra grid is pure
// work amplification, not a memory stall).
//
// Both kernels are F32-input -> F16-output, expect F16 K/V cache, and
// support an optional sliding-window lower bound (window=0 -> causal only).

#define FA2_D 256
#define FA2_BR 64
#define FA2_BC 32
#define FA2_CAUSAL 1
#define FA2_DK (FA2_D / 16)
#define FA2_BCN (FA2_BC / 8)
#define FA2_BCK (FA2_BC / 16)
#define FA2_DN (FA2_D / 8)
#define FA2_DP (FA2_D + 8)
#define FA2_NTHR (FA2_BR / 16 * 32)
#define FA2_MMA "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
#define FA2_LB __launch_bounds__(FA2_NTHR)

typedef unsigned short dt_t;
typedef unsigned int uint32_t;

__device__ __forceinline__ float half_to_float(dt_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ dt_t f2dt(float f) {
    dt_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    uint32_t r;
    asm("{ .reg .b16 a, b; cvt.rn.f16.f32 a, %1; cvt.rn.f16.f32 b, %2;"
        "  mov.b32 %0, {a, b}; }" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

extern "C" __global__ FA2_LB
void fa2_attn_f(dt_t *O,
                const dt_t *Q,
                const dt_t *K,
                const dt_t *V,
                int S, int D_in, int hstride_q, int hstride_kv, int gqa_ratio, float scale,
                int window, int n_q, int q_off) {
    /* n_q   = number of query rows in Q/O (queries are rows [0, n_q)).
     * q_off = global position of query row 0 (query row gqr sits at position
     *         q_off+gqr); keys are [0, S). For start_pos==0 prefill n_q==S and
     *         q_off==0, byte-identical to the original single-S kernel. q_off>0
     *         lets a chunk's queries attend a [history ++ chunk] key buffer
     *         (chunked long-context prefill). */
    const float LOG2E = 1.4426950408889634f;
    float lscale = scale * LOG2E;
    int bh = blockIdx.y;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warp_q0 = blockIdx.x * FA2_BR + wid * 16;

    int row_a = lane >> 2;
    int row_b = row_a + 8;
    int col_g = (lane & 3) * 2;

    // Flat layout: Qh points to head bh's data, stride between rows is hstride_q
    const dt_t *Qh = Q + (size_t)bh * FA2_D;
    int kv_bh = bh / gqa_ratio;
    const dt_t *Kh = K + (size_t)kv_bh * FA2_D;
    const dt_t *Vh = V + (size_t)kv_bh * FA2_D;
    dt_t *Oh = O + (size_t)bh * FA2_D;

    uint32_t qfrag[FA2_DK][4];
    int gqr_a = warp_q0 + row_a;
    int gqr_b = warp_q0 + row_b;
    int gpos_a = gqr_a + q_off;  /* global query positions for causal/window mask */
    int gpos_b = gqr_b + q_off;
    int valid_a = (gqr_a < n_q);
    int valid_b = (gqr_b < n_q);
    for (int kk = 0; kk < FA2_DK; kk++) {
        int kbase = kk * 16;
        const dt_t *qa_lo = Qh + (size_t)gqr_a * hstride_q + kbase + col_g;
        const dt_t *qb_lo = Qh + (size_t)gqr_b * hstride_q + kbase + col_g;
        qfrag[kk][0] = valid_a ? *(const uint32_t *)qa_lo : 0u;
        qfrag[kk][1] = valid_b ? *(const uint32_t *)qb_lo : 0u;
        qfrag[kk][2] = valid_a ? *(const uint32_t *)(qa_lo + 8) : 0u;
        qfrag[kk][3] = valid_b ? *(const uint32_t *)(qb_lo + 8) : 0u;
    }

    float ofrag[FA2_DN][4];
    for (int n = 0; n < FA2_DN; n++)
        for (int i = 0; i < 4; i++)
            ofrag[n][i] = 0.0f;
    float m_a = -1e30f, m_b = -1e30f, l_a = 0.0f, l_b = 0.0f;

    extern __shared__ dt_t smem[];
    dt_t *sK0 = smem;
    dt_t *sK1 = sK0 + FA2_BC * FA2_DP;
    dt_t *sV0 = sK1 + FA2_BC * FA2_DP;
    dt_t *sV1 = sV0 + FA2_BC * FA2_DP;
    dt_t *sK_buf[2] = { sK0, sK1 };
    dt_t *sV_buf[2] = { sV0, sV1 };

    int total_u4 = (FA2_BC * FA2_D) / 8;
    int row_u4 = FA2_D / 8;

    int s_eff = S;
#if FA2_CAUSAL
    /* Largest key any query in this block attends = q_off + block's max query. */
    int q_blk_max = q_off + blockIdx.x * FA2_BR + FA2_BR;
    if (q_blk_max < s_eff) s_eff = q_blk_max;
#endif
    int n_tiles = (s_eff + FA2_BC - 1) / FA2_BC;
    /* Sliding window: skip tiles entirely below the smallest query's lower bound.
     * The smallest query in this block sits at position q_off + blockIdx.x*FA2_BR.
     * window=0 -> t_lo=0 (causal path byte-identical). */
    int t_lo = 0;
    if (window > 0) {
        int klo = q_off + blockIdx.x * FA2_BR - window + 1;
        if (klo > 0) t_lo = klo / FA2_BC;
    }

    // Prologue: cp.async tile t_lo into buffer 0
    {
        dt_t *dK_ = sK_buf[0];
        dt_t *dV_ = sV_buf[0];
        for (int idx = threadIdx.x; idx < total_u4; idx += blockDim.x) {
            int j_ = idx / row_u4;
            int dd_ = (idx - j_ * row_u4) * 8;
            int kv_ = t_lo * FA2_BC + j_;
            unsigned dst_k = __cvta_generic_to_shared(dK_ + j_ * FA2_DP + dd_);
            unsigned dst_v = __cvta_generic_to_shared(dV_ + j_ * FA2_DP + dd_);
            const void *src_k = (const void *)(Kh + (size_t)kv_ * hstride_kv + dd_);
            const void *src_v = (const void *)(Vh + (size_t)kv_ * hstride_kv + dd_);
            int src_bytes = (kv_ < S) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst_k), "l"(src_k), "r"(src_bytes));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst_v), "l"(src_v), "r"(src_bytes));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);

    int ldm_row_off = lane & 7;
    int ldm_col_xtra = (lane & 8) ? 8 : 0;

    for (int t = t_lo; t < n_tiles; t++) {
        int cur = (t - t_lo) & 1;
        int nxt = 1 - cur;
        int kv0 = t * FA2_BC;

        if (t + 1 < n_tiles) {
            int kv0n = (t + 1) * FA2_BC;
            dt_t *dK_ = sK_buf[nxt];
            dt_t *dV_ = sV_buf[nxt];
            for (int idx = threadIdx.x; idx < total_u4; idx += blockDim.x) {
                int j_ = idx / row_u4;
                int dd_ = (idx - j_ * row_u4) * 8;
                int kv_ = kv0n + j_;
                unsigned dst_k = __cvta_generic_to_shared(dK_ + j_ * FA2_DP + dd_);
                unsigned dst_v = __cvta_generic_to_shared(dV_ + j_ * FA2_DP + dd_);
                const void *src_k = (const void *)(Kh + (size_t)kv_ * hstride_kv + dd_);
                const void *src_v = (const void *)(Vh + (size_t)kv_ * hstride_kv + dd_);
                int src_bytes = (kv_ < S) ? 16 : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst_k), "l"(src_k), "r"(src_bytes));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst_v), "l"(src_v), "r"(src_bytes));
            }
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        dt_t *sK = sK_buf[cur];
        dt_t *sV = sV_buf[cur];

        // Q @ K^T -> sfrag[BCN][4]
        float sfrag[FA2_BCN][4];
        for (int n = 0; n < FA2_BCN; n++)
            for (int i = 0; i < 4; i++)
                sfrag[n][i] = 0.0f;
        for (int kk = 0; kk < FA2_DK; kk++) {
            int kbase = kk * 16;
            for (int nn = 0; nn < FA2_BCN; nn++) {
                int nbase = nn * 8;
                unsigned saddr = __cvta_generic_to_shared(
                    sK + (nbase + ldm_row_off) * FA2_DP + kbase + ldm_col_xtra);
                uint32_t b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                    : "=r"(b0), "=r"(b1) : "r"(saddr));
                uint32_t a0 = qfrag[kk][0], a1 = qfrag[kk][1], a2 = qfrag[kk][2], a3 = qfrag[kk][3];
                float c0 = sfrag[nn][0], c1 = sfrag[nn][1], c2 = sfrag[nn][2], c3 = sfrag[nn][3];
                asm(FA2_MMA
                    " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1));
                sfrag[nn][0] = c0; sfrag[nn][1] = c1; sfrag[nn][2] = c2; sfrag[nn][3] = c3;
            }
        }

        // Scale + causal mask
        for (int nn = 0; nn < FA2_BCN; nn++) {
            int kv_c0 = kv0 + nn * 8 + col_g + 0;
            int kv_c1 = kv0 + nn * 8 + col_g + 1;
            int va0 = (kv_c0 < S), va1 = (kv_c1 < S);
            int vb0 = va0, vb1 = va1;
#if FA2_CAUSAL
            va0 = va0 && (kv_c0 <= gpos_a); va1 = va1 && (kv_c1 <= gpos_a);
            vb0 = vb0 && (kv_c0 <= gpos_b); vb1 = vb1 && (kv_c1 <= gpos_b);
            /* Sliding-window lower bound: query q attends only to keys (q-window, q]. */
            if (window > 0) {
                va0 = va0 && (kv_c0 > gpos_a - window); va1 = va1 && (kv_c1 > gpos_a - window);
                vb0 = vb0 && (kv_c0 > gpos_b - window); vb1 = vb1 && (kv_c1 > gpos_b - window);
            }
#endif
            sfrag[nn][0] = va0 ? sfrag[nn][0] * lscale : -1e30f;
            sfrag[nn][1] = va1 ? sfrag[nn][1] * lscale : -1e30f;
            sfrag[nn][2] = vb0 ? sfrag[nn][2] * lscale : -1e30f;
            sfrag[nn][3] = vb1 ? sfrag[nn][3] * lscale : -1e30f;
        }

        // Row max over the 4-thread row group
        float mx_a = -1e30f, mx_b = -1e30f;
        for (int nn = 0; nn < FA2_BCN; nn++) {
            mx_a = fmaxf(mx_a, fmaxf(sfrag[nn][0], sfrag[nn][1]));
            mx_b = fmaxf(mx_b, fmaxf(sfrag[nn][2], sfrag[nn][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));

        float m_new_a = fmaxf(m_a, mx_a);
        float m_new_b = fmaxf(m_b, mx_b);
        float alpha_a = exp2f(m_a - m_new_a);
        float alpha_b = exp2f(m_b - m_new_b);
        l_a *= alpha_a; l_b *= alpha_b;
        for (int n = 0; n < FA2_DN; n++) {
            ofrag[n][0] *= alpha_a; ofrag[n][1] *= alpha_a;
            ofrag[n][2] *= alpha_b; ofrag[n][3] *= alpha_b;
        }
        m_a = m_new_a; m_b = m_new_b;

        // P = exp2(S - m), pack to pfrag
        uint32_t pfrag[FA2_BCK][4];
        float rs_a = 0.0f, rs_b = 0.0f;
        for (int kk = 0; kk < FA2_BCK; kk++) {
            float p0 = exp2f(sfrag[2*kk    ][0] - m_a);
            float p1 = exp2f(sfrag[2*kk    ][1] - m_a);
            float p2 = exp2f(sfrag[2*kk    ][2] - m_b);
            float p3 = exp2f(sfrag[2*kk    ][3] - m_b);
            float p4 = exp2f(sfrag[2*kk + 1][0] - m_a);
            float p5 = exp2f(sfrag[2*kk + 1][1] - m_a);
            float p6 = exp2f(sfrag[2*kk + 1][2] - m_b);
            float p7 = exp2f(sfrag[2*kk + 1][3] - m_b);
            rs_a += p0 + p1 + p4 + p5;
            rs_b += p2 + p3 + p6 + p7;
            pfrag[kk][0] = pack2(p0, p1);
            pfrag[kk][1] = pack2(p2, p3);
            pfrag[kk][2] = pack2(p4, p5);
            pfrag[kk][3] = pack2(p6, p7);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1);
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        l_a += rs_a; l_b += rs_b;

        // P @ V via ldmatrix.x2.trans on sV[bc, d] row-major
        for (int kk = 0; kk < FA2_BCK; kk++) {
            int kbase = kk * 16;
            int bc_row = kbase + ldm_row_off + ((lane & 8) ? 8 : 0);
            for (int nn = 0; nn < FA2_DN; nn++) {
                int nbase = nn * 8;
                unsigned saddr = __cvta_generic_to_shared(sV + bc_row * FA2_DP + nbase);
                uint32_t b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                    : "=r"(b0), "=r"(b1) : "r"(saddr));
                uint32_t a0 = pfrag[kk][0], a1 = pfrag[kk][1], a2 = pfrag[kk][2], a3 = pfrag[kk][3];
                float c0 = ofrag[nn][0], c1 = ofrag[nn][1], c2 = ofrag[nn][2], c3 = ofrag[nn][3];
                asm(FA2_MMA
                    " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1));
                ofrag[nn][0] = c0; ofrag[nn][1] = c1; ofrag[nn][2] = c2; ofrag[nn][3] = c3;
            }
        }
        __syncthreads();
    }

    // Epilogue: divide by l, store O
    float inv_a = (l_a > 0.0f) ? 1.0f / l_a : 0.0f;
    float inv_b = (l_b > 0.0f) ? 1.0f / l_b : 0.0f;
    for (int nn = 0; nn < FA2_DN; nn++) {
        int d_col = nn * 8 + col_g;
        if (gqr_a < n_q) {
            Oh[(size_t)gqr_a * hstride_q + d_col + 0] = f2dt(ofrag[nn][0] * inv_a);
            Oh[(size_t)gqr_a * hstride_q + d_col + 1] = f2dt(ofrag[nn][1] * inv_a);
        }
        if (gqr_b < n_q) {
            Oh[(size_t)gqr_b * hstride_q + d_col + 0] = f2dt(ofrag[nn][2] * inv_b);
            Oh[(size_t)gqr_b * hstride_q + d_col + 1] = f2dt(ofrag[nn][3] * inv_b);
        }
    }
}

// =====================================================================
// d=512 specialization. BC=16 (vs 32 for d=256) to keep smem under the
// 99 KB optin on sm_120: 4*16*520*2 = 66560 B vs 4*32*520*2 = 133120 B.
// Same FA2 algorithm as the d=256 kernel; only the FA2_D/BC/DK/DN/DP
// constants differ. The two kernels are byte-identical to each other
// when D=256 and BC=32 (i.e. the d=256 path is the canonical).
// =====================================================================
#undef FA2_D
#undef FA2_BR
#undef FA2_BC
#undef FA2_DK
#undef FA2_BCN
#undef FA2_BCK
#undef FA2_DN
#undef FA2_DP
#undef FA2_NTHR
#undef FA2_LB

#define FA2_D 512
#define FA2_BR 64
#define FA2_BC 16
#define FA2_CAUSAL 1
#define FA2_DK (FA2_D / 16)
#define FA2_BCN (FA2_BC / 8)
#define FA2_BCK (FA2_BC / 16)
#define FA2_DN (FA2_D / 8)
#define FA2_DP (FA2_D + 8)
#define FA2_NTHR (FA2_BR / 16 * 32)
#define FA2_MMA "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
#define FA2_LB __launch_bounds__(FA2_NTHR)

extern "C" __global__ FA2_LB
void fa2_attn_d512_f(dt_t *O,
                     const dt_t *Q,
                     const dt_t *K,
                     const dt_t *V,
                     int S, int D_in, int hstride_q, int hstride_kv, int gqa_ratio, float scale,
                     int window, int n_q, int q_off) {
    const float d5_LOG2E = 1.4426950408889634f;
    float d5_lscale = scale * d5_LOG2E;
    int d5_bh = blockIdx.y;
    int d5_wid = threadIdx.x >> 5;
    int d5_lane = threadIdx.x & 31;
    int d5_warp_q0 = blockIdx.x * FA2_BR + d5_wid * 16;

    int d5_row_a = d5_lane >> 2;
    int d5_row_b = d5_row_a + 8;
    int d5_col_g = (d5_lane & 3) * 2;

    const dt_t *d5_Qh = Q + (size_t)d5_bh * FA2_D;
    int d5_kv_bh = d5_bh / gqa_ratio;
    const dt_t *d5_Kh = K + (size_t)d5_kv_bh * FA2_D;
    const dt_t *d5_Vh = V + (size_t)d5_kv_bh * FA2_D;
    dt_t *d5_Oh = O + (size_t)d5_bh * FA2_D;

    uint32_t d5_qfrag[FA2_DK][4];
    int d5_gqr_a = d5_warp_q0 + d5_row_a;
    int d5_gqr_b = d5_warp_q0 + d5_row_b;
    int d5_gpos_a = d5_gqr_a + q_off;
    int d5_gpos_b = d5_gqr_b + q_off;
    int d5_valid_a = (d5_gqr_a < n_q);
    int d5_valid_b = (d5_gqr_b < n_q);
    for (int d5_kk = 0; d5_kk < FA2_DK; d5_kk++) {
        int d5_kbase = d5_kk * 16;
        const dt_t *d5_qa_lo = d5_Qh + (size_t)d5_gqr_a * hstride_q + d5_kbase + d5_col_g;
        const dt_t *d5_qb_lo = d5_Qh + (size_t)d5_gqr_b * hstride_q + d5_kbase + d5_col_g;
        d5_qfrag[d5_kk][0] = d5_valid_a ? *(const uint32_t *)d5_qa_lo : 0u;
        d5_qfrag[d5_kk][1] = d5_valid_b ? *(const uint32_t *)d5_qb_lo : 0u;
        d5_qfrag[d5_kk][2] = d5_valid_a ? *(const uint32_t *)(d5_qa_lo + 8) : 0u;
        d5_qfrag[d5_kk][3] = d5_valid_b ? *(const uint32_t *)(d5_qb_lo + 8) : 0u;
    }

    float d5_ofrag[FA2_DN][4];
    for (int n = 0; n < FA2_DN; n++)
        for (int i = 0; i < 4; i++)
            d5_ofrag[n][i] = 0.0f;
    float d5_m_a = -1e30f, d5_m_b = -1e30f, d5_l_a = 0.0f, d5_l_b = 0.0f;

    extern __shared__ dt_t d5_smem[];
    dt_t *d5_sK0 = d5_smem;
    dt_t *d5_sK1 = d5_sK0 + FA2_BC * FA2_DP;
    dt_t *d5_sV0 = d5_sK1 + FA2_BC * FA2_DP;
    dt_t *d5_sV1 = d5_sV0 + FA2_BC * FA2_DP;
    dt_t *d5_sK_buf[2] = { d5_sK0, d5_sK1 };
    dt_t *d5_sV_buf[2] = { d5_sV0, d5_sV1 };

    int d5_total_u4 = (FA2_BC * FA2_D) / 8;
    int d5_row_u4 = FA2_D / 8;

    int d5_s_eff = S;
#if FA2_CAUSAL
    int d5_q_blk_max = q_off + blockIdx.x * FA2_BR + FA2_BR;
    if (d5_q_blk_max < d5_s_eff) d5_s_eff = d5_q_blk_max;
#endif
    int d5_n_tiles = (d5_s_eff + FA2_BC - 1) / FA2_BC;
    int d5_t_lo = 0;
    if (window > 0) {
        int d5_klo = q_off + blockIdx.x * FA2_BR - window + 1;
        if (d5_klo > 0) d5_t_lo = d5_klo / FA2_BC;
    }

    {
        dt_t *d5_dK_ = d5_sK_buf[0];
        dt_t *d5_dV_ = d5_sV_buf[0];
        for (int d5_idx = threadIdx.x; d5_idx < d5_total_u4; d5_idx += blockDim.x) {
            int d5_j_ = d5_idx / d5_row_u4;
            int d5_dd_ = (d5_idx - d5_j_ * d5_row_u4) * 8;
            int d5_kv_ = d5_t_lo * FA2_BC + d5_j_;
            unsigned d5_dst_k = __cvta_generic_to_shared(d5_dK_ + d5_j_ * FA2_DP + d5_dd_);
            unsigned d5_dst_v = __cvta_generic_to_shared(d5_dV_ + d5_j_ * FA2_DP + d5_dd_);
            const void *d5_src_k = (const void *)(d5_Kh + (size_t)d5_kv_ * hstride_kv + d5_dd_);
            const void *d5_src_v = (const void *)(d5_Vh + (size_t)d5_kv_ * hstride_kv + d5_dd_);
            int d5_src_bytes = (d5_kv_ < S) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d5_dst_k), "l"(d5_src_k), "r"(d5_src_bytes));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d5_dst_v), "l"(d5_src_v), "r"(d5_src_bytes));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);

    int d5_ldm_row_off = d5_lane & 7;
    int d5_ldm_col_xtra = (d5_lane & 8) ? 8 : 0;

    for (int t = d5_t_lo; t < d5_n_tiles; t++) {
        int d5_cur = (t - d5_t_lo) & 1;
        int d5_nxt = 1 - d5_cur;
        int d5_kv0 = t * FA2_BC;

        if (t + 1 < d5_n_tiles) {
            int d5_kv0n = (t + 1) * FA2_BC;
            dt_t *d5_dK_ = d5_sK_buf[d5_nxt];
            dt_t *d5_dV_ = d5_sV_buf[d5_nxt];
            for (int d5_idx = threadIdx.x; d5_idx < d5_total_u4; d5_idx += blockDim.x) {
                int d5_j_ = d5_idx / d5_row_u4;
                int d5_dd_ = (d5_idx - d5_j_ * d5_row_u4) * 8;
                int d5_kv_ = d5_kv0n + d5_j_;
                unsigned d5_dst_k = __cvta_generic_to_shared(d5_dK_ + d5_j_ * FA2_DP + d5_dd_);
                unsigned d5_dst_v = __cvta_generic_to_shared(d5_dV_ + d5_j_ * FA2_DP + d5_dd_);
                const void *d5_src_k = (const void *)(d5_Kh + (size_t)d5_kv_ * hstride_kv + d5_dd_);
                const void *d5_src_v = (const void *)(d5_Vh + (size_t)d5_kv_ * hstride_kv + d5_dd_);
                int d5_src_bytes = (d5_kv_ < S) ? 16 : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d5_dst_k), "l"(d5_src_k), "r"(d5_src_bytes));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d5_dst_v), "l"(d5_src_v), "r"(d5_src_bytes));
            }
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        dt_t *d5_sK = d5_sK_buf[d5_cur];
        dt_t *d5_sV = d5_sV_buf[d5_cur];

        // Q @ K^T
        float d5_sfrag[FA2_BCN][4];
        for (int n = 0; n < FA2_BCN; n++)
            for (int i = 0; i < 4; i++)
                d5_sfrag[n][i] = 0.0f;
        for (int d5_kk = 0; d5_kk < FA2_DK; d5_kk++) {
            int d5_kbase = d5_kk * 16;
            for (int d5_nn = 0; d5_nn < FA2_BCN; d5_nn++) {
                int d5_nbase = d5_nn * 8;
                unsigned d5_saddr = __cvta_generic_to_shared(
                    d5_sK + (d5_nbase + d5_ldm_row_off) * FA2_DP + d5_kbase + d5_ldm_col_xtra);
                uint32_t d5_b0, d5_b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                    : "=r"(d5_b0), "=r"(d5_b1) : "r"(d5_saddr));
                uint32_t d5_a0 = d5_qfrag[d5_kk][0], d5_a1 = d5_qfrag[d5_kk][1], d5_a2 = d5_qfrag[d5_kk][2], d5_a3 = d5_qfrag[d5_kk][3];
                float d5_c0 = d5_sfrag[d5_nn][0], d5_c1 = d5_sfrag[d5_nn][1], d5_c2 = d5_sfrag[d5_nn][2], d5_c3 = d5_sfrag[d5_nn][3];
                asm(FA2_MMA
                    " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(d5_c0), "+f"(d5_c1), "+f"(d5_c2), "+f"(d5_c3)
                    : "r"(d5_a0), "r"(d5_a1), "r"(d5_a2), "r"(d5_a3),
                      "r"(d5_b0), "r"(d5_b1));
                d5_sfrag[d5_nn][0] = d5_c0; d5_sfrag[d5_nn][1] = d5_c1; d5_sfrag[d5_nn][2] = d5_c2; d5_sfrag[d5_nn][3] = d5_c3;
            }
        }

        // Scale + causal mask
        for (int d5_nn = 0; d5_nn < FA2_BCN; d5_nn++) {
            int d5_kv_c0 = d5_kv0 + d5_nn * 8 + d5_col_g + 0;
            int d5_kv_c1 = d5_kv0 + d5_nn * 8 + d5_col_g + 1;
            int d5_va0 = (d5_kv_c0 < S), d5_va1 = (d5_kv_c1 < S);
            int d5_vb0 = d5_va0, d5_vb1 = d5_va1;
#if FA2_CAUSAL
            d5_va0 = d5_va0 && (d5_kv_c0 <= d5_gpos_a); d5_va1 = d5_va1 && (d5_kv_c1 <= d5_gpos_a);
            d5_vb0 = d5_vb0 && (d5_kv_c0 <= d5_gpos_b); d5_vb1 = d5_vb1 && (d5_kv_c1 <= d5_gpos_b);
            if (window > 0) {
                d5_va0 = d5_va0 && (d5_kv_c0 > d5_gpos_a - window); d5_va1 = d5_va1 && (d5_kv_c1 > d5_gpos_a - window);
                d5_vb0 = d5_vb0 && (d5_kv_c0 > d5_gpos_b - window); d5_vb1 = d5_vb1 && (d5_kv_c1 > d5_gpos_b - window);
            }
#endif
            d5_sfrag[d5_nn][0] = d5_va0 ? d5_sfrag[d5_nn][0] * d5_lscale : -1e30f;
            d5_sfrag[d5_nn][1] = d5_va1 ? d5_sfrag[d5_nn][1] * d5_lscale : -1e30f;
            d5_sfrag[d5_nn][2] = d5_vb0 ? d5_sfrag[d5_nn][2] * d5_lscale : -1e30f;
            d5_sfrag[d5_nn][3] = d5_vb1 ? d5_sfrag[d5_nn][3] * d5_lscale : -1e30f;
        }

        // Row max
        float d5_mx_a = -1e30f, d5_mx_b = -1e30f;
        for (int d5_nn = 0; d5_nn < FA2_BCN; d5_nn++) {
            d5_mx_a = fmaxf(d5_mx_a, fmaxf(d5_sfrag[d5_nn][0], d5_sfrag[d5_nn][1]));
            d5_mx_b = fmaxf(d5_mx_b, fmaxf(d5_sfrag[d5_nn][2], d5_sfrag[d5_nn][3]));
        }
        d5_mx_a = fmaxf(d5_mx_a, __shfl_xor_sync(0xffffffff, d5_mx_a, 1));
        d5_mx_a = fmaxf(d5_mx_a, __shfl_xor_sync(0xffffffff, d5_mx_a, 2));
        d5_mx_b = fmaxf(d5_mx_b, __shfl_xor_sync(0xffffffff, d5_mx_b, 1));
        d5_mx_b = fmaxf(d5_mx_b, __shfl_xor_sync(0xffffffff, d5_mx_b, 2));

        float d5_m_new_a = fmaxf(d5_m_a, d5_mx_a);
        float d5_m_new_b = fmaxf(d5_m_b, d5_mx_b);
        float d5_alpha_a = exp2f(d5_m_a - d5_m_new_a);
        float d5_alpha_b = exp2f(d5_m_b - d5_m_new_b);
        d5_l_a *= d5_alpha_a; d5_l_b *= d5_alpha_b;
        for (int n = 0; n < FA2_DN; n++) {
            d5_ofrag[n][0] *= d5_alpha_a; d5_ofrag[n][1] *= d5_alpha_a;
            d5_ofrag[n][2] *= d5_alpha_b; d5_ofrag[n][3] *= d5_alpha_b;
        }
        d5_m_a = d5_m_new_a; d5_m_b = d5_m_new_b;

        // P = exp2(S - m)
        uint32_t d5_pfrag[FA2_BCK][4];
        float d5_rs_a = 0.0f, d5_rs_b = 0.0f;
        for (int d5_kk = 0; d5_kk < FA2_BCK; d5_kk++) {
            float p0 = exp2f(d5_sfrag[2*d5_kk    ][0] - d5_m_a);
            float p1 = exp2f(d5_sfrag[2*d5_kk    ][1] - d5_m_a);
            float p2 = exp2f(d5_sfrag[2*d5_kk    ][2] - d5_m_b);
            float p3 = exp2f(d5_sfrag[2*d5_kk    ][3] - d5_m_b);
            float p4 = exp2f(d5_sfrag[2*d5_kk + 1][0] - d5_m_a);
            float p5 = exp2f(d5_sfrag[2*d5_kk + 1][1] - d5_m_a);
            float p6 = exp2f(d5_sfrag[2*d5_kk + 1][2] - d5_m_b);
            float p7 = exp2f(d5_sfrag[2*d5_kk + 1][3] - d5_m_b);
            d5_rs_a += p0 + p1 + p4 + p5;
            d5_rs_b += p2 + p3 + p6 + p7;
            d5_pfrag[d5_kk][0] = pack2(p0, p1);
            d5_pfrag[d5_kk][1] = pack2(p2, p3);
            d5_pfrag[d5_kk][2] = pack2(p4, p5);
            d5_pfrag[d5_kk][3] = pack2(p6, p7);
        }
        d5_rs_a += __shfl_xor_sync(0xffffffff, d5_rs_a, 1);
        d5_rs_a += __shfl_xor_sync(0xffffffff, d5_rs_a, 2);
        d5_rs_b += __shfl_xor_sync(0xffffffff, d5_rs_b, 1);
        d5_rs_b += __shfl_xor_sync(0xffffffff, d5_rs_b, 2);
        d5_l_a += d5_rs_a; d5_l_b += d5_rs_b;

        // P @ V
        for (int d5_kk = 0; d5_kk < FA2_BCK; d5_kk++) {
            int d5_kbase = d5_kk * 16;
            int d5_bc_row = d5_kbase + d5_ldm_row_off + ((d5_lane & 8) ? 8 : 0);
            for (int d5_nn = 0; d5_nn < FA2_DN; d5_nn++) {
                int d5_nbase = d5_nn * 8;
                unsigned d5_saddr = __cvta_generic_to_shared(d5_sV + d5_bc_row * FA2_DP + d5_nbase);
                uint32_t d5_b0, d5_b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                    : "=r"(d5_b0), "=r"(d5_b1) : "r"(d5_saddr));
                uint32_t d5_a0 = d5_pfrag[d5_kk][0], d5_a1 = d5_pfrag[d5_kk][1], d5_a2 = d5_pfrag[d5_kk][2], d5_a3 = d5_pfrag[d5_kk][3];
                float d5_c0 = d5_ofrag[d5_nn][0], d5_c1 = d5_ofrag[d5_nn][1], d5_c2 = d5_ofrag[d5_nn][2], d5_c3 = d5_ofrag[d5_nn][3];
                asm(FA2_MMA
                    " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(d5_c0), "+f"(d5_c1), "+f"(d5_c2), "+f"(d5_c3)
                    : "r"(d5_a0), "r"(d5_a1), "r"(d5_a2), "r"(d5_a3),
                      "r"(d5_b0), "r"(d5_b1));
                d5_ofrag[d5_nn][0] = d5_c0; d5_ofrag[d5_nn][1] = d5_c1; d5_ofrag[d5_nn][2] = d5_c2; d5_ofrag[d5_nn][3] = d5_c3;
            }
        }
        __syncthreads();
    }

    // Epilogue
    float d5_inv_a = (d5_l_a > 0.0f) ? 1.0f / d5_l_a : 0.0f;
    float d5_inv_b = (d5_l_b > 0.0f) ? 1.0f / d5_l_b : 0.0f;
    for (int d5_nn = 0; d5_nn < FA2_DN; d5_nn++) {
        int d5_d_col = d5_nn * 8 + d5_col_g;
        if (d5_gqr_a < n_q) {
            d5_Oh[(size_t)d5_gqr_a * hstride_q + d5_d_col + 0] = f2dt(d5_ofrag[d5_nn][0] * d5_inv_a);
            d5_Oh[(size_t)d5_gqr_a * hstride_q + d5_d_col + 1] = f2dt(d5_ofrag[d5_nn][1] * d5_inv_a);
        }
        if (d5_gqr_b < n_q) {
            d5_Oh[(size_t)d5_gqr_b * hstride_q + d5_d_col + 0] = f2dt(d5_ofrag[d5_nn][2] * d5_inv_b);
            d5_Oh[(size_t)d5_gqr_b * hstride_q + d5_d_col + 1] = f2dt(d5_ofrag[d5_nn][3] * d5_inv_b);
        }
    }
}
