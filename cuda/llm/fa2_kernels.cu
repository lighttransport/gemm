// Flat-layout FA2 kernel for Gemma4 LLM runner
// Compiled as AOT cubin, loaded at runtime
// Supports [S, H*D] flat KV cache layout with GQA
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
