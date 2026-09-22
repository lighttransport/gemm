/* Q2_K x Q8_1 MMVQ arithmetic from llama.cpp 1859b5209, gfx1201.
 * Copyright (c) 2023-2026 The ggml authors, MIT license (see
 * reference_sampler.cpp for the complete retained license).
 * Compile without fast-math. One 256-thread block computes one output row,
 * matching the reference's eight-warp reduction order. */
extern "C" __global__ void qwen35_quantize_q81(
        signed char *q, float *scales, const float *x, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i >= n) return;
    float v = x[i], a = fabsf(v);
    for (int offset = 16; offset; offset >>= 1)
        a = fmaxf(a, __shfl_xor(a, offset, 32));
    float d = a / 127.0f;
    q[i] = a == 0.0f ? 0 : (signed char)roundf(v / d);
    if (threadIdx.x == 0) scales[blockIdx.x] = (float)(_Float16)d;
}

/* DFlash2's Q4_K projections evaluate at most eight adjacent draft rows.
 * Quantizing those activations once per 32 values replaces the scalar F32
 * inner products with packed gfx1201 dot products while each output warp
 * still reuses one decoded weight block across every row.  This path is only
 * used to propose tokens; the target model remains the authority. */
extern "C" __global__ void qwen35_matvec_q4k_q81_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    int lane=threadIdx.x&31;
    int row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(row>=rows)return;
    int blocks=cols/256;
    const unsigned char *w=weights+(size_t)row*blocks*144;
    float sum[8]={};
    for(int chunk=lane;chunk<blocks*4;chunk+=32){
        int b=chunk>>2,c=chunk&3,is=2*c;
        const unsigned char *p=w+(size_t)b*144;
        const unsigned char *sc=p+4;
        int sv0,mv0,sv1,mv1;
        if(is<4){sv0=sc[is]&63;mv0=sc[is+4]&63;}
        else{sv0=(sc[is+4]&15)|((sc[is-4]>>6)<<4);
             mv0=(sc[is+4]>>4)|((sc[is]>>6)<<4);}
        if(is+1<4){sv1=sc[is+1]&63;mv1=sc[is+5]&63;}
        else{sv1=(sc[is+5]&15)|((sc[is-3]>>6)<<4);
             mv1=(sc[is+5]>>4)|((sc[is+1]>>6)<<4);}
        const unsigned int *qw=(const unsigned int *)(p+16+c*32);
        int dot0[8]={},dot1[8]={},xsum0[8]={},xsum1[8]={};
#pragma unroll
        for(int j=0;j<8;++j){
            unsigned int packed=qw[j];
            int lo=(int)(packed&0x0f0f0f0fu);
            int hi=(int)((packed>>4)&0x0f0f0f0fu);
#pragma unroll
            for(int m=0;m<8;++m)if(m<count){
                const int *u=(const int *)(q+(size_t)m*cols+b*256+c*64);
                dot0[m]=__builtin_amdgcn_sudot4(true,lo,true,u[j],dot0[m],false);
                dot1[m]=__builtin_amdgcn_sudot4(true,hi,true,u[j+8],dot1[m],false);
                xsum0[m]=__builtin_amdgcn_sudot4(true,0x01010101,true,u[j],xsum0[m],false);
                xsum1[m]=__builtin_amdgcn_sudot4(true,0x01010101,true,u[j+8],xsum1[m],false);
            }
        }
        float d=(float)*(const _Float16 *)p;
        float dm=(float)*(const _Float16 *)(p+2);
#pragma unroll
        for(int m=0;m<8;++m)if(m<count){
            const float *sx=scales+(size_t)m*(cols/32)+b*8+is;
            sum[m]+=sx[0]*(d*sv0*dot0[m]-dm*mv0*xsum0[m])+
                    sx[1]*(d*sv1*dot1[m]-dm*mv1*xsum1[m]);
        }
    }
#pragma unroll
    for(int m=0;m<8;++m)if(m<count){
        for(int offset=16;offset;offset>>=1)
            sum[m]+=__shfl_down(sum[m],offset,32);
        if(lane==0)out[(size_t)m*rows+row]=sum[m];
    }
}

/* K=4 DFlash2 windows contain five rows (anchor plus four mask rows).  Keep
 * the exact multi8 arithmetic, but make the row count compile-time constant
 * so HIPRTC can discard three unused accumulators and their branches. */
extern "C" __global__ void qwen35_matvec_q4k_q81_multi5(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    (void) count;
    int lane=threadIdx.x&31;
    int row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(row>=rows)return;
    int blocks=cols/256;
    const unsigned char *w=weights+(size_t)row*blocks*144;
    float sum[5]={};
    for(int chunk=lane;chunk<blocks*4;chunk+=32){
        int b=chunk>>2,c=chunk&3,is=2*c;
        const unsigned char *p=w+(size_t)b*144;
        const unsigned char *sc=p+4;
        int sv0,mv0,sv1,mv1;
        if(is<4){sv0=sc[is]&63;mv0=sc[is+4]&63;}
        else{sv0=(sc[is+4]&15)|((sc[is-4]>>6)<<4);
             mv0=(sc[is+4]>>4)|((sc[is]>>6)<<4);}
        if(is+1<4){sv1=sc[is+1]&63;mv1=sc[is+5]&63;}
        else{sv1=(sc[is+5]&15)|((sc[is-3]>>6)<<4);
             mv1=(sc[is+5]>>4)|((sc[is+1]>>6)<<4);}
        const unsigned int *qw=(const unsigned int *)(p+16+c*32);
        int dot0[5]={},dot1[5]={},xsum0[5]={},xsum1[5]={};
#pragma unroll
        for(int j=0;j<8;++j){
            unsigned int packed=qw[j];
            int lo=(int)(packed&0x0f0f0f0fu);
            int hi=(int)((packed>>4)&0x0f0f0f0fu);
#pragma unroll
            for(int m=0;m<5;++m){
                const int *u=(const int *)(q+(size_t)m*cols+b*256+c*64);
                dot0[m]=__builtin_amdgcn_sudot4(true,lo,true,u[j],dot0[m],false);
                dot1[m]=__builtin_amdgcn_sudot4(true,hi,true,u[j+8],dot1[m],false);
                xsum0[m]=__builtin_amdgcn_sudot4(true,0x01010101,true,u[j],xsum0[m],false);
                xsum1[m]=__builtin_amdgcn_sudot4(true,0x01010101,true,u[j+8],xsum1[m],false);
            }
        }
        float d=(float)*(const _Float16 *)p;
        float dm=(float)*(const _Float16 *)(p+2);
#pragma unroll
        for(int m=0;m<5;++m){
            const float *sx=scales+(size_t)m*(cols/32)+b*8+is;
            sum[m]+=sx[0]*(d*sv0*dot0[m]-dm*mv0*xsum0[m])+
                    sx[1]*(d*sv1*dot1[m]-dm*mv1*xsum1[m]);
        }
    }
#pragma unroll
    for(int m=0;m<5;++m){
        for(int offset=16;offset;offset>>=1)
            sum[m]+=__shfl_down(sum[m],offset,32);
        if(lane==0)out[(size_t)m*rows+row]=sum[m];
    }
}

/* DFlash2's Q/K/V projections consume the same activation rows and input
 * width.  Keep their independent row ranges in one grid so the sidecar pays
 * one launch after the shared Q8_1 quantization.  Each output warp retains
 * qwen35_matvec_q4k_q81_multi8's ordered dot, affine correction, and
 * __shfl_down reduction; only the selected output range changes. */
template<int MAX_COUNT>
__device__ __forceinline__ void qwen35_matvec_q4k_q81_qkv_impl(
        float *qout, float *kout, float *vout,
        const unsigned char *qw, const unsigned char *kw,
        const unsigned char *vw, const signed char *q,
        const float *scales, int qrows, int krows, int vrows,
        int cols, int count) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int total = qrows + krows + vrows;
    if (row >= total) return;

    float *out;
    const unsigned char *weights;
    int output_rows, local_row;
    if (row < qrows) {
        out = qout; weights = qw; output_rows = qrows; local_row = row;
    } else if (row < qrows + krows) {
        out = kout; weights = kw; output_rows = krows;
        local_row = row - qrows;
    } else {
        out = vout; weights = vw; output_rows = vrows;
        local_row = row - qrows - krows;
    }

    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)local_row * blocks * 144;
    float sum[MAX_COUNT] = {};
    for (int chunk = lane; chunk < blocks * 4; chunk += 32) {
        int b = chunk >> 2, c = chunk & 3, is = 2 * c;
        const unsigned char *p = w + (size_t)b * 144;
        const unsigned char *sc = p + 4;
        int sv0, mv0, sv1, mv1;
        if (is < 4) {
            sv0 = sc[is] & 63; mv0 = sc[is + 4] & 63;
        } else {
            sv0 = (sc[is + 4] & 15) | ((sc[is - 4] >> 6) << 4);
            mv0 = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4);
        }
        if (is + 1 < 4) {
            sv1 = sc[is + 1] & 63; mv1 = sc[is + 5] & 63;
        } else {
            sv1 = (sc[is + 5] & 15) | ((sc[is - 3] >> 6) << 4);
            mv1 = (sc[is + 5] >> 4) | ((sc[is + 1] >> 6) << 4);
        }
        const unsigned int *qw32 = (const unsigned int *)(p + 16 + c * 32);
        int dot0[MAX_COUNT] = {}, dot1[MAX_COUNT] = {};
        int xsum0[MAX_COUNT] = {}, xsum1[MAX_COUNT] = {};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned int packed = qw32[j];
            int lo = (int)(packed & 0x0f0f0f0fu);
            int hi = (int)((packed >> 4) & 0x0f0f0f0fu);
#pragma unroll
            for (int m = 0; m < MAX_COUNT; ++m) if (m < count) {
                const int *u = (const int *)(q + (size_t)m * cols +
                                             b * 256 + c * 64);
                dot0[m] = __builtin_amdgcn_sudot4(true, lo, true, u[j],
                                                  dot0[m], false);
                dot1[m] = __builtin_amdgcn_sudot4(true, hi, true, u[j + 8],
                                                  dot1[m], false);
                xsum0[m] = __builtin_amdgcn_sudot4(true, 0x01010101, true,
                                                   u[j], xsum0[m], false);
                xsum1[m] = __builtin_amdgcn_sudot4(true, 0x01010101, true,
                                                   u[j + 8], xsum1[m], false);
            }
        }
        float d = (float)*(const _Float16 *)p;
        float dm = (float)*(const _Float16 *)(p + 2);
#pragma unroll
        for (int m = 0; m < MAX_COUNT; ++m) if (m < count) {
            const float *sx = scales + (size_t)m * (cols / 32) + b * 8 + is;
            sum[m] += sx[0] * (d * sv0 * dot0[m] - dm * mv0 * xsum0[m]) +
                      sx[1] * (d * sv1 * dot1[m] - dm * mv1 * xsum1[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < MAX_COUNT; ++m) if (m < count) {
        for (int offset = 16; offset; offset >>= 1)
            sum[m] += __shfl_down(sum[m], offset, 32);
        if (lane == 0) out[(size_t)m * output_rows + local_row] = sum[m];
    }
}

extern "C" __global__ void qwen35_matvec_q4k_q81_qkv(
        float *qout, float *kout, float *vout,
        const unsigned char *qw, const unsigned char *kw,
        const unsigned char *vw, const signed char *q,
        const float *scales, int qrows, int krows, int vrows,
        int cols, int count) {
    qwen35_matvec_q4k_q81_qkv_impl<8>(qout, kout, vout, qw, kw, vw, q,
                                      scales, qrows, krows, vrows, cols,
                                      count);
}

extern "C" __global__ void qwen35_matvec_q4k_q81_qkv5(
        float *qout, float *kout, float *vout,
        const unsigned char *qw, const unsigned char *kw,
        const unsigned char *vw, const signed char *q,
        const float *scales, int qrows, int krows, int vrows,
        int cols, int count) {
    (void)count;
    qwen35_matvec_q4k_q81_qkv_impl<5>(qout, kout, vout, qw, kw, vw, q,
                                      scales, qrows, krows, vrows, cols, 5);
}

extern "C" __global__ void qwen35_argmax_parts(const float *x, int n,
                                               float *scores, int *indices) {
    __shared__ float values[256];
    __shared__ int ids[256];
    int t=threadIdx.x, id=0, begin=blockIdx.x*4096;
    float value=-3.402823466e38f;
    for (int i=begin+t; i<n && i<begin+4096; i+=256)
        if (x[i]>value || (x[i]==value && i<id)) { value=x[i]; id=i; }
    values[t]=value; ids[t]=id;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (t<s && (values[t+s]>values[t] || (values[t+s]==values[t] && ids[t+s]<ids[t]))) {
            values[t]=values[t+s]; ids[t]=ids[t+s];
        }
        __syncthreads();
    }
    if (!t) { scores[blockIdx.x]=values[0]; indices[blockIdx.x]=ids[0]; }
}

extern "C" __global__ void qwen35_argmax_finish(const float *scores,
        const int *indices, int n, int *out) {
    __shared__ float values[256];
    __shared__ int ids[256];
    int t=threadIdx.x;
    values[t]=t<n ? scores[t] : -3.402823466e38f;
    ids[t]=t<n ? indices[t] : 0;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (t<s && (values[t+s]>values[t] || (values[t+s]==values[t] && ids[t+s]<ids[t]))) {
            values[t]=values[t+s]; ids[t]=ids[t+s];
        }
        __syncthreads();
    }
    if (!t) *out=ids[0];
}

extern "C" __global__ void qwen35_matvec_q2k(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (row >= rows) return;
    int blocks = cols / 256, iqs = tid & 15;
    const unsigned char *w = weights + (size_t)row * blocks * 84;
    float sum = 0.0f;
    for (int b = tid / 16; b < blocks; b += 16) {
        const unsigned char *p = w + b * 84;
        const unsigned char *s = p + (iqs & ~7) + (iqs & 7) / 4;
        int v = ((const int *)(p + 16))[iqs];
        float sd = 0.0f, sm = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qb = b * 8 + 4 * (iqs / 8) + j;
            int u = ((const int *)(q + qb * 32))[iqs & 7];
            int z = __builtin_amdgcn_sudot4(true, (v >> (2*j)) & 0x03030303,
                                          true, u, 0, false);
            sd += scales[qb] * (z * (s[2*j] & 15));
            int m = s[2*j] >> 4;
            m |= m << 8; m |= m << 16;
            int zm = __builtin_amdgcn_sudot4(true, m, true, u, 0, false);
            sm += scales[qb] * zm;
        }
        float d = (float)*(const _Float16 *)(p + 80);
        float dm = (float)*(const _Float16 *)(p + 82);
        sum += d * sd - dm * sm;
    }
    __shared__ float partial[7][32];
    if (warp > 0) partial[warp - 1][lane] = sum;
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (int widx = 0; widx < 7; ++widx) sum += partial[widx][lane];
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}

/* One warp per row, evaluated as eight virtual reference warps. Keeping
 * their accumulators separate preserves the eight-warp reduction order.
 * This improves tall 5120-column projections; the original kernel remains
 * preferable for the 17408-column down projections. */
extern "C" __global__ void qwen35_matvec_q2k_rows(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    int blocks = cols / 256, iqs = lane & 15;
    const unsigned char *w = weights + (size_t)row * blocks * 84;
    float parts[8] = {};
#pragma unroll
    for (int vw = 0; vw < 8; ++vw) {
        for (int b = lane / 16 + 2 * vw; b < blocks; b += 16) {
            const unsigned char *p = w + b * 84;
            const unsigned char *s = p + (iqs & ~7) + (iqs & 7) / 4;
            int v = ((const int *)(p + 16))[iqs];
            float sd = 0.0f, sm = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int qb = b * 8 + 4 * (iqs / 8) + j;
                int u = ((const int *)(q + qb * 32))[iqs & 7];
                int z = __builtin_amdgcn_sudot4(true, (v >> (2*j)) & 0x03030303,
                                              true, u, 0, false);
                sd += scales[qb] * (z * (s[2*j] & 15));
                int m = s[2*j] >> 4;
                m |= m << 8; m |= m << 16;
                int zm = __builtin_amdgcn_sudot4(true, m, true, u, 0, false);
                sm += scales[qb] * zm;
            }
            float d = (float)*(const _Float16 *)(p + 80);
            float dm = (float)*(const _Float16 *)(p + 82);
            parts[vw] += d * sd - dm * sm;
        }
    }
    float sum = parts[0];
#pragma unroll
    for (int vw = 1; vw < 8; ++vw) sum += parts[vw];
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}

template<int MAX_COUNT>
__device__ __forceinline__ void qwen35_matvec_q2k_multi(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    int lane=threadIdx.x&31, row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (row>=rows) return;
    int blocks=cols/256, iqs=lane&15;
    const unsigned char *w=weights+(size_t)row*blocks*84;
    float sum[MAX_COUNT]={};
#pragma unroll
    for (int vw=0; vw<8; ++vw) {
        float partial[MAX_COUNT]={};
        for (int b=lane/16+2*vw; b<blocks; b+=16) {
            const unsigned char *p=w+b*84;
            const unsigned char *s=p+(iqs&~7)+(iqs&7)/4;
            int v=((const int *)(p+16))[iqs];
            float sd[MAX_COUNT]={}, sm[MAX_COUNT]={};
#pragma unroll
            for (int j=0; j<4; ++j) {
                int qb=b*8+4*(iqs/8)+j;
                int low=s[2*j]&15, high=(s[2*j]>>4)*0x01010101;
#pragma unroll
                for (int t=0; t<MAX_COUNT; ++t) if (t<count) {
                    int u=((const int *)(q+(size_t)t*cols+qb*32))[iqs&7];
                    int z=__builtin_amdgcn_sudot4(true,(v>>(2*j))&0x03030303,true,u,0,false);
                    int zm=__builtin_amdgcn_sudot4(true,high,true,u,0,false);
                    float scale=scales[(size_t)t*(cols/32)+qb];
                    sd[t]+=scale*(z*low); sm[t]+=scale*zm;
                }
            }
            float d=(float)*(const _Float16 *)(p+80), dm=(float)*(const _Float16 *)(p+82);
#pragma unroll
            for (int t=0; t<MAX_COUNT; ++t) if (t<count) partial[t]+=d*sd[t]-dm*sm[t];
        }
#pragma unroll
        for (int t=0; t<MAX_COUNT; ++t) if (t<count) sum[t]+=partial[t];
    }
#pragma unroll
    for (int t=0; t<MAX_COUNT; ++t) if (t<count) {
        for (int offset=16; offset; offset>>=1) sum[t]+=__shfl_xor(sum[t],offset,32);
        if (lane==0) out[(size_t)t*rows+row]=sum[t];
    }
}

extern "C" __global__ void qwen35_matvec_q2k_multi4(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    qwen35_matvec_q2k_multi<4>(out,weights,q,scales,rows,cols,count);
}

extern "C" __global__ void qwen35_matvec_q2k_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    qwen35_matvec_q2k_multi<8>(out,weights,q,scales,rows,cols,count);
}

/* The K=7 verifier evaluates exactly eight rows.  Keep the eight reference
 * virtual-warp partials, but finish one query's four dot products before
 * starting the next query.  This avoids keeping sixteen sd/sm accumulators
 * live across the inner loop while still decoding each weight block once. */
extern "C" __global__ void qwen35_matvec_q2k_fixed8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    (void)count;
    int lane=threadIdx.x&31, row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (row>=rows) return;
    int blocks=cols/256, iqs=lane&15;
    const unsigned char *w=weights+(size_t)row*blocks*84;
    float sum[8]={};
#pragma unroll
    for (int vw=0; vw<8; ++vw) {
        float partial[8]={};
        for (int b=lane/16+2*vw; b<blocks; b+=16) {
            const unsigned char *p=w+b*84;
            const unsigned char *s=p+(iqs&~7)+(iqs&7)/4;
            int v=((const int *)(p+16))[iqs];
            float d=(float)*(const _Float16 *)(p+80);
            float dm=(float)*(const _Float16 *)(p+82);
#pragma unroll
            for (int t=0; t<8; ++t) {
                float sd=0.0f, sm=0.0f;
#pragma unroll
                for (int j=0; j<4; ++j) {
                    int qb=b*8+4*(iqs/8)+j;
                    int u=((const int *)(q+(size_t)t*cols+qb*32))[iqs&7];
                    int low=s[2*j]&15;
                    int high=(s[2*j]>>4)*0x01010101;
                    int z=__builtin_amdgcn_sudot4(true,
                        (v>>(2*j))&0x03030303,true,u,0,false);
                    int zm=__builtin_amdgcn_sudot4(true,high,true,u,0,false);
                    float scale=scales[(size_t)t*(cols/32)+qb];
                    sd+=scale*(z*low);
                    sm+=scale*zm;
                }
                partial[t]+=d*sd-dm*sm;
            }
        }
#pragma unroll
        for (int t=0; t<8; ++t) sum[t]+=partial[t];
    }
#pragma unroll
    for (int t=0; t<8; ++t) {
        for (int offset=16; offset; offset>>=1)
            sum[t]+=__shfl_xor(sum[t],offset,32);
        if (lane==0) out[(size_t)t*rows+row]=sum[t];
    }
}
