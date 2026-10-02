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
    replace("int M, int N, int K)", "int M, int N, int K, int T, int H, int Wd, int C, int KT, int KH, int KW, int replicate)");
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
__device__ __forceinline__ long long conv_offset(int r, int k, int T,int H,int W,int C,int KT,int KH,int KW,int replicate) {
    int channel=k%C, point=k/C;
    int dx=point%KW-KW/2,dy=(point/KW)%KH-KH/2,dt=point/(KW*KH)-(replicate?KT-1:0);
    int x=r%W+dx,y=(r/W)%H+dy,t=r/(W*H)+dt;
    if(replicate){x=max(0,min(W-1,x));y=max(0,min(H-1,y));t=max(0,min(T-1,t));}
    if(x<0||x>=W||y<0||y>=H||t<0||t>=T)return -1;
    return ((long long)t*H*W+y*W+x)*C+channel;
}
)CUDA" + source;
}
}
#endif
