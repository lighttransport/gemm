/* Standalone correctness test for the sm_120a NVFP4 block-scale mma.
 *
 * Goal: nail the (undocumented) A/B/C fragment layout and block-scale operand
 * mapping for
 *   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
 *     .f32.e2m1.e2m1.f32.ue4m3
 * by computing one 16x8 output tile (A[16x64] @ B[8x64]^T, e2m1 codes, per
 * group-16 ue4m3 scales) on the tensor core and comparing to a CPU reference.
 *
 * Phase 1 (scales=1): isolate the A/B/C fragment layout.
 * Phase 2 (random scales): solve the {byte-id,thread-id} scale selector.
 *
 * Build: gcc -O2 -I.. -o fp4_gemm_test fp4_gemm_test.c ../cuew.c -ldl -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuew.h"

#define CHECK_CUDA(call) do { CUresult e=(call); if(e!=CUDA_SUCCESS){ \
    const char*s; cuGetErrorString(e,&s); \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,s); exit(1);} } while(0)

static const float E2M1[16] = {0,0.5f,1,1.5f,2,3,4,6, -0.f,-0.5f,-1,-1.5f,-2,-3,-4,-6};

/* ue4m3 (==e4m3 unsigned) decode of a byte: sign ignored (always +). */
static float ue4m3_decode(unsigned char b) {
    int e = (b >> 3) & 0xF;     /* 4-bit exponent, bias 7 */
    int m = b & 0x7;            /* 3-bit mantissa */
    if (e == 0) return ldexpf((float)m / 8.0f, 1 - 7);     /* subnormal */
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}
/* encode 1.0 -> e=7,m=0 -> 0b0111000 = 0x38 */

static const char *KSRC =
"extern \"C\" __global__ void fp4tile(const unsigned int* A, const unsigned int* B,\n"
"        const unsigned int* sfa_arr, const unsigned int* sfb_arr, float* D) {\n"
"  int lane = threadIdx.x & 31;\n"
"  int g = lane >> 2;       /* groupID 0..7 */\n"
"  int t = lane & 3;        /* tid 0..3 */\n"
"  /* A: row-major, 16 uints/row stride (8 used = 64 e2m1, 8 codes/uint). */\n"
"  /* a0:row g, k[8t..8t+7](uint t); a1:row g+8; a2:row g k[8t+32..](uint t+4); a3:row g+8 */\n"
"  unsigned int a0 = A[g*16     + t];\n"
"  unsigned int a1 = A[(g+8)*16 + t];\n"
"  unsigned int a2 = A[g*16     + t + 4];\n"
"  unsigned int a3 = A[(g+8)*16 + t + 4];\n"
"  /* B: row-major [8][16] uint. b0:n g k[8t..](uint t); b1:n g k[8t+32..](uint t+4) */\n"
"  unsigned int b0 = B[g*16 + t];\n"
"  unsigned int b1 = B[g*16 + t + 4];\n"
"  unsigned int sfa = sfa_arr[lane], sfb = sfb_arr[lane];\n"
"  float d0=0.f,d1=0.f,d2=0.f,d3=0.f;\n"
"  asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
"    \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
"    : \"+f\"(d0),\"+f\"(d1),\"+f\"(d2),\"+f\"(d3)\n"
"    : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1),\"r\"(sfa),\"r\"(sfb));\n"
"  /* C layout: d0=(g,t*2) d1=(g,t*2+1) d2=(g+8,t*2) d3=(g+8,t*2+1) */\n"
"  D[g*8 + t*2]       = d0;\n"
"  D[g*8 + t*2 + 1]   = d1;\n"
"  D[(g+8)*8 + t*2]   = d2;\n"
"  D[(g+8)*8 + t*2+1] = d3;\n"
"}\n";

static CUmodule compile_sm120a(const char* src){
    nvrtcProgram p; if(nvrtcCreateProgram(&p,src,"k",0,NULL,NULL)!=NVRTC_SUCCESS) return NULL;
    const char* o[]={"--gpu-architecture=sm_120a"};
    if(nvrtcCompileProgram(p,1,o)!=NVRTC_SUCCESS){
        size_t l=0; nvrtcGetProgramLogSize(p,&l); char*b=malloc(l+1); nvrtcGetProgramLog(p,b); b[l]=0;
        fprintf(stderr,"NVRTC REJECTED:\n%s\n",b); free(b); return NULL; }
    CUmodule m=NULL; size_t bs=0;
    if(nvrtcGetCUBINSize&&nvrtcGetCUBINSize(p,&bs)==NVRTC_SUCCESS&&bs>0){
        char*bl=malloc(bs); nvrtcGetCUBIN(p,bl); nvrtcDestroyProgram(&p);
        if(cuModuleLoadData(&m,bl)!=CUDA_SUCCESS) m=NULL; free(bl);
    } else { size_t ps=0; nvrtcGetPTXSize(p,&ps); char*x=malloc(ps); nvrtcGetPTX(p,x);
        nvrtcDestroyProgram(&p); if(cuModuleLoadData(&m,x)!=CUDA_SUCCESS) m=NULL; free(x); }
    return m;
}

static CUfunction g_fn; static CUdeviceptr g_dA,g_dB,g_dsfa,g_dsfb,g_dD;
static void run(unsigned int*sfa,unsigned int*sfb,float*out){
    CHECK_CUDA(cuMemcpyHtoD(g_dsfa,sfa,32*sizeof(unsigned int)));
    CHECK_CUDA(cuMemcpyHtoD(g_dsfb,sfb,32*sizeof(unsigned int)));
    void* args[]={&g_dA,&g_dB,&g_dsfa,&g_dsfb,&g_dD};
    CHECK_CUDA(cuLaunchKernel(g_fn,1,1,1,32,1,1,0,0,args,NULL));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(out,g_dD,16*8*sizeof(float)));
}

int main(int argc, char** argv){
    if(cuewInit(CUEW_INIT_CUDA|CUEW_INIT_NVRTC)!=CUEW_SUCCESS){fprintf(stderr,"cuewInit fail\n");return 1;}
    CHECK_CUDA(cuInit(0));
    CUdevice dev; CUcontext ctx; CHECK_CUDA(cuDeviceGet(&dev,0)); CHECK_CUDA(cuCtxCreate(&ctx,0,dev));
    CUmodule mod=compile_sm120a(KSRC); if(!mod){fprintf(stderr,"compile failed\n");return 1;}
    CHECK_CUDA(cuModuleGetFunction(&g_fn,mod,"fp4tile"));
    CHECK_CUDA(cuMemAlloc(&g_dA,16*16*sizeof(unsigned int)));
    CHECK_CUDA(cuMemAlloc(&g_dB,8*16*sizeof(unsigned int)));
    CHECK_CUDA(cuMemAlloc(&g_dsfa,32*sizeof(unsigned int)));
    CHECK_CUDA(cuMemAlloc(&g_dsfb,32*sizeof(unsigned int)));
    CHECK_CUDA(cuMemAlloc(&g_dD,16*8*sizeof(float)));

    /* ---- Phase 1 sanity: random codes, all scales=1 -> A@B^T exact ---- */
    srand(1234);
    unsigned char Ac[16][64], Bc[8][64];
    for(int m=0;m<16;m++) for(int k=0;k<64;k++) Ac[m][k]=rand()&0xF;
    for(int n=0;n<8;n++)  for(int k=0;k<64;k++) Bc[n][k]=rand()&0xF;
    unsigned int Au[16*16], Bu[8*16]; memset(Au,0,sizeof(Au)); memset(Bu,0,sizeof(Bu));
    for(int m=0;m<16;m++) for(int u=0;u<8;u++){unsigned v=0;for(int i=0;i<8;i++)v|=((unsigned)Ac[m][u*8+i]&0xF)<<(i*4);Au[m*16+u]=v;}
    for(int n=0;n<8;n++)  for(int u=0;u<8;u++){unsigned v=0;for(int i=0;i<8;i++)v|=((unsigned)Bc[n][u*8+i]&0xF)<<(i*4);Bu[n*16+u]=v;}
    CHECK_CUDA(cuMemcpyHtoD(g_dA,Au,sizeof(Au))); CHECK_CUDA(cuMemcpyHtoD(g_dB,Bu,sizeof(Bu)));
    unsigned int sfa[32],sfb[32]; for(int i=0;i<32;i++){sfa[i]=0x38383838u;sfb[i]=0x38383838u;}
    float got[128];
    run(sfa,sfb,got);
    { double num=0,den=0; for(int m=0;m<16;m++)for(int n=0;n<8;n++){double r=0;
        for(int k=0;k<64;k++)r+=(double)E2M1[Ac[m][k]]*E2M1[Bc[n][k]];
        double d=got[m*8+n]-r; num+=d*d; den+=r*r;}
      printf("Phase1 (scales=1, random codes): rel_L2=%.6f %s\n",sqrt(num/(den+1e-12)),num/den<1e-6?"PASS":"FAIL"); }

    /* ---- Probe: A=B=code2(=1.0), sfb=1.0. For each thread T set sfa[T]=2.0 (0x40), rest 1.0.
       out[m][n] = 16 * sum_gq sA(m,gq); baseline 64. delta/16 = #A-groups of row m fed by thread T,
       each contributing (2-1)=1. So delta=16*c. Reveals row<-thread and group-count. ---- */
    for(int m=0;m<16;m++) for(int u=0;u<8;u++) Au[m*16+u]=0x22222222u; /* code2 everywhere =1.0 */
    for(int n=0;n<8;n++)  for(int u=0;u<8;u++) Bu[n*16+u]=0x22222222u;
    CHECK_CUDA(cuMemcpyHtoD(g_dA,Au,sizeof(Au))); CHECK_CUDA(cuMemcpyHtoD(g_dB,Bu,sizeof(Bu)));
    printf("\n=== A-scale probe (sfa[T]=2.0, others 1.0; delta=out-64; c=delta/16 groups) ===\n");
    for(int T=0;T<32;T++){
        for(int i=0;i<32;i++){sfa[i]=0x38383838u;sfb[i]=0x38383838u;}
        sfa[T]=0x40404040u; /* 2.0 in all 4 bytes */
        run(sfa,sfb,got);
        char buf[256]; int p=0; p+=sprintf(buf,"T%2d(g%d,t%d):",T,T>>2,T&3);
        int any=0;
        for(int m=0;m<16;m++){ float v=got[m*8+0]; if(fabsf(v-64.f)>0.5f){ p+=sprintf(buf+p," r%d=%.0f(c%.0f)",m,v,(v-64.f)/16.f); any=1; } }
        if(any) printf("%s\n",buf);
    }

    /* ---- byte->group probe. B group gq gets distinct value: g0=1,g1=2,g2=4,g3=0.5
       (codes 2,4,6,1). baseline out=16*(1+2+4+0.5)=120. Doubling sfa byte b of the
       row's feeding thread adds 16*v_{group(b)} -> identify group. ---- */
    { int gcode[4]={2,4,6,1}; float gval[4]={1,2,4,0.5f};
      for(int n=0;n<8;n++) for(int k=0;k<64;k++) Bc[n][k]=gcode[k/16];
      for(int n=0;n<8;n++){unsigned v;for(int u=0;u<8;u++){v=0;for(int i=0;i<8;i++)v|=((unsigned)Bc[n][u*8+i]&0xF)<<(i*4);Bu[n*16+u]=v;}}
      CHECK_CUDA(cuMemcpyHtoD(g_dB,Bu,sizeof(Bu)));
      /* A stays all code2 (=1.0). */
      printf("\n=== byte->group probe (row0 via lane0; delta/16 = B-group value of that byte) ===\n");
      printf("    (group values: g0=1 g1=2 g2=4 g3=0.5; so delta/16: 1->g0 2->g1 4->g2 0.5->g3)\n");
      for(int b=0;b<4;b++){
        for(int i=0;i<32;i++){sfa[i]=0x38383838u;sfb[i]=0x38383838u;}
        unsigned int byteval=0x40; /* 2.0 */
        sfa[0]=0x38383838u & ~(0xFFu<<(b*8)); sfa[0]|=(byteval<<(b*8));
        run(sfa,sfb,got);
        float base=120.f, v=got[0*8+0]; printf("  byte%d: out[0][0]=%.1f delta/16=%.3f\n",b,v,(v-base)/16.f);
      }
    }
    /* ---- B-scale thread probe: A=B=1.0, sfa=1.0, sfb[T]=2.0 -> which column n changes ---- */
    for(int m=0;m<16;m++) for(int u=0;u<8;u++) Au[m*16+u]=0x22222222u;
    for(int n=0;n<8;n++)  for(int u=0;u<8;u++) Bu[n*16+u]=0x22222222u;
    CHECK_CUDA(cuMemcpyHtoD(g_dA,Au,sizeof(Au))); CHECK_CUDA(cuMemcpyHtoD(g_dB,Bu,sizeof(Bu)));
    printf("\n=== B-scale thread probe (sfb[T]=2.0; report cols n with out[0][n]!=64) ===\n");
    for(int T=0;T<32;T++){
        for(int i=0;i<32;i++){sfa[i]=0x38383838u;sfb[i]=0x38383838u;}
        sfb[T]=0x40404040u; run(sfa,sfb,got);
        char buf[256]; int p=0; p+=sprintf(buf,"T%2d(g%d,t%d):",T,T>>2,T&3); int any=0;
        for(int n=0;n<8;n++){ float v=got[0*8+n]; if(fabsf(v-64.f)>0.5f){p+=sprintf(buf+p," c%d=%.0f",n,v);any=1;} }
        if(any) printf("%s\n",buf);
    }
    /* ---- B byte->group probe: A groups distinct, scan sfb byte of col0's feeding thread ---- */
    { int gcode[4]={2,4,6,1}; float gval[4]={1,2,4,0.5f}; (void)gval;
      for(int m=0;m<16;m++) for(int k=0;k<64;k++) Ac[m][k]=gcode[k/16];
      for(int m=0;m<16;m++){unsigned v;for(int u=0;u<8;u++){v=0;for(int i=0;i<8;i++)v|=((unsigned)Ac[m][u*8+i]&0xF)<<(i*4);Au[m*16+u]=v;}}
      CHECK_CUDA(cuMemcpyHtoD(g_dA,Au,sizeof(Au)));
      printf("\n=== B byte->group probe (col0; delta/16 = A-group value of that byte) ===\n");
      for(int b=0;b<4;b++){
        for(int i=0;i<32;i++){sfa[i]=0x38383838u;sfb[i]=0x38383838u;}
        sfb[0]=0x38383838u & ~(0xFFu<<(b*8)); sfb[0]|=(0x40u<<(b*8));
        run(sfa,sfb,got);
        printf("  byte%d: out[0][0]=%.1f delta/16=%.3f\n",b,got[0],(got[0]-120.f)/16.f);
      }
    }

    /* ============ FULL correctness: random codes + random ue4m3 scales ============
       sfa[lane]: lane%4==0 -> row lane/4 (0..7); lane%4==1 -> row lane/4+8; else unused.
       sfb[lane]: lane%4==0 -> col lane/4 (0..7); else unused.  byte gq = group gq scale. */
    {
      unsigned char sAb[16][4], sBb[8][4];
      for(int m=0;m<16;m++)for(int q=0;q<4;q++) sAb[m][q]=0x30+(rand()%16); /* random ue4m3 ~[.5,2) */
      for(int n=0;n<8;n++) for(int q=0;q<4;q++) sBb[n][q]=0x30+(rand()%16);
      for(int m=0;m<16;m++)for(int k=0;k<64;k++) Ac[m][k]=rand()&0xF;
      for(int n=0;n<8;n++) for(int k=0;k<64;k++) Bc[n][k]=rand()&0xF;
      for(int m=0;m<16;m++){unsigned v;for(int u=0;u<8;u++){v=0;for(int i=0;i<8;i++)v|=((unsigned)Ac[m][u*8+i]&0xF)<<(i*4);Au[m*16+u]=v;}}
      for(int n=0;n<8;n++){unsigned v;for(int u=0;u<8;u++){v=0;for(int i=0;i<8;i++)v|=((unsigned)Bc[n][u*8+i]&0xF)<<(i*4);Bu[n*16+u]=v;}}
      CHECK_CUDA(cuMemcpyHtoD(g_dA,Au,sizeof(Au))); CHECK_CUDA(cuMemcpyHtoD(g_dB,Bu,sizeof(Bu)));
      for(int i=0;i<32;i++){sfa[i]=0x38383838u;sfb[i]=0x38383838u;}
      for(int L=0;L<32;L++){
        if(L%4==0){int r=L/4;       sfa[L]=sAb[r][0]|(sAb[r][1]<<8)|(sAb[r][2]<<16)|(sAb[r][3]<<24);
                                    sfb[L]=sBb[r][0]|(sBb[r][1]<<8)|(sBb[r][2]<<16)|(sBb[r][3]<<24);}
        if(L%4==1){int r=L/4+8;     sfa[L]=sAb[r][0]|(sAb[r][1]<<8)|(sAb[r][2]<<16)|(sAb[r][3]<<24);}
      }
      run(sfa,sfb,got);
      double num=0,den=0; int nbad=0;
      for(int m=0;m<16;m++)for(int n=0;n<8;n++){ double r=0;
        for(int k=0;k<64;k++) r+=(double)E2M1[Ac[m][k]]*ue4m3_decode(sAb[m][k/16])
                                  *E2M1[Bc[n][k]]*ue4m3_decode(sBb[n][k/16]);
        double d=got[m*8+n]-r; num+=d*d; den+=r*r;
        if(fabs(d)>1e-3*(1+fabs(r))&&nbad<6){printf("  mism[%d,%d] got %.4f ref %.4f\n",m,n,got[m*8+n],r);nbad++;} }
      printf("\n=== FULL W4A4 tile (random codes+scales): rel_L2=%.6f %s ===\n",
             sqrt(num/(den+1e-12)), num/den<1e-5?"PASS":"FAIL");
    }
    return 0;
}
