// Minimal 16x16x16 WMMA: C[16x16] f32 = A[16x16] x B[16x16], wave32, gfx12.
// Lane l: half=l/16 id=l%16. A[i]=A[id][half*8+i], B[i]=B[half*8+i][id], C[i]=C[half*8+i][id].
#define WMMA_KSRC \
"typedef unsigned int u32;\n" \
"typedef float f8 __attribute__((ext_vector_type(8)));\n" \
"#ifdef T_BF16\n typedef unsigned short wt; typedef wt v8 __attribute__((ext_vector_type(8)));\n #define MMA(a,b,c) c=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c)\n" \
"#else\n typedef _Float16 wt; typedef wt v8 __attribute__((ext_vector_type(8)));\n #define MMA(a,b,c) c=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c)\n #endif\n" \
"extern \"C\" __global__ void mm(float*C,const wt*A,const wt*B){\n" \
"  int l=threadIdx.x,half=l>>4,id=l&15; v8 a,b; f8 c;\n" \
"  for(int i=0;i<8;i++){a[i]=A[id*16+half*8+i]; b[i]=B[(half*8+i)*16+id]; c[i]=0;}\n" \
"  MMA(a,b,c);\n" \
"  for(int i=0;i<8;i++) C[(half*8+i)*16+id]=c[i];\n}\n"
