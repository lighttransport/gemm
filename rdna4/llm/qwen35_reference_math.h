/* RMSNorm reduction/operation order from llama.cpp 1859b5209 norm.cu.
 * Copyright (c) 2023-2026 The ggml authors, MIT license (see
 * reference_sampler.cpp for the complete retained license).
 * Compile this separate module without fast-math. */
static const char *qwen35_reference_math_source =
"extern \"C\" __global__ void qwen35_rmsnorm_reference(float *dst, const float *x,\n"
"    const float *weight, int n, int stride, float eps) {\n"
"    int tid=threadIdx.x,lane=tid&31,warp=tid>>5;\n"
"    x+=(size_t)blockIdx.x*stride;dst+=(size_t)blockIdx.x*stride;\n"
"    float sum=0.0f;\n"
"    for(int col=tid;col<n;col+=blockDim.x){float v=x[col];sum+=v*v;}\n"
"    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_xor(sum,offset,32);\n"
"    __shared__ float partial[32];\n"
"    if(lane==0)partial[warp]=sum;__syncthreads();\n"
"    sum=lane<(int)blockDim.x/32?partial[lane]:0.0f;\n"
"    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_xor(sum,offset,32);\n"
"    float mean=sum/n,scale=rsqrtf(mean+eps);\n"
"    for(int col=tid;col<n;col+=blockDim.x){float v=scale*x[col];dst[col]=weight?v*weight[col]:v;}\n"
"}\n"
"extern \"C\" __global__ void qwen35_conv_reference(float *out,float *state,\n"
"    const float *input,const float *weight,int channels,int width){\n"
"    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=channels)return;\n"
"    float sum=0.0f;\n"
"    for(int j=0;j<width-1;++j)sum+=state[j*channels+c]*weight[c*width+j];\n"
"    sum+=input[c]*weight[c*width+width-1];sum+=0.0f;\n"
"    out[c]=sum/(1.0f+expf(-sum));\n"
"    for(int j=0;j<width-2;++j)state[j*channels+c]=state[(j+1)*channels+c];\n"
"    state[(width-2)*channels+c]=input[c];\n"
"}\n"
"extern \"C\" __global__ void qwen35_silu_gate_reference(float *x,const float *gate,int n){\n"
"    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float g=gate[i];x[i]*=g/(1.0f+expf(-g));}\n"
"}\n";
