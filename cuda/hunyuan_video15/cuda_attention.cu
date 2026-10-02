/* Repo-owned long-sequence attention. Keep the value accumulator in FP32.
 * The pinned ggml NVIDIA MMA implementation accumulates V*P in half2.
 * Hook only unmasked, 128-channel, batch-one CUDA attention with >=8192 queries.
 * Other operators continue through the original backend without alteration.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include "cuda_attention.h"

namespace {
constexpr int K_TILE=128, HEAD=128, THREADS=256;
std::atomic<uint64_t> calls{0};
using graph_fn = ggml_status (*)(ggml_backend_t, ggml_cgraph *);
using free_fn = void (*)(ggml_backend_t);
struct Hook { graph_fn graph; free_fn free; };
std::mutex hooks_mutex;
std::unordered_map<ggml_backend_t,Hook> hooks;

__global__ void pack_queries(const float *q, half *packed, float *acc,
        float *maxima, float *sums, int nq, int heads, size_t q1, size_t q2) {
    const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=static_cast<size_t>(nq)*heads*HEAD) return;
    const size_t row=i/HEAD, channel=i%HEAD;
    packed[i]=__float2half(q[(row/nq)*q2+(row%nq)*q1+channel]);
    acc[i]=0.f;
    if (channel==0) { maxima[row]=-CUDART_INF_F; sums[row]=0.f; }
}
__global__ void softmax_tile(const float *scores, half *p, float *acc,
        float *maxima, float *sums, int rows, int keys, float scale) {
    const int lane=threadIdx.x%32;
    const int row=blockIdx.x*(THREADS/32)+threadIdx.x/32;
    if (row>=rows) return;
    float maximum=-CUDART_INF_F;
    for (int j=lane;j<keys;j+=32) maximum=fmaxf(maximum,scores[row*K_TILE+j]*scale);
    for (int offset=16;offset;offset/=2)
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
    maximum=__shfl_sync(0xffffffff,maximum,0);
    const float next=fmaxf(maxima[row],maximum);
    const float rescale=isfinite(maxima[row]) ? expf(maxima[row]-next) : 0.f;
    for (int c=lane;c<HEAD;c+=32) acc[row*HEAD+c]*=rescale;
    float sum=0.f;
    for (int j=lane;j<keys;j+=32) {
        const float value=expf(scores[row*K_TILE+j]*scale-next);
        p[row*K_TILE+j]=__float2half(value); sum+=value;
    }
    for (int offset=16;offset;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if (lane==0) { maxima[row]=next; sums[row]=sums[row]*rescale+sum; }
}
__global__ void finish_attention(const float *acc, const float *sums, float *out,
        int nq, int heads, size_t o1, size_t o2) {
    const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=static_cast<size_t>(nq)*heads*HEAD) return;
    const size_t row=i/HEAD, channel=i%HEAD;
    out[(row/nq)*o1+(row%nq)*o2+channel]=acc[i]/sums[row];
}
struct Resources {
    void *memory=nullptr;
    cublasHandle_t blas=nullptr;
    ~Resources() { if (blas) cublasDestroy(blas); if (memory) cudaFree(memory); }
};
ggml_status attention(ggml_tensor *node) {
    auto q=node->src[0], k=node->src[1], v=node->src[2];
    const int nq=q->ne[1], nk=k->ne[1], heads=q->ne[2], rows=nq*heads;
    const size_t elements=static_cast<size_t>(rows)*HEAD;
    const size_t tile_elements=static_cast<size_t>(rows)*K_TILE;
    // Workspace is linear in query count, not the square of sequence length.
    Resources resources;
    auto error=cudaMalloc(&resources.memory,elements*6+tile_elements*6+static_cast<size_t>(rows)*8);
    if (error!=cudaSuccess) return GGML_STATUS_ALLOC_FAILED;
    half *packed=static_cast<half *>(resources.memory);
    half *p=packed+elements;
    float *scores=reinterpret_cast<float *>(p+tile_elements);
    float *acc=scores+tile_elements;
    float *maxima=acc+elements;
    float *sums=maxima+rows;
    auto status=cublasCreate(&resources.blas);
    if (status!=CUBLAS_STATUS_SUCCESS) return GGML_STATUS_FAILED;
    status=cublasSetMathMode(resources.blas,static_cast<cublasMath_t>(
        CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    if (status!=CUBLAS_STATUS_SUCCESS) return GGML_STATUS_FAILED;
    pack_queries<<<(elements+THREADS-1)/THREADS,THREADS>>>(static_cast<const float *>(q->data),
        packed,acc,maxima,sums,nq,heads,q->nb[1]/4,q->nb[2]/4);
    const float one=1.f, zero=0.f;
    float scale=0;std::memcpy(&scale,node->op_params,sizeof(float));
    for (int start=0;start<nk;start+=K_TILE) {
        const int keys=std::min(K_TILE,nk-start);
        status=cublasGemmStridedBatchedEx(resources.blas,CUBLAS_OP_T,CUBLAS_OP_N,
            keys,nq,HEAD,&one,static_cast<const half *>(k->data)+start*(k->nb[1]/2),CUDA_R_16F,k->nb[1]/2,k->nb[2]/2,
            packed,CUDA_R_16F,HEAD,static_cast<long long>(nq)*HEAD,&zero,
            scores,CUDA_R_32F,K_TILE,static_cast<long long>(nq)*K_TILE,heads,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status!=CUBLAS_STATUS_SUCCESS) break;
        softmax_tile<<<(rows+7)/8,THREADS>>>(scores,p,acc,maxima,sums,rows,keys,scale);
        status=cublasGemmStridedBatchedEx(resources.blas,CUBLAS_OP_N,CUBLAS_OP_N,
            HEAD,nq,keys,&one,static_cast<const half *>(v->data)+start*(v->nb[1]/2),CUDA_R_16F,v->nb[1]/2,v->nb[2]/2,
            p,CUDA_R_16F,K_TILE,static_cast<long long>(nq)*K_TILE,&one,
            acc,CUDA_R_32F,HEAD,static_cast<long long>(nq)*HEAD,heads,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status!=CUBLAS_STATUS_SUCCESS) break;
    }
    if (status!=CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,"HV15 attention cuBLAS error: %d\n",static_cast<int>(status));
        return GGML_STATUS_FAILED;
    }
    finish_attention<<<(elements+THREADS-1)/THREADS,THREADS>>>(acc,sums,static_cast<float *>(node->data),
        nq,heads,node->nb[1]/4,node->nb[2]/4);
    error=cudaGetLastError();
    if (error==cudaSuccess) error=cudaDeviceSynchronize();
    if (error!=cudaSuccess) {
        std::fprintf(stderr,"HV15 attention CUDA error: %s\n",cudaGetErrorString(error));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

bool supported(const ggml_tensor *node) {
    if (node->op!=GGML_OP_FLASH_ATTN_EXT || node->src[3]!=nullptr || node->src[4]!=nullptr) return false;
    auto q=node->src[0], k=node->src[1], v=node->src[2];
    float max_bias=0,softcap=0;
    std::memcpy(&max_bias,node->op_params+1,sizeof(float));
    std::memcpy(&softcap,node->op_params+2,sizeof(float));
    return q->type==GGML_TYPE_F32 && k->type==GGML_TYPE_F16 && v->type==GGML_TYPE_F16
        && node->type==GGML_TYPE_F32 && q->ne[0]==HEAD && k->ne[0]==HEAD && v->ne[0]==HEAD
        && q->nb[0]==sizeof(float) && k->nb[0]==sizeof(half) && v->nb[0]==sizeof(half)
        && node->nb[0]==sizeof(float)
        && q->ne[1]>=8192 && q->ne[2]==k->ne[2] && k->ne[2]==v->ne[2]
        && k->ne[1]==v->ne[1] && q->ne[3]==1 && k->ne[3]==1 && v->ne[3]==1
        && max_bias==0.f && softcap==0.f;
}
Hook hook_for(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(hooks_mutex);
    return hooks.at(backend);
}
ggml_status compute(ggml_backend_t backend,ggml_cgraph *graph) {
    const auto hook=hook_for(backend);
    int begin=0;
    for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto node=ggml_graph_node(graph,i);
        if (!supported(node)) continue;
        if (begin<i) {
            auto prefix=ggml_graph_view(graph,begin,i);
            auto status=hook.graph(backend,&prefix);
            if (status!=GGML_STATUS_SUCCESS) return status;
        }
        ggml_backend_synchronize(backend);
        const auto status=attention(node);
        if (status!=GGML_STATUS_SUCCESS) return status;
        ++calls;begin=i+1;
    }
    if (begin<ggml_graph_n_nodes(graph)) {
        auto tail=ggml_graph_view(graph,begin,ggml_graph_n_nodes(graph));
        return hook.graph(backend,&tail);
    }
    return GGML_STATUS_SUCCESS;
}
void free_backend(ggml_backend_t backend) {
    Hook hook;
    { std::lock_guard<std::mutex> lock(hooks_mutex);hook=hooks.at(backend);hooks.erase(backend); }
    backend->iface.graph_compute=hook.graph;backend->iface.free=hook.free;
    hook.free(backend);
}
}
extern "C" SD_API bool hv15_cuda_attention_install(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    std::lock_guard<std::mutex> lock(hooks_mutex);
    if (hooks.count(backend)) return true;
    hooks.emplace(backend,Hook{backend->iface.graph_compute,backend->iface.free});
    backend->iface.graph_compute=compute;backend->iface.free=free_backend;
    std::fprintf(stderr,"HV15 long-sequence CUDA attention uses FP32 value accumulation\n");
    return true;
}
extern "C" SD_API uint64_t hv15_cuda_attention_calls(void) {return calls.load();}
