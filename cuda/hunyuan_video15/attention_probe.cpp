/* Exercise both heads and partial Q/K tiles against a double-accumulated reference. */
#include <cmath>
#include <cstdio>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "cuda_attention.h"

int main() {
    constexpr int d=128, nq=8193, nk=129, heads=2;
    ggml_backend_load_all();
    auto backend=ggml_backend_init_by_name("CUDA0",nullptr);
    if (!backend || !hv15_cuda_attention_install(backend)) return 1;
    auto ctx=ggml_init({16*1024*1024,nullptr,true});
    auto q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,d,nq,heads,1);
    auto k=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,d,nk,heads,1);
    auto v=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,d,nk,heads,1);
    const float scale=1.f/std::sqrt(static_cast<float>(d));
    auto out=ggml_flash_attn_ext(ctx,q,k,v,nullptr,scale,0.f,0.f);
    auto graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if (!buffer) return 1;
    std::vector<float> qdata(d*nq*heads), result(d*nq*heads);
    std::vector<ggml_fp16_t> kdata(d*nk*heads),vdata(d*nk*heads);
    for (size_t i=0;i<qdata.size();++i) qdata[i]=.25f*std::sin(i*.017);
    for (size_t i=0;i<kdata.size();++i) {
        kdata[i]=ggml_fp32_to_fp16(.25f*std::cos(i*.013));
        vdata[i]=ggml_fp32_to_fp16(std::sin(i*.023));
    }
    ggml_backend_tensor_set(q,qdata.data(),0,qdata.size()*sizeof(float));
    ggml_backend_tensor_set(k,kdata.data(),0,kdata.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v,vdata.data(),0,vdata.size()*sizeof(ggml_fp16_t));
    if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 1;
    ggml_backend_tensor_get(out,result.data(),0,result.size()*sizeof(float));
    double squared_error=0, squared_reference=0, maximum_error=0;
    std::vector<double> probabilities(nk), reference(d);
    for (int h=0;h<heads;++h) for (int row=0;row<nq;++row) {
        double sum=0;
        for (int j=0;j<nk;++j) {
            double dot=0;
            for (int c=0;c<d;++c) dot+=qdata[(h*nq+row)*d+c]*
                static_cast<double>(ggml_fp16_to_fp32(kdata[(h*nk+j)*d+c]));
            probabilities[j]=std::exp(dot*scale); sum+=probabilities[j];
        }
        std::fill(reference.begin(),reference.end(),0.);
        for (int j=0;j<nk;++j) for (int c=0;c<d;++c)
            reference[c]+=probabilities[j]/sum*ggml_fp16_to_fp32(vdata[(h*nk+j)*d+c]);
        for (int c=0;c<d;++c) {
            const float actual=result[(row*heads+h)*d+c];
            if (!std::isfinite(actual)) return 1;
            const double error=actual-reference[c];
            squared_error+=error*error; squared_reference+=reference[c]*reference[c];
            maximum_error=std::fmax(maximum_error,std::fabs(error));
        }
    }
    const double relative=std::sqrt(squared_error/squared_reference);
    const auto count=hv15_cuda_attention_calls();
    std::printf("precise_attention calls=%llu relative_l2=%.9g max_abs=%.9g\n",
                static_cast<unsigned long long>(count),relative,maximum_error);
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
    return count==1 && relative<=.002 && maximum_error<=.001 ? 0 : 1;
}
