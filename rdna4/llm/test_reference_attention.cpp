/* Differential test against the actual pinned llama.cpp Q8/Q8 HIP kernels. */
#include "fattn-vec.cuh"
#include "qwen35_attention_q8.hip"
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>

#define CHECK(x) do { auto e=(x); if(e!=hipSuccess) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 2; } } while(0)
template<class T> static T *upload(const std::vector<T> &v) {
    T *p = nullptr;
    if (hipMalloc(&p, v.size()*sizeof(T)) != hipSuccess ||
        hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice) != hipSuccess) return nullptr;
    return p;
}

int main() {
    constexpr int heads=24, kv_heads=4, dim=256;
    std::mt19937 rng(7319);
    size_t checked=0;
    int ref_occupancy=0, ours_occupancy=0;
    int batch_occupancy=0;
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&ref_occupancy,
        flash_attn_ext_vec<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,false>,128,0));
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&ours_occupancy,qwen35_attention_q8_decode,128,0));
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&batch_occupancy,
        flash_attn_ext_vec<256,2,GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,false>,128,0));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props,0));
    printf("nsm=%d reference_occupancy=%d ours_occupancy=%d\n",props.multiProcessorCount,ref_occupancy,ours_occupancy);
    printf("reference_batch_occupancy=%d\n",batch_occupancy);
    for(int queries : {1,2,7,8,512})
    for(int pattern : {0,1,2})
    for(int length : {1,31,127,128,129,255,256,257,511,512,513,4096,4097,8192,65536}) {
        if(queries>1 && (pattern!=0 ||
           (length!=512 && length!=4097 && !(queries==8 && length==65536)))) continue;
        if(length==65536 && !((queries==1 && pattern==2) ||
                              (queries==8 && pattern==0))) continue;
        const int output_size=heads*dim*queries;
        int occupancy=queries==1?ref_occupancy:batch_occupancy;
        const int padded=(length+255)/256*256;
        std::vector<float> q(output_size),ks(padded*kv_heads*8),vs(ks.size());
        std::vector<signed char> k(padded*kv_heads*dim),v(k.size());
        std::vector<block_q8_0> rk(ks.size()),rv(ks.size());
        std::vector<half> mask(padded*queries);
        for(auto &x:q) x=(int(rng()%20000)-10000)*.001f;
        if(pattern==1) std::fill(q.begin(),q.end(),0.0f);
        for(size_t i=0;i<ks.size();++i) {
            ks[i]=__half2float(__float2half((rng()%1000+1)*.0001f));
            vs[i]=__half2float(__float2half((rng()%1000+1)*.0001f));
            rk[i].d=__float2half(ks[i]);rv[i].d=__float2half(vs[i]);
            for(int j=0;j<32;++j) {
                k[i*32+j]=rk[i].qs[j]=pattern==2 ? 0 : int(rng()%255)-127;
                v[i*32+j]=rv[i].qs[j]=pattern==2 ? 0 : int(rng()%255)-127;
            }
        }
        for(int row=0;row<queries;++row)
            for(int i=0;i<padded;++i) mask[row*padded+i]=__float2half(i<length-queries+row+1?0.0f:-INFINITY);
        float *dq=upload(q),*dks=upload(ks),*dvs=upload(vs);
        auto *dk=upload(k),*dv=upload(v);
        auto *drk=upload(rk),*drv=upload(rv);
        auto *dm=upload(mask);
        int *dp=upload(std::vector<int>{length-1});
        float *ours=nullptr,*ref=nullptr,*op=nullptr,*rp=nullptr;
        float2 *om=nullptr,*rm=nullptr;
        CHECK(hipMalloc(&ours,output_size*4));CHECK(hipMalloc(&ref,output_size*4));
        constexpr int max_splits=256;
        CHECK(hipMalloc(&op,output_size*max_splits*4));CHECK(hipMalloc(&rp,output_size*max_splits*4));
        CHECK(hipMalloc(&om,queries*heads*max_splits*8));CHECK(hipMalloc(&rm,queries*heads*max_splits*8));
        for(int requested : {0,1,2,4,8,12,14,16,32,64,128,256}) {
            if(length==65536 && requested!=8 && requested!=12 &&
               requested!=16 && requested!=32 && requested!=64 &&
               requested!=128 && requested!=256) continue;
            if(length!=65536 && requested>32) continue;
            int splits=requested;
            if(!splits) {
                int tiles=padded/256, best=0, waves_best=0, tiles_dst=heads*((queries+1)/2);
                splits=std::min(occupancy,tiles);
                for(int trial=splits;trial<=tiles;++trial) {
                    int waves=(tiles_dst*trial+props.multiProcessorCount*occupancy-1)/(props.multiProcessorCount*occupancy);
                    int efficiency=100*tiles_dst*trial/(waves*props.multiProcessorCount*occupancy);
                    if(best>=95 && waves>waves_best) break;
                    if(efficiency>best) { best=efficiency;waves_best=waves;splits=trial; }
                }
            }
            if(splits>(padded+127)/128) continue;
            auto run=[&](bool reference, bool gqa3=false, bool reuse8=false,
                         bool gqa3_reuse=false, bool adaptive=false) {
                if(reference) {
                    if(queries==1) flash_attn_ext_vec<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,false><<<dim3(1,splits,heads),dim3(32,4)>>>(
                        (char *)dq,(char *)drk,(char *)drv,(char *)dm,nullptr,nullptr,splits==1?ref:rp,rm,
                        .0625f,0,0,0,0,0,256,init_fastdiv_values(1),heads,1,heads*1024,1024,heads*1024,
                        256,padded,kv_heads,1,kv_heads*272,272,padded*kv_heads*272,
                        kv_heads*272,272,padded*kv_heads*272,1,1,1,padded*2,padded*2,padded*2);
                    else flash_attn_ext_vec<256,2,GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,false><<<dim3((queries+1)/2,splits,heads),dim3(32,4)>>>(
                        (char *)dq,(char *)drk,(char *)drv,(char *)dm,nullptr,nullptr,splits==1?ref:rp,rm,
                        .0625f,0,0,0,0,0,256,init_fastdiv_values(queries),heads,1,heads*1024,1024,queries*heads*1024,
                        256,padded,kv_heads,1,kv_heads*272,272,padded*kv_heads*272,
                        kv_heads*272,272,padded*kv_heads*272,queries,1,1,padded*2,queries*padded*2,queries*padded*2);
                    if(splits>1) flash_attn_combine_results<256><<<dim3(queries,heads),256,splits*8>>>(rp,rm,ref,splits);
                } else {
                    int force=queries==1?requested:splits;
                    dim3 grid = queries==1 ?
                        dim3((gqa3||gqa3_reuse||adaptive?2*kv_heads:heads)*
                             (force?splits:128),1,1) :
                        dim3(reuse8?1:queries,force?splits:128,heads);
                    if(adaptive) {
                        qwen35_attention_q8_decode_gqa3<<<grid,dim3(32,12)>>>(
                            ours,op,om,dq,dk,dv,dks,dvs,dp,heads,kv_heads,
                            props.multiProcessorCount,occupancy,force,queries,-1,1);
                        qwen35_attention_q8_decode_gqa3_reuse<<<grid,dim3(32,4)>>>(
                            ours,op,om,dq,dk,dv,dks,dvs,dp,heads,kv_heads,
                            props.multiProcessorCount,occupancy,force,queries,-1,1);
                    } else if(gqa3_reuse) qwen35_attention_q8_decode_gqa3_reuse<<<grid,dim3(32,4)>>>(
                            ours,op,om,dq,dk,dv,dks,dvs,dp,heads,kv_heads,
                            props.multiProcessorCount,occupancy,force,queries,-1,0);
                    else if(gqa3) qwen35_attention_q8_decode_gqa3<<<grid,dim3(32,12)>>>(
                            ours,op,om,dq,dk,dv,dks,dvs,dp,heads,kv_heads,
                            props.multiProcessorCount,occupancy,force,queries,-1,0);
                    else if(reuse8) qwen35_attention_q8_decode_reuse8<<<grid,dim3(32,4)>>>(
                            ours,op,om,dq,dk,dv,dks,dvs,dp,heads,kv_heads,
                            props.multiProcessorCount,occupancy,force,
                            queries,queries==1?-1:length-queries);
                    else qwen35_attention_q8_decode<<<grid,dim3(32,4)>>>(
                            ours,op,om,dq,dk,dv,dks,dvs,dp,heads,kv_heads,
                            props.multiProcessorCount,occupancy,force,
                            queries,queries==1?-1:length-queries);
                    qwen35_attention_q8_combine<<<dim3(heads,queries),256,max_splits*8>>>(ours,op,om,dp,heads,props.multiProcessorCount,occupancy,force);
                }
            };
            run(true);run(false);
            CHECK(hipDeviceSynchronize());
            std::vector<float> a(output_size),b(output_size);
            CHECK(hipMemcpy(a.data(),ours,output_size*4,hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(b.data(),ref,output_size*4,hipMemcpyDeviceToHost));
            size_t wrong=0;float worst=0;
            for(int i=0;i<output_size;++i) {
                if(memcmp(&a[i],&b[i],4) || !std::isfinite(a[i])) {
                    if(wrong++<3) fprintf(stderr,"length=%d splits=%d i=%d ours=%.9g ref=%.9g\n",length,splits,i,a[i],b[i]);
                    worst=fmaxf(worst,fabsf(a[i]-b[i]));
                }
                ++checked;
            }
            if(wrong) { fprintf(stderr,"mismatches=%zu/%d max=%.9g\n",wrong,output_size,worst);return 1; }
            if(queries>1 && queries<=8) {
                run(false,false,true);CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(a.data(),ours,output_size*4,hipMemcpyDeviceToHost));
                wrong=0;worst=0;
                for(int i=0;i<output_size;++i) {
                    if(memcmp(&a[i],&b[i],4) || !std::isfinite(a[i])) {
                        if(wrong++<3) fprintf(stderr,
                            "reuse8 length=%d splits=%d i=%d ours=%.9g ref=%.9g\n",
                            length,splits,i,a[i],b[i]);
                        worst=fmaxf(worst,fabsf(a[i]-b[i]));
                    }
                    ++checked;
                }
                if(wrong) { fprintf(stderr,"reuse8 mismatches=%zu/%d max=%.9g\n",wrong,output_size,worst);return 1; }
            }
            if(queries==1) {
                run(false,true);CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(a.data(),ours,output_size*4,hipMemcpyDeviceToHost));
                wrong=0;worst=0;
                for(int i=0;i<output_size;++i) {
                    if(memcmp(&a[i],&b[i],4) || !std::isfinite(a[i])) {
                        if(wrong++<3) fprintf(stderr,
                            "gqa3 length=%d splits=%d i=%d ours=%.9g ref=%.9g\n",
                            length,splits,i,a[i],b[i]);
                        worst=fmaxf(worst,fabsf(a[i]-b[i]));
                    }
                    ++checked;
                }
                if(wrong) { fprintf(stderr,"gqa3 mismatches=%zu/%d max=%.9g\n",wrong,output_size,worst);return 1; }
                run(false,false,false,true);CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(a.data(),ours,output_size*4,hipMemcpyDeviceToHost));
                wrong=0;worst=0;
                for(int i=0;i<output_size;++i) {
                    if(memcmp(&a[i],&b[i],4) || !std::isfinite(a[i])) {
                        if(wrong++<3) fprintf(stderr,
                            "gqa3_reuse length=%d splits=%d i=%d ours=%.9g ref=%.9g\n",
                            length,splits,i,a[i],b[i]);
                        worst=fmaxf(worst,fabsf(a[i]-b[i]));
                    }
                    ++checked;
                }
                if(wrong) { fprintf(stderr,"gqa3_reuse mismatches=%zu/%d max=%.9g\n",wrong,output_size,worst);return 1; }
                run(false,false,false,false,true);CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(a.data(),ours,output_size*4,hipMemcpyDeviceToHost));
                wrong=0;worst=0;
                for(int i=0;i<output_size;++i) {
                    if(memcmp(&a[i],&b[i],4) || !std::isfinite(a[i])) {
                        if(wrong++<3) fprintf(stderr,
                            "adaptive length=%d splits=%d i=%d ours=%.9g ref=%.9g\n",
                            length,splits,i,a[i],b[i]);
                        worst=fmaxf(worst,fabsf(a[i]-b[i]));
                    }
                    ++checked;
                }
                if(wrong) { fprintf(stderr,"adaptive mismatches=%zu/%d max=%.9g\n",wrong,output_size,worst);return 1; }
            }
            if(queries==512 && length==4097 && pattern==0 && requested==0) {
                qwen35_attention_q8_prefill_wmma<<<dim3(heads,(queries+127)/128),512>>>(
                    ours,dq,dk,dv,dks,dvs,heads,kv_heads,queries,length-queries);
                CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(a.data(),ours,output_size*4,hipMemcpyDeviceToHost));
                double error2=0.0, reference2=0.0;float approximate_worst=0.0f;
                for(int i=0;i<output_size;++i) {
                    if(!std::isfinite(a[i])) {
                        fprintf(stderr,"WMMA prefill produced non-finite output at %d\n",i);
                        return 1;
                    }
                    double error=(double)a[i]-b[i];
                    error2+=error*error;reference2+=(double)b[i]*b[i];
                    approximate_worst=fmaxf(approximate_worst,fabsf(a[i]-b[i]));
                }
                double relative_l2=sqrt(error2/reference2);
                printf("WMMA prefill queries=512 length=4097 rel_l2=%.9g max=%.9g\n",
                       relative_l2,approximate_worst);
                if(relative_l2>0.01 || approximate_worst>0.1f) {
                    fprintf(stderr,"WMMA prefill exceeded numerical tolerance\n");
                    return 1;
                }
            }
            if(length==65536) {
                int modes = queries == 1 ? 3 : 2;
                for(int mode=0;mode<modes;++mode) {
                    hipEvent_t start,stop;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&stop));
                    CHECK(hipEventRecord(start));
                    for(int i=0;i<20;++i)
                        run(false,queries==1 && mode==1,queries>1 && mode==1,
                            queries==1 && mode==2);
                    CHECK(hipEventRecord(stop));CHECK(hipEventSynchronize(stop));
                    float elapsed=0;CHECK(hipEventElapsedTime(&elapsed,start,stop));
                    printf("%s attention length=65536 splits=%d %.3f us\n",
                           mode==0?"ours":queries==1?
                               (mode==1?"gqa3":"gqa3_reuse"):"reuse8",
                           splits,elapsed*50);
                    CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(stop));
                }
            }
            if(pattern==0 && ((queries==1 && length==4096 && requested==16) || (queries==512 && length==4097 && requested==0))) {
                for(bool reference : {true,false}) {
                    hipEvent_t start,stop;
                    CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&stop));
                    CHECK(hipEventRecord(start));
                    for(int i=0;i<200;++i) run(reference);
                    CHECK(hipEventRecord(stop));CHECK(hipEventSynchronize(stop));
                    float elapsed=0;CHECK(hipEventElapsedTime(&elapsed,start,stop));
                    printf("%s attention queries=%d length=%d splits=%d %.3f us\n",reference?"llama":"ours",queries,length,splits,elapsed*5);
                    CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(stop));
                }
            }
            if(pattern==0 && queries==8 && length==4097) {
                for(bool reuse8 : {false,true}) {
                    hipEvent_t start,stop;
                    CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&stop));
                    CHECK(hipEventRecord(start));
                    for(int i=0;i<100;++i) run(false,false,reuse8);
                    CHECK(hipEventRecord(stop));CHECK(hipEventSynchronize(stop));
                    float elapsed=0;CHECK(hipEventElapsedTime(&elapsed,start,stop));
                    printf("%s attention queries=8 length=4097 splits=%d %.3f us\n",
                           reuse8?"reuse8":"ours",splits,elapsed*10);
                    CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(stop));
                }
            }
        }
        for(void *p : {static_cast<void *>(dq),static_cast<void *>(dks),static_cast<void *>(dvs),
            static_cast<void *>(dk),static_cast<void *>(dv),static_cast<void *>(drk),static_cast<void *>(drv),
            static_cast<void *>(dm),static_cast<void *>(dp),static_cast<void *>(ours),static_cast<void *>(ref),
            static_cast<void *>(op),static_cast<void *>(rp),static_cast<void *>(om),static_cast<void *>(rm)}) CHECK(hipFree(p));
    }
    printf("PASS: %zu bitwise Q8/Q8 attention comparisons\n",checked);
}
