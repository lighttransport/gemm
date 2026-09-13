/* Real-weight bounded validation of lossless signed-palette caches. */
#include <omp.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"
#include "iq4_decode_cache.h"

int main(int argc,char **argv) {
    if (argc!=2 || svcntb()!=64) return 2;
    gguf_context *g=gguf_open(argv[1],3);
    int fd=open(argv[1],O_RDONLY),bad=0,seen[64]={0};
    if (!g || fd<0) return 2;
    for (uint64_t ti=0;ti<g->n_tensors;ti++) {
        const gguf_tensor_info *t=g->tensors+ti;
        if (!tf_iq4_cache_palette(t->type) || seen[t->type] || t->n_dims!=2 ||
            t->dims[0]%256 || t->dims[1]<13) continue;
        seen[t->type]=1;
        int n=t->dims[0],rows=t->dims[1]<4096?t->dims[1]:4096,nb=n/256;
        size_t rb=tf_row_bytes(t->type,n),bytes=(size_t)rows*rb;
        void *raw=malloc(bytes);
        tf_iq4_cache_block *cache=tf_aligned_alloc_notouch(256,(size_t)rows*nb*sizeof(*cache));
        float *x=malloc(n*sizeof(float)),*y=malloc((rows+7)*sizeof(float)),*dw=malloc(n*sizeof(float));
        float *local_y=malloc((rows+7)*sizeof(float));
        if (!raw || !cache || !x || !y || !dw || !local_y ||
            pread(fd,raw,bytes,g->data_offset+t->offset)!=(ssize_t)bytes) return 2;
        int packing_bad=0;
        #pragma omp parallel for num_threads(48) schedule(static) reduction(+:packing_bad)
        for (int r=0;r<rows;r++) packing_bad+=tf_iq4_cache_pack_row(cache+(size_t)r*nb,
            (const uint8_t *)raw+(size_t)r*rb,t->type,n)!=0;
        bad+=packing_bad;
        printf("PACK type=%u rows=%d K=%d exact=%s\n",t->type,rows,n,packing_bad?"FAIL":"PASS");
        if (packing_bad) return 1;
        transformer_model m={0};
        transformer_layer layer={0};
        layer.ffn_gate=(qtensor){.data=raw,.type=t->type,.n_rows=rows,.n_cols=n};
        m.n_layers=1;m.layers=&layer;m.n_threads=48;
        m.numa.enabled=1;m.numa.n_cmgs=4;
        setenv("NUMA_INTERLEAVE","1",1);
        tf_pool_start(&m);
        size_t expected=(size_t)rows*nb*sizeof(*cache);
        if (transformer_cache_iq4_decode(&m,expected)!=expected) return 1;
        /* A repeated request must not allocate another representation. */
        if (transformer_cache_iq4_decode(&m,expected)!=0) return 1;
        tf_mixed_q8_block *qx=malloc(nb*sizeof(*qx));
        if (!qx) return 2;
        for (int q8=0;q8<2;q8++) {
            transformer_set_mixed_iq_q8(q8);
            double worst=0;
            for (int pass=0;pass<3;pass++) {
                for (int k=0;k<n;k++) x[k]=pass==2?0:((k*37+pass*53)%255-127)/127.f;
                for (int r=0;r<rows+7;r++) y[r]=local_y[r]=12345.f;
                tf_mixed_quant_q8(qx,x,n);
                tf_iq4_cache_rows(y,cache,x,n,t->type,0,5,q8);
                tf_iq4_cache_rows(y,cache,x,n,t->type,5,5,q8);
                tf_iq4_cache_rows(y,cache,x,n,t->type,5,rows,q8);
                tf_matvec_qtensor_rows(local_y,&layer.ffn_gate,x,0,5);
                tf_matvec_qtensor_rows(local_y,&layer.ffn_gate,x,5,5);
                tf_matvec_qtensor_rows(local_y,&layer.ffn_gate,x,5,rows);
                for (int r=0;r<rows;r++) if (local_y[r]!=y[r]) bad++;
                for (int r=0;r<13;r++) {
                    dequant_row(t->type,(const uint8_t *)raw+(size_t)r*rb,dw,n);
                    double ref=0,mag=0;
                    for (int k=0;k<n;k++) {
                        double a=(q8 || t->type==GGML_TYPE_IQ3_XXS)?(double)qx[k/256].q[k%256]*qx[k/256].d:x[k];
                        double v=(double)dw[k]*a; ref+=v;mag+=fabs(v);
                    }
                    double err=fabs(y[r]-ref)/fmax(1.,mag);
                    if(err>worst) worst=err;
                    if(!isfinite(y[r]) || err>2e-6) bad++;
                }
                for (int r=rows;r<rows+7;r++) if(y[r]!=12345.f || local_y[r]!=12345.f) bad++;
            }
            printf("CHECK type=%u q8=%d scaled_error=%.3g %s\n",t->type,q8,worst,bad?"FAIL":"PASS");
            for (int k=0;k<n;k++) x[k]=(k%255-127)/127.f;
            qtensor mat={.data=raw,.type=t->type,.n_rows=rows,.n_cols=n};
            transformer_set_mixed_iq_q8(q8);
            for (int cached=0;cached<3;cached++) {
                double start=omp_get_wtime();
                for (int rep=0;rep<20;rep++) {
                    #pragma omp parallel num_threads(48)
                    {
                        int id=omp_get_thread_num(),nt=omp_get_num_threads();
                        if(cached==2) tf_matvec_qtensor_rows(y,&layer.ffn_gate,x,rows*id/nt,rows*(id+1)/nt);
                        else if(cached) tf_iq4_cache_rows(y,cache,x,n,t->type,rows*id/nt,rows*(id+1)/nt,q8);
                        else tf_matvec_qtensor_rows(y,&mat,x,rows*id/nt,rows*(id+1)/nt);
                    }
                }
                printf("BENCH type=%u q8=%d cache4=%d ms=%.3f checksum=%g\n",
                    t->type,q8,cached,(omp_get_wtime()-start)*50,y[0]);
            }
        }
        tf_pool_shutdown(&m);
        for (int i=0;i<m.decode_owned_count;i++) free(m.decode_owned[i]);
        free(m.decode_owned);
        free(qx);free(raw);free(cache);free(x);free(y);free(dw);free(local_y);
        fflush(stdout);
    }
    close(fd);gguf_close(g);
    return bad?1:0;
}
