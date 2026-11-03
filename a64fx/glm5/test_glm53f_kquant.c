/* Compare fused kernels against independently dequantized weights and Q8 input. */
#define _GNU_SOURCE
#include "glm53f_iq_bridge.c"

static int check_expert_batch(void) {
    enum { H=4096, S=9, I=256 };
    float *input=malloc(4*H*sizeof(float)), *output=malloc(4*H*sizeof(float));
    float *reference=calloc(4*H,sizeof(float));
    float *gu=malloc(4*S*1024*sizeof(float)), *act=malloc(4*S*512*sizeof(float));
    float *expert=malloc(4*S*H*sizeof(float));
    float *ref_gu=calloc(4*S*1024,sizeof(float)),*ref_act=calloc(4*S*512,sizeof(float));
    glm53f_iq_part parts[4*S]={{0}}, unique[2];
    int counts[4]={2,1,0,2};float weights[4*S]={0};
    if(!input||!output||!reference||!gu||!act||!expert||!ref_gu||!ref_act)return 2;
    for(int i=0;i<4*H;i++)input[i]=(float)((i*17+31)%251-125)/125;
    for(int e=0;e<2;e++) {
        int type=GLM53F_GGML_Q4_K+e;
        size_t gr=dequant_row_size(type,H),dr=dequant_row_size(type,I);
        uint8_t *gate=malloc(2*I*gr),*down=malloc(H*dr);
        if(!gate||!down)return 2;
        for(int side=0;side<2;side++) {
            uint8_t *p=side?down:gate;size_t bytes=side?H*dr:2*I*gr;
            for(size_t j=0;j<bytes;j++)p[j]=(uint8_t)(j*37+e*19+11);
            int blocks=(int)(bytes/(e?sizeof(block_q5_K):sizeof(block_q4_K)));
            for(int b=0;b<blocks;b++) {
                if(e) {((block_q5_K*)p)[b].d=ggml_fp32_to_fp16(0.002f);((block_q5_K*)p)[b].dmin=ggml_fp32_to_fp16(0.001f);}
                else {((block_q4_K*)p)[b].d=ggml_fp32_to_fp16(0.002f);((block_q4_K*)p)[b].dmin=ggml_fp32_to_fp16(0.001f);}
            }
        }
        unique[e]=(glm53f_iq_part){gate,down,type,type,I};
    }
    for(int t=0;t<4;t++)for(int k=0;k<counts[t];k++) {
        parts[t*S+k]=unique[t==3?1-k:k];weights[t*S+k]=0.13f*(k+1);
    }
    for(int t=0;t<4;t++)if(counts[t]&&glm53f_iq_expert_weighted(
        reference+(size_t)t*H,parts+t*S,weights+t*S,counts[t],input+(size_t)t*H,
        ref_gu+(size_t)t*S*1024,ref_act+(size_t)t*S*512))return 1;
    for(int tokens=1;tokens<=4;tokens++) {
        if(glm53f_iq_expert_weighted_batch(output,parts,weights,counts,tokens,input,gu,act,expert))return 1;
        if(memcmp(output,reference,(size_t)tokens*H*sizeof(float))) {
            for(int t=0;t<tokens;t++)for(int k=0;k<counts[t];k++) {
                int slot=t*S+k;
                if(memcmp(gu+(size_t)slot*1024,ref_gu+(size_t)slot*1024,2*I*sizeof(float)))
                    fprintf(stderr,"gate mismatch slot=%d\n",slot);
                if(memcmp(act+(size_t)slot*512,ref_act+(size_t)slot*512,I*sizeof(float)))
                    fprintf(stderr,"activation mismatch slot=%d\n",slot);
            }
            for(int i=0;i<tokens*H;i++)if(output[i]!=reference[i]) {
                fprintf(stderr,"FAIL expert batch tokens=%d i=%d got=%.9g expected=%.9g\n",tokens,i,output[i],reference[i]);break;
            }
            return 1;
        }
    }
    for(int mode=0;mode<=2;mode++){
        char value[2]={(char)('0'+mode),0};setenv("GLM53F_Q4_ROW_TILE",value,1);
        double begin=omp_get_wtime();
        for(int repeat=0;repeat<8;repeat++)
            if(glm53f_iq_expert_weighted_batch(output,parts,weights,counts,2,input,gu,act,expert))return 1;
        if(memcmp(output,reference,2UL*H*sizeof(float)))return 1;
        printf("KQUANT_TILE_MICRO mode=%d tokens=2 ms=%.6f BIT_EXACT\n",mode,(omp_get_wtime()-begin)*1e3/8);
    }
    setenv("GLM53F_Q4_ROW_TILE","0",1);
    for(int budget=0;budget<2;budget++){
        size_t limit=(size_t)(budget?64:1)*1024*1024;
        glm53f_iq_cache *cache=glm53f_iq_cache_create(limit);
        if(!cache)return 2;
        setenv("GLM53F_Q4_ACT_SUMS",budget?"1":"0",1);
        for(int tokens=1;tokens<=4;tokens++){
            if(glm53f_iq_expert_weighted_batch_cached(output,parts,weights,counts,tokens,input,gu,act,expert,cache))return 1;
            if(memcmp(output,reference,(size_t)tokens*H*sizeof(float)))return 1;
        }
        if(glm53f_iq_cache_bytes(cache)>limit)return 1;
        double begin=omp_get_wtime();
        for(int repeat=0;repeat<8;repeat++)
            if(glm53f_iq_expert_weighted_batch_cached(output,parts,weights,counts,2,input,gu,act,expert,cache))return 1;
        printf("KQUANT_METADATA budget=%zu used=%zu ms=%.6f BIT_EXACT\n",limit,
               glm53f_iq_cache_bytes(cache),(omp_get_wtime()-begin)*1e3/8);
        glm53f_iq_cache_free(cache);
    }
    for(int e=0;e<2;e++){free((void*)unique[e].gate_up);free((void*)unique[e].down);}
    free(ref_gu);free(ref_act);free(expert);free(act);free(gu);free(reference);free(output);free(input);
    puts("PASS expert batch tokens=1..4 shared/disjoint/reordered/empty BIT_EXACT");
    return 0;
}

int main(void) {
    unsigned char weights[16 * sizeof(block_q6_K)];
    float input[4096], decoded[4096];
    glm5_iq_q8_block xq[16];
    unsigned rng = 7;
    double worst = 0.0;
    for (int type = GLM53F_GGML_Q4_K; type <= GLM53F_GGML_Q6_K; ++type) {
        for (int trial = 0; trial < 100; ++trial) {
            int n = trial % 2 ? 4096 : 256;
            size_t bytes = dequant_row_size(type, n);
            for (size_t i = 0; i < bytes; ++i) {
                rng = rng * 1664525u + 1013904223u;
                weights[i] = rng >> 24;
            }
            for (int i = 0; i < n; ++i) {
                rng = rng * 1664525u + 1013904223u;
                input[i] = ((int)(rng >> 16) - 32768) / 32768.0f;
            }
            for (int b = 0; b < n / 256; ++b) {
                if (type == GLM53F_GGML_Q4_K) {
                    block_q4_K *w = (block_q4_K *)weights;
                    w[b].d = ggml_fp32_to_fp16(0.01f);
                    w[b].dmin = ggml_fp32_to_fp16(0.005f);
                } else if (type == GLM53F_GGML_Q5_K) {
                    block_q5_K *w = (block_q5_K *)weights;
                    w[b].d = ggml_fp32_to_fp16(0.01f);
                    w[b].dmin = ggml_fp32_to_fp16(0.005f);
                } else ((block_q6_K *)weights)[b].d = ggml_fp32_to_fp16(0.01f);
            }
            glm5_iq_quant_q8(xq, input, n);
            if (type == GLM53F_GGML_Q4_K) dequantize_row_q4_K(weights, decoded, n);
            else if (type == GLM53F_GGML_Q5_K) dequantize_row_q5_K(weights, decoded, n);
            else dequantize_row_q6_K(weights, decoded, n);
            double ref = 0.0, magnitude = 0.0;
            for (int i = 0; i < n; ++i) {
                double term = (double)decoded[i] * xq[i / 256].d * xq[i / 256].q[i % 256];
                ref += term;
                magnitude += fabs(term);
            }
            float got = iq_row(type, weights, xq, n / 256);
            glm5_iq_q8_block multi_q[4][16];
            const glm5_iq_q8_block *multi_x[4];
            for(int t=0;t<4;t++) {
                for(int i=0;i<n;i++)decoded[i]=input[i]*(t+1)+0.13f*t;
                glm5_iq_quant_q8(multi_q[t],decoded,n);multi_x[t]=multi_q[t];
            }
            for(int tokens=1;tokens<=4;tokens++) {
                float values[4];iq_rows_multi(values,type,weights,multi_x,tokens,n/256);
                for(int t=0;t<tokens;t++) {
                    float expected=iq_row(type,weights,multi_x[t],n/256);
                    if(memcmp(&values[t],&expected,sizeof(float))) {
                        fprintf(stderr,"FAIL multi-row type=%d tokens=%d token=%d\n",type,tokens,t);
                        return 1;
                    }
                }
            }
            double error = fabs(got - ref) / (1.0 + magnitude);
            if (!isfinite(got) || error > 2e-6) {
                fprintf(stderr, "FAIL type=%d n=%d ref=%.9g got=%.9g error=%g\n", type, n, ref, got, error);
                return 1;
            }
            if (error > worst) worst = error;
        }
    }
    printf("PASS K-quants cases=300 worst_normalized_error=%g\n", worst);
    if(check_expert_batch())return 1;
    if (getenv("GLM53F_KQUANT_BENCH")) {
        enum { ROWS = 8192, COLS = 4096, REPS = 20 };
        float *out = malloc(ROWS * sizeof(float));
        if (!out) return 2;
        for (int type = GLM53F_GGML_Q4_K; type <= GLM53F_GGML_Q5_K; ++type) {
            size_t rb = dequant_row_size(type, COLS);
            unsigned char *matrix = malloc(ROWS * rb);
            if (!matrix) return 2;
            /* A repeated finite block exercises the real row stride and HBM
             * footprint without allowing the compiler to fold the dot. */
            memset(weights, 0x55, sizeof(weights));
            for (int b = 0; b < COLS / 256; ++b) {
                if (type == GLM53F_GGML_Q4_K) {
                    ((block_q4_K *)weights)[b].d = ggml_fp32_to_fp16(0.01f);
                    ((block_q4_K *)weights)[b].dmin = ggml_fp32_to_fp16(0.005f);
                } else {
                    ((block_q5_K *)weights)[b].d = ggml_fp32_to_fp16(0.01f);
                    ((block_q5_K *)weights)[b].dmin = ggml_fp32_to_fp16(0.005f);
                }
            }
#pragma omp parallel for schedule(static)
            for (int r = 0; r < ROWS; ++r) memcpy(matrix + r * rb, weights, rb);
            double start = omp_get_wtime();
            for (int rep = 0; rep < REPS; ++rep) {
#pragma omp parallel for schedule(static)
                for (int r = 0; r < ROWS; ++r)
                    out[r] = iq_row(type, matrix + r * rb, xq, COLS / 256);
            }
            double seconds = omp_get_wtime() - start;
            printf("BENCH type=%d Gweights_s=%.3f check=%.9g\n", type,
                   (double)ROWS * COLS * REPS / seconds / 1e9, out[ROWS-1]);
            free(matrix);
        }
        free(out);
    }
    return 0;
}
