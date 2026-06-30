/* Standalone Gemma4UV vision-encoder test: load ONLY mmproj, run g4v_encode on a
 * synthetic image, time it + checksum the output embeddings (finite + stable). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"
#define GEMMA4_VISION_IMPLEMENTATION
#include "gemma4_vision_encoder.h"

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

int main(int argc,char**argv){
    const char *mm = argc>1?argv[1]:"/local/u14346/mmproj-F32.gguf";
    int W = argc>2?atoi(argv[2]):224, H = argc>3?atoi(argv[3]):224;
    gguf_context *g = getenv("TF_FORCE_MMAP") ? gguf_open(mm, 1) : g4v_open_mmproj(mm);
    if(!g){ fprintf(stderr,"open fail %s\n",mm); return 1; }
    g4v_model *vm = g4v_load(g);
    if(!vm){ fprintf(stderr,"g4v_load fail\n"); return 1; }
    /* synthetic RGB checkerboard */
    uint8_t *rgb = (uint8_t*)malloc((size_t)W*H*3);
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){ int c=((x/16)+(y/16))&1; uint8_t v=c?200:40;
        rgb[((size_t)y*W+x)*3+0]=v; rgb[((size_t)y*W+x)*3+1]=(uint8_t)(v/2); rgb[((size_t)y*W+x)*3+2]=(uint8_t)(255-v); }
    /* warm (faults cold weight pages) then time the 2nd call */
    float *emb = g4v_encode(vm, rgb, W, H);
    double t0=wall();
    emb = g4v_encode(vm, rgb, W, H);
    double t1=wall();
    if(!emb){ fprintf(stderr,"encode returned NULL\n"); return 1; }
    int ntok = vm->n_merged, dim = vm->proj_dim;
    double sum=0,sa=0; int nan=0; float mn=1e30f,mx=-1e30f;
    for(size_t i=0;i<(size_t)ntok*dim;i++){ float v=emb[i]; if(!isfinite(v))nan++; sum+=v; sa+=fabs(v); if(v<mn)mn=v; if(v>mx)mx=v; }
    printf("g4v_encode: %dx%d -> ntok=%d dim=%d  %.2f ms\n", W,H,ntok,dim,(t1-t0)*1e3);
    printf("  sum=%.6f sumabs=%.6f min=%.4f max=%.4f NaN=%d  emb[0..3]=%.5f %.5f %.5f %.5f\n",
           sum,sa,mn,mx,nan,emb[0],emb[1],emb[2],emb[3]);
    g4v_free(vm); gguf_close(g);
    return nan?2:0;
}
