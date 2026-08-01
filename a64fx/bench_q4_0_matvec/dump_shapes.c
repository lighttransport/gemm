/* Dump GGUF tensor shapes (name, type, dims). */
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#include <stdio.h>

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf\n",argv[0]); return 1; }
    gguf_context *g = gguf_open(argv[1], 1);
    if(!g){ fprintf(stderr,"open failed\n"); return 1; }
    printf("n_tensors=%llu\n", (unsigned long long)g->n_tensors);
    for(uint64_t i=0;i<g->n_tensors;i++){
        gguf_tensor_info *t=&g->tensors[i];
        printf("%-40s type=%-10s ndim=%u dims=[", t->name.str?t->name.str:"?",
               ggml_type_name(t->type), t->n_dims);
        for(uint32_t d=0;d<t->n_dims;d++) printf("%llu%s",(unsigned long long)t->dims[d], d+1<t->n_dims?",":"");
        printf("]\n");
    }
    gguf_close(g);
    return 0;
}
