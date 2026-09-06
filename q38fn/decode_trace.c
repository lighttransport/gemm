#define _GNU_SOURCE
#define GLM5_BPE_IMPLEMENTATION
#include "../common/glm5_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char**argv)
{
    if(argc!=3){fprintf(stderr,"usage: %s TOKENIZER_JSON RANK_TRACE\n",argv[0]);return 2;}
    glm5_bpe bpe={0};FILE*f=NULL;char line[8192],piece[8192];
    if(glm5_bpe_load(argv[1],&bpe)||(f=fopen(argv[2],"r"))==NULL)return 1;
    bpe.im_start=248045;bpe.im_end=248046;bpe.think=248068;
    bpe.end_think=248069;bpe.endoftext=248044;
    while(fgets(line,sizeof(line),f)){
        int id;if(sscanf(line,"token pos=%*d id=%d",&id)==1){
            if(id==bpe.im_start||id==bpe.im_end||id==bpe.think||
               id==bpe.end_think||id==bpe.endoftext)continue;
            int n=glm5_bpe_decode_token(&bpe,id,piece,sizeof(piece));
            if(n<0||fwrite(piece,1,(size_t)n,stdout)!=(size_t)n){fclose(f);glm5_bpe_free(&bpe);return 1;}
        }
    }
    fclose(f);glm5_bpe_free(&bpe);return ferror(stdout)?1:0;
}
