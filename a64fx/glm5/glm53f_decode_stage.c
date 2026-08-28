/* Stage four-way sliced GLM-5.3F routed experts to node-local storage. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_arch.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ALIGN 256u
static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int env_i(const char*n,int d){const char*s=getenv(n);return s&&*s?atoi(s):d;}
static int rank_id(void){const char*k[]={"PMIX_RANK","PJM_MPI_RANK","OMPI_COMM_WORLD_RANK",NULL};for(int i=0;k[i];i++){const char*s=getenv(k[i]);if(s&&*s)return atoi(s);}return env_i("GLM53F_RANK",0);}
static int write_all(int fd,const void*p,size_t n){const unsigned char*b=p;while(n){ssize_t z=write(fd,b,n);if(z<0){if(errno==EINTR)continue;return -1;}b+=z;n-=(size_t)z;}return 0;}
static int align_fd(int fd,uint64_t*off){uint64_t a=(*off+ALIGN-1)&~(uint64_t)(ALIGN-1);if(a!=*off&&lseek(fd,(off_t)a,SEEK_SET)<0)return-1;*off=a;return 0;}
static int owned_part(int expert,int rank,int parts,int ranks){for(int p=0;p<parts;p++)if(glm53f_expert_part_owner(expert,p,parts,ranks)==rank)return p;return-1;}
static int fp8_has_nan(const unsigned char*p,size_t n){for(size_t i=0;i<n;i++)if((p[i]&0x7f)==0x7f)return 1;return 0;}

static int read_tensor(glm53f_st_context*st,const char*name,void**buf,size_t*cap,const st_tensor_info**ti){
    const st_tensor_info*t=glm53f_st_find(st,name,NULL);if(!t)return-1;
    if(*cap<t->nbytes){void*p=realloc(*buf,t->nbytes);if(!p)return-1;*buf=p;*cap=t->nbytes;}
    if(glm53f_st_read(st,name,0,*buf,t->nbytes))return-1;
    *ti=t;return 0;
}
static int put_rows(int fd,uint64_t*off,FILE*mf,glm53f_st_context*st,const char*src,
                    const char*dst,int r0,int nr,int manifest_rows,int fuse_append,void**buf,size_t*cap){
    const st_tensor_info*t;size_t rowb,nb;uint64_t begin;
    t=glm53f_st_find(st,src,NULL);if(!t||t->n_dims!=2)return-1;
    rowb=t->nbytes/(size_t)t->shape[0];begin=(uint64_t)r0*rowb;nb=(size_t)nr*rowb;
    if(*cap<nb){void*p=realloc(*buf,nb);if(!p)return-1;*buf=p;*cap=nb;}
    if(glm53f_st_read(st,src,(size_t)begin,*buf,nb))return-1;
    if(!strcmp(t->dtype_str,"F8_E4M3")&&fp8_has_nan(*buf,nb)){errno=EDOM;return-1;}
    if(!fuse_append&&align_fd(fd,off))return-1;
    if(write_all(fd,*buf,nb))return-1;
    if(!fuse_append)fprintf(mf,"%"PRIu64" %s %d %d %"PRIu64" axis=0 begin=%d global=%"PRIu64" %s\n",
                            *off,t->dtype_str,2,manifest_rows,t->shape[1],r0,t->shape[0],dst);
    *off+=nb;return 0;
}
static int put_cols(int fd,uint64_t*off,FILE*mf,glm53f_st_context*st,const char*src,
                    const char*dst,int c0,int nc,void**buf,size_t*cap){
    const st_tensor_info*t;size_t es,rowb,outrow;
    if(read_tensor(st,src,buf,cap,&t)||t->n_dims!=2)return-1;
    es=safetensors_dtype_size(t->dtype_str);rowb=(size_t)t->shape[1]*es;outrow=(size_t)nc*es;
    uint64_t start;
    if(!es||align_fd(fd,off))return-1;
    start=*off;
    for(uint64_t r=0;r<t->shape[0];r++){
        unsigned char*p=(unsigned char*)*buf+r*rowb+(size_t)c0*es;
        if(!strcmp(t->dtype_str,"F8_E4M3")&&fp8_has_nan(p,outrow)){errno=EDOM;return-1;}
        if(write_all(fd,p,outrow))return-1;
    }
    fprintf(mf,"%"PRIu64" %s 2 %"PRIu64" %d axis=1 begin=%d global=%"PRIu64" %s\n",
            start,t->dtype_str,t->shape[0],nc,c0,t->shape[1],dst);*off+=(uint64_t)t->shape[0]*outrow;return 0;
}

int main(int argc,char**argv){
    const char*model=argc>1?argv[1]:getenv("GLM53F_MODEL_DIR");const char*out=getenv("GLM53F_STAGE_DIR");
    int rank=rank_id(),ranks=env_i("GLM53F_RANKS",12),parts=env_i("GLM53F_EXPERT_PARTS",4);
    int shared_only=env_i("GLM53F_STAGE_SHARED_ONLY",0);
    int first=env_i("GLM53F_STAGE_FIRST_LAYER",3),last=env_i("GLM53F_STAGE_LAYERS",45);
    char model_dflt[256],out_dflt[256],bp[512],mp[512],name[512],virt[512];glm53f_st_context*st;void*buf=NULL;size_t cap=0;
    uint64_t off=0,last_sync=0,flush=1ull<<30;int nt=0,fd=-1;FILE*mf=NULL;double t0=now_sec();
    if(!model){const char*h=getenv("HOME");snprintf(model_dflt,sizeof model_dflt,"%s/models/glm53f",h?h:".");model=model_dflt;}
    if(!out||!*out){snprintf(out_dflt,sizeof out_dflt,"/local/glm53f-decode-%s",getenv("PJM_JOBID")?getenv("PJM_JOBID"):"manual");out=out_dflt;}
    if(rank<0||rank>=ranks||parts<1||parts>16||ranks%parts||first<3||last>45||first>=last)return 2;
    mkdir(out,0755);snprintf(bp,sizeof bp,"%s/rank%02d.blob",out,rank);snprintf(mp,sizeof mp,"%s/rank%02d.manifest",out,rank);
    st=glm53f_st_open(model);if(!st||glm53f_st_validate_contract(st,0)){fprintf(stderr,"checkpoint failed\n");return 2;}
    fd=open(bp,O_CREAT|O_TRUNC|O_WRONLY,0644);mf=fopen(mp,"w");if(fd<0||!mf){perror("stage output");return 2;}
    fprintf(mf,"# GLM53F_DECODE rank=%d ranks=%d expert_parts=%d layers=%d:%d\n",rank,ranks,parts,first,last);
    if(!shared_only)for(int l=first;l<last;l++)for(int e=0;e<288;e++){
        int p=owned_part(e,rank,parts,ranks),b,n;if(p<0)continue;
        /* F8 scales cover 128x128 blocks, so expert partitions must start and
         * end on a scale-block boundary. This is identical to equal slicing
         * for 1/2/4 parts and gives valid 128/256-row shards for 12 parts. */
        if(glm53f_block_aligned_slice(2048,128,p,parts,&b,&n))goto fail;
#define NM(S) snprintf(name,sizeof name,"model.language_model.layers.%d.mlp.experts.%d.%s",l,e,S)
#define VM(S) snprintf(virt,sizeof virt,"model.language_model.layers.%d.mlp.experts.%d.%s",l,e,S)
        NM("gate_proj.weight");VM("gate_up_fused.weight");if(put_rows(fd,&off,mf,st,name,virt,b,n,2*n,0,&buf,&cap))goto fail;
        NM("up_proj.weight");if(put_rows(fd,&off,mf,st,name,virt,b,n,2*n,1,&buf,&cap))goto fail;nt++;
        NM("gate_proj.weight_scale_inv");VM("gate_up_fused.weight_scale_inv");if(put_rows(fd,&off,mf,st,name,virt,b/128,n/128,2*(n/128),0,&buf,&cap))goto fail;
        NM("up_proj.weight_scale_inv");if(put_rows(fd,&off,mf,st,name,virt,b/128,n/128,2*(n/128),1,&buf,&cap))goto fail;nt++;
        NM("down_proj.weight");VM("down_proj.weight");if(put_cols(fd,&off,mf,st,name,virt,b,n,&buf,&cap))goto fail;nt++;
        NM("down_proj.weight_scale_inv");VM("down_proj.weight_scale_inv");if(put_cols(fd,&off,mf,st,name,virt,b/128,n/128,&buf,&cap))goto fail;nt++;
#undef NM
#undef VM
        if(off-last_sync>=flush){fdatasync(fd);posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);last_sync=off;}
        if(e%48==47){printf("rank=%d layer=%d expert=%d staged=%.3fGiB sec=%.1f\n",rank,l,e,off/1073741824.0,now_sec()-t0);fflush(stdout);}
    }
    if(shared_only)for(int l=first;l<last;l++){
        int b,n;glm53f_block_aligned_slice(2048,128,rank,ranks,&b,&n);
#define SN(S) snprintf(name,sizeof name,"model.language_model.layers.%d.mlp.shared_experts.%s",l,S)
#define SV(S) snprintf(virt,sizeof virt,"model.language_model.layers.%d.mlp.shared_experts.%s",l,S)
        SN("gate_proj.weight");SV("gate_up_fused.weight");if(put_rows(fd,&off,mf,st,name,virt,b,n,2*n,0,&buf,&cap))goto fail;
        SN("up_proj.weight");if(put_rows(fd,&off,mf,st,name,virt,b,n,2*n,1,&buf,&cap))goto fail;nt++;
        SN("gate_proj.weight_scale_inv");SV("gate_up_fused.weight_scale_inv");if(put_rows(fd,&off,mf,st,name,virt,b/128,n/128,2*(n/128),0,&buf,&cap))goto fail;
        SN("up_proj.weight_scale_inv");if(put_rows(fd,&off,mf,st,name,virt,b/128,n/128,2*(n/128),1,&buf,&cap))goto fail;nt++;
        SN("down_proj.weight");SV("down_proj.weight");if(put_cols(fd,&off,mf,st,name,virt,b,n,&buf,&cap))goto fail;nt++;
        SN("down_proj.weight_scale_inv");SV("down_proj.weight_scale_inv");if(put_cols(fd,&off,mf,st,name,virt,b/128,n/128,&buf,&cap))goto fail;nt++;
#undef SN
#undef SV
    }
    fdatasync(fd);posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);fclose(mf);close(fd);free(buf);glm53f_st_close(st);
    { const char*sd=getenv("GLM53F_STATUS_DIR"); if(sd&&*sd){char sp[512];snprintf(sp,sizeof sp,"%s/rank%02d.status",sd,rank);
      FILE*sf=fopen(sp,"w");if(sf){fprintf(sf,"rank=%d tensors=%d bytes=%"PRIu64" sec=%.1f blob=%s OK\n",rank,nt,off,now_sec()-t0,bp);fclose(sf);}} }
    printf("SENTINEL glm53f_decode_stage=OK rank=%d tensors=%d bytes=%"PRIu64" sec=%.1f\n",rank,nt,off,now_sec()-t0);return 0;
fail:
    fprintf(stderr,"stage failed rank=%d name=%s: %s\n",rank,name,strerror(errno));if(mf)fclose(mf);if(fd>=0)close(fd);free(buf);glm53f_st_close(st);return 1;
}
