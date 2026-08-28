/* Benchmark one staged GLM-5.3F quarter-expert using real FP8 weights. */
#define _GNU_SOURCE
#define GLM5_IMPL
#include "glm5.h"
#include "glm53f_expert_kern.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {size_t off;int rows,cols;const char*dtype;} ent;
static double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
/* GLM-5.3F stores one F32 scale per 128x128 FP8 block.  The GLM-5.2
 * runtime expands that grid to one scale row per output row; staged 5.3F
 * weights deliberately retain the compact checkpoint layout. */
static void mv_fp8_block128(glm5_model*m,float*restrict y,const uint8_t*W,
                            const float*S,const float*x,int rows,int cols){
    const int sb=(cols+127)/128,nb=rows/8;
#if defined(__ARM_FEATURE_SVE)
    static int use_bits=-1;if(use_bits<0)use_bits=getenv("FP8_DECODE_BITS")?atoi(getenv("FP8_DECODE_BITS")):0;
    #pragma omp parallel for schedule(static)
    for(int bi=0;bi<nb;bi++){
        int r=bi*8;const uint8_t*w=W+(size_t)r*cols;
        const float*s=S+(size_t)(r/128)*sb;
        if(use_bits)glm53f_matvec_fp8_bits_8(y+r,w,s,x,cols);else glm5_matvec_mxfp8_f32scale_8row(y+r,
            w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
            w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
            s,s,s,s,s,s,s,s,x,cols,m->fp8_lut);
    }
#else
    for(int r=0;r<nb*8;r++)
        y[r]=glm5_dot_mxfp8_f32scale_row(W+(size_t)r*cols,
            S+(size_t)(r/128)*sb,x,cols,m->fp8_lut);
#endif
    for(int r=nb*8;r<rows;r++)
        y[r]=glm5_dot_mxfp8_f32scale_row(W+(size_t)r*cols,
            S+(size_t)(r/128)*sb,x,cols,m->fp8_lut);
}
static int find_ent(const char*path,const char*name,ent*out){
    FILE*f=fopen(path,"r");char line[2048],dt[16];unsigned long long off;int nd,r,c;
    if(!f)return-1;while(fgets(line,sizeof line,f)){if(line[0]=='#')continue;
        if(strstr(line,name)&&sscanf(line,"%llu %15s %d %d %d",&off,dt,&nd,&r,&c)==5){
            out->off=(size_t)off;out->rows=r;out->cols=c;out->dtype=!strcmp(dt,"F8_E4M3")?"fp8":"f32";fclose(f);return 0;}}
    fclose(f);return-1;
}
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s BLOB MANIFEST [layer=3] [expert=0]\n",argv[0]);return 2;}
    int l=argc>3?atoi(argv[3]):3,e=argc>4?atoi(argv[4]):0,reps=getenv("REPS")?atoi(getenv("REPS")):100;
    char base[512],n_gu[640],n_gs[640],n_dn[640],n_ds[640];ent gu,gs,dn,ds;struct stat sb;
    snprintf(base,sizeof base,"model.language_model.layers.%d.mlp.experts.%d",l,e);
    snprintf(n_gu,sizeof n_gu,"%s.gate_up_fused.weight",base);snprintf(n_gs,sizeof n_gs,"%s.gate_up_fused.weight_scale_inv",base);
    snprintf(n_dn,sizeof n_dn,"%s.down_proj.weight",base);snprintf(n_ds,sizeof n_ds,"%s.down_proj.weight_scale_inv",base);
    if(find_ent(argv[2],n_gu,&gu)||find_ent(argv[2],n_gs,&gs)||find_ent(argv[2],n_dn,&dn)||find_ent(argv[2],n_ds,&ds)){fprintf(stderr,"entries not found\n");return 2;}
    int stream=getenv("STREAM_TASKS")?atoi(getenv("STREAM_TASKS")):0,nt=stream>0?stream:1;
    int batch=getenv("BATCH_TASKS")?atoi(getenv("BATCH_TASKS")):1;if(batch<1)batch=1;
    if(nt>96)nt=96;ent guv[96],gsv[96],dnv[96],dsv[96];
    guv[0]=gu;gsv[0]=gs;dnv[0]=dn;dsv[0]=ds;
    int expert_step=getenv("EXPERT_STEP")?atoi(getenv("EXPERT_STEP")):3;
    for(int k=1;k<nt;k++){
        int ek=(e+expert_step*k)%288;snprintf(base,sizeof base,"model.language_model.layers.%d.mlp.experts.%d",l,ek);
        snprintf(n_gu,sizeof n_gu,"%s.gate_up_fused.weight",base);snprintf(n_gs,sizeof n_gs,"%s.gate_up_fused.weight_scale_inv",base);
        snprintf(n_dn,sizeof n_dn,"%s.down_proj.weight",base);snprintf(n_ds,sizeof n_ds,"%s.down_proj.weight_scale_inv",base);
        if(find_ent(argv[2],n_gu,&guv[k])||find_ent(argv[2],n_gs,&gsv[k])||find_ent(argv[2],n_dn,&dnv[k])||find_ent(argv[2],n_ds,&dsv[k])){fprintf(stderr,"stream entry expert=%d not found\n",ek);return 2;}
    }
    int max_gu=gu.rows,max_inter=dn.cols;
    for(int k=1;k<nt;k++){if(guv[k].rows>max_gu)max_gu=guv[k].rows;if(dnv[k].cols>max_inter)max_inter=dnv[k].cols;}
    int fd=open(argv[1],O_RDONLY);if(fd<0||fstat(fd,&sb))return 2;
    int anon=getenv("ANON_WEIGHTS")?atoi(getenv("ANON_WEIGHTS")):1;
    unsigned char*map;
    if(anon){
        map=glm5_amalloc((size_t)sb.st_size);if(!map)return 2;
        const size_t chunk=16u<<20;double lt=sec();
        for(size_t off=0;off<(size_t)sb.st_size;off+=chunk){size_t n=(size_t)sb.st_size-off;if(n>chunk)n=chunk;
            ssize_t got=pread(fd,map+off,n,(off_t)off);if(got!=(ssize_t)n){perror("pread");return 2;}
            posix_fadvise(fd,(off_t)off,(off_t)n,POSIX_FADV_DONTNEED);
        }
        printf("GLM53F_REAL_EXPERT_LOAD mode=anon bytes=%zu seconds=%.3f\n",(size_t)sb.st_size,sec()-lt);
    }else{map=mmap(NULL,(size_t)sb.st_size,PROT_READ,MAP_SHARED,fd,0);if(map==MAP_FAILED)return 2;}
    int scratch_gu=batch>1&&max_gu<1024?1024:max_gu,scratch_inter=batch>1&&max_inter<512?512:max_inter;
    float*x=glm5_amalloc((size_t)gu.cols*4),*up=glm5_amalloc((size_t)scratch_gu*batch*4),*act=glm5_amalloc((size_t)scratch_inter*batch*4),*y=glm5_amalloc((size_t)dn.rows*batch*4);
    glm5_model m;memset(&m,0,sizeof m);glm5_init_fp8_lut(m.fp8_lut);for(int i=0;i<gu.cols;i++)x[i]=(float)((i%29)-14)*0.001f;
    {const float*s1=(const float*)(map+gs.off),*s2=(const float*)(map+ds.off);float a=s1[0],b=s1[0],c=s2[0],d=s2[0];
     for(int i=0;i<gs.rows*gs.cols;i++){if(s1[i]<a)a=s1[i];if(s1[i]>b)b=s1[i];}
     for(int i=0;i<ds.rows*ds.cols;i++){if(s2[i]<c)c=s2[i];if(s2[i]>d)d=s2[i];}
     printf("GLM53F_REAL_EXPERT_SCALE gate_up=[%.8g,%.8g] down=[%.8g,%.8g] offsets=%zu,%zu,%zu,%zu\n",a,b,c,d,gu.off,gs.off,dn.off,ds.off);}
    double best=1e30,sum=0;int bad_up=0,bad_y=0;float max_up=0,max_y=0;
#if defined(__ARM_FEATURE_SVE)
    if(batch>1){
        if(nt%batch){fprintf(stderr,"STREAM_TASKS must be divisible by BATCH_TASKS\n");return 2;}
        int groups=nt/batch,runs=reps*groups;
        for(int q=0;q<runs;q++){int k=(q%groups)*batch;glm53f_expert_part p[96];
            for(int j=0;j<batch;j++){p[j].gate_up=map+guv[k+j].off;p[j].gate_up_scale=(const float*)(map+gsv[k+j].off);
                p[j].down=map+dnv[k+j].off;p[j].down_scale=(const float*)(map+dsv[k+j].off);p[j].inter=dnv[k+j].cols;}
            double t=sec();glm53f_expert_batch_bits(p,batch,x,up,act,y);t=sec()-t;
            if(t<best)best=t;sum+=t;
        }
        double checksum=0;for(int i=0;i<dn.rows*batch;i++){float a=fabsf(y[i]);bad_y+=!isfinite(y[i]);if(a>max_y&&isfinite(a))max_y=a;checksum+=y[i];}
        {int k=(groups-1)*batch;for(int j=0;j<batch;j++)for(int i=0;i<guv[k+j].rows;i++){
            float a=fabsf(up[(size_t)j*1024+i]);bad_up+=!isfinite(up[(size_t)j*1024+i]);if(a>max_up&&isfinite(a))max_up=a;}}
        printf("GLM53F_REAL_EXPERT_BATCH layer=%d tasks=%d batch=%d cycles=%d best_batch_ms=%.4f mean_batch_ms=%.4f mean_task_ms=%.4f bad_up=%d bad_y=%d max_up=%.6g max_y=%.6g checksum=%.9g\n",
               l,nt,batch,reps,best*1e3,sum/runs*1e3,sum/runs*1e3/batch,bad_up,bad_y,max_up,max_y,checksum);
        glm5_afree(y);glm5_afree(act);glm5_afree(up);glm5_afree(x);if(anon)glm5_afree(map);else munmap(map,(size_t)sb.st_size);close(fd);
        return bad_up||bad_y||!isfinite(checksum)?1:0;
    }
#endif
    int runs=reps*nt;for(int q=0;q<runs;q++){ent*qgu=&guv[q%nt],*qgs=&gsv[q%nt],*qdn=&dnv[q%nt],*qds=&dsv[q%nt];double t=sec();
        mv_fp8_block128(&m,up,map+qgu->off,(const float*)(map+qgs->off),x,qgu->rows,qgu->cols);
        int h=qgu->rows/2;for(int i=0;i<qgu->rows;i++){float a=fabsf(up[i]);bad_up+=!isfinite(up[i]);if(a>max_up&&isfinite(a))max_up=a;}
        for(int i=0;i<h;i++){float g=up[i]>10?10:up[i],u=up[h+i];if(g<-100)g=-100;if(u>10)u=10;if(u<-10)u=-10;act[i]=(g/(1+expf(-g)))*u;}
        mv_fp8_block128(&m,y,map+qdn->off,(const float*)(map+qds->off),act,qdn->rows,qdn->cols);
        for(int i=0;i<dn.rows;i++){float a=fabsf(y[i]);bad_y+=!isfinite(y[i]);if(a>max_y&&isfinite(a))max_y=a;}
        t=sec()-t;if(t<best)best=t;sum+=t;
    }
    double checksum=0;for(int i=0;i<dn.rows;i++)checksum+=y[i];
    printf("GLM53F_REAL_EXPERT layer=%d expert=%d part_inter=%d tasks=%d cycles=%d best_ms=%.4f mean_ms=%.4f bad_up=%d bad_y=%d max_up=%.6g max_y=%.6g checksum=%.9g\n",l,e,dn.cols,nt,reps,best*1e3,sum/runs*1e3,bad_up,bad_y,max_up,max_y,checksum);
    glm5_afree(y);glm5_afree(act);glm5_afree(up);glm5_afree(x);if(anon)glm5_afree(map);else munmap(map,(size_t)sb.st_size);close(fd);return bad_up||bad_y||!isfinite(checksum)?1:0;
}
