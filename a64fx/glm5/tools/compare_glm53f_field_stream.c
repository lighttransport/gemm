/* Bounded host-only canonical field comparator. Requests are tab-separated:
 * kind, field, pathA, offsetA, pathB, offsetB, count, selection_limit.
 * F=float32, M=four-stream mean, B=exact bytes, S=indices, R=expert routes.
 * No model construction, MPI or NumPy dependency. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctype.h>

enum { CHUNK = 16384, FILES = 16 };
typedef struct { char *path; FILE *file; uint64_t bytes, age; int side; } cached_file;
static cached_file cache[FILES];
static uint64_t age;
static int number(const char *s,uint64_t *value) {
    char *end;errno=0;if(!*s||*s=='-')return-1;
    *value=strtoull(s,&end,10);return errno||*end?-1:0;
}
static FILE *open_at(const char *path,uint64_t offset,uint64_t bytes,int side) {
    int slot=-1,oldest=0;
    for(int i=0;i<FILES;i++) {
        if(cache[i].path&&cache[i].side==side&&!strcmp(cache[i].path,path)){slot=i;break;}
        if(!cache[i].path)oldest=i;
        else if(cache[oldest].path&&cache[i].age<cache[oldest].age)oldest=i;
    }
    if(slot<0) {
        slot=oldest;
        if(cache[slot].file)fclose(cache[slot].file);
        free(cache[slot].path);memset(&cache[slot],0,sizeof(cache[slot]));
        FILE *f=fopen(path,"rb");struct stat st;
        if(!f||fstat(fileno(f),&st)||st.st_size<0){if(f)fclose(f);return NULL;}
        cache[slot].path=strdup(path);
        if(!cache[slot].path){fclose(f);return NULL;}
        cache[slot].file=f;cache[slot].bytes=(uint64_t)st.st_size;cache[slot].side=side;
    }
    cache[slot].age=++age;
    if(offset>cache[slot].bytes||bytes>cache[slot].bytes-offset||offset>INT64_MAX||
       fseeko(cache[slot].file,(off_t)offset,SEEK_SET))return NULL;
    return cache[slot].file;
}
static int finite_float(float x){uint32_t bits;memcpy(&bits,&x,4);return(bits&UINT32_C(0x7f800000))!=UINT32_C(0x7f800000);}
static int int_order(const void *a,const void *b){int32_t x,y;memcpy(&x,a,4);memcpy(&y,b,4);return(x>y)-(x<y);}
int main(int argc,char **argv) {
    int exact=argc==2&&!strcmp(argv[1],"--bit-exact"),failed=0;
    if(argc!=1&&!exact)return 2;
    uint16_t endian=1;if(*(unsigned char *)&endian!=1||sizeof(float)!=4)return 2;
    char line[16384];float a[CHUNK],b[CHUNK];
    while(fgets(line,sizeof(line),stdin)) {
        if(!strchr(line,'\n'))return 2;
        line[strcspn(line,"\r\n")]=0;
        char *parts[8],*save=NULL;int n=0;
        for(char *p=strtok_r(line,"\t",&save);p;p=strtok_r(NULL,"\t",&save)){if(n==8)return 2;parts[n++]=p;}
        uint64_t oa,ob,count,limit;
        if(n!=8||strlen(parts[0])!=1||number(parts[3],&oa)||number(parts[5],&ob)||number(parts[6],&count)||number(parts[7],&limit))return 2;
        for(const char *p=parts[1];*p;p++)if(!isalnum((unsigned char)*p)&&*p!='.'&&*p!='_')return 2;
        char kind=parts[0][0];uint64_t unit=kind=='B'?1:kind=='R'?64:kind=='M'?16:4;
        if(!strchr("FMBSR",kind)||count>UINT64_MAX/unit||(kind=='M'&&count>CHUNK/4)||(kind=='S'&&count>CHUNK))return 2;
        FILE *fa=open_at(parts[2],oa,count*unit,0),*fb=open_at(parts[4],ob,count*unit,1);
        int valid=fa&&fb,same=1;uint64_t changes=0,done=0;double denominator=0,error=0;
        const char *reason="none";
        if(!valid)reason="io_shape";
        if(kind=='M'&&valid) {
            size_t total=(size_t)count*4;
            if(fread(a,4,total,fa)!=total||fread(b,4,total,fb)!=total){valid=0;reason="io";}
            else for(size_t i=0;i<count;i++) {
                float x=a[i]+a[count+i],y=b[i]+b[count+i];
                x=x+a[2*count+i];y=y+b[2*count+i];x=x+a[3*count+i];y=y+b[3*count+i];x=x*.25f;y=y*.25f;
                if(!finite_float(x)||!finite_float(y)){valid=0;reason="nonfinite";}
                same&=!memcmp(&x,&y,4);double delta=(double)y-x;denominator+=(double)x*x;error+=delta*delta;
            }
        } else if(kind=='S'&&valid) {
            if(fread(a,4,(size_t)count,fa)!=count||fread(b,4,(size_t)count,fb)!=count){valid=0;reason="io";}
            else {
                for(size_t i=0;i<count;i++){int32_t x,y;memcpy(&x,a+i,4);memcpy(&y,b+i,4);changes+=x!=y;if(x<0||y<0||(uint64_t)x>=limit||(uint64_t)y>=limit){valid=0;reason="invalid_indices";}}
                same=!memcmp(a,b,(size_t)count*4);
                qsort(a,(size_t)count,4,int_order);qsort(b,(size_t)count,4,int_order);
                for(size_t i=1;i<count;i++)if(!memcmp(a+i-1,a+i,4)||!memcmp(b+i-1,b+i,4)){valid=0;reason="duplicate_indices";}
            }
        } else while(done<count*unit&&valid) {
            size_t bytes=(size_t)(count*unit-done);if(bytes>sizeof(a))bytes=sizeof(a);
            if(fread(a,1,bytes,fa)!=bytes||fread(b,1,bytes,fb)!=bytes){valid=0;reason="io";break;}
            same&=!memcmp(a,b,bytes);
            if(kind=='F')for(size_t i=0;i<bytes/4;i++) {
                if(!finite_float(a[i])||!finite_float(b[i])){valid=0;reason="nonfinite";}
                double delta=(double)b[i]-a[i];denominator+=(double)a[i]*a[i];error+=delta*delta;
            }
            if(kind=='R')for(size_t row=0;row<bytes/64;row++) {
                int32_t x[8],y[8];memcpy(x,a+row*16,32);memcpy(y,b+row*16,32);
                for(int i=0;i<8;i++) {
                    changes+=x[i]!=y[i];
                    if(x[i]<0||x[i]>=288||y[i]<0||y[i]>=288||!finite_float(a[row*16+8+i])||!finite_float(b[row*16+8+i])){valid=0;reason="invalid_routes";}
                    for(int j=0;j<i;j++)if(x[i]==x[j]||y[i]==y[j]){valid=0;reason="duplicate_routes";}
                }
            }
            done+=bytes;
        }
        double relative=denominator?sqrt(error/denominator):0;
        if(valid&&(kind=='F'||kind=='M')) {
            if(!denominator&&!same){valid=0;reason="zero_norm_requires_exact";}
            else if(relative>1e-3){valid=0;reason="relative_l2";}
        }
        if(valid&&!same&&(kind=='B'||exact)){valid=0;reason="bit_exact";}
        if(!valid)failed=1;
        printf("{\"field\":\"%s\",\"pass\":%s,\"relative_l2\":%.17g,\"changes\":%" PRIu64 ",\"reason\":\"%s\"}\n",parts[1],valid?"true":"false",isfinite(relative)?relative:0,changes,reason);
    }
    if(ferror(stdin))return 2;
    for(int i=0;i<FILES;i++){if(cache[i].file)fclose(cache[i].file);free(cache[i].path);}
    return failed;
}
