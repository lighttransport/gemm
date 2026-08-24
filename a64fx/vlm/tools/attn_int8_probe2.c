// Precision probe for INT8 attention, PER-ROW scales (Q per-query, K per-key,
// V per-row, attn per-query). Loads L00_qkv.vlmd (384 x 3072 f32).
//   fcc -Nclang -O3 -std=c11 -D_GNU_SOURCE tools/attn_int8_probe2.c -lm -o /tmp/ap2 \
//       && /tmp/ap2 /tmp/dump_1/L00_qkv.vlmd
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static float *load_f32(const char *path, int *n) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "open fail %s\n", path); exit(1); }
    char magic[4]; uint32_t version, dtype, ndim, dims[8]; char name[64];
    if (fread(magic,1,4,f)!=4 || fread(&version,4,1,f)!=1 || fread(&dtype,4,1,f)!=1 ||
        fread(&ndim,4,1,f)!=1 || fread(dims,4,8,f)!=8 || fread(name,64,1,f)!=1) { fprintf(stderr,"hdr fail\n"); exit(1); }
    if (memcmp(magic,"VLMD",4)) { fprintf(stderr,"bad magic\n"); exit(1); }
    size_t ne = 1; for (int i=0;i<ndim;i++) ne *= dims[i];
    float *d = malloc(ne*sizeof(float)); if (fread(d,4,ne,f)!=ne) { fprintf(stderr,"data fail\n"); exit(1); }
    fclose(f); *n = (int)ne; printf("loaded %s: %u x %u\n", path, dims[0], dims[1]);
    return d;
}
static int8_t q8(float x, float s){ int a=(int)lroundf(x/s); return (int8_t)(a<-127?-127:(a>127?127:a)); }
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s qkv.vlmd\n", argv[0]); return 1; }
    int ne; float *qkv = load_f32(argv[1], &ne);
    const int np = 384, dim = 1024, nh = 16, hd = 64;
    float scale = 1.0f/sqrtf((float)hd);
    double max_out_rel=0; int worst_h=-1; float acc_out_rel=0; float max_qk=0,acc_qk=0;
    for (int h=0; h<nh; h++) {
        float *Q=malloc((size_t)np*hd*4),*K=malloc((size_t)np*hd*4),*V=malloc((size_t)np*hd*4);
        for (int p=0;p<np;p++) for (int d=0;d<hd;d++){
            Q[(size_t)p*hd+d]=qkv[(size_t)p*3072+h*64+d];
            K[(size_t)p*hd+d]=qkv[(size_t)p*3072+1024+h*64+d];
            V[(size_t)p*hd+d]=qkv[(size_t)p*3072+2048+h*64+d];
        }
        // per-row scales
        float *sq=malloc(np*4),*sk=malloc(np*4),*sv=malloc(np*4);
        for (int p=0;p<np;p++){ float a=0,b=0,c=0; for (int d=0;d<hd;d++){a=fmaxf(a,fabsf(Q[(size_t)p*hd+d]));b=fmaxf(b,fabsf(K[(size_t)p*hd+d]));c=fmaxf(c,fabsf(V[(size_t)p*hd+d]));}
            sq[p]=a/127.0f; if(sq[p]<1e-9f)sq[p]=1e-9f; sk[p]=b/127.0f; if(sk[p]<1e-9f)sk[p]=1e-9f; sv[p]=c/127.0f; if(sv[p]<1e-9f)sv[p]=1e-9f; }
        int8_t *Qi=malloc(np*hd),*Ki=malloc(np*hd),*Vi=malloc(np*hd);
        for (int p=0;p<np;p++) for (int d=0;d<hd;d++){ Qi[(size_t)p*hd+d]=q8(Q[(size_t)p*hd+d],sq[p]); Ki[(size_t)p*hd+d]=q8(K[(size_t)p*hd+d],sk[p]); Vi[(size_t)p*hd+d]=q8(V[(size_t)p*hd+d],sv[p]); }
        // QK^T
        float *Sfp=malloc((size_t)np*np*4), *Si=malloc((size_t)np*np*4);
        for (int q=0;q<np;q++) for (int k=0;k<np;k++){
            int32_t s=0; for (int d=0;d<hd;d++) s += (int)Qi[(size_t)q*hd+d]*(int)Ki[(size_t)k*hd+d];
            Si[(size_t)q*np+k]=(float)s*sq[q]*sk[k]*scale;
            float acc=0; for (int d=0;d<hd;d++) acc += Q[(size_t)q*hd+d]*K[(size_t)k*hd+d];
            Sfp[(size_t)q*np+k]=acc*scale;
        }
        // softmax both
        float *Wfp=malloc((size_t)np*np*4), *Wi=malloc((size_t)np*np*4);
        for (int q=0;q<np;q++){
            float m=1e30f; for (int k=0;k<np;k++){ if(Sfp[q*np+k]<m)m=Sfp[q*np+k]; if(Si[q*np+k]<m)m=Si[q*np+k]; }
            float sp=0,si=0; for (int k=0;k<np;k++){ sp+=expf(Sfp[q*np+k]-m); si+=expf(Si[q*np+k]-m); }
            for (int k=0;k<np;k++){ Wfp[q*np+k]=expf(Sfp[q*np+k]-m)/sp; Wi[q*np+k]=expf(Si[q*np+k]-m)/si; }
        }
        // AV (int8 attn per-query, int8 V per-HEAD so sv factors out of the sum)
        float *Ofp=malloc((size_t)np*hd*4), *Oi=malloc((size_t)np*hd*4);
        float vhm=0; for (int i=0;i<np*hd;i++) vhm=fmaxf(vhm,fabsf(V[i])); float svh=vhm/127.0f; if(svh<1e-9f)svh=1e-9f;
        for (int i=0;i<np*hd;i++) Vi[i]=q8(V[i],svh);
        for (int q=0;q<np;q++){
            float wmx=0; for (int k=0;k<np;k++) wmx=fmaxf(wmx,Wi[q*np+k]);
            float sA=wmx/127.0f; if(sA<1e-9f)sA=1e-9f;
            for (int d=0;d<hd;d++){
                float acc=0; for (int k=0;k<np;k++) acc += Wfp[q*np+k]*V[(size_t)k*hd+d];
                int32_t s=0; for (int k=0;k<np;k++) s += (int)lroundf(Wi[q*np+k]/sA)*(int)Vi[(size_t)k*hd+d];
                Oi[(size_t)q*hd+d]=(float)s*sA*svh;
                Ofp[(size_t)q*hd+d]=acc;
            }
        }
        float oden=0,onum=0,qknum=0; for (int i=0;i<np*hd;i++){ oden+=Ofp[i]*Ofp[i]; float e=Oi[i]-Ofp[i]; onum+=e*e; }
        // QK^T-only int8 output = Wi (int8 scores softmaxed) x V (fp32 AV)
        for (int q=0;q<np;q++) for (int d=0;d<hd;d++){ float qi=0; for (int k=0;k<np;k++) qi += Wi[(size_t)q*np+k]*V[(size_t)k*hd+d];
            float e=qi-Ofp[(size_t)q*hd+d]; qknum+=e*e; }
        float out_rel=sqrtf(onum/(oden+1e-20f));
        float qk_only_rel=sqrtf(qknum/(oden+1e-20f));
        acc_out_rel+=out_rel; acc_qk+=qk_only_rel;
        if (out_rel>max_out_rel){ max_out_rel=out_rel; worst_h=h; }
        if (qk_only_rel>max_qk) max_qk=qk_only_rel;
        if (h<4) printf("  head %d: full_int8_out=%.4f%%  qk_only_int8_out=%.4f%%\n", h, out_rel*100, qk_only_rel*100);
        free(Q);free(K);free(V);free(Qi);free(Ki);free(Vi);free(sq);free(sk);free(sv);free(Sfp);free(Si);free(Wfp);free(Wi);free(Ofp);free(Oi);
    }
    printf("\nfull int8 : worst attn_out_rel=%.4f%% (head %d), avg=%.4f%%\n", max_out_rel*100, worst_h, acc_out_rel/nh*100);
    printf("qk-only int8: worst attn_out_rel=%.4f%%\n", max_qk*100);
    free(qkv); return 0;
}
