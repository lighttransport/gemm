/* Standalone MHR inference. Build from cpu/sam3d_body or cuda/sam3d_body.
 * NumPy is only an interchange format: no Python/framework runtime is linked. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <time.h>
#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"
#define SAM3D_BODY_MHR_IMPLEMENTATION
#include "sam3d_body_mhr.h"
#ifdef MHR_HAVE_AVX2
#include "../ryzen/gemm_avx2.h"
#endif
#ifdef MHR_CUDA
#include "../cuda/sam3d_body/mhr_decode_cuda.h"
#endif

/* Read only the bounded, little-endian C-order float32 matrix contract. */
static float *read_matrix(const char *path, int width, int *rows)
{
    FILE *f = fopen(path, "rb");
    unsigned char pre[10]; char hdr[4097]; float *data = NULL;
    if (!f) return NULL;
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY\x01\x00", 8)) goto done;
    unsigned len = pre[8] | ((unsigned)pre[9] << 8);
    if (!len || len > 4096 || fread(hdr, 1, len, f) != len) goto done;
    hdr[len] = 0;
    if (!strstr(hdr, "'descr': '<f4'") || !strstr(hdr, "'fortran_order': False")) goto done;
    char *p = strstr(hdr, "'shape': (");
    long r, c; char *end;
    if (!p) goto done;
    p += strlen("'shape': ("); errno = 0; r = strtol(p, &end, 10);
    if (errno || end == p || *end != ',' || r < 1 || r > 65536) goto done;
    p = end + 1; errno = 0; c = strtol(p, &end, 10);
    if (errno || end == p || c != width) goto done;
    while (*end == ' ') end++;
    if (*end != ')') goto done;
    size_t count = (size_t)r * width;
    data = malloc(count * sizeof(float));
    if (!data) goto done;
    if (fread(data, sizeof(float), count, f) != count || fgetc(f) != EOF) {
        free(data); data = NULL; goto done;
    }
    for (size_t i = 0; i < count; i++) if (!isfinite(data[i])) {
        free(data); data = NULL; goto done;
    }
    *rows = (int)r;
done:
    fclose(f);
    return data;
}

static FILE *open_array(const char *dir, const char *name, int b, int n, int d)
{
    char path[4096], hdr[256];
    if (snprintf(path, sizeof(path), "%s/%s.npy", dir, name) >= (int)sizeof(path)) return NULL;
    int len = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d), }", b,n,d);
    while ((10 + len + 1) % 64) hdr[len++] = ' ';
    hdr[len++] = '\n';
    unsigned char pre[10] = {0x93,'N','U','M','P','Y',1,0,(unsigned char)len,(unsigned char)(len>>8)};
    FILE *f = fopen(path, "wb");
    if (!f) return NULL;
    if (fwrite(pre, 1, 10, f) != 10 || fwrite(hdr, 1, len, f) != (size_t)len) {
        fclose(f); return NULL;
    }
    return f;
}

static int write_finite(FILE *f, const float *v, size_t n)
{
    for (size_t i=0; i<n; i++) if (!isfinite(v[i])) return -1;
    return fwrite(v, sizeof(float), n, f) == n ? 0 : -1;
}

/* Validate fixed model shapes before any native pointer arithmetic. */
static int validate_assets(sam3d_body_mhr_assets *a)
{
    st_context *st = a->_st;
    struct { const char *name; int nd; uint64_t dims[3]; } specs[] = {
        {"blend_shape.shape_vectors",3,{45,18439,3}},
        {"blend_shape.base_shape",2,{18439,3}},
        {"face_expressions.shape_vectors",3,{72,18439,3}},
        {"parameter_transform",2,{889,249}},
        {"skeleton.joint_translation_offsets",2,{127,3}},
        {"skeleton.joint_prerotations",2,{127,4}},
        {"skeleton.pmi",2,{2,266}}, {"skeleton.joint_parents",1,{127}},
        {"lbs.inverse_bind_pose",2,{127,8}},
        {"lbs.skin_indices_flattened",1,{51337}},
        {"lbs.skin_weights_flattened",1,{51337}},
        {"lbs.vert_indices_flattened",1,{51337}},
        {"pose_correctives.sparse_indices",2,{2,53136}},
        {"pose_correctives.sparse_weight",1,{53136}},
        {"pose_correctives.linear_weight",2,{55317,3000}},
    };
    for (size_t s=0; s<sizeof(specs)/sizeof(specs[0]); s++) {
        int i = safetensors_find(st, specs[s].name);
        if (i < 0 || safetensors_ndims(st,i) != specs[s].nd) return -1;
        size_t bytes = safetensors_dtype_size(safetensors_dtype(st,i));
        const uint64_t *dims = safetensors_shape(st,i);
        for (int d=0; d<specs[s].nd; d++) {
            if (dims[d] != specs[s].dims[d]) return -1;
            bytes *= dims[d];
        }
        if (bytes != safetensors_nbytes(st,i)) return -1;
    }
    int total = 0;
    for (int i=0; i<4; i++) {
        if (a->pmi_buffer_sizes[i] < 0 || a->pmi_buffer_sizes[i] > 266) return -1;
        total += a->pmi_buffer_sizes[i];
    }
    if (total != 266) return -1;
    const int64_t *pmi=a->pmi.data, *spi=a->pc_sparse_indices.data, *vi=a->vert_indices_flat.data;
    const int32_t *si=a->skin_indices_flat.data, *parents=a->joint_parents.data;
    for (int i=0; i<532; i++) if (pmi[i]<0 || pmi[i]>=127) return -1;
    for (int i=0; i<127; i++) if (parents[i]<-1 || parents[i]>=i) return -1;
    for (int i=0; i<53136; i++)
        if (spi[i]<0 || spi[i]>=3000 || spi[i+53136]<0 || spi[i+53136]>=750) return -1;
    for (int i=0; i<51337; i++)
        if (si[i]<0 || si[i]>=127 || vi[i]<0 || vi[i]>=18439) return -1;
    return 0;
}

#ifdef MHR_HAVE_AVX2
static int cpu_pc(void *user, const float *h, float *out)
{
    const sam3d_body_mhr_assets *a = user;
    /* W[55317,3000] @ h[3000,1] avoids transposing the 633 MiB matrix. */
    memset(out, 0, 55317*sizeof(float));
    sgemm_avx2(55317,1,3000,1,a->pc_linear_weight.data,3000,h,1,0,out,1);
    return 0;
}
#endif

static int integer(const char *s, int min, int max)
{
    char *end; errno=0; long n=strtol(s,&end,10);
    return errno || *end || end==s || n<min || n>max ? -1 : (int)n;
}

int main(int argc, char **argv)
{
    const char *dir=NULL,*params_path=NULL,*shape_path=NULL,*face_path=NULL,*outdir=NULL;
    const char *backend="cpu";
    int device=0,threads=1,skeleton_only=0,rc=1;
    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i],"--skeleton-only")) { skeleton_only=1; continue; }
        if (i+1>=argc) goto usage;
        const char *flag=argv[i], *value=argv[++i];
        if (!strcmp(flag,"--mhr-assets")) dir=value;
        else if (!strcmp(flag,"--params")) params_path=value;
        else if (!strcmp(flag,"--shape")) shape_path=value;
        else if (!strcmp(flag,"--face")) face_path=value;
        else if (!strcmp(flag,"--output-dir")) outdir=value;
        else if (!strcmp(flag,"--backend")) backend=value;
        else if (!strcmp(flag,"--device")) device=integer(value,0,1024);
        else if (!strcmp(flag,"--threads")) threads=integer(value,1,256);
        else goto usage;
    }
    if (!dir || !params_path || !shape_path || !outdir || device<0 || threads<1 ||
        (strcmp(backend,"cpu") && strcmp(backend,"cuda"))) goto usage;
#ifndef MHR_CUDA
    if (!strcmp(backend,"cuda")) { fprintf(stderr,"Use the CUDA mhr_decode build\n"); return 2; }
#endif
    int b=0,sb=0,fb=0;
    float *params=read_matrix(params_path,204,&b), *shape=read_matrix(shape_path,45,&sb);
    float *face=face_path ? read_matrix(face_path,72,&fb) : NULL;
    sam3d_body_mhr_assets *a=NULL;
    FILE *verts_file=NULL,*state_file=NULL;
    float *verts=NULL,*scratch=NULL;
#ifdef MHR_CUDA
    mhr_cuda cuda={0};
#endif
    if (!params || !shape || sb!=b || (face_path && (!face || fb!=b))) {
        fprintf(stderr,"Expected finite C-order float32 matrices with matching batch sizes\n"); goto done;
    }
    char sft[4096],json[4096];
    if (snprintf(sft,sizeof(sft),"%s/sam3d_body_mhr_jit.safetensors",dir)>=(int)sizeof(sft) ||
        snprintf(json,sizeof(json),"%s/sam3d_body_mhr_jit.json",dir)>=(int)sizeof(json)) goto done;
    FILE *sidecar=fopen(json,"rb");
    if (!sidecar) { fprintf(stderr,"Missing MHR JSON sidecar\n"); goto done; }
    fclose(sidecar);
    a=sam3d_body_mhr_load(sft,json);
    if (!a || validate_assets(a)) { fprintf(stderr,"Invalid MHR assets/shapes/indices\n"); goto done; }
    const char *math="portable_c";
#ifdef MHR_HAVE_AVX2
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        a->pc_matvec_user=a; a->pc_matvec_fn=cpu_pc; math="avx2_gemm";
    }
#endif
    size_t resident=0;
#ifdef MHR_CUDA
    if (!strcmp(backend,"cuda") && !skeleton_only) {
        if (mhr_cuda_init(&cuda,a,device)) goto done;
        resident=cuda.resident_bytes; math="cuda_own_fp32";
    }
#endif
    if (skeleton_only) math="portable_c_skeleton";
    if (!skeleton_only) {
        verts=malloc(55317*sizeof(float));
        scratch=malloc((889+127*8*2+55317*3)*sizeof(float));
        if (!verts || !scratch) goto done;
        verts_file=open_array(outdir,"vertices",b,18439,3);
        if (!verts_file) goto done;
    }
    state_file=open_array(outdir,"skeleton",b,127,8);
    if (!state_file) goto done;
    struct timespec start,end; clock_gettime(CLOCK_MONOTONIC,&start);
    for (int i=0; i<b; i++) {
        float state[127*8]; int r;
        if (skeleton_only) {
            float jp[889],local[127*8];
            r=sam3d_body_mhr_parameter_transform(a,params+(size_t)i*204,1,threads,jp);
            if (!r) r=sam3d_body_mhr_joint_params_to_local_skel(a,jp,1,local);
            if (!r) r=sam3d_body_mhr_local_to_global_skel(a,local,1,state);
        } else {
            r=sam3d_body_mhr_forward(a,params+(size_t)i*204,shape+(size_t)i*45,
                    face ? face+(size_t)i*72 : NULL,1,1,threads,scratch,verts,state);
        }
        if (r || write_finite(state_file,state,127*8) ||
            (verts_file && write_finite(verts_file,verts,55317))) {
            fprintf(stderr,"Decode/write failed at sample %d\n",i); goto done;
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&end);
    int close_error=fclose(state_file); state_file=NULL;
    if (verts_file) { close_error |= fclose(verts_file); verts_file=NULL; }
    if (close_error) goto done;
    printf("{\"backend\":\"%s\",\"math\":\"%s\",\"frames\":%d,\"resident_bytes\":%zu,"
           "\"decode_seconds\":%.6f,\"units\":\"cm\"}\n",
           skeleton_only ? "cpu" : backend,math,b,resident,
           (end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9);
    rc=0;
done:
    if (state_file) fclose(state_file);
    if (verts_file) fclose(verts_file);
#ifdef MHR_CUDA
    mhr_cuda_free(&cuda);
#endif
    sam3d_body_mhr_free(a);
    free(params); free(shape); free(face); free(verts); free(scratch);
    return rc;
usage:
    fprintf(stderr,"mhr_decode --mhr-assets DIR --params POSES.npy --shape SHAPE.npy "
                   "[--face FACE.npy] --output-dir DIR [--backend cpu|cuda] "
                   "[--device N] [--threads N] [--skeleton-only]\n");
    return 2;
}
