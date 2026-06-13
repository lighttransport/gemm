/* Per-stage timing of sam3d_body_mhr_forward on real assets, to find the
 * residual CPU MHR bottleneck now that pose_correctives' dense matvec is on GPU.
 * Synthetic inputs (timing only, not correctness). 32-thread OpenMP like prod.
 *
 * Build: gcc -O3 -march=native -fopenmp -I. -I.. -o bench_mhr_stages \
 *          bench_mhr_stages.c ../rocew.c -ldl -lm   (rocew unused but links)
 */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define SAM3D_BODY_MHR_IMPLEMENTATION
#include "../../common/sam3d_body_mhr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0 + t.tv_nsec*1e-6; }

int main(int argc, char **argv){
    const char *dir = argc>1 ? argv[1] : "/mnt/disk1/models/sam3d-body/safetensors";
    char sft[512], js[512];
    snprintf(sft,sizeof sft,"%s/sam3d_body_mhr_jit.safetensors",dir);
    snprintf(js, sizeof js, "%s/sam3d_body_mhr_jit.json",dir);
    sam3d_body_mhr_assets *a = sam3d_body_mhr_load(sft, js);
    if(!a){ fprintf(stderr,"load failed: %s\n",sft); return 2; }

    int nthr = 1;
#if defined(_OPENMP)
    nthr = omp_get_max_threads();
#endif
    const int J=S3DM_N_JOINTS, V=S3DM_N_VERTS;
    float *mp   = calloc(204,sizeof(float));
    float *shape= calloc(S3DM_N_SHAPE,sizeof(float));
    float *face = calloc(S3DM_N_FACE,sizeof(float));
    float *jp   = calloc((size_t)889,sizeof(float));
    float *lskel= calloc((size_t)J*8,sizeof(float));
    float *gskel= calloc((size_t)J*8,sizeof(float));
    float *vrest= calloc((size_t)V*3,sizeof(float));
    float *vface= calloc((size_t)V*3,sizeof(float));
    float *vpc  = calloc((size_t)V*3,sizeof(float));
    float *vskin= calloc((size_t)V*3,sizeof(float));
    unsigned s=7u;
    for(int i=0;i<204;i++){s=s*1103515245u+12345u; mp[i]=((float)((s>>16)&0x7fff)/16384.f-1.f)*0.1f;}
    for(int i=0;i<S3DM_N_SHAPE;i++){s=s*1103515245u+12345u; shape[i]=((float)((s>>16)&0x7fff)/16384.f-1.f);}
    for(int i=0;i<S3DM_N_FACE;i++){s=s*1103515245u+12345u; face[i]=((float)((s>>16)&0x7fff)/16384.f-1.f);}

    int reps = 10;
    /* warm caches once */
    sam3d_body_mhr_parameter_transform(a, mp, 1, nthr, jp);
    sam3d_body_mhr_joint_params_to_local_skel(a, jp, 1, lskel);
    sam3d_body_mhr_local_to_global_skel(a, lskel, 1, gskel);
    sam3d_body_mhr_blend_shape(a, shape, 1, nthr, vrest);
    sam3d_body_mhr_face_expressions(a, face, 1, nthr, vface);
    sam3d_body_mhr_pose_correctives(a, jp, 1, nthr, vpc);
    sam3d_body_mhr_skin_points(a, gskel, vrest, 1, vskin);

    double t;
    #define TIME(label, call) do { double t0=now_ms(); \
        for(int r=0;r<reps;r++){ call; } \
        t=(now_ms()-t0)/reps; printf("  %-26s %7.3f ms\n", label, t); } while(0)

    printf("MHR per-stage (B=1, %d threads, avg of %d, warm):\n", nthr, reps);
    TIME("parameter_transform",  sam3d_body_mhr_parameter_transform(a,mp,1,nthr,jp));
    TIME("joint->local_skel",    sam3d_body_mhr_joint_params_to_local_skel(a,jp,1,lskel));
    TIME("local->global (walker)",sam3d_body_mhr_local_to_global_skel(a,lskel,1,gskel));
    TIME("blend_shape",          sam3d_body_mhr_blend_shape(a,shape,1,nthr,vrest));
    TIME("face_expressions",     sam3d_body_mhr_face_expressions(a,face,1,nthr,vface));
    TIME("pose_correctives(CPU)",sam3d_body_mhr_pose_correctives(a,jp,1,nthr,vpc));
    TIME("skin_points",          sam3d_body_mhr_skin_points(a,gskel,vrest,1,vskin));
    sam3d_body_mhr_free(a);
    return 0;
}
