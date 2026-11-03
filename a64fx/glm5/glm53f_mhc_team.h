#ifndef GLM53F_MHC_TEAM_H
#define GLM53F_MHC_TEAM_H
/* Included by glm53f_mhc_sve.h after its types and dot kernel. */
static inline void glm53f_mhc_pre_team(
        glm53f_mhc_scratch *scratch,const float *streams,
        const glm53f_mhc_site *site,const uint16_t *norm,int tokens,
        size_t stride,float *normalized){
    double sums[5][256];float inv[5],logits[5][GLM53F_MHC_MIX];
#pragma omp parallel
    {
        int tid=omp_get_thread_num(),nt=omp_get_num_threads();
        int begin=GLM53F_MHC_FLAT*tid/nt,end=GLM53F_MHC_FLAT*(tid+1)/nt;
        for(int t=0;t<tokens;t++){
            const float *s=streams+(size_t)t*GLM53F_MHC_FLAT;double ss=0;
            for(int i=begin;i<end;i++)ss+=(double)s[i]*s[i];
            sums[t][tid]=ss;
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for(int t=0;t<tokens;t++){
            double ss=0;for(int j=0;j<nt;j++)ss+=sums[t][j];
            inv[t]=1.0f/sqrtf((float)(ss/GLM53F_MHC_FLAT)+1e-5f);
        }
#pragma omp for collapse(2) schedule(static)
        for(int t=0;t<tokens;t++)for(int r=0;r<GLM53F_MHC_MIX;r++)
            logits[t][r]=glm53f_mhc_dot_bf16_sve(site->fn+(size_t)r*GLM53F_MHC_FLAT,
                streams+(size_t)t*GLM53F_MHC_FLAT,GLM53F_MHC_FLAT)*inv[t];
#pragma omp for schedule(static)
        for(int t=0;t<tokens;t++){
            glm53f_mhc_scratch *q=(glm53f_mhc_scratch*)((unsigned char*)scratch+(size_t)t*stride);
            float *z=logits[t];
            for(int k=0;k<4;k++){
                z[k]=glm53f_sigmoid(z[k]*site->scale[0]+site->base[k])+1e-6f;
                q->post[k]=2.0f*glm53f_sigmoid(z[4+k]*site->scale[1]+site->base[4+k]);
            }
            for(int i=0;i<16;i++)q->combine[i]=z[8+i]*site->scale[2]+site->base[8+i];
            glm53f_mhc_sinkhorn(q->combine,4,20,1e-6f);
        }
#pragma omp for collapse(2) schedule(static)
        for(int t=0;t<tokens;t++)for(int d=0;d<GLM53F_MHC_WIDTH;d++){
            glm53f_mhc_scratch *q=(glm53f_mhc_scratch*)((unsigned char*)scratch+(size_t)t*stride);
            const float *s=streams+(size_t)t*GLM53F_MHC_FLAT;float v=0;
            for(int k=0;k<4;k++){
                float x=s[(size_t)k*GLM53F_MHC_WIDTH+d];
                v+=logits[t][k]*x;q->residual[(size_t)k*GLM53F_MHC_WIDTH+d]=x;
            }
            q->collapsed[d]=v;
        }
        /* Use the identical normalization loop, including FP multiplication
         * order under the site's fast-math compiler. Splitting its element loop
         * into another worksharing region can change reassociation decisions. */
#pragma omp for schedule(static)
        for(int t=0;t<tokens;t++){
            glm53f_mhc_scratch *q=(glm53f_mhc_scratch*)((unsigned char*)scratch+(size_t)t*stride);
            glm53f_rmsnorm_bf16(q->normalized,q->collapsed,norm,GLM53F_MHC_WIDTH,1e-5f);
            memcpy(normalized+(size_t)t*GLM53F_MHC_WIDTH,q->normalized,sizeof(q->normalized));
        }
    }
}
#endif
