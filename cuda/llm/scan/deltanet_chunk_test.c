/* Standalone CPU prototype: validate the chunked-parallel gated-delta-rule scan
   against the exact sequential recurrence used by batch_deltanet_scan_f32.

   Sequential recurrence (per head, state S is [d_state x d_state], S[r][c]):
     S       = gamma_t * S            (decay BEFORE read)
     sk[r]   = sum_c S[r][c] * k[c]
     delta[r]= beta_t * (v[r] - sk[r])
     S[r][c]+= delta[r] * k[c]        (rank-1 update)
     o[r]    = sum_c S[r][c] * q[c]   (q already pre-scaled by 1/sqrt(d_state))

   Chunked form (UT transform), per chunk of L tokens with incoming state S_in:
     Gamma_t = prod_{i=0..t} gamma_i              (cumulative decay, inclusive)
     P(j->t) = Gamma_t / Gamma_j                  (decay from after j to t)
     D[t][j] = beta_t * P(j->t) * (k_j . k_t)     for j<t  (strict lower-tri)
     B[t]    = beta_t * (v_t - Gamma_t * (S_in @ k_t))
     Delta   = (I + D)^{-1} B                      (forward substitution, L x d_state)
     M[t][j] = P(j->t) * (k_j . q_t)              for j<=t (lower-tri incl diag)
     O[t]    = Gamma_t * (S_in @ q_t) + sum_{j<=t} M[t][j] * Delta[j]
     S_out[r][c] = Gamma_{L-1}*S_in[r][c] + sum_j P(j->L-1)*Delta[j][r]*k_j[c]

   Build: gcc -O2 -o deltanet_chunk_test deltanet_chunk_test.c -lm
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define DS 128          /* d_state */
#define NT 512          /* n_tokens */

static float frand(unsigned int *s) {
    *s = *s * 1664525u + 1013904223u;
    return ((float)(*s >> 8) / (float)(1u << 24)) * 2.0f - 1.0f; /* [-1,1) */
}

/* ---- exact sequential reference ---- */
static void seq_scan(const float *Q, const float *K, const float *V,
                     const float *gamma, const float *beta,
                     float *Out, float *S /* [DS*DS], in/out */) {
    for (int t = 0; t < NT; t++) {
        const float *q = Q + (size_t)t*DS, *k = K + (size_t)t*DS, *v = V + (size_t)t*DS;
        float g = gamma[t], b = beta[t];
        for (int i = 0; i < DS*DS; i++) S[i] *= g;
        float sk[DS];
        for (int r = 0; r < DS; r++) { float s=0; for (int c=0;c<DS;c++) s+=S[r*DS+c]*k[c]; sk[r]=s; }
        float delta[DS];
        for (int r = 0; r < DS; r++) delta[r] = b*(v[r]-sk[r]);
        for (int r = 0; r < DS; r++) { float dr=delta[r]; for (int c=0;c<DS;c++) S[r*DS+c]+=dr*k[c]; }
        float *o = Out + (size_t)t*DS;
        for (int r = 0; r < DS; r++) { float s=0; for (int c=0;c<DS;c++) s+=S[r*DS+c]*q[c]; o[r]=s; }
    }
}

/* ---- chunked form ---- */
static void chunk_scan(const float *Q, const float *K, const float *V,
                       const float *gamma, const float *beta,
                       float *Out, float *S /* [DS*DS], in/out */, int L) {
    for (int c0 = 0; c0 < NT; c0 += L) {
        int Lc = (c0 + L <= NT) ? L : (NT - c0);
        const float *Qc = Q + (size_t)c0*DS, *Kc = K + (size_t)c0*DS, *Vc = V + (size_t)c0*DS;
        const float *gc = gamma + c0, *bc = beta + c0;
        /* cumulative decay Gamma_t (inclusive) */
        double Gam[1024];
        double acc = 1.0;
        for (int t = 0; t < Lc; t++) { acc *= gc[t]; Gam[t] = acc; }
        /* S_in @ k_t  and S_in @ q_t  -> [Lc][DS] */
        float *Sk = malloc((size_t)Lc*DS*sizeof(float));
        float *Sq = malloc((size_t)Lc*DS*sizeof(float));
        for (int t = 0; t < Lc; t++) {
            const float *k = Kc + (size_t)t*DS, *q = Qc + (size_t)t*DS;
            for (int r = 0; r < DS; r++) {
                float ak=0, aq=0;
                const float *Sr = S + (size_t)r*DS;
                for (int cc = 0; cc < DS; cc++) { ak += Sr[cc]*k[cc]; aq += Sr[cc]*q[cc]; }
                Sk[t*DS+r]=ak; Sq[t*DS+r]=aq;
            }
        }
        /* B[t][r] = beta_t (v_t[r] - Gamma_t * Sk[t][r]) */
        float *Delta = malloc((size_t)Lc*DS*sizeof(float));
        for (int t = 0; t < Lc; t++)
            for (int r = 0; r < DS; r++)
                Delta[t*DS+r] = bc[t]*(Vc[t*DS+r] - (float)Gam[t]*Sk[t*DS+r]);
        /* Forward substitution: (I + D) Delta = B, D[t][j]=beta_t P(j->t)(k_j.k_t), j<t.
           Delta[t] -= sum_{j<t} D[t][j] Delta[j]  (since diag = 1) */
        for (int t = 1; t < Lc; t++) {
            const float *kt = Kc + (size_t)t*DS;
            for (int j = 0; j < t; j++) {
                const float *kj = Kc + (size_t)j*DS;
                float kk=0; for (int cc=0;cc<DS;cc++) kk+=kj[cc]*kt[cc];
                float coef = bc[t]*(float)(Gam[t]/Gam[j])*kk;
                float *Dt = Delta + (size_t)t*DS; const float *Dj = Delta + (size_t)j*DS;
                for (int r = 0; r < DS; r++) Dt[r] -= coef*Dj[r];
            }
        }
        /* Output: O[t] = Gamma_t * Sq[t] + sum_{j<=t} P(j->t)(k_j.q_t) Delta[j] */
        for (int t = 0; t < Lc; t++) {
            const float *qt = Qc + (size_t)t*DS;
            float *o = Out + (size_t)(c0+t)*DS;
            for (int r = 0; r < DS; r++) o[r] = (float)Gam[t]*Sq[t*DS+r];
            for (int j = 0; j <= t; j++) {
                const float *kj = Kc + (size_t)j*DS;
                float kq=0; for (int cc=0;cc<DS;cc++) kq+=kj[cc]*qt[cc];
                float coef = (float)(Gam[t]/Gam[j])*kq;
                const float *Dj = Delta + (size_t)j*DS;
                for (int r = 0; r < DS; r++) o[r] += coef*Dj[r];
            }
        }
        /* State carry: S_out[r][c] = Gamma_{Lc-1} S_in[r][c] + sum_j P(j->Lc-1) Delta[j][r] k_j[c] */
        double GL = Gam[Lc-1];
        float *Snew = malloc((size_t)DS*DS*sizeof(float));
        for (int r = 0; r < DS; r++)
            for (int cc = 0; cc < DS; cc++)
                Snew[r*DS+cc] = (float)GL * S[r*DS+cc];
        for (int j = 0; j < Lc; j++) {
            const float *kj = Kc + (size_t)j*DS;
            float pj = (float)(GL/Gam[j]);
            const float *Dj = Delta + (size_t)j*DS;
            for (int r = 0; r < DS; r++) {
                float dr = pj*Dj[r];
                float *Sr = Snew + (size_t)r*DS;
                for (int cc = 0; cc < DS; cc++) Sr[cc] += dr*kj[cc];
            }
        }
        memcpy(S, Snew, (size_t)DS*DS*sizeof(float));
        free(Sk); free(Sq); free(Delta); free(Snew);
    }
}

static double rel_l2(const float *a, const float *b, size_t n) {
    double num=0, den=0;
    for (size_t i=0;i<n;i++){ double d=(double)a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; }
    return sqrt(num/(den+1e-30));
}

int main(int argc, char **argv) {
    int L = argc>1 ? atoi(argv[1]) : 64;
    unsigned int s = 12345;
    float *Q=malloc((size_t)NT*DS*4), *K=malloc((size_t)NT*DS*4), *V=malloc((size_t)NT*DS*4);
    float *gamma=malloc(NT*4), *beta=malloc(NT*4);
    float scale = 1.0f/sqrtf((float)DS);
    for (int t=0;t<NT;t++){
        /* L2-normalize q,k per token (as the real pipeline does), pre-scale q */
        float q[DS],k[DS]; float nq=0,nk=0;
        for (int i=0;i<DS;i++){ q[i]=frand(&s); k[i]=frand(&s); nq+=q[i]*q[i]; nk+=k[i]*k[i]; }
        float iq=1.0f/sqrtf(nq+1e-6f), ik=1.0f/sqrtf(nk+1e-6f);
        for (int i=0;i<DS;i++){ Q[t*DS+i]=q[i]*iq*scale; K[t*DS+i]=k[i]*ik; V[t*DS+i]=frand(&s); }
        /* decay near 1 (alpha=softplus*a, a<0): gamma in ~[0.90,1.0] */
        gamma[t] = 0.90f + 0.099f*((frand(&s)+1)*0.5f);
        beta[t]  = (frand(&s)+1)*0.5f; /* sigmoid output in (0,1) */
    }
    float *Oseq=calloc((size_t)NT*DS,4), *Ochunk=calloc((size_t)NT*DS,4);
    float *Sseq=calloc((size_t)DS*DS,4), *Schunk=calloc((size_t)DS*DS,4);
    seq_scan(Q,K,V,gamma,beta,Oseq,Sseq);
    chunk_scan(Q,K,V,gamma,beta,Ochunk,Schunk,L);
    printf("chunk L=%d  output rel_L2=%.3e   state rel_L2=%.3e\n",
           L, rel_l2(Ochunk,Oseq,(size_t)NT*DS), rel_l2(Schunk,Sseq,(size_t)DS*DS));
    /* sample */
    printf("  o[0][0..3] seq=%.5f %.5f %.5f  chunk=%.5f %.5f %.5f\n",
           Oseq[0],Oseq[1],Oseq[2],Ochunk[0],Ochunk[1],Ochunk[2]);
    printf("  o[511][0..3] seq=%.5f %.5f %.5f  chunk=%.5f %.5f %.5f\n",
           Oseq[511*DS],Oseq[511*DS+1],Oseq[511*DS+2],Ochunk[511*DS],Ochunk[511*DS+1],Ochunk[511*DS+2]);
    return 0;
}
