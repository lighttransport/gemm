#include "ggml.h"
#include "ggml-cpu.h"
#include "../../common/glm53f_stage_artifact.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum { D = 64, I = 96, V = 257, HC = 4, HEADS = 1, KD = 16, TOKENS = 32, SEL = 5, EXPERTS = 3 };

static void put(glm53f_stage_artifact * a, const char * stage, const char * name,
                const float * x, size_t n, const size_t * shape, size_t rank) {
    if (a->manifest && glm53f_stage_write_f32(a, stage, name, x, n, shape, rank)) std::exit(1);
}

static void fill(std::vector<float> & x, float scale, int salt) {
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = scale * (float)((((int)i * 17 + salt * 29) % 101) - 50);
}

static float silu(float x) { return x / (1.0f + std::exp(-x)); }

static void custom_mla(const std::vector<float> & q, const std::vector<float> & k,
                       const std::vector<float> & v, std::vector<float> & out) {
    for (int t = 0; t < TOKENS; ++t) for (int h = 0; h < HEADS; ++h) {
        float score[SEL], prob[SEL], mx = -INFINITY, den = 0.0f;
        for (int p = 0; p < SEL; ++p) {
            float z = 0.0f;
            for (int d = 0; d < KD; ++d)
                z += q[(size_t)d + (size_t)KD * (h + HEADS*t)] * k[(size_t)d + (size_t)KD*p];
            score[p] = z / std::sqrt((float)KD); mx = std::max(mx, score[p]);
        }
        for (int p = 0; p < SEL; ++p) { prob[p] = std::exp(score[p] - mx); den += prob[p]; }
        for (int d = 0; d < KD; ++d) {
            float z = 0.0f;
            for (int p = 0; p < SEL; ++p) z += prob[p] / den * v[(size_t)d + (size_t)KD * (h + HEADS*p)];
            out[(size_t)d + (size_t)KD * (h + HEADS*t)] = z;
        }
    }
}

static void custom_moe(const std::vector<float> & x, const std::vector<float> & gate,
                       const std::vector<float> & up, const std::vector<float> & down,
                       const std::vector<float> & weights, std::vector<float> & out) {
    std::fill(out.begin(), out.end(), 0.0f);
    for (int e = 0; e < EXPERTS; ++e) for (int t = 0; t < TOKENS; ++t) {
        std::vector<float> a(I, 0.0f);
        for (int r = 0; r < I; ++r) {
            for (int d = 0; d < D; ++d) {
                float xv = x[(size_t)d + (size_t)D*t];
                a[r] += gate[(size_t)e*I*D + (size_t)r*D + (size_t)d] * xv;
            }
            float u = 0.0f;
            for (int d = 0; d < D; ++d) u += up[(size_t)e*I*D + (size_t)r*D + d] * x[(size_t)d + (size_t)D*t];
            a[r] = silu(a[r]) * u;
        }
        for (int d = 0; d < D; ++d) for (int r = 0; r < I; ++r)
            out[(size_t)d + (size_t)D*t] += weights[(size_t)e*TOKENS+t] *
                down[(size_t)e*D*I + (size_t)d*I + r] * a[r];
    }
}

static double compare(const float * a, const float * b, size_t n, float * max_abs) {
    double e = 0.0, r = 0.0; *max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) { double d = (double)a[i]-b[i]; e += d*d; r += (double)a[i]*a[i]; *max_abs = std::max(*max_abs, (float)std::fabs(d)); }
    return std::sqrt(e/(r+1e-30));
}

int main() {
    const int variant = [] {
        const char *s = std::getenv("GLM53F_TAIL_VARIANT");
        if (!s || !*s) return 0;
        char *end = nullptr;
        long v = std::strtol(s, &end, 10);
        return end && *end == '\0' && v >= 0 && v <= 100000 ? (int)v : 0;
    }();
    ggml_init_params ip = { 256ull*1024ull*1024ull, nullptr, false };
    ggml_context * ctx = ggml_init(ip); if (!ctx) return 1;
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KD, HEADS, TOKENS);
    ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, KD, SEL);
    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SEL, KD);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, TOKENS);
    ggml_tensor * norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    ggml_tensor * vocab = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, V);
    std::vector<float> qv(KD*HEADS*TOKENS), kv(KD*SEL), vv(KD*HEADS*SEL), xv(D*TOKENS), nw(D), vw(D*V);
    fill(qv, .01f, 1 + variant * 7); fill(kv, .012f, 2 + variant * 7);
    fill(vv, .014f, 3 + variant * 7); fill(xv, .01f, 4 + variant * 7);
    fill(nw, .01f, 5 + variant * 7); fill(vw, .003f, 6 + variant * 7);
    std::vector<float> vgg(vv.size()); for (int h=0; h<HEADS; ++h) for (int p=0; p<SEL; ++p) for (int d=0; d<KD; ++d) vgg[(size_t)d*SEL+p]=vv[(size_t)d+KD*(h+HEADS*p)];
    std::memcpy(q->data,qv.data(),qv.size()*4); std::memcpy(k->data,kv.data(),kv.size()*4); std::memcpy(v->data,vgg.data(),vgg.size()*4); std::memcpy(x->data,xv.data(),xv.size()*4); std::memcpy(norm_w->data,nw.data(),nw.size()*4); std::memcpy(vocab->data,vw.data(),vw.size()*4);
    ggml_tensor * scores = ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f/std::sqrt((float)KD));
    ggml_tensor * probs = ggml_soft_max(ctx, scores);
    ggml_tensor * mla = ggml_mul_mat(ctx, v, probs);
    ggml_cgraph * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, mla);
    if (ggml_graph_compute_with_ctx(ctx, graph, 48) != GGML_STATUS_SUCCESS) return 1;
    std::vector<float> cmla(KD*HEADS*TOKENS); custom_mla(qv,kv,vv,cmla);
    float ma; double mla_rel=compare((float*)mla->data,cmla.data(),cmla.size(),&ma);

    std::vector<float> gw((size_t)EXPERTS*I*D), uw((size_t)EXPERTS*I*D), dw((size_t)EXPERTS*D*I), rw((size_t)EXPERTS*TOKENS, 0.25f), cmoe(D*TOKENS);
    fill(gw,.002f,7 + variant * 7); fill(uw,.002f,8 + variant * 7); fill(dw,.002f,9 + variant * 7);
    std::vector<ggml_tensor*> es;
    ggml_tensor * moe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, TOKENS); std::memset(moe->data,0,ggml_nbytes(moe));
    for (int e=0;e<EXPERTS;++e) {
        ggml_tensor * eg=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,D,I), *eu=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,D,I), *ed=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,I,D);
        std::memcpy(eg->data,gw.data()+(size_t)e*I*D,I*D*4); std::memcpy(eu->data,uw.data()+(size_t)e*I*D,I*D*4); std::memcpy(ed->data,dw.data()+(size_t)e*D*I,D*I*4);
        ggml_tensor * ea=ggml_swiglu_clamp(ctx,ggml_mul_mat(ctx,eg,x),ggml_mul_mat(ctx,eu,x),10.0f);
        ggml_tensor * eo=ggml_mul_mat(ctx,ed,ea); moe=ggml_add(ctx,moe,ggml_scale(ctx,eo,0.25f));
    }
    ggml_build_forward_expand(graph,moe); if (ggml_graph_compute_with_ctx(ctx,graph,48)!=GGML_STATUS_SUCCESS)return 1;
    custom_moe(xv,gw,uw,dw,rw,cmoe); /* deterministic weights are equalized to 0.25 in GGML lane */
    float moe_max; double moe_rel=compare((float*)moe->data,cmoe.data(),cmoe.size(),&moe_max);
    ggml_tensor * merged = ggml_add(ctx, x, moe);
    ggml_tensor * mhc = ggml_repeat(ctx, ggml_reshape_3d(ctx, merged, D, 1, TOKENS), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HC, TOKENS));
    ggml_tensor * xn = ggml_rms_norm(ctx, merged, 1e-5f);
    xn = ggml_mul(ctx, xn, ggml_repeat(ctx, norm_w, xn));
    ggml_tensor * logits = ggml_mul_mat(ctx, vocab, xn);
    ggml_build_forward_expand(graph, mhc);
    ggml_build_forward_expand(graph, logits);
    if (ggml_graph_compute_with_ctx(ctx, graph, 48) != GGML_STATUS_SUCCESS) return 1;
    std::vector<float> base(D*TOKENS); for(size_t i=0;i<base.size();++i) base[i]=xv[i]+cmoe[i];
    std::vector<float> cmhc(D*HC*TOKENS), lmhc((float *)mhc->data,(float *)mhc->data+D*HC*TOKENS); for(int t=0;t<TOKENS;++t)for(int h=0;h<HC;++h)for(int d=0;d<D;++d) cmhc[(size_t)d+D*(h+HC*t)]=base[(size_t)d+D*t];
    float mhc_max; double mhc_rel=compare(lmhc.data(),cmhc.data(),cmhc.size(),&mhc_max);
    std::vector<float> clog(V*TOKENS), llog((float*)logits->data,(float*)logits->data+V*TOKENS); for(int t=0;t<TOKENS;++t){double ss=0;for(int j=0;j<D;++j)ss+=(double)base[(size_t)j+D*t]*base[(size_t)j+D*t];for(int z=0;z<V;++z){double s=0;for(int d=0;d<D;++d){float a=base[(size_t)d+D*t]*nw[d]/std::sqrt(ss/D+1e-5);s+=(double)vw[(size_t)z*D+d]*a;}clog[(size_t)z+V*t]=(float)s;}}
    int lt[ TOKENS ], ct[ TOKENS ]; for(int t=0;t<TOKENS;++t){lt[t]=ct[t]=0;for(int z=1;z<V;++z){if(llog[(size_t)z+V*t]>llog[(size_t)lt[t]+V*t])lt[t]=z;if(clog[(size_t)z+V*t]>clog[(size_t)ct[t]+V*t])ct[t]=z;}}
    float logit_max; double logit_rel=compare(llog.data(),clog.data(),llog.size(),&logit_max);
    std::vector<float> cnorm(D*TOKENS); for(int t=0;t<TOKENS;++t){double ss=0;for(int d=0;d<D;++d)ss+=(double)base[(size_t)d+D*t]*base[(size_t)d+D*t];for(int d=0;d<D;++d)cnorm[(size_t)d+D*t]=base[(size_t)d+D*t]*nw[d]/std::sqrt(ss/D+1e-5);}
    float norm_max; double norm_rel=compare((float *)xn->data,cnorm.data(),cnorm.size(),&norm_max);
    bool same_tokens=!std::memcmp(lt,ct,sizeof(lt));
    std::printf("TAIL_STAGE variant=%d mla_rel_l2=%.9g moe_rel_l2=%.9g mhc_rel_l2=%.9g norm_rel_l2=%.9g logits_rel_l2=%.9g token_count=%d first_token=%d/%d last_token=%d/%d exact_tokens=%s %s\n",variant,mla_rel,moe_rel,mhc_rel,norm_rel,logit_rel,TOKENS,lt[0],ct[0],lt[TOKENS-1],ct[TOKENS-1],same_tokens?"YES":"NO",mla_rel<2e-5&&moe_rel<1e-5&&mhc_rel<2e-5&&norm_rel<2e-5&&logit_rel<3e-3&&same_tokens?"PASS":"FAIL");
    glm53f_stage_artifact la={},ca={}; const char *lr=getenv("GLM53F_TAIL_LLAMA_OUT"),*cr=getenv("GLM53F_TAIL_CUSTOM_OUT");
    if(lr&&glm53f_stage_artifact_open(&la,lr,"llama_cpp","glm53f-tail-primitives","tail-stage",44,0,TOKENS))return 1; if(cr&&glm53f_stage_artifact_open(&ca,cr,"custom_adapter","glm53f-tail-primitives","tail-stage",44,0,TOKENS))return 1;
    const size_t as[]={TOKENS,HEADS,KD}, ms[]={TOKENS,D}, hs[]={TOKENS,HC,D}, ls[]={TOKENS,V};
    const size_t ns[]={TOKENS,D};
    if(la.manifest){put(&la,"mla","output",(float*)mla->data,cmla.size(),as,3);put(&la,"moe","output",(float*)moe->data,cmoe.size(),ms,2);put(&la,"mhc","output",lmhc.data(),lmhc.size(),hs,3);put(&la,"norm","output",(float*)xn->data,cnorm.size(),ns,2);put(&la,"logits","output",llog.data(),llog.size(),ls,2);} if(ca.manifest){put(&ca,"mla","output",cmla.data(),cmla.size(),as,3);put(&ca,"moe","output",cmoe.data(),cmoe.size(),ms,2);put(&ca,"mhc","output",cmhc.data(),cmhc.size(),hs,3);put(&ca,"norm","output",cnorm.data(),cnorm.size(),ns,2);put(&ca,"logits","output",clog.data(),clog.size(),ls,2);} if(la.manifest&&glm53f_stage_artifact_close(&la))return 1;if(ca.manifest&&glm53f_stage_artifact_close(&ca))return 1;
    ggml_free(ctx); return mla_rel<2e-5&&moe_rel<1e-5&&mhc_rel<2e-5&&norm_rel<2e-5&&logit_rel<3e-3&&same_tokens?0:1;
}
