#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../../common/glm53f_ref.h"

int main(void) {
    float s[6] = {0}, q[2] = {3, 4}, k[2] = {0, 1}, v[3] = {2, 4, 6}, out[3];
    float s2[6] = {0}, out2[3], work[3];
    float sv1[6] = {.1f,-.2f,.3f,.4f,-.5f,.6f};
    float sv2[6], ov1[3], ov2[3], ld[2] = {-.1f, -.7f};
    float gate[2] = {-2.0f, 3.0f}, dt[2] = {.5f, -.25f}, safe[2];
    float cs[6] = {0}, cs_checkpoint[6], ci[2] = {1, 2}, co[2];
    float co_replay[2], cx[4] = {1,2,3,4}, cg[4] = {0,0,0,0}, cgo[4];
    uint16_t cw[6] = {0x3f80,0x3f80,0x3f80, 0x3f80,0,0x3f80};
    int sd[3] = {11,12,13}, st0[4] = {21,22,23,24};
    int st1[4] = {11,22,23,24}, st3[4] = {11,12,13,24};
    float logits[6] = {-3, 2, 0, 4, -1, 1}, bias[6] = {0};
    float w[2], comb[4] = {1, 2, 3, 4};
    int ids[2];
    uint16_t one[4] = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
    uint16_t zero[4] = {0, 0, 0, 0};
    float nx[4] = {1, 2, 3, 4}, nr[4], ln[4];
    float tied[4] = {2, 3, 3, 1};
    int ti[2];
    float iq[2] = {1, 0}, iw[1] = {1};
    float ik[6] = {1, 0, 3, 0, 0, 2};
    float ig[6] = {0}, ia[4] = {0}, ip[2];
    int isel[3];
    float fe[2] = {3, 4}, fh[2] = {0, 5}, fused[2], fs[4];
    uint16_t fw[8] = {0x3f80, 0, 0, 0, 0, 0, 0, 0x3f80};
    uint16_t head[6] = {0x3f80, 0, 0, 0x3f80, 0xbf80, 0};
    float hn[2], best_logit;
    float mx[4] = {1, 2, 3, 4}, mb[8] = {0}, ms[3] = {0};
    uint16_t mf[32] = {0};
    float mc[2] = {0}, mp[2] = {0}, mm[4] = {0}, mr[4], mo[2] = {10, 20};
    float aq[2] = {.25f, -.5f}, az[6] = {.1f,.2f,.3f, -.2f,.4f,.1f};
    uint16_t aw[12] = {0x3f80,0,0, 0,0x3f80,0, 0,0,0x3f80, 0x3f80,0x3f80,0};
    int ai[2] = {0, 1}; float ad[1] = {0}, aa[1] = {0};
    glm53f_l2norm(q, 2, 1e-6f); glm53f_l2norm(k, 2, 1e-6f);
    glm53f_kda_step(s, q, k, v, 0.0f, 1.0f, 2, 3, out);
    glm53f_kda_step_streamed(s2, q, k, v, 0.0f, 1.0f, 2, 3, out2, work);
    if (fabsf(out[0] - 1.1313708f) > 2e-5f || fabsf(out[2] - 3.3941125f) > 2e-5f) return 1;
    if (fabsf(out2[0] - out[0]) > 2e-6f || fabsf(out2[2] - out[2]) > 2e-6f) return 5;
    memcpy(sv2, sv1, sizeof(sv1));
    glm53f_kda_step_vec(sv1, q, k, v, ld, .75f, 2, 3, ov1);
    glm53f_kda_step_vec_streamed(sv2, q, k, v, ld, .75f, 2, 3,
                                 ov2, work);
    for (int z = 0; z < 6; ++z)
        if (fabsf(sv1[z] - sv2[z]) > 2e-6f) return 14;
    for (int z = 0; z < 3; ++z)
        if (fabsf(ov1[z] - ov2[z]) > 2e-6f) return 15;
    /* A scalar decay would make both state rows shrink equally; require the
     * channel-wise oracle to preserve the official unequal decay. */
    if (fabsf(sv1[0] - sv1[3]) < 1e-3f) return 16;
    glm53f_kda_safe_log_decay(safe, gate, dt, logf(2.0f), -5.0f, 2);
    if (!(safe[0] < 0.0f && safe[0] > -5.0f &&
          safe[1] < safe[0] && safe[1] > -5.0f)) return 17;
    glm53f_causal_conv1d_silu_bf16(co, cs, ci, cw, 2, 3);
    if (fabsf(co[0] - glm53f_sigmoid(1.0f)) > 1e-6f ||
        fabsf(co[1] - 2.0f * glm53f_sigmoid(2.0f)) > 1e-6f) return 18;
    memcpy(cs_checkpoint, cs, sizeof(cs));
    ci[0] = -3.0f; ci[1] = .5f;
    glm53f_causal_conv1d_silu_bf16(co, cs, ci, cw, 2, 3);
    memcpy(cs, cs_checkpoint, sizeof(cs));
    glm53f_causal_conv1d_silu_bf16(co_replay, cs, ci, cw, 2, 3);
    if (memcmp(co, co_replay, sizeof(co)) != 0) return 19;
    glm53f_rmsnorm_gated_bf16(cgo, cx, cg, one, 2, 2, 1e-6f);
    if (fabsf(cgo[0] - .5f / sqrtf(2.5f + 1e-6f)) > 1e-6f ||
        fabsf(cgo[3] - 2.0f / sqrtf(12.5f + 1e-6f)) > 1e-6f) return 20;
    glm53f_spec_result sr = glm53f_spec_verify_greedy(sd, st0, 3);
    if (sr.accepted != 0 || sr.next_token != 21 || sr.committed_steps != 1) return 21;
    sr = glm53f_spec_verify_greedy(sd, st1, 3);
    if (sr.accepted != 1 || sr.next_token != 22 || sr.committed_steps != 2) return 22;
    sr = glm53f_spec_verify_greedy(sd, st3, 3);
    if (sr.accepted != 3 || sr.next_token != 24 || sr.committed_steps != 4) return 23;
    glm53f_router_topk(logits, bias, 6, 2, 2.5f, ids, w);
    if (ids[0] != 3 || ids[1] != 1 || fabsf(w[0] + w[1] - 2.5f) > 1e-6f) return 2;
    glm53f_mhc_sinkhorn(comb, 2, 20, 1e-6f);
    if (fabsf(comb[0] + comb[1] - 1.0f) > 1e-4f || fabsf(comb[0] + comb[2] - 1.0f) > 1e-4f) return 3;
    if (glm53f_cp_owner(25, 12) != 1 || glm53f_cp_slot(25, 12) != 2 || glm53f_cp_slots(26, 12) != 3) return 4;
    glm53f_rmsnorm_bf16(nr, nx, one, 4, 1e-5f);
    if (fabsf(nr[3] - 4.0f / sqrtf(7.5f + 1e-5f)) > 1e-6f) return 6;
    glm53f_layernorm_bf16(ln, nx, one, zero, 4, 1e-5f);
    if (fabsf(ln[0] + 1.3416355f) > 2e-5f || fabsf(ln[3] - 1.3416355f) > 2e-5f) return 7;
    glm53f_topk_stable(tied, 4, 2, ti);
    if (ti[0] != 1 || ti[1] != 2) return 8;
    if (glm53f_index_select_decode(ip, isel, iq, iw, ik, ig, ia,
                                   3, 2, 2, 1, 2) != 3 ||
        isel[0] != 0 || isel[1] != 1 || isel[2] != 2) return 9;
    glm53f_mtp_fuse_bf16(fused, fs, fe, fh, one, one, fw, 2, 1e-5f);
    if (fabsf(fused[0] - 3.0f / sqrtf(12.5f + 1e-5f)) > 1e-6f ||
        fabsf(fused[1] - 5.0f / sqrtf(12.5f + 1e-5f)) > 1e-6f) return 10;
    if (glm53f_vocab_argmax_bf16(fused, one, head, 2, 7, 3, 1e-5f,
                                  hn, &best_logit) != 8 ||
        !isfinite(best_logit)) return 11;
    memcpy(mr, mx, sizeof(mx));
    glm53f_mhc_pre(mc, mp, mm, mx, mf, mb, ms, 2, 2, 20, 1e-5f, 1e-6f);
    glm53f_mhc_post(mx, mr, mo, mp, mm, 2, 2);
    if (fabsf(mc[0] - 2.0f) > 1e-5f || fabsf(mc[1] - 3.0f) > 1e-5f ||
        fabsf(mx[0] - 12.0f) > 2e-5f || fabsf(mx[1] - 23.0f) > 2e-5f ||
        fabsf(mx[2] - 12.0f) > 2e-5f || fabsf(mx[3] - 23.0f) > 2e-5f) return 12;
    glm53f_mla_selected_bf16(ad, aq, az, aw, ai, 2, 1, 2, 1, 3);
    if (glm53f_mla_selected_absorbed_bf16(aa, aq, az, aw, ai, 2, 1, 2, 1, 3) ||
        fabsf(ad[0] - aa[0]) > 2e-7f) return 13;
    printf("GLM53F_REF kda=ok kda_vec=ok safe_gate=ok conv_state=ok gated_norm=ok rollback=ok spec_commit=ok router=ok mhc=ok cp=ok norm=ok topk=ok index=ok fusion=ok head=ok mhc_site=ok mla_absorb=ok\n");
    return 0;
}
