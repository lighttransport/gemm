/* laguna_batch.inc - continuous batching for the HTTP serve path.
 * Included from laguna_serve.inc.
 *
 * WHY: decode reads ~13 GB of weights per token and is bandwidth-bound, so K
 * sequences stepped together share one pass over the weights.  Throughput should
 * scale close to K x until it turns compute-bound; a single stream leaves most of
 * that on the floor.
 *
 * LOCKSTEP: every rank must run the identical sequence of forward passes or the
 * KV caches diverge.  Rank 0 alone decides the batch composition -- which slots
 * are active, which are admitted this step, and at what position each sits -- and
 * broadcasts that decision before any compute.  Ranks never infer composition
 * from their own state.  The broadcast rides the existing allreduce exactly as
 * lsrv_bcast does (rank 0 contributes, everyone else contributes zero).
 *
 * Each slot owns one sequence's KV cache (slot s uses seq index s), so slots are
 * independent and a finished slot can be refilled without disturbing the others.
 */

#define LBAT_HDR 4              /* quit, n_active, n_admit, spare */
#define LBAT_PER_ACTIVE 2       /* slot, pos */
#define LBAT_PER_ADMIT  2       /* slot, n_prompt */

typedef struct {
    int active, fd;
    int n_prompt, n_cached, pos, ngen, max_new, hit_eos;
    int stream, client_gone;
    int *ids, *gen;
    laguna_sampler smp;
    double t_admit, t_prefill_done;
} lbat_slot;

typedef struct {
    lbat_slot *slot;
    int nslot;
    float *bcast;               /* header + per-slot records + prompt payload */
    size_t bcast_cap;
    float *X;                   /* [nslot * hidden] batched activations */
    int *seqs, *poss;
} lbat_state;

static void lbat_init(lbat_state *B, int nslot, int maxpos) {
    B->nslot = nslot;
    B->slot = laguna_xalloc((size_t)nslot*sizeof(lbat_slot), &g_mem_other, "batch slots");
    memset(B->slot, 0, (size_t)nslot*sizeof(lbat_slot));
    for (int i=0;i<nslot;i++) {
        B->slot[i].ids = laguna_xalloc((size_t)maxpos*sizeof(int), &g_mem_other, "slot prompt ids");
        B->slot[i].gen = laguna_xalloc((size_t)maxpos*sizeof(int), &g_mem_other, "slot output ids");
    }
    B->bcast_cap = (size_t)(LBAT_HDR + nslot*(LBAT_PER_ACTIVE+LBAT_PER_ADMIT)) + (size_t)maxpos;
    B->bcast = laguna_xalloc(B->bcast_cap*sizeof(float), &g_mem_other, "batch broadcast");
    B->X    = laguna_xalloc((size_t)nslot*LAGUNA_HIDDEN*sizeof(float), &g_mem_other, "batch activations");
    B->seqs = laguna_xalloc((size_t)nslot*sizeof(int), &g_mem_other, "batch seq ids");
    B->poss = laguna_xalloc((size_t)nslot*sizeof(int), &g_mem_other, "batch positions");
}

/* One step's composition, as decided by rank 0 and agreed by everyone. */
typedef struct {
    int quit, n_active, n_admit;
    int act_slot[LSRV_MAX_SLOTS], act_pos[LSRV_MAX_SLOTS];
    int adm_slot[LSRV_MAX_SLOTS], adm_len[LSRV_MAX_SLOTS];
} lbat_step;

/* Broadcast the step descriptor and any admitted prompts.  Two allreduces: a
 * fixed header so everyone learns the counts, then one variable payload. */
static int lbat_bcast(lbat_state *B, lbat_step *st) {
    if (N <= 1) return 0;
    float *b = B->bcast;
    if (MyRank == 0) {
        b[0]=(float)st->quit; b[1]=(float)st->n_active; b[2]=(float)st->n_admit; b[3]=0.0f;
    } else {
        memset(b, 0, LBAT_HDR*sizeof(float));
    }
    ep_ar(&g_comm, b, LBAT_HDR);
    st->quit     = (int)(b[0]+0.5f);
    st->n_active = (int)(b[1]+0.5f);
    st->n_admit  = (int)(b[2]+0.5f);
    if (st->quit) return 0;
    if (st->n_active < 0 || st->n_active > B->nslot ||
        st->n_admit  < 0 || st->n_admit  > B->nslot) return -1;

    size_t np = (size_t)st->n_active*LBAT_PER_ACTIVE + (size_t)st->n_admit*LBAT_PER_ADMIT;
    size_t total_ids = 0;
    if (MyRank == 0) {
        size_t o = 0;
        for (int i=0;i<st->n_active;i++){ b[o++]=(float)st->act_slot[i]; b[o++]=(float)st->act_pos[i]; }
        for (int i=0;i<st->n_admit;i++){ b[o++]=(float)st->adm_slot[i]; b[o++]=(float)st->adm_len[i]; }
        for (int i=0;i<st->n_admit;i++) {
            const lbat_slot *s=&B->slot[st->adm_slot[i]];
            for (int j=0;j<st->adm_len[i];j++) b[np+total_ids+j]=(float)s->ids[j];
            total_ids += (size_t)st->adm_len[i];
        }
    } else {
        memset(b, 0, np*sizeof(float));
    }
    ep_ar(&g_comm, b, (int)np);           /* records first: lengths are in them */
    { size_t o=0;
      for (int i=0;i<st->n_active;i++){ st->act_slot[i]=(int)(b[o]+0.5f); st->act_pos[i]=(int)(b[o+1]+0.5f); o+=2; }
      for (int i=0;i<st->n_admit;i++){ st->adm_slot[i]=(int)(b[o]+0.5f); st->adm_len[i]=(int)(b[o+1]+0.5f); o+=2; } }
    total_ids = 0;
    for (int i=0;i<st->n_admit;i++) {
        if (st->adm_len[i] < 0) return -1;
        total_ids += (size_t)st->adm_len[i];
    }
    if (np + total_ids > B->bcast_cap) return -1;
    if (total_ids) {
        if (MyRank != 0) memset(b+np, 0, total_ids*sizeof(float));
        ep_ar(&g_comm, b+np, (int)total_ids);
        size_t o=0;
        for (int i=0;i<st->n_admit;i++) {
            lbat_slot *s=&B->slot[st->adm_slot[i]];
            for (int j=0;j<st->adm_len[i];j++) s->ids[j]=(int)(b[np+o+j]+0.5f);
            s->n_prompt=st->adm_len[i];
            o += (size_t)st->adm_len[i];
        }
    }
    return 0;
}

/* Prefill an admitted sequence into its own slot's KV, then take its first token.
 * Identical on every rank: the ids arrived through the broadcast. */
static int lbat_prefill(const laguna_model *m, laguna_scratch *sc, lbat_slot *s, int slot,
                        float *Xc, float *x, long *disagree,
                        const laguna_prompt_cache *pc) {
    int nprefill = s->n_prompt - 1, p = 0;
    if(pc){
        char err[256];
        if(lpc_restore(pc,(laguna_model *)m,slot,err,sizeof err)){
            logmsg("FATAL: prompt cache restore: %s\n",err);exit(1);
        }
        p=pc->n_tokens;
    }
    s->n_cached=p;
    for (; p + 1 <= nprefill; ) {
        int C = nprefill - p; if (C > LAGUNA_PCHUNK) C = LAGUNA_PCHUNK; if (C <= 0) break;
        for (int c=0;c<C;++c) embed_token(m, s->ids[p+c], Xc+(size_t)c*LAGUNA_HIDDEN);
        forward_prefill_chunk(m, sc, Xc, slot, p, C, &g_aar);
        p += C;
        if (MyRank == 0 && s->stream && !s->client_gone) {
            char ev[160];
            int n = snprintf(ev, sizeof ev,
                             "{\"event\":\"prefill\",\"done\":%d,\"total\":%d}\n",
                             p, nprefill);
            if (lsrv_write_all(s->fd, ev, (size_t)n) != 0) s->client_gone = 1;
        }
    }
    for (; p < s->n_prompt; ++p) {
        embed_token(m, s->ids[p], x);
        forward_token(m, sc, x, slot, p, &g_aar, p == s->n_prompt-1);
    }
    s->pos = s->n_prompt;
    return pick_token(sc, &s->smp, disagree);
}
