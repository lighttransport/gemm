/* laguna_serve.inc - HTTP/1.1 interface for the 12-node EP runner.
 * Included at the bottom of laguna_tofu.inc (one translation unit).
 *
 * Only rank 0 touches sockets.  Every rank must run the identical sequence of
 * forward passes or the KV caches diverge, so the parsed request is broadcast to
 * the whole EP group before any compute starts -- the same discipline pick_token
 * uses for the chosen token.  The broadcast rides the existing allreduce:
 * rank 0 contributes the value and everyone else contributes zero, so the sum IS
 * rank 0's value.  Ints are carried as floats, which is exact below 2^24 (token
 * ids reach 100352); the 64-bit seed is split into four 16-bit limbs.
 *
 * INVARIANT: every accepted connection reaches lsrv_bcast exactly once, even a
 * health check or a rejected request.  The other ranks sit blocked inside that
 * collective whenever they are idle, so skipping it would strand them there until
 * their allreduce timed out.
 *
 * Endpoints
 *   GET  /health    -> {"status":"ok",...}
 *   POST /generate  -> {"ids":[...], "max_new":N, "sample":bool, "temp":f,
 *                       "top_k":i, "top_p":f, "min_p":f, "seed":u}
 *                      response {"ids":[...],"n":N,"stop":"eos"|"length",...}
 *   POST /shutdown  -> stops every rank cleanly
 *
 * The runner has no tokenizer, so the API is ids-in/ids-out; tools/laguna_cli.py
 * does the chat templating and tokenisation on the client side.
 */
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define LSRV_MAX_PROMPT   262144        /* ids per request */
#define LSRV_HDR          12            /* control words ahead of the ids */
#define LSRV_REQ_BODY_MAX (64u<<20)
#define LSRV_MAX_SLOTS    16            /* concurrent sequences (--max-batch) */

/* ---- tiny JSON scanners (only the shapes this API accepts) ---- */
static const char *ljs_find(const char *s, const char *key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(s, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p && *p != ':') ++p;
    return *p ? p + 1 : NULL;
}
static double ljs_num(const char *s, const char *key, double dflt) {
    const char *p = ljs_find(s, key);
    if (!p) return dflt;
    while (*p == ' ' || *p == '\t') ++p;
    if (!strncmp(p, "true", 4))  return 1;
    if (!strncmp(p, "false", 5)) return 0;
    char *end; double v = strtod(p, &end);
    return end == p ? dflt : v;
}
/* parse "ids":[1,2,3] into out; returns count, or -1 if absent/too long */
static int ljs_ids(const char *s, int *out, int cap) {
    const char *p = ljs_find(s, "ids");
    if (!p) return -1;
    while (*p && *p != '[') ++p;
    if (!*p) return -1;
    ++p;
    int n = 0;
    for (;;) {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        if (*p == ']' || !*p) break;
        char *end; long v = strtol(p, &end, 10);
        if (end == p) return -1;
        if (n >= cap) return -1;
        out[n++] = (int)v;
        p = end;
    }
    return n;
}

/* ---- broadcast of the request to the whole EP group ---- */
typedef struct {
    int quit, n_prompt, max_new;
    laguna_sampler smp;
} lsrv_req;

static int lsrv_bcast(lsrv_req *r, int *ids) {
    if (N <= 1) return 0;
    static float *buf = NULL;
    if (!buf) buf = laguna_xalloc((size_t)(LSRV_HDR + LSRV_MAX_PROMPT) * sizeof(float),
                                  &g_mem_other, "serve broadcast buffer");
    int n = r->n_prompt < 0 ? 0 : r->n_prompt;
    if (MyRank == 0) {
        buf[0]=(float)r->quit;        buf[1]=(float)n;
        buf[2]=(float)r->max_new;     buf[3]=(float)r->smp.do_sample;
        buf[4]=r->smp.temp;           buf[5]=(float)r->smp.top_k;
        buf[6]=r->smp.top_p;          buf[7]=r->smp.min_p;
        for (int i=0;i<4;i++) buf[8+i] = (float)((r->smp.rng >> (16*i)) & 0xFFFFu);
        for (int i=0;i<n;i++) buf[LSRV_HDR+i] = (float)ids[i];
    } else {
        memset(buf, 0, (size_t)(LSRV_HDR + n) * sizeof(float));
    }
    /* header first: everyone needs n_prompt before the ids length is known */
    ep_ar(&g_comm, buf, LSRV_HDR);
    r->quit         = (int)(buf[0]+0.5f);
    r->n_prompt     = (int)(buf[1]+0.5f);
    r->max_new      = (int)(buf[2]+0.5f);
    r->smp.do_sample= (int)(buf[3]+0.5f);
    r->smp.temp     = buf[4];
    r->smp.top_k    = (int)(buf[5]+0.5f);
    r->smp.top_p    = buf[6];
    r->smp.min_p    = buf[7];
    r->smp.rng = 0;
    for (int i=0;i<4;i++) r->smp.rng |= ((uint64_t)(uint32_t)(buf[8+i]+0.5f) & 0xFFFFu) << (16*i);
    if (r->quit) return 0;
    if (r->n_prompt < 0 || r->n_prompt > LSRV_MAX_PROMPT) return -1;
    if (MyRank != 0) memset(buf+LSRV_HDR, 0, (size_t)r->n_prompt*sizeof(float));
    ep_ar(&g_comm, buf+LSRV_HDR, r->n_prompt);
    for (int i=0;i<r->n_prompt;i++) ids[i] = (int)(buf[LSRV_HDR+i]+0.5f);
    return 0;
}

/* ---- rank-0 socket helpers ---- */
static int lsrv_listen(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }
    int one = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a,0,sizeof a);
    a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons((uint16_t)port);
    if (bind(fd,(struct sockaddr*)&a,sizeof a) != 0) { perror("bind"); close(fd); return -1; }
    if (listen(fd, 64) != 0) { perror("listen"); close(fd); return -1; }
    return fd;
}
static void lsrv_write_all(int fd, const char *p, size_t n) {
    while (n) { ssize_t w = write(fd, p, n); if (w <= 0) return; p += w; n -= (size_t)w; }
}
static void lsrv_respond(int fd, int code, const char *status, const char *body) {
    char hdr[256];
    int hn = snprintf(hdr, sizeof hdr,
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n", code, status, strlen(body));
    lsrv_write_all(fd, hdr, (size_t)hn);
    lsrv_write_all(fd, body, strlen(body));
}
/* read headers + body; returns malloc'd request text (caller frees) or NULL */
static char *lsrv_read_request(int fd, char **body_out) {
    size_t cap = 65536, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    char *hdr_end = NULL;
    for (;;) {
        if (len + 4096 > cap) {
            if (cap >= LSRV_REQ_BODY_MAX) break;
            cap *= 2; char *nb = realloc(buf, cap); if (!nb) { free(buf); return NULL; } buf = nb;
        }
        ssize_t r = read(fd, buf + len, cap - len - 1);
        if (r <= 0) break;
        len += (size_t)r; buf[len] = 0;
        if (!hdr_end) hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) {
            const char *cl = strcasestr(buf, "Content-Length:");
            long need = cl ? strtol(cl + 15, NULL, 10) : 0;
            size_t have = len - (size_t)(hdr_end + 4 - buf);
            if ((long)have >= need) break;
        }
    }
    if (!hdr_end) hdr_end = strstr(buf, "\r\n\r\n");
    *body_out = hdr_end ? hdr_end + 4 : buf + len;
    return buf;
}

#include "laguna_batch.inc"

/* ---- the serve loop ---- */
/* Parse one HTTP request on rank 0 into a free slot.  Returns 1 if the slot was
 * filled and should be admitted, 0 if the connection was already answered
 * (health / error) and needs no compute. */
static int lsrv_accept_into(lbat_slot *s, int cfd, char *raw, char *body,
                            int maxpos, int nranks, int nlayers,
                            laguna_sampler defaults) {
    if (!strncmp(raw,"GET /health",11)) {
        char b[192];
        snprintf(b,sizeof b,"{\"status\":\"ok\",\"ranks\":%d,\"maxpos\":%d,\"layers\":%d}",
                 nranks, maxpos, nlayers);
        lsrv_respond(cfd,200,"OK",b); return 0;
    }
    if (strncmp(raw,"POST /generate",14)) {
        lsrv_respond(cfd,400,"Bad Request",
                     "{\"error\":\"unknown endpoint; use GET /health or POST /generate\"}");
        return 0;
    }
    int n = ljs_ids(body, s->ids, maxpos);
    if (n <= 0) {
        lsrv_respond(cfd,400,"Bad Request","{\"error\":\"missing or malformed ids array\"}");
        return 0;
    }
    s->smp = defaults;
    s->max_new       = (int)ljs_num(body,"max_new",32);
    s->smp.do_sample = (int)ljs_num(body,"sample",defaults.do_sample);
    s->smp.temp      = (float)ljs_num(body,"temp", defaults.temp);
    s->smp.top_k     = (int)ljs_num(body,"top_k",defaults.top_k);
    s->smp.top_p     = (float)ljs_num(body,"top_p",defaults.top_p);
    s->smp.min_p     = (float)ljs_num(body,"min_p",defaults.min_p);
    s->smp.rng       = (uint64_t)ljs_num(body,"seed",(double)defaults.rng);
    if (s->max_new < 1) s->max_new = 1;
    if (n + s->max_new > maxpos) {
        lsrv_respond(cfd,400,"Bad Request",
                     "{\"error\":\"prompt + max_new exceeds the server's --maxpos\"}");
        return 0;
    }
    s->n_prompt = n; s->fd = cfd; s->pos = 0; s->ngen = 0; s->hit_eos = 0;
    return 1;
}

static void lsrv_finish(lbat_slot *s, double t_now, long disagree) {
    if (s->fd < 0) { s->active = 0; return; }
    size_t cap = (size_t)s->ngen*8 + 512;
    char *b = malloc(cap);
    if (b) {
        double pf = s->t_prefill_done > s->t_admit ? s->n_prompt/(s->t_prefill_done-s->t_admit) : 0.0;
        double dt = t_now > s->t_prefill_done ? s->ngen/(t_now-s->t_prefill_done) : 0.0;
        int k = snprintf(b, cap, "{\"n\":%d,\"stop\":\"%s\",\"lockstep_disagree\":%ld,"
                "\"prefill_tok_s\":%.2f,\"decode_tok_s\":%.2f,\"ids\":[",
                s->ngen, s->hit_eos?"eos":"length", disagree, pf, dt);
        for (int i=0;i<s->ngen && k < (int)cap-16; ++i)
            k += snprintf(b+k, cap-(size_t)k, i?",%d":"%d", s->gen[i]);
        k += snprintf(b+k, cap-(size_t)k, "]}");
        lsrv_respond(s->fd, 200, "OK", b);
        free(b);
    } else lsrv_respond(s->fd, 500, "Internal Server Error", "{\"error\":\"oom\"}");
    close(s->fd); s->fd = -1; s->active = 0;
}

static int lsrv_run(laguna_model *m, laguna_scratch *sc, int port, int maxpos,
                    laguna_sampler defaults, int max_batch) {
    int listen_fd = -1;
    if (MyRank == 0) {
        listen_fd = lsrv_listen(port);
        if (listen_fd < 0) fprintf(stderr,"FATAL: cannot listen on port %d\n", port);
        else logmsg("serving HTTP on port %d (maxpos %d, %d ranks, max batch %d)\n",
                    port, maxpos, N, max_batch);
    }
    { lsrv_req probe = { .quit = (MyRank==0 && listen_fd<0) ? 1 : 0,
                         .n_prompt = 0, .max_new = 0, .smp = defaults };
      static int dummy_ids[1];
      lsrv_bcast(&probe, dummy_ids);
      if (probe.quit) return 2; }

    lbat_state B; lbat_init(&B, max_batch, maxpos);
    float *x  = laguna_xalloc(LAGUNA_HIDDEN*sizeof(float), &g_mem_other, "serve activation");
    float *Xc = laguna_xalloc((size_t)LAGUNA_PCHUNK*LAGUNA_HIDDEN*sizeof(float),
                              &g_mem_other, "serve chunk activations");
    const int eos0 = 2, eos1 = 24;
    long disagree = 0;
    int shutting_down = 0;

    for (;;) {
        lbat_step st; memset(&st, 0, sizeof st);

        if (MyRank == 0) {
            /* Admit whatever is waiting into free slots.  Block on accept() only
             * when the batch is empty, so an idle server does not spin and a busy
             * one never stalls a step waiting for a new client. */
            int nactive = 0;
            for (int i=0;i<B.nslot;i++) if (B.slot[i].active) nactive++;
            for (;;) {
                int free_slot = -1;
                for (int i=0;i<B.nslot;i++) if (!B.slot[i].active) { free_slot = i; break; }
                if (free_slot < 0) break;
                int fl = fcntl(listen_fd, F_GETFL, 0);
                fcntl(listen_fd, F_SETFL, nactive ? (fl|O_NONBLOCK) : (fl&~O_NONBLOCK));
                int cfd = accept(listen_fd, NULL, NULL);
                if (cfd < 0) break;                       /* nothing pending */
                int one=1; setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                char *body=NULL; char *raw = lsrv_read_request(cfd, &body);
                if (!raw) { close(cfd); continue; }
                if (!strncmp(raw,"POST /shutdown",14)) {
                    lsrv_respond(cfd,200,"OK","{\"status\":\"stopping\"}");
                    close(cfd); free(raw); shutting_down = 1; break;
                }
                lbat_slot *s = &B.slot[free_slot];
                int admit = lsrv_accept_into(s, cfd, raw, body, maxpos, N, m->n_layers, defaults);
                free(raw);
                if (!admit) { close(cfd); continue; }
                s->active = 1; s->t_admit = now_sec();
                st.adm_slot[st.n_admit] = free_slot;
                st.adm_len[st.n_admit]  = s->n_prompt;
                st.n_admit++;
                nactive++;
            }
            /* everything already running steps this round */
            for (int i=0;i<B.nslot;i++) {
                if (!B.slot[i].active) continue;
                int newly = 0;
                for (int j=0;j<st.n_admit;j++) if (st.adm_slot[j]==i) newly=1;
                if (newly) continue;
                st.act_slot[st.n_active] = i;
                st.act_pos[st.n_active]  = B.slot[i].pos;
                st.n_active++;
            }
            st.quit = shutting_down && st.n_active==0 && st.n_admit==0;
        }

        if (lbat_bcast(&B, &st) != 0) break;
        if (st.quit) break;
        if (st.n_active == 0 && st.n_admit == 0) continue;   /* nothing to do */

        /* --- admitted sequences: prefill each into its own slot's KV --- */
        for (int i=0;i<st.n_admit;i++) {
            int slot = st.adm_slot[i];
            lbat_slot *s = &B.slot[slot];
            if (MyRank != 0) { s->active = 1; s->fd = -1; s->ngen = 0; s->hit_eos = 0;
                               s->max_new = maxpos; s->smp = defaults; }
            int nxt = lbat_prefill(m, sc, s, slot, Xc, x, &disagree);
            s->t_prefill_done = now_sec();
            if (nxt==eos0 || nxt==eos1) { s->hit_eos = 1; }
            else { s->gen[s->ngen++] = nxt; embed_token(m, nxt, B.X + (size_t)slot*LAGUNA_HIDDEN); }
            /* a freshly prefilled slot joins the decode batch below */
            if (!s->hit_eos && s->ngen < s->max_new) {
                st.act_slot[st.n_active] = slot;
                st.act_pos[st.n_active]  = s->pos;
                st.n_active++;
            }
        }

        /* --- one decode step over every active slot ---
         * K==1 goes through forward_token, which fuses q/k/v/g into a single OMP
         * region and uses matvec rather than a 1-column GEMM; measured 27.8 vs
         * 15.1 tok/s, so a lone request must not pay for the batching machinery. */
        if (st.n_active == 1) {
            int slot = st.act_slot[0];
            lbat_slot *s = &B.slot[slot];
            memcpy(x, B.X + (size_t)slot*LAGUNA_HIDDEN, LAGUNA_HIDDEN*sizeof(float));
            forward_token(m, sc, x, slot, st.act_pos[0], &g_aar, 1);
            int nxt = pick_token(sc, &s->smp, &disagree);
            s->pos++;
            if (nxt==eos0 || nxt==eos1) s->hit_eos = 1;
            else if (s->ngen < s->max_new) {
                s->gen[s->ngen++] = nxt;
                embed_token(m, nxt, B.X + (size_t)slot*LAGUNA_HIDDEN);
            }
        } else if (st.n_active > 1) {
            for (int i=0;i<st.n_active;i++) {
                B.seqs[i] = st.act_slot[i];
                B.poss[i] = st.act_pos[i];
                if (i != st.act_slot[i])   /* pack activations contiguously for the batch */
                    memcpy(B.X + (size_t)i*LAGUNA_HIDDEN,
                           B.X + (size_t)st.act_slot[i]*LAGUNA_HIDDEN,
                           LAGUNA_HIDDEN*sizeof(float));
            }
            forward_decode_batch(m, sc, B.X, B.seqs, B.poss, st.n_active, &g_aar);
            /* all K choices in ONE allreduce, and each slot keeps its own sampler
             * state so a seeded request is unaffected by who it batches with */
            { laguna_sampler smps[LAGUNA_MAX_BATCH]; int nxt[LAGUNA_MAX_BATCH];
              for (int i=0;i<st.n_active;i++) smps[i] = B.slot[st.act_slot[i]].smp;
              pick_tokens_batch(sc, smps, nxt, st.n_active, &disagree);
              for (int i=0;i<st.n_active;i++) {
                lbat_slot *s = &B.slot[st.act_slot[i]];
                s->smp = smps[i];                       /* advanced RNG state */
                s->pos++;
                if (nxt[i]==eos0 || nxt[i]==eos1) s->hit_eos = 1;
                else if (s->ngen < s->max_new) {
                    s->gen[s->ngen++] = nxt[i];
                    embed_token(m, nxt[i], B.X + (size_t)i*LAGUNA_HIDDEN);
                }
              } }
            /* unpack activations back to slot order for the next step */
            for (int i=st.n_active-1;i>=0;i--)
                if (i != st.act_slot[i])
                    memcpy(B.X + (size_t)st.act_slot[i]*LAGUNA_HIDDEN,
                           B.X + (size_t)i*LAGUNA_HIDDEN, LAGUNA_HIDDEN*sizeof(float));
        }

        /* --- retire finished slots (rank 0 answers; all ranks free the slot) --- */
        for (int i=0;i<B.nslot;i++) {
            lbat_slot *s = &B.slot[i];
            if (!s->active) continue;
            if (s->hit_eos || s->ngen >= s->max_new) {
                if (MyRank == 0) {
                    logmsg("slot %d done: %d prompt + %d gen tok%s\n",
                           i, s->n_prompt, s->ngen, s->hit_eos?" (eos)":"");
                    lsrv_finish(s, now_sec(), disagree);
                } else { s->active = 0; s->fd = -1; }
            }
        }
    }
    if (MyRank == 0 && listen_fd >= 0) { close(listen_fd); logmsg("server stopped\n"); }
    return 0;
}
