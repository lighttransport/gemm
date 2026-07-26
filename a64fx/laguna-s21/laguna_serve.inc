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
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define LSRV_MAX_PROMPT   262144        /* ids per request */
#define LSRV_HDR          12            /* control words ahead of the ids */
#define LSRV_REQ_BODY_MAX (64u<<20)

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
    if (listen(fd, 16) != 0) { perror("listen"); close(fd); return -1; }
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

/* ---- the serve loop ---- */
static int lsrv_run(laguna_model *m, laguna_scratch *sc, int port, int maxpos,
                    laguna_sampler defaults) {
    int listen_fd = -1;
    if (MyRank == 0) {
        listen_fd = lsrv_listen(port);
        if (listen_fd < 0) { fprintf(stderr,"FATAL: cannot listen on port %d\n", port); }
        else logmsg("serving HTTP on port %d (maxpos %d, %d ranks)\n", port, maxpos, N);
    }
    /* a listen failure on rank 0 must stop every rank, not just rank 0 */
    { lsrv_req probe = { .quit = (MyRank==0 && listen_fd<0) ? 1 : 0,
                         .n_prompt = 0, .max_new = 0, .smp = defaults };
      static int dummy_ids[1];
      lsrv_bcast(&probe, dummy_ids);
      if (probe.quit) return 2; }

    int *ids = laguna_xalloc((size_t)LSRV_MAX_PROMPT*sizeof(int), &g_mem_other, "serve prompt ids");
    int *gen = laguna_xalloc((size_t)LSRV_MAX_PROMPT*sizeof(int), &g_mem_other, "serve output ids");
    float *x = laguna_xalloc(LAGUNA_HIDDEN*sizeof(float), &g_mem_other, "serve activation");
    float *Xc = laguna_xalloc((size_t)LAGUNA_PCHUNK*LAGUNA_HIDDEN*sizeof(float),
                              &g_mem_other, "serve chunk activations");
    const int eos0 = 2, eos1 = 24;

    for (;;) {
        int cfd = -1; char *raw = NULL, *body = NULL;
        lsrv_req req = { .quit=0, .n_prompt=0, .max_new=0, .smp=defaults };
        int bad = 0, answered = 0; const char *bad_msg = NULL;

        if (MyRank == 0) {
            struct sockaddr_in ca; socklen_t cl = sizeof ca;
            cfd = accept(listen_fd, (struct sockaddr*)&ca, &cl);
            if (cfd < 0) { if (errno==EINTR) continue; perror("accept"); req.quit = 1; }
            else {
                int one=1; setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                raw = lsrv_read_request(cfd, &body);
                if (!raw) { bad=1; bad_msg="could not read request"; }
                else if (!strncmp(raw,"GET /health",11)) {
                    char b[256];
                    snprintf(b,sizeof b,"{\"status\":\"ok\",\"ranks\":%d,\"maxpos\":%d,\"layers\":%d}",
                             N, maxpos, m->n_layers);
                    lsrv_respond(cfd,200,"OK",b);
                    answered = 1;   /* still falls through to the broadcast below */
                }
                else if (!strncmp(raw,"POST /shutdown",14)) {
                    lsrv_respond(cfd,200,"OK","{\"status\":\"stopping\"}");
                    close(cfd); free(raw); raw=NULL; req.quit = 1;
                }
                else if (strncmp(raw,"POST /generate",14)) {
                    bad=1; bad_msg="unknown endpoint; use GET /health or POST /generate";
                }
                else {
                    int n = ljs_ids(body, ids, LSRV_MAX_PROMPT);
                    if (n <= 0) { bad=1; bad_msg="missing or malformed ids array"; }
                    else {
                        req.n_prompt = n;
                        req.max_new  = (int)ljs_num(body,"max_new",32);
                        req.smp.do_sample = (int)ljs_num(body,"sample",defaults.do_sample);
                        req.smp.temp  = (float)ljs_num(body,"temp", defaults.temp);
                        req.smp.top_k = (int)ljs_num(body,"top_k",defaults.top_k);
                        req.smp.top_p = (float)ljs_num(body,"top_p",defaults.top_p);
                        req.smp.min_p = (float)ljs_num(body,"min_p",defaults.min_p);
                        req.smp.rng   = (uint64_t)ljs_num(body,"seed",(double)defaults.rng);
                        if (req.max_new < 1) req.max_new = 1;
                        if (n + req.max_new > maxpos) {
                            bad=1; bad_msg="prompt + max_new exceeds the server's --maxpos";
                        }
                    }
                }
                if (bad) { req.n_prompt = 0; req.max_new = 0; }
            }
        }
        /* every rank learns the request (or the quit/no-op) together */
        if (lsrv_bcast(&req, ids) != 0) { req.quit = 1; }
        if (req.quit) {
            if (cfd >= 0) close(cfd);
            free(raw);
            break;
        }
        if (req.n_prompt == 0) {          /* health check, or rejected: no compute */
            if (MyRank == 0 && cfd >= 0) {
                if (!answered) {
                    char b[256];
                    snprintf(b,sizeof b,"{\"error\":\"%s\"}", bad_msg ? bad_msg : "bad request");
                    lsrv_respond(cfd,400,"Bad Request",b);
                }
                close(cfd);
            }
            free(raw);
            continue;
        }

        /* ---- identical compute on every rank ---- */
        double t0 = now_sec();
        int nprefill = req.n_prompt - 1, p = 0, nan_total = 0;
        for (; p + 1 <= nprefill; ) {
            int C = nprefill - p; if (C > LAGUNA_PCHUNK) C = LAGUNA_PCHUNK; if (C <= 0) break;
            for (int c=0;c<C;++c) embed_token(m, ids[p+c], Xc+(size_t)c*LAGUNA_HIDDEN);
            forward_prefill_chunk(m, sc, Xc, p, C, &g_aar);
            p += C;
        }
        for (; p < req.n_prompt; ++p) {
            embed_token(m, ids[p], x);
            forward_token(m, sc, x, p, &g_aar, p == req.n_prompt-1);
            nan_total += count_nan(x, LAGUNA_HIDDEN);
        }
        double t1 = now_sec();
        long disagree = 0;
        int next = pick_token(sc, &req.smp, &disagree), ngen = 0, hit_eos = 0;
        int pos = req.n_prompt;
        for (int g=0; g<req.max_new; ++g) {
            if (next==eos0 || next==eos1) { hit_eos = 1; break; }
            gen[ngen++] = next;
            embed_token(m, next, x);
            forward_token(m, sc, x, pos, &g_aar, 1);
            nan_total += count_nan(x, LAGUNA_HIDDEN);
            next = pick_token(sc, &req.smp, &disagree);
            ++pos;
        }
        double t2 = now_sec();

        if (MyRank == 0 && cfd >= 0) {
            size_t cap = (size_t)ngen*8 + 512;
            char *b = malloc(cap);
            if (b) {
                int k = snprintf(b, cap, "{\"n\":%d,\"stop\":\"%s\",\"nan\":%d,"
                    "\"lockstep_disagree\":%ld,\"prefill_tok_s\":%.2f,\"decode_tok_s\":%.2f,\"ids\":[",
                    ngen, hit_eos?"eos":"length", nan_total, disagree,
                    t1>t0 ? req.n_prompt/(t1-t0) : 0.0, t2>t1 ? ngen/(t2-t1) : 0.0);
                for (int i=0;i<ngen && k < (int)cap-16; ++i)
                    k += snprintf(b+k, cap-(size_t)k, i?",%d":"%d", gen[i]);
                k += snprintf(b+k, cap-(size_t)k, "]}");
                lsrv_respond(cfd, 200, "OK", b);
                free(b);
            } else lsrv_respond(cfd, 500, "Internal Server Error", "{\"error\":\"oom\"}");
            close(cfd);
            logmsg("served %d prompt + %d gen tok, prefill %.1f tok/s, decode %.1f tok/s%s\n",
                   req.n_prompt, ngen, t1>t0?req.n_prompt/(t1-t0):0.0,
                   t2>t1?ngen/(t2-t1):0.0, hit_eos?" (eos)":"");
        }
        free(raw);
    }
    if (MyRank == 0 && listen_fd >= 0) { close(listen_fd); logmsg("server stopped\n"); }
    return 0;
}
