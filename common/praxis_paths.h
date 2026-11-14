/* Default data locations for the praxis tools (no machine-specific paths):
 *   px_root("sub") -> $PRAXIS_ROOT/sub   (PRAXIS_ROOT defaults to $HOME/praxis)   models, weights, reference code
 *   px_out("sub")  -> $PRAXIS_OUT/sub    (PRAXIS_OUT defaults to ./tmp, i.e. <repo>/tmp when run from the repo root)  scratch/outputs
 * The returned pointer is valid until 8 further calls (rotating static buffers). */
#ifndef PRAXIS_PATHS_H
#define PRAXIS_PATHS_H
#include <stdio.h>
#include <stdlib.h>

static inline const char *px_path_(const char *env, const char *home_sub, const char *sub) {
    static char buf[8][1024]; static int idx;
    char *o = buf[idx++ & 7];
    const char *e = getenv(env), *h = getenv("HOME");
    if (e && *e) snprintf(o, 1024, "%s/%s", e, sub);
    else if (home_sub) snprintf(o, 1024, "%s/%s/%s", h ? h : ".", home_sub, sub);
    else snprintf(o, 1024, "tmp/%s", sub);
    return o;
}
static inline const char *px_root(const char *sub) { return px_path_("PRAXIS_ROOT", "praxis", sub); }
static inline const char *px_out(const char *sub)  { return px_path_("PRAXIS_OUT", NULL, sub); }
#endif
