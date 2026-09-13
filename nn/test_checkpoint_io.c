/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include "gn.h"
#undef NDEBUG
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

static unsigned char *read_checkpoint(const char *path, size_t *bytes) {
    struct stat s;
    assert(!stat(path, &s) && s.st_size > 0);
    *bytes = (size_t)s.st_size;
    unsigned char *data = malloc(*bytes);
    assert(data);
    FILE *f = fopen(path, "rb");
    assert(f);
    assert(fread(data, 1, *bytes, f) == *bytes);
    assert(!fclose(f));
    return data;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    struct rlimit original;
    assert(!getrlimit(RLIMIT_FSIZE, &original));
    struct sigaction ignored = {0}, old_signal;
    ignored.sa_handler = SIG_IGN;
    assert(!sigemptyset(&ignored.sa_mask));
    assert(!sigaction(SIGXFSZ, &ignored, &old_signal));
    /* Both a sub-buffer-size checkpoint (failure deferred until fclose),
     * and a larger checkpoint with short fwrite and final-flush failures. */
    for (int large = 0; large < 2; large++) {
        gn_config c = gn_default_config();
        c.side = c.inputs = c.actions = c.head_dim = 1;
        c.channels = large ? 32 : 1;
        c.value_channels = c.value_hidden = 1;
        c.blocks = c.attention_every = 0;
        gn_model *m = gn_create(&c, "cpu", 0);
        assert(m && !gn_save(m, argv[1]));
        size_t before_bytes;
        unsigned char *before = read_checkpoint(argv[1], &before_bytes);
        gn_random(m); /* Failed replacement must preserve the old RNG, too. */
        for (int final_byte = 0; final_byte < 2; final_byte++) {
            struct rlimit limited = original;
            limited.rlim_cur = final_byte ? before_bytes - 1 : 1;
            assert(!setrlimit(RLIMIT_FSIZE, &limited));
            int rc = gn_save(m, argv[1]);
            assert(!setrlimit(RLIMIT_FSIZE, &original));
            assert(rc != 0);
            size_t after_bytes;
            unsigned char *after = read_checkpoint(argv[1], &after_bytes);
            assert(before_bytes == after_bytes && !memcmp(before, after, before_bytes));
            free(after);
            gn_model *loaded = gn_load(argv[1], "cpu", 0);
            assert(loaded);
            gn_destroy(loaded);
            char partial[4096];
            snprintf(partial, sizeof(partial), "%s.partial.%ld", argv[1], (long)getpid());
            errno = 0;
            assert(access(partial, F_OK) == -1 && errno == ENOENT);
        }
        /* A later successful save must still work after the I/O errors. */
        assert(!gn_save(m, argv[1]));
        gn_model *loaded = gn_load(argv[1], "cpu", 0);
        assert(loaded && gn_random(loaded) == gn_random(m));
        gn_destroy(loaded);
        gn_destroy(m);
        free(before);
    }
    assert(!sigaction(SIGXFSZ, &old_signal, NULL));
    puts("PASS: short/final-flush write failures preserve checkpoint bytes and reload; retry succeeds");
    return 0;
}
