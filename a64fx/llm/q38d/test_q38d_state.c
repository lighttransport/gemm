/* Portable snapshot/layout test; the implementation under test is included.
 * cc -O2 -Wall -Wextra test_q38d_state.c -o test_q38d_state
 * ./test_q38d_state NEW_TEST_DIR (parent must exist)
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <assert.h>
#include "q38d_kv_i6.h"
#define NLAYER 2
#define NT 48
#define DS 128
#define NVH 48
#define NKV 4
#define HD 256
#define EMBD 5120
#define KVB 16
static int tp_n = 1, tp_r, ssq_valid, tp_ssq_ok, tp_prod_ok;
static int pf_kv_i6_mode, kv_i6_mode;
static size_t kv_row_bytes(void) { return q38d_kv_i6_row_bytes(HD); }
static struct {
    int8_t *k[1][NKV][4], *v[1][NKV][4];
    float *ks[1][NKV][4], *vs[1][NKV][4];
} KI;
static struct { int n_prompt; int32_t *tok; } JOB;
static struct { uint64_t n_tensors; } fake_g, *G = &fake_g;
static struct {
    int fmt, arith, n_vocab, next_token;
    float x[EMBD];
    struct { int ssm, ai; } L[NLAYER];
    float *conv_hist[NLAYER][NVH];
    float *kcp[1][NKV][4], *vcp[1][NKV][4];
} E;
static float *states[NVH];
static float *ssm_st(int layer, int h, int i) { (void)layer; (void)i; return states[h]; }
static int ssm_head_of(int tid) {
    int gc = 4 / tp_n, hc = 3 * gc, c = tid / 12, lane = tid % 12;
    return lane < hc ? tp_r * (16 / tp_n) + c * gc + lane % gc + 16 * (lane / gc) : -1;
}
static int kv_cpk(void) { return tp_n; }
static void kv_map(int t, int cpk, int *k, int *u) {
    int block = t / KVB; *k = block % cpk; *u = block / cpk * KVB + t % KVB;
}
static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9;
}
#include "q38d_state.inc"

static float value(int kind, int head, int index) { return kind * 1000000 + head * 20000 + index; }
static void clear_state(void) {
    memset(E.x, 0, sizeof(E.x));
    for (int h = 0; h < NVH; h++) {
        memset(states[h], 0, (DS * DS + 2 * DS) * 4);
        memset(E.conv_hist[0][h], 0, 8 * 384 * 4);
    }
    for (int h = 0; h < NKV; h++) for (int k = 0; k < 4; k++) {
        memset(E.kcp[0][h][k], 0, (size_t)JOB.n_prompt * HD * 4);
        memset(E.vcp[0][h][k], 0, (size_t)JOB.n_prompt * HD * 4);
        if (KI.k[0][h][k]) {
            memset(KI.k[0][h][k], 0, (size_t)JOB.n_prompt * kv_row_bytes());
            memset(KI.v[0][h][k], 0, (size_t)JOB.n_prompt * kv_row_bytes());
            memset(KI.ks[0][h][k], 0, (size_t)JOB.n_prompt * sizeof(float));
            memset(KI.vs[0][h][k], 0, (size_t)JOB.n_prompt * sizeof(float));
        }
    }
}
int main(int argc, char **argv) {
    (void)state_identity; (void)state_write_sec;
    assert(argc == 2);
    state_out = state_in = argv[1];
    assert(!mkdir(state_out, 0700));
    JOB.n_prompt = 65;  /* partial block, crosses all TP4 position partitions */
    E.L[0].ssm = 1; E.L[1].ai = 0;
    E.fmt = 4; E.arith = 16; E.n_vocab = 100; E.next_token = 42;
    state_model_id = 123; state_prompt_id = 456;
    for (int i = 0; i < EMBD; i++) E.x[i] = (float)i;
    for (int h = 0; h < NVH; h++) {
        states[h] = malloc((DS * DS + 2 * DS) * 4);
        E.conv_hist[0][h] = malloc(8 * 384 * 4);
        for (int i = 0; i < DS * DS + 2 * DS; i++) states[h][i] = value(1, h, i);
        for (int i = 0; i < 8 * 384; i++) E.conv_hist[0][h][i] = value(2, h, i);
    }
    for (int h = 0; h < NKV; h++) for (int k = 0; k < 4; k++) {
        E.kcp[0][h][k] = calloc((size_t)JOB.n_prompt * HD, 4);
        E.vcp[0][h][k] = calloc((size_t)JOB.n_prompt * HD, 4);
        if (!k) for (int i = 0; i < JOB.n_prompt * HD; i++) {
            E.kcp[0][h][k][i] = value(3, h, i);
            E.vcp[0][h][k][i] = value(4, h, i);
        }
    }
    state_export_layer(0); state_export_layer(1); state_export_meta();
    for (tp_n = 1; tp_n <= 4; tp_n *= 2) for (tp_r = 0; tp_r < tp_n; tp_r++) {
        clear_state(); state_import();
        assert(state_first_token == 42);
        for (int i = 0; i < EMBD; i++) assert(E.x[i] == i);
        for (int h = 0; h < NVH; h++) {
            int owned = h % 16 / (16 / tp_n) == tp_r;
            for (int i = 0; i < DS * DS + 2 * DS; i++) assert(states[h][i] == (owned ? value(1, h, i) : 0));
            for (int i = 0; i < 8 * 384; i++) assert(E.conv_hist[0][h][i] == (owned ? value(2, h, i) : 0));
        }
        for (int h = 0; h < NKV / tp_n; h++) for (int t = 0; t < JOB.n_prompt; t++) {
            int part = t / 16 % tp_n, offset = (t / (16 * tp_n) * 16 + t % 16) * HD;
            for (int i = 0; i < HD; i++) {
                assert(E.kcp[0][h][part][offset + i] == value(3, tp_r * (4 / tp_n) + h, t * HD + i));
                assert(E.vcp[0][h][part][offset + i] == value(4, tp_r * (4 / tp_n) + h, t * HD + i));
            }
        }
    }
    tp_n = 1; tp_r = 0;
    state_prefix_tokens = 65; JOB.n_prompt = 96;
    state_import(); /* the snapshot remains 65 tokens inside a longer request */
    assert(state_first_token == 42);
    state_prefix_tokens = 0; JOB.n_prompt = 65;
    /* A wrong prompt must fail before state is consumed. */
    pid_t pid = fork(); assert(pid >= 0);
    if (!pid) { state_prompt_id++; state_import(); _exit(0); }
    int status; assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    /* A corrupted KV payload must fail the checksum. */
    char path[4096]; snprintf(path, sizeof(path), "%s/layer01.bin", state_out);
    int fd = open(path, O_WRONLY); assert(fd >= 0);
    uint32_t bad = 0; assert(pwrite(fd, &bad, 4, 80) == 4); close(fd);
    pid = fork(); assert(pid >= 0);
    if (!pid) { state_import(); _exit(0); }
    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    /* Version 2: export INT6 from the FP32 source and import its packed rows
     * into every TP partition, including the final partial KVB block. */
    char i6_dir[4096];
    assert(snprintf(i6_dir, sizeof(i6_dir), "%s.i6", argv[1]) < (int)sizeof(i6_dir));
    assert(!mkdir(i6_dir, 0700));
    state_in = state_out = i6_dir;
    state_out_i6 = 1;
    state_export_layer(0); state_export_layer(1); state_export_meta();
    state_out_i6 = 0;
    kv_i6_mode = 1;
    for (int h = 0; h < NKV; h++) for (int k = 0; k < 4; k++) {
        KI.k[0][h][k] = calloc((size_t)JOB.n_prompt, kv_row_bytes());
        KI.v[0][h][k] = calloc((size_t)JOB.n_prompt, kv_row_bytes());
        KI.ks[0][h][k] = calloc((size_t)JOB.n_prompt, sizeof(float));
        KI.vs[0][h][k] = calloc((size_t)JOB.n_prompt, sizeof(float));
        assert(KI.k[0][h][k] && KI.v[0][h][k] && KI.ks[0][h][k] && KI.vs[0][h][k]);
    }
    for (tp_n = 1; tp_n <= 4; tp_n *= 2) for (tp_r = 0; tp_r < tp_n; tp_r++) {
        clear_state(); state_import();
        assert(state_first_token == 42);
        for (int h = 0; h < NKV / tp_n; h++) for (int t = 0; t < JOB.n_prompt; t++) {
            int global_h = tp_r * (NKV / tp_n) + h, part, u;
            float source[HD], expected_scale;
            uint8_t expected[192];
            kv_map(t, tp_n, &part, &u);
            for (int i = 0; i < HD; i++) source[i] = value(3, global_h, t * HD + i);
            assert(!q38d_kv_i6_pack_row(source, expected, HD, &expected_scale));
            assert(!memcmp(KI.k[0][h][part] + (size_t)u * kv_row_bytes(), expected, sizeof(expected)));
            assert(KI.ks[0][h][part][u] == expected_scale);
            for (int i = 0; i < HD; i++) source[i] = value(4, global_h, t * HD + i);
            assert(!q38d_kv_i6_pack_row(source, expected, HD, &expected_scale));
            assert(!memcmp(KI.v[0][h][part] + (size_t)u * kv_row_bytes(), expected, sizeof(expected)));
            assert(KI.vs[0][h][part][u] == expected_scale);
        }
    }
    puts("PASS: FP32 and INT6 state, TP1/2/4 all ranks, KV tail, identity/corruption rejection");
    return 0;
}
