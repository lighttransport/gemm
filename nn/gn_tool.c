/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include "gn.h"
#include "gn_replay.h"
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <time.h>
static volatile sig_atomic_t stopped;
static void stop(int signal_number) {
    (void)signal_number;
    stopped = 1;
}
static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static unsigned number(const char *s) {
    char *e;
    errno = 0;
    unsigned long n = strtoul(s, &e, 10);
    if (errno || *e || !n || n > UINT_MAX) {
        fprintf(stderr, "invalid positive integer: %s\n", s);
        exit(2);
    }
    return (unsigned)n;
}
static int error(void) {
    fprintf(stderr, "gn: %s\n", gn_error());
    return 1;
}
int main(int argc, char **argv) {
    signal(SIGINT, stop);
    signal(SIGTERM, stop);
    if (argc >= 3 && !strcmp(argv[1], "init")) {
        gn_config c = gn_default_config();
        if (argc == 4 && !strcmp(argv[3], "silu"))
            c.version = 2;
        else if (argc == 5 || argc == 6) {
            c.channels = number(argv[3]);
            c.blocks = number(argv[4]);
            c.head_dim = c.channels < 32 ? c.channels : 32;
            c.attention_every = c.blocks < 5 ? c.blocks : 5;
            if (argc == 6) {
                if (strcmp(argv[5], "silu"))
                    return 2;
                c.version = 2;
            }
        } else if (argc != 3)
            return 2;
        gn_model *m = gn_create(&c, "cpu", 0);
        if (!m)
            return error();
        int rc = gn_save(m, argv[2]);
        printf("{\"parameters\":%zu,\"channels\":%u,\"blocks\":%u,\"activation\":\"%s\"}\n",
               gn_parameter_count(m), c.channels, c.blocks, c.version >= 2 ? "silu" : "relu");
        gn_destroy(m);
        return rc ? error() : 0;
    }
    if ((argc == 9 || argc == 11) && !strcmp(argv[1], "train")) {
        /* Optional MICRO CHECKPOINT_SECONDS; BN uses microbatch statistics. */
        unsigned steps = number(argv[5]), batch = number(argv[6]);
        unsigned micro = argc == 11 ? number(argv[9]) : batch;
        unsigned checkpoint = argc == 11 ? number(argv[10]) : 3600;
        if (micro > batch)
            micro = batch;
        if (batch > 4096)
            return 2;
        char *end;
        float lr = strtof(argv[7], &end);
        if (*end || !isfinite(lr) || lr <= 0)
            return 2;
        gn_model *m = gn_load(argv[3], argv[8], 0);
        if (!m)
            return error();
        FILE *f = fopen(argv[2], "rb");
        gn_replay_schema s;
        int rc = 1;
        float *x = NULL, *p = NULL;
        uint32_t *labels = NULL;
        long *offsets = NULL;
        size_t count = 0, cap = 0, oldest = 0;
        if (!f || !gnr_read_header(f, &s)) {
            fprintf(stderr, "invalid replay header\n");
            goto done;
        }
        const gn_config *c = gn_configuration(m);
        if (s.side != c->side || s.channels != c->inputs || s.actions != c->actions) {
            fprintf(stderr, "replay/network schema mismatch\n");
            goto done;
        }
        size_t X = (size_t)s.side * s.side * s.channels, A = (size_t)s.side * s.side * s.actions;
        if ((X + A) * micro > c->memory_limit / sizeof(float) / 2) {
            fprintf(stderr, "batch exceeds host memory budget\n");
            goto done;
        }
        x = malloc(X * micro * sizeof(float));
        p = malloc(A * micro * sizeof(float));
        labels = malloc(micro * sizeof(*labels));
        if (!x || !p || !labels)
            goto done;
        gn_replay_record record;
        int status;
        long offset;
        /* The most recent million records form the replay window. Validation
         * games (game id divisible by 20) never enter the training index. */
        while ((offset = ftell(f)), (status = gnr_read(f, &s, &record, x, p)) > 0) {
            if (record.game % 20 == 0)
                continue;
            if (count == cap) {
                size_t next = cap ? cap * 2 : 1024;
                if (next > 1000000)
                    next = 1000000;
                if (cap == next) {
                    offsets[oldest] = offset;
                    oldest = (oldest + 1) % cap;
                    continue;
                } else {
                    long *v = realloc(offsets, next * sizeof(*v));
                    if (!v)
                        goto done;
                    offsets = v;
                    cap = next;
                }
            }
            offsets[count++] = offset;
        }
        if (status < 0 || !count) {
            fprintf(stderr, "malformed replay or no training games\n");
            goto done;
        }
        gn_metrics metrics = {0};
        double last_save = seconds();
        for (unsigned step = 0; step < steps && !stopped; step++) {
            double policy_loss = 0, value_loss = 0, begin = seconds();
            for (unsigned base = 0; base < batch; base += micro) {
                unsigned n = batch - base < micro ? batch - base : micro;
                for (unsigned b = 0; b < n; b++) {
                    size_t i = (size_t)(gn_random(m) % count);
                    if (fseek(f, offsets[i], SEEK_SET) ||
                        gnr_read(f, &s, &record, x + b * X, p + b * A) != 1)
                        goto done;
                    labels[b] = record.label;
                }
                if (gn_backward(m, n, x, p, labels, &metrics)) {
                    error();
                    goto done;
                }
                policy_loss += metrics.policy * n;
                value_loss += metrics.value * n;
            }
            if (gn_update(m, lr, 1e-4f, 1, &metrics)) {
                error();
                goto done;
            }
            printf("{\"step\":%llu,\"policy_loss\":%.9g,\"value_loss\":%.9g,\"gradient_norm\":%.9g,"
                   "\"memory_bytes\":%zu,\"examples_per_second\":%.6g}\n",
                   (unsigned long long)metrics.step, policy_loss / batch, value_loss / batch,
                   metrics.grad_norm, gn_memory_used(m), batch / (seconds() - begin));
            fflush(stdout);
            if (seconds() - last_save >= checkpoint) {
                if (gn_save(m, argv[4])) {
                    error();
                    goto done;
                }
                last_save = seconds();
            }
        }
        if (gn_save(m, argv[4])) {
            error();
            goto done;
        }
        rc = 0;
    done:
        if (f)
            fclose(f);
        free(offsets);
        free(x);
        free(p);
        free(labels);
        gn_destroy(m);
        return rc;
    }
    if (argc == 6 && !strcmp(argv[1], "validate")) {
        unsigned limit = number(argv[5]);
        gn_model *m = gn_load(argv[3], argv[4], 0);
        if (!m)
            return error();
        FILE *f = fopen(argv[2], "rb");
        gn_replay_schema schema;
        float *x = NULL, *target = NULL, *policy = NULL;
        int rc = 1;
        if (!f || !gnr_read_header(f, &schema))
            goto validation_done;
        const gn_config *c = gn_configuration(m);
        if (schema.side != c->side || schema.channels != c->inputs || schema.actions != c->actions)
            goto validation_done;
        size_t X = (size_t)c->side * c->side * c->inputs,
               A = (size_t)c->side * c->side * c->actions;
        x = malloc(X * sizeof(float));
        target = malloc(A * sizeof(float));
        policy = malloc(A * sizeof(float));
        if (!x || !target || !policy)
            goto validation_done;
        double lp = 0, lv = 0, brier = 0;
        unsigned count = 0;
        gn_replay_record record;
        int status = 0;
        while (count < limit && !stopped &&
               (status = gnr_read(f, &schema, &record, x, target)) > 0) {
            if (record.game % 20 != 0)
                continue;
            float wdl[3], top = -INFINITY;
            if (gn_infer(m, 1, x, policy, wdl))
                goto validation_done;
            for (size_t i = 0; i < A; i++)
                if (target[i] >= 0 && policy[i] > top)
                    top = policy[i];
            double sum = 0;
            for (size_t i = 0; i < A; i++)
                if (target[i] >= 0)
                    sum += exp((double)policy[i] - top);
            for (size_t i = 0; i < A; i++)
                if (target[i] > 0)
                    lp += target[i] * (log(sum) + top - policy[i]);
            lv -= log(fmax((double)wdl[record.label], 1e-30));
            for (int i = 0; i < 3; i++) {
                double d = wdl[i] - (i == (int)record.label);
                brier += d * d;
            }
            count++;
        }
        if (status < 0)
            goto validation_done;
        printf("{\"validation_positions\":%u,\"policy_loss\":%.9g,\"value_loss\":%.9g,\"wdl_"
               "brier\":%.9g,\"step\":%llu}\n",
               count, count ? lp / count : 0, count ? lv / count : 0, count ? brier / count : 0,
               (unsigned long long)gn_step(m));
        rc = 0;
    validation_done:
        if (f)
            fclose(f);
        free(x);
        free(target);
        free(policy);
        gn_destroy(m);
        if (rc)
            fprintf(stderr, "validation failed: malformed replay/model or %s\n", gn_error());
        return rc;
    }
    if ((argc == 6 || argc == 8) && !strcmp(argv[1], "bench")) {
        double bf16_peak = 195, int8_peak = 389;
        if (argc == 8) {
            char *end;
            bf16_peak = strtod(argv[6], &end);
            if (*end || !isfinite(bf16_peak) || bf16_peak <= 0)
                return 2;
            int8_peak = strtod(argv[7], &end);
            if (*end || !isfinite(int8_peak) || int8_peak <= 0)
                return 2;
        }
        unsigned batch = number(argv[4]), iterations = number(argv[5]);
        if (batch > 256)
            return 2;
        gn_model *m = gn_load(argv[2], argv[3], 0);
        if (!m)
            return error();
        const gn_config *c = gn_configuration(m);
        size_t X = (size_t)c->side * c->side * c->inputs * batch;
        size_t A = (size_t)c->side * c->side * c->actions * batch;
        float *x = calloc(X, sizeof(float)), *p = calloc(A, sizeof(float));
        float *wdl = calloc(batch * 3, sizeof(float));
        uint32_t *labels = calloc(batch, sizeof(uint32_t));
        int rc = 1;
        if (!x || !p || !wdl || !labels)
            goto bench_done;
        for (size_t i = 0; i < X; i++)
            x[i] = sinf((float)i * .1f);
        for (int i = 0; i < 10; i++)
            if (gn_infer(m, batch, x, p, wdl))
                goto bench_done;
        double start = seconds();
        for (unsigned i = 0; i < iterations; i++)
            if (gn_infer(m, batch, x, p, wdl))
                goto bench_done;
        double inference = (seconds() - start) / iterations;
        for (size_t i = 0; i < A; i++)
            p[i] = 1.0f / (A / batch);
        gn_metrics metrics;
        for (int i = 0; i < 2; i++)
            if (gn_backward(m, batch, x, p, labels, &metrics) ||
                gn_update(m, .001f, .0001f, 1, &metrics))
                goto bench_done;
        start = seconds();
        for (unsigned i = 0; i < iterations; i++)
            if (gn_backward(m, batch, x, p, labels, &metrics) ||
                gn_update(m, .001f, .0001f, 1, &metrics))
                goto bench_done;
        double train_seconds = seconds() - start, flops = gn_matrix_flops(m);
        int integer = strstr(argv[3], "int16") ? 16 : strstr(argv[3], "int8") ? 8 : 0;
        double gemm_rate = gn_gemm_flops(m) * iterations / train_seconds / 1e12;
        double total_rate = flops * iterations / train_seconds / 1e12;
        printf(
            "{\"backend\":\"%s\",\"batch\":%u,\"inference_ms\":%.6g,\"positions_per_second\":%.6g,"
            "\"train_examples_per_second\":%.6g,\"host_tensor_bytes\":%zu,"
            "\"matrix_equivalent_ops_per_example\":%.0f,\"useful_train_matrix_equivalent_tops\":%."
            "6g",
            argv[3], batch, inference * 1000, batch / inference, batch * iterations / train_seconds,
            gn_memory_used(m), flops / batch, total_rate);
        if (!integer)
            printf(",\"matrix_flops_per_example\":%.0f,\"useful_train_matrix_tflops\":%.6g",
                   flops / batch, total_rate);
        else
            printf(",\"useful_integer_gemm_tiops\":%.6g,\"int8_product_tiops\":%.6g", gemm_rate,
                   gemm_rate * (integer == 16 ? 4 : 1));
        printf(strstr(argv[3], "fp16back") ? ",\"attention_matrix_equivalent_tflops\":%.6g"
                                            : ",\"fp32_attention_tflops\":%.6g",
               total_rate - gemm_rate);
        if (!strncmp(argv[3], "hip", 3) && !strstr(argv[3], "fp32")) {
            double products = gn_gemm_product_ops(m, argv[3]) * iterations / train_seconds / 1e12;
            double peak = integer ? int8_peak : bf16_peak;
            printf(
                ",\"peak_reference\":\"%s\",\"dense_peak_tops\":%.6g,"
                "\"gemm_product_tops\":%.6g,\"gemm_product_peak_pct\":%.6g,"
                "\"useful_gemm_peak_pct\":%.6g,\"timing_scope\":\"whole_training_step\","
                "\"product_count_excludes_padding\":true,\"training_qualification\":\"unresolved\","
                "\"95pct_training_target_met\":false,\"75pct_product_rate_met\":%s,"
                "\"1000_examples_per_second_rate_met\":%s,\"qualified_target_met\":false",
                argc == 8 ? "user-supplied dense" : "RX 9070 XT nominal dense", peak, products,
                products / peak * 100, gemm_rate / peak * 100,
                products / peak >= .75 ? "true" : "false",
                batch * iterations / train_seconds >= 1000 ? "true" : "false");
        }
        printf("}\n");
        rc = 0;
    bench_done:
        free(x);
        free(p);
        free(wdl);
        free(labels);
        gn_destroy(m);
        return rc ? error() : 0;
    }
    fprintf(
        stderr,
        "usage: gn_tool init MODEL [CHANNELS BLOCKS]\n       gn_tool train REPLAY "
        "INPUT OUTPUT STEPS BATCH LR BACKEND [MICRO CHECKPOINT_SECONDS]\n"
        "       gn_tool bench MODEL BACKEND BATCH ITERATIONS [BF16_PEAK_TFLOPS INT8_PEAK_TOPS]\n"
        "       gn_tool validate REPLAY MODEL BACKEND LIMIT\n");
    return 2;
}
