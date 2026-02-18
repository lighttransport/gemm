/*
 * pmu_example.c - Usage example for the PMU counter library
 *
 * Demonstrates measuring a simple compute kernel with different
 * predefined event groups.
 */

#include "pmu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple DAXPY kernel for testing */
static void daxpy(int n, double a, const double *x, double *y)
{
    for (int i = 0; i < n; i++)
        y[i] += a * x[i];
}

/* Simple matrix-vector multiply for more compute */
static void matvec(int n, const double *A, const double *x, double *y)
{
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * x[j];
        y[i] = sum;
    }
}

static void run_group(const char *name, const uint16_t *events, int nevents)
{
    pmu_ctx_t ctx;
    if (pmu_init(&ctx, events, nevents) < 0) {
        fprintf(stderr, "Failed to init PMU group: %s\n", name);
        return;
    }

    const int N = 1024;
    double *A = (double *)malloc(N * N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));

    /* Initialize data */
    for (int i = 0; i < N * N; i++) A[i] = (double)(i % 17) * 0.1;
    for (int i = 0; i < N; i++) { x[i] = 1.0; y[i] = 0.0; }

    /* Warmup */
    matvec(N, A, x, y);

    /* Measure */
    pmu_start(&ctx);
    matvec(N, A, x, y);
    pmu_stop(&ctx);
    pmu_read(&ctx);
    pmu_print(&ctx, name);

    free(A);
    free(x);
    free(y);
    pmu_fini(&ctx);
}

int main(void)
{
    printf("PMU Counter Library - Example\n");
    printf("Timer frequency: %lu Hz\n\n", (unsigned long)pmu_freq());

    /* Test each predefined group */
    run_group("BASIC (matvec 1024x1024)",
              PMU_GROUP_BASIC, PMU_GROUP_BASIC_N);

    run_group("L2 (matvec 1024x1024)",
              PMU_GROUP_L2, PMU_GROUP_L2_N);

    run_group("PREFETCH (matvec 1024x1024)",
              PMU_GROUP_PREFETCH, PMU_GROUP_PREFETCH_N);

    run_group("ENERGY (matvec 1024x1024)",
              PMU_GROUP_ENERGY, PMU_GROUP_ENERGY_N);

    run_group("FP (matvec 1024x1024)",
              PMU_GROUP_FP, PMU_GROUP_FP_N);

    /* CSV output demo */
    printf("--- CSV output demo ---\n");
    {
        pmu_ctx_t ctx;
        if (pmu_init(&ctx, PMU_GROUP_BASIC, PMU_GROUP_BASIC_N) == 0) {
            pmu_print_csv_header(&ctx);

            const int N = 4096;
            double *x = (double *)malloc(N * sizeof(double));
            double *y = (double *)malloc(N * sizeof(double));
            for (int i = 0; i < N; i++) { x[i] = 1.0; y[i] = 0.0; }

            int sizes[] = { 256, 512, 1024, 2048, 4096 };
            for (int s = 0; s < 5; s++) {
                char label[64];
                snprintf(label, sizeof(label), "daxpy_%d", sizes[s]);

                /* Warmup */
                daxpy(sizes[s], 2.0, x, y);

                pmu_start(&ctx);
                daxpy(sizes[s], 2.0, x, y);
                pmu_stop(&ctx);
                pmu_read(&ctx);
                pmu_print_csv(&ctx, label);
            }

            free(x);
            free(y);
            pmu_fini(&ctx);
        }
    }

    printf("\nDone.\n");
    return 0;
}
