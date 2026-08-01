/*
 * uTofu initialization probe for allocations where mpiexec stdout is swallowed.
 *
 * This binary intentionally makes no MPI calls. mpiexec is used only for process
 * placement; each rank discovers its rank from launcher environment variables
 * and writes a per-rank log file in UTOFU_PROBE_DIR.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <utofu.h>

#include "tofu_demo.h"

static FILE *Log;

static int env_int_any(const char **names, int defval)
{
    for (int i = 0; names[i]; i++) {
        const char *s = getenv(names[i]);
        if (s && *s) return atoi(s);
    }
    return defval;
}

static void log_line(const char *tag, int rc)
{
    fprintf(Log, "%s rc=%d\n", tag, rc);
    fflush(Log);
}

static void log_env(const char *name)
{
    const char *v = getenv(name);
    fprintf(Log, "ENV %s=%s\n", name, v ? v : "-");
}

int main(void)
{
    const char *rank_names[] = {
        "PMIX_RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", NULL
    };
    const char *size_names[] = {
        "PMIX_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", NULL
    };
    int rank = env_int_any(rank_names, -1);
    int size = env_int_any(size_names, -1);

    const char *dir = getenv("UTOFU_PROBE_DIR");
    if (!dir || !*dir) dir = ".";
    char path[512];
    snprintf(path, sizeof(path), "%s/utofu_probe_rank%05d_pid%ld.log",
             dir, rank, (long)getpid());
    Log = fopen(path, "w");
    if (!Log) {
        fprintf(stderr, "utofu_init_probe: fopen %s failed: %s\n", path, strerror(errno));
        return 90;
    }

    fprintf(Log, "UTOFU_PROBE rank=%d size=%d pid=%ld\n", rank, size, (long)getpid());
    log_env("PJM_JOBID");
    log_env("PJM_NODE");
    log_env("PJM_MPI_PROC");
    log_env("PMIX_RANK");
    log_env("PMIX_SIZE");
    log_env("PMI_RANK");
    log_env("PMI_SIZE");
    fflush(Log);

    utofu_tni_id_t *tni_ids = NULL;
    size_t num_tnis = 0;
    int rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    fprintf(Log, "utofu_get_onesided_tnis rc=%d num_tnis=%zu\n", rc, num_tnis);
    fflush(Log);
    if (rc != UTOFU_SUCCESS || num_tnis <= DEMO_TNI_INDEX) {
        free(tni_ids);
        fclose(Log);
        return 10;
    }

    uint8_t coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(coords);
    fprintf(Log, "utofu_query_my_coords rc=%d coords=%u %u %u %u %u %u\n",
            rc, coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]);
    fflush(Log);
    if (rc != UTOFU_SUCCESS) {
        free(tni_ids);
        fclose(Log);
        return 11;
    }

    utofu_tni_id_t tni = tni_ids[DEMO_TNI_INDEX];
    utofu_vcq_hdl_t vcq;
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &vcq);
    log_line("utofu_create_vcq_with_cmp_id", rc);
    if (rc != UTOFU_SUCCESS) {
        free(tni_ids);
        fclose(Log);
        return 12;
    }

    utofu_vcq_id_t real_id;
    rc = utofu_query_vcq_id(vcq, &real_id);
    log_line("utofu_query_vcq_id", rc);
    if (rc != UTOFU_SUCCESS) {
        utofu_free_vcq(vcq);
        free(tni_ids);
        fclose(Log);
        return 13;
    }

    utofu_vcq_id_t constructed;
    rc = utofu_construct_vcq_id(coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &constructed);
    log_line("utofu_construct_vcq_id_self", rc);
    if (rc != UTOFU_SUCCESS) {
        utofu_free_vcq(vcq);
        free(tni_ids);
        fclose(Log);
        return 14;
    }
    utofu_set_vcq_id_path(&real_id, NULL);
    utofu_set_vcq_id_path(&constructed, NULL);

    struct demo_region *region = NULL;
    rc = posix_memalign((void **)&region, DEMO_CACHE_LINE, sizeof(*region));
    fprintf(Log, "posix_memalign rc=%d region=%p bytes=%zu\n", rc, (void *)region, sizeof(*region));
    fflush(Log);
    if (rc || !region) {
        utofu_free_vcq(vcq);
        free(tni_ids);
        fclose(Log);
        return 15;
    }
    memset(region, 0, sizeof(*region));

    utofu_stadd_t base;
    rc = utofu_reg_mem_with_stag(vcq, region, sizeof(*region), DEMO_STAG, 0, &base);
    fprintf(Log, "utofu_reg_mem_with_stag rc=%d base=0x%llx\n",
            rc, (unsigned long long)base);
    fflush(Log);
    if (rc != UTOFU_SUCCESS) {
        free(region);
        utofu_free_vcq(vcq);
        free(tni_ids);
        fclose(Log);
        return 16;
    }

    utofu_stadd_t qbase;
    rc = utofu_query_stadd(real_id, DEMO_STAG, &qbase);
    fprintf(Log, "utofu_query_stadd_self rc=%d base=0x%llx\n",
            rc, (unsigned long long)qbase);
    fflush(Log);

    utofu_dereg_mem(vcq, base, 0);
    utofu_free_vcq(vcq);
    free(region);
    free(tni_ids);
    fprintf(Log, "UTOFU_PROBE_DONE rank=%d rc=%d\n", rank, rc);
    fclose(Log);
    return (rc == UTOFU_SUCCESS) ? 0 : 17;
}
