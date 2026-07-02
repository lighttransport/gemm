/* utofu.h — shim for cross-compiling uTofu programs (tp_allreduce.h, the glm5
 * sim harness) as static aarch64 ELFs run under the qlair simulator.
 *
 * qlair intercepts these functions BY ELF SYMBOL NAME at call entry
 * (tools/qlair/syscall/qlair-syscall.cc ~:2949-3360) and services them in the
 * TofuSimulator, so the linked-in bodies (utofu_stubs.c) never execute. Types
 * and constants below match the simulator's (tools/qlair/tofu/qlair-tofu.hh).
 * NOT the real Fugaku utofu.h — for qlair guest builds only.
 */
#ifndef QLAIR_UTOFU_SHIM_H
#define QLAIR_UTOFU_SHIM_H

#include <stddef.h>
#include <stdint.h>

typedef uint16_t  utofu_tni_id_t;
typedef uintptr_t utofu_vcq_hdl_t;
typedef uint64_t  utofu_vcq_id_t;
typedef uint64_t  utofu_stadd_t;

#define UTOFU_SUCCESS        0
#define UTOFU_ERR_NOT_FOUND  (-1)
/* qlair's put returns ERR_FULL(-6) when the 512-deep TOQ is full; map BUSY to
 * it so tp_ar_put's BUSY-retry loop drains and retries exactly as on Fugaku. */
#define UTOFU_ERR_BUSY       (-6)
#define UTOFU_ERR_INVALID_ARG (-512)

#define UTOFU_ONESIDED_FLAG_TCQ_NOTICE        (1UL << 14)
#define UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE  (1UL << 15)
#define UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE (1UL << 2)

#define UTOFU_MRQ_TYPE_LCL_PUT 0
#define UTOFU_MRQ_TYPE_RMT_PUT 1

/* 64-byte layout matching qlair's TofuMRQNotice */
struct utofu_mrq_notice {
    uint8_t        notice_type;
    uint8_t        padding1[7];
    utofu_vcq_id_t vcq_id;
    uint64_t       edata;
    uint64_t       rmt_value;
    utofu_stadd_t  lcl_stadd;
    utofu_stadd_t  rmt_stadd;
    uint64_t       reserved[2];
};

#ifdef __cplusplus
extern "C" {
#endif

int utofu_get_onesided_tnis(utofu_tni_id_t **tni_ids, size_t *num_tnis);
int utofu_create_vcq(utofu_tni_id_t tni_id, unsigned long flags,
                     utofu_vcq_hdl_t *vcq_hdl);
int utofu_free_vcq(utofu_vcq_hdl_t vcq_hdl);
int utofu_query_vcq_id(utofu_vcq_hdl_t vcq_hdl, utofu_vcq_id_t *vcq_id);
int utofu_reg_mem(utofu_vcq_hdl_t vcq_hdl, void *addr, size_t size,
                  unsigned long flags, utofu_stadd_t *stadd);
int utofu_reg_mem_with_stag(utofu_vcq_hdl_t vcq_hdl, void *addr, size_t size,
                            unsigned int stag, unsigned long flags,
                            utofu_stadd_t *stadd);
int utofu_query_stadd(utofu_vcq_id_t vcq_id, unsigned int stag,
                      utofu_stadd_t *stadd);
int utofu_dereg_mem(utofu_vcq_hdl_t vcq_hdl, utofu_stadd_t stadd,
                    unsigned long flags);
int utofu_put(utofu_vcq_hdl_t vcq_hdl, utofu_vcq_id_t rmt_vcq_id,
              utofu_stadd_t lcl_stadd, utofu_stadd_t rmt_stadd, size_t length,
              uint64_t edata, unsigned long flags, void *cbdata);
int utofu_poll_tcq(utofu_vcq_hdl_t vcq_hdl, unsigned long flags, void **cbdata);
int utofu_poll_mrq(utofu_vcq_hdl_t vcq_hdl, unsigned long flags,
                   struct utofu_mrq_notice *notice);

#ifdef __cplusplus
}
#endif

/* guest cycle counter: qlair CNTVCT delta == simulated nanoseconds */
static inline uint64_t qlair_rd_cyc(void) {
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

#endif /* QLAIR_UTOFU_SHIM_H */
