/* utofu_stubs.c — link-time bodies for the uTofu API under qlair.
 *
 * qlair intercepts by ELF symbol name at call entry, so these bodies never
 * run in the simulator; they exist so a static gcc link resolves. Running the
 * binary OUTSIDE qlair returns -7 (no Tofu) from every call — a loud failure.
 * (The .S stub tricks documented in README.md were clair-compiler workarounds;
 * with aarch64-linux-gnu-gcc plain C stubs suffice.)
 */
#include "utofu.h"

#define QLAIR_STUB { return -7; /* only meaningful under qlair interception */ }

int utofu_get_onesided_tnis(utofu_tni_id_t **tni_ids, size_t *num_tnis) QLAIR_STUB
int utofu_create_vcq(utofu_tni_id_t tni_id, unsigned long flags,
                     utofu_vcq_hdl_t *vcq_hdl) QLAIR_STUB
int utofu_free_vcq(utofu_vcq_hdl_t vcq_hdl) QLAIR_STUB
int utofu_query_vcq_id(utofu_vcq_hdl_t vcq_hdl, utofu_vcq_id_t *vcq_id) QLAIR_STUB
int utofu_reg_mem(utofu_vcq_hdl_t vcq_hdl, void *addr, size_t size,
                  unsigned long flags, utofu_stadd_t *stadd) QLAIR_STUB
int utofu_reg_mem_with_stag(utofu_vcq_hdl_t vcq_hdl, void *addr, size_t size,
                            unsigned int stag, unsigned long flags,
                            utofu_stadd_t *stadd) QLAIR_STUB
int utofu_query_stadd(utofu_vcq_id_t vcq_id, unsigned int stag,
                      utofu_stadd_t *stadd) QLAIR_STUB
int utofu_dereg_mem(utofu_vcq_hdl_t vcq_hdl, utofu_stadd_t stadd,
                    unsigned long flags) QLAIR_STUB
int utofu_put(utofu_vcq_hdl_t vcq_hdl, utofu_vcq_id_t rmt_vcq_id,
              utofu_stadd_t lcl_stadd, utofu_stadd_t rmt_stadd, size_t length,
              uint64_t edata, unsigned long flags, void *cbdata) QLAIR_STUB
int utofu_poll_tcq(utofu_vcq_hdl_t vcq_hdl, unsigned long flags,
                   void **cbdata) QLAIR_STUB
int utofu_poll_mrq(utofu_vcq_hdl_t vcq_hdl, unsigned long flags,
                   struct utofu_mrq_notice *notice) QLAIR_STUB
