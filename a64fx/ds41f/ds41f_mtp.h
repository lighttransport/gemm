#ifndef DS41F_MTP_H
#define DS41F_MTP_H
#include "ds41f_weights.h"
#define DS41F_DRAFT_BLOCK 5
/* Draft state is separate from backbone KV. Only committed backbone taps
 * enter these windows; the five noisy draft keys are temporary. TP4/EP12. */
typedef struct {
    ds41f_weights weights;
    const ds41f_weights *backbone;
    const char *logits_prefix;
    float *window,*workspace;
    size_t committed;
    int rank,hc_mode,expert_fused;
} ds41f_mtp;
/* Conservative peak estimate from the small index, before loading payload. */
int ds41f_mtp_admission(const char *stage,int int8,size_t *bytes);
int ds41f_mtp_load(ds41f_mtp *mtp,const ds41f_weights *backbone,const char *stage,
                   int rank,size_t budget,int int8,int expert_sdot,int hc_mode);
void ds41f_mtp_free(ds41f_mtp *mtp);
/* Taps are BF16-rounded means of h over its four streams immediately before
 * layers 37/38/39 attention, concatenated on all ranks (15360 floats). */
int ds41f_mtp_commit(ds41f_mtp *mtp,const float *taps,size_t position);
/* Calculate all five positions at once, seeded by the next uncached token.
 * outputs[0] is seed, outputs[1..5] are proposals, never verified emissions. */
int ds41f_mtp_draft(ds41f_mtp *mtp,int seed,int outputs[6],float confidence[5]);
#endif
