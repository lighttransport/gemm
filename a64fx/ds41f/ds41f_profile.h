#ifndef DS41F_PROFILE_H
#define DS41F_PROFILE_H
#include <stddef.h>

/* Inclusive spans are intentionally separate from their nested components.
 * The report reconstructs the critical path instead of summing nested spans
 * or summing collective wait time over ranks. All recording is main-thread. */
#define DS41F_PROFILE_PHASES(X) \
    X(TOKEN) X(EMBED) X(EMBED_BCAST) X(ENGRAM_IO) X(ENGRAM_SUM) X(ENGRAM_PROJECT) \
    X(HC_ATTN_MIX) X(HC_ATTN_PRE) X(ATTENTION) X(HC_ATTN_POST) \
    X(HC_FFN_MIX) X(HC_FFN_PRE) X(GATE) X(ATTN_SYNC) X(FFN_BCAST) \
    X(EXPERTS) X(EXPERT_SUM) X(SHARED_EXPERT) X(HC_FFN_POST) X(RESIDUAL_BCAST) \
    X(HEAD_PRE) X(HEAD_LINEAR) X(HEAD_SELECT) X(NEXT_BCAST) \
    X(ATTN_QA) X(ATTN_QB) X(ATTN_Q_ROPE) X(ATTN_KV) X(ATTN_COMPRESS) \
    X(ATTN_INDEX) X(INDEX_QUERY) X(INDEX_SCORE) X(INDEX_SELECT) \
    X(ATTN_ROWS) X(ATTN_SPARSE) X(ATTN_INVERSE_ROPE) X(ATTN_WOA) X(ATTN_WOB) \
    X(LINEAR_QUANT) X(LINEAR_FP8) X(LINEAR_BF16) X(LINEAR_F32) X(LINEAR_ROUND) \
    X(NORM) X(EXPERT_QUANT) X(EXPERT_W13) X(EXPERT_SWIGLU) X(EXPERT_W2) \
    X(EXPERT_ROUND) X(EXPERT_COUNT) X(FP8_BYTES) X(BF16_BYTES) X(F32_BYTES) X(FP4_BYTES) \
    X(ENGRAM_READ) X(ENGRAM_DECODE) X(HC_NORM) X(HC_MATVEC) X(HC_SPLIT) X(ENGRAM_PREFETCH)
#define DS41F_PROFILE_ENUM(name) DS41F_P_##name,
enum ds41f_profile_phase { DS41F_PROFILE_PHASES(DS41F_PROFILE_ENUM) DS41F_P_COUNT };
#undef DS41F_PROFILE_ENUM

#ifdef DS41F_ENABLE_PROFILE
extern _Thread_local double *ds41f_profile_current;
double ds41f_profile_clock(void);
int ds41f_profile_init(size_t start,size_t count);
void ds41f_profile_at(size_t position,int layer);
int ds41f_profile_write(int rank);
void ds41f_profile_free(void);
static inline double ds41f_profile_begin(void)
{return ds41f_profile_current?ds41f_profile_clock():0;}
static inline void ds41f_profile_end(enum ds41f_profile_phase phase,double start)
{if(ds41f_profile_current)ds41f_profile_current[phase]+=ds41f_profile_clock()-start;}
static inline void ds41f_profile_value(enum ds41f_profile_phase phase,double value)
{if(ds41f_profile_current)ds41f_profile_current[phase]+=value;}
#else
static inline double ds41f_profile_begin(void){return 0;}
static inline void ds41f_profile_end(enum ds41f_profile_phase phase,double start)
{(void)phase;(void)start;}
static inline void ds41f_profile_value(enum ds41f_profile_phase phase,double value)
{(void)phase;(void)value;}
#endif
#define P_BEGIN() ds41f_profile_begin()
#define P_END(name,start) ds41f_profile_end(DS41F_P_##name,start)
#define P_VALUE(name,value) ds41f_profile_value(DS41F_P_##name,value)
#endif
