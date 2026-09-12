#ifndef DS41F_JOURNAL_H
#define DS41F_JOURNAL_H
#include "ds41f_attention.h"
#include "ds41f_prefetch.h"
typedef struct ds41f_journal ds41f_journal;
/* Main-thread-only transaction for at most six consecutive verifier inputs.
 * Record immediately BEFORE each complete causal forward. Finish retains the
 * first keep inputs and restores all overwritten state from the suffix.
 * The scratch attention rows and content-keyed projection cache are not state.
 * Allocation is bounded by six window rows/layer, six compressed rows/source,
 * and six copies of the small selection/pool/Engram state, never full KV. */
size_t ds41f_journal_bytes(size_t capacity,size_t inputs);
int ds41f_journal_create(ds41f_journal **out,ds41f_attention *attention,
                         ds41f_engram *engram,ds41f_prefetch *prefetch,
                         size_t start,size_t inputs,size_t budget);
int ds41f_journal_record(ds41f_journal *journal,size_t position);
int ds41f_journal_finish(ds41f_journal *journal,size_t keep);
void ds41f_journal_free(ds41f_journal *journal);
#endif
