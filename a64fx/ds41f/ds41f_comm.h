#ifndef DS41F_COMM_H
#define DS41F_COMM_H
#include <stddef.h>
int ds41f_comm_init(int *argc,char ***argv,int *rank,int *ranks);
void ds41f_comm_sum(float *values,size_t count);
void ds41f_comm_ready(void);
/* Select identically on every rank, before issuing broadcasts. */
void ds41f_comm_use_mpi_broadcast(int enabled);
void ds41f_comm_broadcast(float *values,size_t count,int owner);
void ds41f_comm_bytes(void *values,size_t bytes,int owner);
/* Exact BF16 transport, followed by an unquantized FP32 tail. Reject values
 * with nonzero low bits instead of silently quantizing a residual. */
void ds41f_comm_bf16_broadcast(float *values,size_t count,size_t tail,int owner);
void ds41f_comm_bf16_handoff(float *values,size_t count,size_t tail,int owner,int next);
/* Token-major strided rows, batch 1..6. The BF16 prefix and FP32 tail keep
 * the same representation and signed-zero rule as the single-row forms. */
void ds41f_comm_bf16_broadcast_batch(float *values,size_t stride,size_t count,size_t tail,size_t batch,int owner);
void ds41f_comm_bf16_handoff_batch(float *values,size_t stride,size_t count,size_t tail,size_t batch,int owner,int next);
void ds41f_comm_tp_gather_batch(float *out,size_t out_stride,float *part,size_t part_stride,size_t count,size_t batch,int owner,int all);

void ds41f_comm_abort(const char *message,int error);
void ds41f_comm_set_tp(int tp);
void ds41f_comm_tp_bytes(void *values,size_t bytes,int owner);
void ds41f_comm_tp_allgather(float *out,float *part,size_t count);
void ds41f_comm_tp_gather(float *out,float *part,size_t count,int owner);
void ds41f_comm_argmax(float *value,int *index);
void ds41f_comm_head_logits(float *out,const float *part,size_t count);
void ds41f_comm_free(void);
/* The shared expert may use a communicator independent of dense attention.
 * A TP12 shared path uses every rank while dense attention remains TP4. */
void ds41f_comm_set_shared_tp(int tp);
int ds41f_comm_shared_member(int owner);
void ds41f_comm_shared_range(size_t global_count,size_t *first,size_t *count);
void ds41f_comm_shared_range_aligned(size_t global_count,size_t alignment,
                                     size_t *first,size_t *count);
void ds41f_comm_shared_allgather(float *out,const float *part,size_t count);
void ds41f_comm_shared_gather(float *out,const float *part,size_t count,
                              int owner,size_t global_count);
void ds41f_comm_shared_gather_aligned(float *out,const float *part,size_t count,
                                      int owner,size_t global_count,size_t alignment);
void ds41f_comm_shared_reduce_scatter(float *out,const float *in,size_t global_count);
void ds41f_comm_shared_reduce_scatter_aligned(float *out,const float *in,
                                              size_t global_count,size_t alignment);
#endif
