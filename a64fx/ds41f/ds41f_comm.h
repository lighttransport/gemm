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
void ds41f_comm_abort(const char *message,int error);
void ds41f_comm_set_tp(int tp);
void ds41f_comm_tp_bytes(void *values,size_t bytes,int owner);
void ds41f_comm_tp_allgather(float *out,float *part,size_t count);
void ds41f_comm_tp_gather(float *out,float *part,size_t count,int owner);
void ds41f_comm_argmax(float *value,int *index);
void ds41f_comm_head_logits(float *out,const float *part,size_t count);
void ds41f_comm_free(void);
#endif
