#ifndef GLM53F_COLLECTIVE_12N_H
#define GLM53F_COLLECTIVE_12N_H

int glm53f_collective_init_12n(const char *topology_path, int max_count);
void glm53f_collective_free_12n(void);
int glm53f_collective_is_utofu_12n(void);
int glm53f_collective_capacity_12n(void);
int glm53f_collective_prefill_algorithm_12n(int algorithm);
int glm53f_sum_allreduce_12n(const float *input, float *output, int count);
int glm53f_sum_allreduce_prefill_12n(const float *input, float *output, int count);
int glm53f_sum_allreduce_mpi_12n(const float *input, float *output, int count);
/* Async slab reduction on a spare core (uTofu multi-TNI only). */
int glm53f_async_available_12n(void);
int glm53f_allgather_bytes_12n(const void *input, void *output, int bytes);
int glm53f_async_begin_12n(const float *input, float *output, int tokens, int width, int slab_tokens);
void glm53f_async_ready_12n(int tokens_ready);
int glm53f_async_finish_12n(void);
int glm53f_sum_allreduce_slabs_12n(const float *input, float *output,
                                  int tokens, int width, int slab_tokens);

#endif
