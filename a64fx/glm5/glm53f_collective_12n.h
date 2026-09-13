#ifndef GLM53F_COLLECTIVE_12N_H
#define GLM53F_COLLECTIVE_12N_H

int glm53f_collective_init_12n(const char *topology_path, int max_count);
void glm53f_collective_free_12n(void);
int glm53f_collective_is_utofu_12n(void);
int glm53f_sum_allreduce_12n(const float *input, float *output, int count);

#endif
