#ifndef GLM53F_Q8_RESIDENT_H
#define GLM53F_Q8_RESIDENT_H

#include <stddef.h>
#include <stdint.h>

typedef struct glm53f_q8_resident glm53f_q8_resident;
typedef struct {
    uint64_t tensor_index;
    uint32_t type;
    uint32_t n_dims;
    uint64_t dims[4];
    uint64_t source_offset;
    uint64_t bytes;
    uint64_t data_offset;
    int mode;
    int expert;
    char name[192];
} glm53f_q8_resident_entry;

/* Load one completed rank image into anonymous memory.  The source image is
 * read in bounded chunks and its page cache is discarded as it is consumed. */
glm53f_q8_resident * glm53f_q8_resident_load(const char *image_dir, int rank);
void glm53f_q8_resident_free(glm53f_q8_resident *image);
const void * glm53f_q8_resident_data(const glm53f_q8_resident *image);
size_t glm53f_q8_resident_size(const glm53f_q8_resident *image);
uint64_t glm53f_q8_resident_hash(const glm53f_q8_resident *image);
size_t glm53f_q8_resident_entry_count(const glm53f_q8_resident *image);
const glm53f_q8_resident_entry * glm53f_q8_resident_entry_at(
        const glm53f_q8_resident *image, size_t index);
const glm53f_q8_resident_entry * glm53f_q8_resident_find(
        const glm53f_q8_resident *image, const char *name);

#endif
