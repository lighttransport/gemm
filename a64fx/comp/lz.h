#ifndef LZ_H
#define LZ_H

#include <stddef.h>

/* Compression levels */
#define LZ_LEVEL_FAST   3
#define LZ_LEVEL_BEST   9

/* Upper bound on compressed size for src_size input bytes */
size_t lz_compress_bound(size_t src_size);

/*
 * Compress src[0..src_size) into dst[0..dst_capacity).
 * Returns compressed size, or 0 on failure.
 * level: LZ_LEVEL_FAST (3) or LZ_LEVEL_BEST (9).
 */
size_t lz_compress(const void *src, size_t src_size,
                   void *dst, size_t dst_capacity, int level);

/*
 * Decompress src into dst (C implementation).
 * dst_capacity must be >= original uncompressed size.
 * Returns decompressed size, or 0 on error.
 */
size_t lz_decompress(const void *src, size_t src_size,
                     void *dst, size_t dst_capacity);

/*
 * Decompress src into dst (ARM64 asm fast path).
 * Same interface as lz_decompress.
 * Requires LZ_SAFETY_MARGIN extra bytes in both src and dst buffers.
 */
size_t lz_decompress_asm(const void *src, size_t src_size,
                         void *dst, size_t dst_capacity);

#endif /* LZ_H */
