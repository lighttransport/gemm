#ifndef LZ_FMT_H
#define LZ_FMT_H

#include <stdint.h>

/* Magic number: "LZ77" in LE */
#define LZ_MAGIC         0x37375A4Cu

/* Block size: 64 KB */
#define LZ_BLOCK_SIZE    (64 * 1024)

/* Token format constants */
#define LZ_MIN_MATCH     4
#define LZ_MAX_OFFSET    65535
#define LZ_EXT_THRESHOLD 15

/* Block header bit 31: stored (uncompressed) block */
#define LZ_BLOCK_STORED  (1u << 31)

/* Safety margin for overrun-safe ldp/stp in asm decompressor */
#define LZ_SAFETY_MARGIN 32

/* Frame header: 8 bytes */
typedef struct {
    uint32_t magic;
    uint32_t original_size;
} lz_frame_header_t;

static inline void lz_write16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
}

static inline uint16_t lz_read16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t lz_read32(const uint8_t *p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

static inline void lz_write32(uint8_t *p, uint32_t v) {
    __builtin_memcpy(p, &v, 4);
}

#endif /* LZ_FMT_H */
