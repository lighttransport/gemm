#ifndef SERVER_IMAGE_DECODE_H
#define SERVER_IMAGE_DECODE_H

#include <stddef.h>

/* Decode a PNG/JPEG/WebP image into a packed interleaved RGB8 buffer.
 * Returns NULL on failure. On success, *W and *H receive the dimensions and
 * the caller must release the buffer with plain free().
 *
 * WebP support is provided by common/tiny_webp.h; everything else routes
 * through stb_image. The implementation lives in server_sam3.c (the TU that
 * already carries STB_IMAGE_IMPLEMENTATION). */
unsigned char *server_decode_image_rgb(const unsigned char *bytes,
                                       size_t len,
                                       int *W, int *H);

#endif
