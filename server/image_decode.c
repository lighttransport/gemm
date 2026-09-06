#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

/* Shared image decoder for the VLM and SAM3 server paths. Keeping this in its
 * own TU means LLM builds do not need to enable the SAM3 runner merely to
 * satisfy server_llm.c's image-decoding symbol. */
#define STB_IMAGE_IMPLEMENTATION
#include "../common/stb_image.h"

#define twp_NO_STDIO
#define twp_IMPLEMENTATION
#include "../common/tiny_webp.h"

#include "image_decode.h"

unsigned char *server_decode_image_rgb(const unsigned char *bytes,
                                       size_t len, int *W, int *H) {
    if (len >= 12 &&
        bytes[0] == 'R' && bytes[1] == 'I' && bytes[2] == 'F' && bytes[3] == 'F' &&
        bytes[8] == 'W' && bytes[9] == 'E' && bytes[10] == 'B' && bytes[11] == 'P') {
        int w = 0, h = 0;
        unsigned char *rgb = twp_read_from_memory((void *)bytes, (int)len, &w, &h,
                                                  twp_FORMAT_RGB, 0);
        if (!rgb) return NULL;
        *W = w; *H = h;
        return rgb;
    }
    int w = 0, h = 0, c = 0;
    unsigned char *rgb = stbi_load_from_memory(bytes, (int)len, &w, &h, &c, 3);
    if (!rgb) return NULL;
    *W = w; *H = h;
    return rgb;
}
