// pack_p_sve.h - SVE optimized P packing
#ifndef PACK_P_SVE_H
#define PACK_P_SVE_H

#include <stdint.h>

// Pack P[6][64] row-major to Pp[16][6][4] for kernel
// Current scalar version is 48% of fused attention time!
void pack_P_sve(const int8_t* P, int8_t* Pp);

#endif
