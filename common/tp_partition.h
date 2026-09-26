#ifndef TF_TP_PARTITION_H
#define TF_TP_PARTITION_H

/* Partition n elements in aligned tiles, distributing whole tiles evenly.
 * The final endpoint is clipped to n; callers must ensure each tile boundary
 * is valid for their tensor format (or use a final partial tile). */
static inline void tf_tp_range_aligned(int n, int parts, int rank, int alignment,
                                       int *lo, int *hi) {
    if (alignment < 1) alignment = 1;
    int tiles = (n + alignment - 1) / alignment;
    int base = tiles / parts, rem = tiles % parts;
    int tile_lo = rank * base + (rank < rem ? rank : rem);
    int tile_count = base + (rank < rem);
    *lo = tile_lo * alignment;
    *hi = (tile_lo + tile_count) * alignment;
    if (*lo > n) *lo = n;
    if (*hi > n) *hi = n;
}

#endif
