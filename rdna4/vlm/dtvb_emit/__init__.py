"""DTVB asm emitter for mm0 BF16 GEMM on RDNA4 (gfx1201).

Reproduces hipBLASLt algo-73624's SIA3+DTVB1 schedule pattern.
Ground truth annotation: hipblaslt_mm0_alik_bljk_dtvb_groundtruth.s.annot
"""
