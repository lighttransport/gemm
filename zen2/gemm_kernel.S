/*
 * Zen2 AVX2 FP32 6×16 GEMM microkernel (BLIS-style)
 *
 * Optimized for AMD Zen2 with HW prefetch OFF:
 * - 2-broadcast + 4-FMA grouping per row pair
 * - B loads cleanly after all 12 FMAs per K step
 * - B pointer pre-biased by +128 for compact encoding
 * - C prefetch (prefetchw for overwrite, prefetcht0 for accum) before K-loop
 * - A prefetch at 256, 304 bytes ahead
 * - Full B prefetch coverage: 4 prefetches per 4-CL unrolled block
 *
 * Register allocation (BLIS convention):
 *   ymm0, ymm1:   B vectors (NR=16 = 2 × 8-wide YMM)
 *   ymm2, ymm3:   A broadcast temps
 *   ymm4–ymm15:   12 accumulators
 *
 * System V AMD64:
 *   rdi=A_packed, rsi=B_packed, rdx=C, rcx=K, r8=ldc_bytes
 */

.intel_syntax noprefix

.text

/* ------------------------------------------------------------------ */
/*  gemm_kernel_6x16:  C = A × B  (overwrite)                        */
/* ------------------------------------------------------------------ */
.globl gemm_kernel_6x16
.type  gemm_kernel_6x16, @function
.align 16
gemm_kernel_6x16:
    push   rbx
    push   r12

    vxorps ymm4,  ymm4,  ymm4
    vxorps ymm5,  ymm5,  ymm5
    vxorps ymm6,  ymm6,  ymm6
    vxorps ymm7,  ymm7,  ymm7
    vxorps ymm8,  ymm8,  ymm8
    vxorps ymm9,  ymm9,  ymm9
    vxorps ymm10, ymm10, ymm10
    vxorps ymm11, ymm11, ymm11
    vxorps ymm12, ymm12, ymm12
    vxorps ymm13, ymm13, ymm13
    vxorps ymm14, ymm14, ymm14
    vxorps ymm15, ymm15, ymm15

    jmp .L_k_loop_setup_ow

/* ------------------------------------------------------------------ */
/*  gemm_kernel_6x16_accum:  C += A × B                              */
/* ------------------------------------------------------------------ */
.globl gemm_kernel_6x16_accum
.type  gemm_kernel_6x16_accum, @function
.align 16
gemm_kernel_6x16_accum:
    push   rbx
    push   r12

    mov    rax, rdx
    vmovups ymm4,  [rax]
    vmovups ymm5,  [rax + 32]
    add    rax, r8
    vmovups ymm6,  [rax]
    vmovups ymm7,  [rax + 32]
    add    rax, r8
    vmovups ymm8,  [rax]
    vmovups ymm9,  [rax + 32]
    add    rax, r8
    vmovups ymm10, [rax]
    vmovups ymm11, [rax + 32]
    add    rax, r8
    vmovups ymm12, [rax]
    vmovups ymm13, [rax + 32]
    add    rax, r8
    vmovups ymm14, [rax]
    vmovups ymm15, [rax + 32]

    /* Prefetch C rows to L1 for accumulate (data already loaded above,
       but prefetch the +32 half-CL if not same line) */
    jmp .L_k_loop_setup

/* ------------------------------------------------------------------ */
/*  K-loop setup: overwrite path (prefetchw for write-allocate)       */
/* ------------------------------------------------------------------ */
.L_k_loop_setup_ow:
    mov    rax, rdi
    mov    r10, rsi

    /* Prefetch C rows with write-intent (PREFETCHW) */
    mov    r12, rdx
    prefetchw [r12]
    add    r12, r8
    prefetchw [r12]
    add    r12, r8
    prefetchw [r12]
    add    r12, r8
    prefetchw [r12]
    add    r12, r8
    prefetchw [r12]
    add    r12, r8
    prefetchw [r12]

    jmp .L_k_loop_common

/* ------------------------------------------------------------------ */
/*  K-loop setup: accum path (prefetcht0 for read)                    */
/* ------------------------------------------------------------------ */
.L_k_loop_setup:
    mov    rax, rdi
    mov    r10, rsi

    /* Prefetch C rows to L1 (read mode) */
    mov    r12, rdx
    prefetcht0 [r12]
    add    r12, r8
    prefetcht0 [r12]
    add    r12, r8
    prefetcht0 [r12]
    add    r12, r8
    prefetcht0 [r12]
    add    r12, r8
    prefetcht0 [r12]
    add    r12, r8
    prefetcht0 [r12]

/* ------------------------------------------------------------------ */
/*  Main K-loop (shared)                                              */
/* ------------------------------------------------------------------ */
.L_k_loop_common:
    /* Pre-bias B pointer */
    add    r10, 128

    mov    rbx, rcx
    shr    rbx, 2
    mov    r11d, ecx
    and    r11d, 3

    test   rbx, rbx
    jz     .L_k_remainder

    /* Preload B for K=0 */
    vmovaps ymm0, [r10 - 4*32]
    vmovaps ymm1, [r10 - 3*32]

    .align 32
.L_k_unrolled:

    /* ============ K step 0 ============ */
    prefetcht0 [rax + 256]
    prefetcht0 [r10 + 384]

    vbroadcastss ymm2, dword ptr [rax + 0*4]
    vbroadcastss ymm3, dword ptr [rax + 1*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 2*4]
    vbroadcastss ymm3, dword ptr [rax + 3*4]
    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 4*4]
    vbroadcastss ymm3, dword ptr [rax + 5*4]
    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    vmovaps ymm0, [r10 - 2*32]
    vmovaps ymm1, [r10 - 1*32]


    /* ============ K step 1 ============ */
    prefetcht0 [r10 + 448]

    vbroadcastss ymm2, dword ptr [rax + 6*4]
    vbroadcastss ymm3, dword ptr [rax + 7*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 8*4]
    vbroadcastss ymm3, dword ptr [rax + 9*4]
    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 10*4]
    vbroadcastss ymm3, dword ptr [rax + 11*4]
    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    vmovaps ymm0, [r10 + 0*32]
    vmovaps ymm1, [r10 + 1*32]


    /* ============ K step 2 ============ */
    prefetcht0 [rax + 304]
    prefetcht0 [r10 + 512]

    vbroadcastss ymm2, dword ptr [rax + 12*4]
    vbroadcastss ymm3, dword ptr [rax + 13*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 14*4]
    vbroadcastss ymm3, dword ptr [rax + 15*4]
    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 16*4]
    vbroadcastss ymm3, dword ptr [rax + 17*4]
    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    vmovaps ymm0, [r10 + 2*32]
    vmovaps ymm1, [r10 + 3*32]


    /* ============ K step 3 ============ */
    prefetcht0 [r10 + 576]

    vbroadcastss ymm2, dword ptr [rax + 18*4]
    vbroadcastss ymm3, dword ptr [rax + 19*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 20*4]
    vbroadcastss ymm3, dword ptr [rax + 21*4]
    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 22*4]
    vbroadcastss ymm3, dword ptr [rax + 23*4]
    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    /* Advance pointers */
    add    rax, 4*6*4
    add    r10, 4*16*4

    /* Preload B for next iteration K=0 */
    vmovaps ymm0, [r10 - 4*32]
    vmovaps ymm1, [r10 - 3*32]

    dec    rbx
    jnz    .L_k_unrolled


/* ---- K remainder (0-3) ---- */
.L_k_remainder:
    test   r11d, r11d
    jz     .L_store

    .align 16
.L_k_remain_loop:
    vmovaps ymm0, [r10 - 4*32]
    vmovaps ymm1, [r10 - 3*32]

    vbroadcastss ymm2, dword ptr [rax + 0*4]
    vbroadcastss ymm3, dword ptr [rax + 1*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 2*4]
    vbroadcastss ymm3, dword ptr [rax + 3*4]
    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1

    vbroadcastss ymm2, dword ptr [rax + 4*4]
    vbroadcastss ymm3, dword ptr [rax + 5*4]
    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    add    rax, 6*4
    add    r10, 16*4

    dec    r11d
    jnz    .L_k_remain_loop


/* ------------------------------------------------------------------ */
/*  Store C[6×16]                                                     */
/* ------------------------------------------------------------------ */
.L_store:
    mov    rax, rdx

    vmovups [rax],      ymm4
    vmovups [rax + 32], ymm5
    add    rax, r8

    vmovups [rax],      ymm6
    vmovups [rax + 32], ymm7
    add    rax, r8

    vmovups [rax],      ymm8
    vmovups [rax + 32], ymm9
    add    rax, r8

    vmovups [rax],      ymm10
    vmovups [rax + 32], ymm11
    add    rax, r8

    vmovups [rax],      ymm12
    vmovups [rax + 32], ymm13
    add    rax, r8

    vmovups [rax],      ymm14
    vmovups [rax + 32], ymm15

    pop    r12
    pop    rbx

    vzeroupper
    ret

.size gemm_kernel_6x16, .-gemm_kernel_6x16
.size gemm_kernel_6x16_accum, .-gemm_kernel_6x16_accum

.section .note.GNU-stack,"",@progbits
