/*
 * Zen2 AVX2 FP32 6×16 GEMM microkernel — v2 optimized scheduling
 *
 * Changes from v1:
 * - Removed B software prefetches (Zen2 HW prefetch handles sequential B)
 * - Broadcasts issued BEFORE prefetches to enter scheduler earlier
 * - Pre-broadcast A[row0,row1] for next K-step at end of current K-step
 * - Interleaved broadcast for rows 2-3 into rows 0-1 FMA latency shadow
 * - Prefetches placed in broadcast→FMA latency gap (dead cycles)
 *
 * Register allocation:
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
    /* Pre-bias B pointer for compact displacement encoding */
    add    r10, 128

    mov    rbx, rcx
    shr    rbx, 2
    mov    r11d, ecx
    and    r11d, 3

    test   rbx, rbx
    jz     .L_k_remainder

    /* Preload B for K=0 and pre-broadcast A[0,1] */
    vmovaps ymm0, [r10 - 4*32]
    vmovaps ymm1, [r10 - 3*32]
    vbroadcastss ymm2, dword ptr [rax + 0*4]
    vbroadcastss ymm3, dword ptr [rax + 1*4]

    .align 32
.L_k_unrolled:

    /* ============ K step 0 ============ */
    /* ymm2=A[0], ymm3=A[1] already pre-broadcast */

    /* Rows 0-1 FMAs (ymm2, ymm3 ready from pre-broadcast) */
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    /* Interleave: start rows 2-3 broadcasts in latency shadow */
    vbroadcastss ymm2, dword ptr [rax + 2*4]
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 3*4]

    /* A prefetch in dead time (broadcasts in flight) */
    prefetcht0 [rax + 256]

    /* Rows 2-3 FMAs */
    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    /* Interleave: start rows 4-5 broadcasts */
    vbroadcastss ymm2, dword ptr [rax + 4*4]
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 5*4]

    /* Rows 4-5 FMAs */
    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    /* Load B for K1 + pre-broadcast K1 A[0,1] */
    vmovaps ymm0, [r10 - 2*32]
    vmovaps ymm1, [r10 - 1*32]
    vbroadcastss ymm2, dword ptr [rax + 6*4]
    vbroadcastss ymm3, dword ptr [rax + 7*4]


    /* ============ K step 1 ============ */
    /* ymm2=A[6], ymm3=A[7] already pre-broadcast */

    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 8*4]
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 9*4]

    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 10*4]
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 11*4]

    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    /* Load B for K2 + pre-broadcast K2 A[0,1] */
    vmovaps ymm0, [r10 + 0*32]
    vmovaps ymm1, [r10 + 1*32]
    vbroadcastss ymm2, dword ptr [rax + 12*4]
    vbroadcastss ymm3, dword ptr [rax + 13*4]


    /* ============ K step 2 ============ */
    /* ymm2=A[12], ymm3=A[13] already pre-broadcast */

    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 14*4]
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 15*4]

    /* A prefetch (second cache line) */
    prefetcht0 [rax + 304]

    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 16*4]
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 17*4]

    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    /* Load B for K3 + pre-broadcast K3 A[0,1] */
    vmovaps ymm0, [r10 + 2*32]
    vmovaps ymm1, [r10 + 3*32]
    vbroadcastss ymm2, dword ptr [rax + 18*4]
    vbroadcastss ymm3, dword ptr [rax + 19*4]


    /* ============ K step 3 ============ */
    /* ymm2=A[18], ymm3=A[19] already pre-broadcast */

    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 20*4]
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 21*4]

    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 22*4]
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 23*4]

    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    /* Advance pointers */
    add    rax, 4*6*4
    add    r10, 4*16*4

    /* Pre-load B for next K0 + pre-broadcast next K0 A[0,1] */
    vmovaps ymm0, [r10 - 4*32]
    vmovaps ymm1, [r10 - 3*32]
    vbroadcastss ymm2, dword ptr [rax + 0*4]
    vbroadcastss ymm3, dword ptr [rax + 1*4]

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


/* ================================================================== */
/*  gemm_kernel_6x16_pf:  C = A × B  (overwrite, A_next prefetch)    */
/*                                                                     */
/*  K-loop unrolled ×4.  A_next prefetched at K steps 0 and 2.       */
/*  A-self prefetched at K steps 0 and 2 (~2.7 iters ahead).         */
/*                                                                     */
/*  System V AMD64:                                                    */
/*    rdi=A, rsi=B, rdx=C, rcx=K, r8=ldc_bytes, r9=A_next            */
/*  r13 holds A_next pointer throughout the K-loop.                   */
/* ================================================================== */
.globl gemm_kernel_6x16_pf
.type  gemm_kernel_6x16_pf, @function
.align 16
gemm_kernel_6x16_pf:
    push   rbx
    push   r12
    push   r13
    mov    r13, r9

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

    mov    rax, rdi
    mov    r10, rsi

    /* Prefetch C rows with write-intent */
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

    /* Pre-bias B pointer */
    add    r10, 128

    mov    rbx, rcx
    shr    rbx, 2            /* K / 4 */
    mov    r11d, ecx
    and    r11d, 3            /* K % 4 */

    test   rbx, rbx
    jz     .L_pf_k_remainder

    /* Preload B[K0] + pre-broadcast A[K0][rows 0,1] */
    vmovaps ymm0, [r10 - 4*32]
    vmovaps ymm1, [r10 - 3*32]
    vbroadcastss ymm2, dword ptr [rax + 0*4]
    vbroadcastss ymm3, dword ptr [rax + 1*4]

    .align 32
.L_pf_k_unrolled:

    /* ============ K step 0 (A-self + A-next prefetch) ============ */
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 2*4]
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 3*4]

    prefetcht0 [rax + 256]
    prefetcht0 [r13]
    add    r13, 64

    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 4*4]
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 5*4]

    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    vmovaps ymm0, [r10 - 2*32]
    vmovaps ymm1, [r10 - 1*32]
    vbroadcastss ymm2, dword ptr [rax + 6*4]
    vbroadcastss ymm3, dword ptr [rax + 7*4]

    /* ============ K step 1 ============ */
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 8*4]
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 9*4]

    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 10*4]
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 11*4]

    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    vmovaps ymm0, [r10 + 0*32]
    vmovaps ymm1, [r10 + 1*32]
    vbroadcastss ymm2, dword ptr [rax + 12*4]
    vbroadcastss ymm3, dword ptr [rax + 13*4]

    /* ============ K step 2 (A-self + A-next prefetch) ============ */
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 14*4]
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 15*4]

    prefetcht0 [rax + 304]
    prefetcht0 [r13]
    add    r13, 64

    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 16*4]
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 17*4]

    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    vmovaps ymm0, [r10 + 2*32]
    vmovaps ymm1, [r10 + 3*32]
    vbroadcastss ymm2, dword ptr [rax + 18*4]
    vbroadcastss ymm3, dword ptr [rax + 19*4]

    /* ============ K step 3 ============ */
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 20*4]
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 21*4]

    vfmadd231ps  ymm8,  ymm2, ymm0
    vfmadd231ps  ymm9,  ymm2, ymm1
    vbroadcastss ymm2, dword ptr [rax + 22*4]
    vfmadd231ps  ymm10, ymm3, ymm0
    vfmadd231ps  ymm11, ymm3, ymm1
    vbroadcastss ymm3, dword ptr [rax + 23*4]

    vfmadd231ps  ymm12, ymm2, ymm0
    vfmadd231ps  ymm13, ymm2, ymm1
    vfmadd231ps  ymm14, ymm3, ymm0
    vfmadd231ps  ymm15, ymm3, ymm1

    /* Advance pointers: A += 4*MR*4=96, B += 4*NR*4=256 */
    add    rax, 4*6*4
    add    r10, 4*16*4

    /* Preload B[K0] + A[K0][rows 0,1] for next iteration */
    vmovaps ymm0, [r10 - 4*32]
    vmovaps ymm1, [r10 - 3*32]
    vbroadcastss ymm2, dword ptr [rax + 0*4]
    vbroadcastss ymm3, dword ptr [rax + 1*4]

    dec    rbx
    jnz    .L_pf_k_unrolled

/* ---- K remainder (0-3) ---- */
.L_pf_k_remainder:
    test   r11d, r11d
    jz     .L_pf_store

    .align 16
.L_pf_k_remain_loop:
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
    jnz    .L_pf_k_remain_loop

.L_pf_store:
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

    pop    r13
    pop    r12
    pop    rbx

    vzeroupper
    ret

.size gemm_kernel_6x16_pf, .-gemm_kernel_6x16_pf


/* ================================================================== */
/*  gemm_kernel_4x16:  C[4×16] = A × B  (overwrite)                  */
/*                                                                     */
/*  Edge kernel for mr=4.  K-loop unrolled ×4.                        */
/*  A_pack layout: k*MR + m (MR=6, but only rows 0-3 used).          */
/*  Accumulators: ymm4-ymm11 (4 rows × 2 columns = 8 regs).         */
/*                                                                     */
/*  System V AMD64:                                                    */
/*    rdi=A, rsi=B, rdx=C, rcx=K, r8=ldc_bytes                       */
/* ================================================================== */
.globl gemm_kernel_4x16
.type  gemm_kernel_4x16, @function
.align 16
gemm_kernel_4x16:
    push   rbx

    vxorps ymm4,  ymm4,  ymm4
    vxorps ymm5,  ymm5,  ymm5
    vxorps ymm6,  ymm6,  ymm6
    vxorps ymm7,  ymm7,  ymm7
    vxorps ymm8,  ymm8,  ymm8
    vxorps ymm9,  ymm9,  ymm9
    vxorps ymm10, ymm10, ymm10
    vxorps ymm11, ymm11, ymm11

    mov    rax, rdi
    mov    r10, rsi
    add    r10, 128

    mov    rbx, rcx
    shr    rbx, 2
    mov    r11d, ecx
    and    r11d, 3

    test   rbx, rbx
    jz     .L_4x16_k_remainder

    .align 32
.L_4x16_k_unrolled:
    /* K step 0 */
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

    /* K step 1 */
    vmovaps ymm0, [r10 - 2*32]
    vmovaps ymm1, [r10 - 1*32]
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

    /* K step 2 */
    vmovaps ymm0, [r10 + 0*32]
    vmovaps ymm1, [r10 + 1*32]
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

    /* K step 3 */
    vmovaps ymm0, [r10 + 2*32]
    vmovaps ymm1, [r10 + 3*32]
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

    add    rax, 4*6*4
    add    r10, 4*16*4

    dec    rbx
    jnz    .L_4x16_k_unrolled

.L_4x16_k_remainder:
    test   r11d, r11d
    jz     .L_4x16_store

    .align 16
.L_4x16_k_remain:
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

    add    rax, 6*4
    add    r10, 16*4
    dec    r11d
    jnz    .L_4x16_k_remain

.L_4x16_store:
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

    pop    rbx
    vzeroupper
    ret

.size gemm_kernel_4x16, .-gemm_kernel_4x16


/* ================================================================== */
/*  gemm_kernel_2x16:  C[2×16] = A × B  (overwrite)                  */
/*                                                                     */
/*  Edge kernel for mr=2.  K-loop unrolled ×4.                        */
/*  Accumulators: ymm4-ymm7 (2 rows × 2 columns = 4 regs).          */
/*                                                                     */
/*  System V AMD64:                                                    */
/*    rdi=A, rsi=B, rdx=C, rcx=K, r8=ldc_bytes                       */
/* ================================================================== */
.globl gemm_kernel_2x16
.type  gemm_kernel_2x16, @function
.align 16
gemm_kernel_2x16:
    push   rbx

    vxorps ymm4,  ymm4,  ymm4
    vxorps ymm5,  ymm5,  ymm5
    vxorps ymm6,  ymm6,  ymm6
    vxorps ymm7,  ymm7,  ymm7

    mov    rax, rdi
    mov    r10, rsi
    add    r10, 128

    mov    rbx, rcx
    shr    rbx, 2
    mov    r11d, ecx
    and    r11d, 3

    test   rbx, rbx
    jz     .L_2x16_k_remainder

    .align 32
.L_2x16_k_unrolled:
    /* K step 0 */
    vmovaps ymm0, [r10 - 4*32]
    vmovaps ymm1, [r10 - 3*32]
    vbroadcastss ymm2, dword ptr [rax + 0*4]
    vbroadcastss ymm3, dword ptr [rax + 1*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    /* K step 1 */
    vmovaps ymm0, [r10 - 2*32]
    vmovaps ymm1, [r10 - 1*32]
    vbroadcastss ymm2, dword ptr [rax + 6*4]
    vbroadcastss ymm3, dword ptr [rax + 7*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    /* K step 2 */
    vmovaps ymm0, [r10 + 0*32]
    vmovaps ymm1, [r10 + 1*32]
    vbroadcastss ymm2, dword ptr [rax + 12*4]
    vbroadcastss ymm3, dword ptr [rax + 13*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    /* K step 3 */
    vmovaps ymm0, [r10 + 2*32]
    vmovaps ymm1, [r10 + 3*32]
    vbroadcastss ymm2, dword ptr [rax + 18*4]
    vbroadcastss ymm3, dword ptr [rax + 19*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    add    rax, 4*6*4
    add    r10, 4*16*4

    dec    rbx
    jnz    .L_2x16_k_unrolled

.L_2x16_k_remainder:
    test   r11d, r11d
    jz     .L_2x16_store

    .align 16
.L_2x16_k_remain:
    vmovaps ymm0, [r10 - 4*32]
    vmovaps ymm1, [r10 - 3*32]
    vbroadcastss ymm2, dword ptr [rax + 0*4]
    vbroadcastss ymm3, dword ptr [rax + 1*4]
    vfmadd231ps  ymm4,  ymm2, ymm0
    vfmadd231ps  ymm5,  ymm2, ymm1
    vfmadd231ps  ymm6,  ymm3, ymm0
    vfmadd231ps  ymm7,  ymm3, ymm1

    add    rax, 6*4
    add    r10, 16*4
    dec    r11d
    jnz    .L_2x16_k_remain

.L_2x16_store:
    mov    rax, rdx
    vmovups [rax],      ymm4
    vmovups [rax + 32], ymm5
    add    rax, r8
    vmovups [rax],      ymm6
    vmovups [rax + 32], ymm7

    pop    rbx
    vzeroupper
    ret

.size gemm_kernel_2x16, .-gemm_kernel_2x16


.section .note.GNU-stack,"",@progbits
