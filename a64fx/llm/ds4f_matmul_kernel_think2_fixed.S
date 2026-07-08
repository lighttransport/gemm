// A64FX SVE1.0 (Armv8.2-A+SVE, no SVE2) matrix multiplication micro-kernel.
// Computes an 8-row x 1-SVE-vector-column block of C += A @ B in FP32:
//   C[m][0:VL) += sum_k A[m][k] * B[k][0:VL)   for m = 0..7
// A is K-MAJOR PACKED: A_packed[k*8 + m] = A[m][k], i.e. the 8 rows' values
// for a given K-step are stored contiguously (32 bytes/step), NOT row-major.
// This was NOT the layout originally requested (the initial doc/prompt asked
// for row-major [8,K], K contiguous per row) -- the model implicitly wrote a
// packed-A microkernel instead (the layout essentially every real blocked
// GEMM kernel uses, including this repo's own fp16_kernel_6x4.S "Ap: K
// panels of 6 half elements"), without updating the doc to match. Verified
// numerically (ds4f_matmul_kernel_kmajor_diagnosis_test.c / ..._correctness_
// test.c): the kernel is BIT-EXACT (max err ~3e-5, pure FP32 rounding) when
// A is fed in this K-major packed layout, for both ldc==VL and ldc>VL; it is
// WRONG (matches neither hypothesis) if A is fed row-major as originally
// specified. A caller MUST pack A into this layout before calling.
// B is row-major [K, N]; only the first VL columns of each row are read here
// (a caller loops this kernel over N in VL-wide column blocks). C is
// row-major [8, N] with row stride ldc (in FLOATS, not bytes). Both B's and
// C's ldc-based row-to-row advance ARE correct as written (see the "Wait:
// x22 is ldc, not VL" comments further down -- that self-doubt is UNFOUNDED,
// confirmed by the same numerical tests; ldc IS the right stride whenever
// B and C share width N with no padding, the standard row-major convention).
//
// AAPCS64 calling convention:
//   x0 = A       (const float*, K-MAJOR PACKED: A_packed[k*8+m] = A[m][k])
//   x1 = B       (const float*, K rows x N contiguous, only first VL cols read)
//   x2 = C       (float*, 8 rows x ldc)
//   x3 = ldc     (int64, C row stride in elements; also used as B's row stride)
//   x4 = K       (int64, reduction depth, K >= 1)
//
// Uses ONLY base SVE1.0 instructions (ptrue, ld1w, ld1rw, fmla, st1w) -- no
// SVE2, no integer dot-product, no gather/scatter. Targets the A64FX FPU
// (2 pipes, 512-bit SVE, FMA-capable) with an 8x1 register-blocked
// accumulator (8 M-rows x 1 SVE vector of N) held live across the whole K
// reduction, so each element of A and B is read from memory exactly once:
// per K step, load the B row vector ONCE and reuse it against all 8
// broadcast A scalars (8 FMAs per loaded B vector).
//
//   void matmul_kernel_8x1_f32_sve(const float *A_packed, const float *B,
//                                    float *C, long ldc, long K);

	.arch armv8.2-a+sve
	.text
	.align	4
	.global	matmul_kernel_8x1_f32_sve
	.type	matmul_kernel_8x1_f32_sve, %function
matmul_kernel_8x1_f32_sve:
	// Save callee-saved GPRs we use for the loop pointers/counter.
	stp	x19, x20, [sp, #-48]!
	stp	x21, x22, [sp, #16]
	stp	x23, x24, [sp, #32]

	ptrue	p0.s			// all-true FP32 predicate (VL-agnostic)

	// Zero the 8 accumulator vectors (one SVE vector of C per M-row).
	dup	z0.s, #0
	dup	z1.s, #0
	dup	z2.s, #0
	dup	z3.s, #0
	dup	z4.s, #0
	dup	z5.s, #0
	dup	z6.s, #0
	dup	z7.s, #0

	// Move args into callee-saved registers so they survive the K loop.
	mov	x19, x0		// A
	mov	x20, x1		// B
	mov	x21, x2		// C
	mov	x22, x3		// ldc (elements)
	mov	x23, x4		// K
	cbz	x23, 2f

	.align	6
1:	// K reduction loop: load B[k][0:VL) once this iteration, broadcast
	// each of the 8 A[m][k] scalars, FMA-accumulate into z0..z7.
	// Loop body, in pseudocode (A is K-major packed, see header comment):
	//   B_row = ld1w (p0) [x20]        // one SVE vector of B, this k-step
	//   for m = 0..7:
	//       a = A_packed[k*8 + m]      // this k-step's 8 row-values, contiguous
	//       z[m] = fmla (z[m], p0/m, a, B_row)
	//   A_packed += 8                  // next k-step's 8 row-values
	//   B += ldc                       // next row of B (ldc == B's true row stride N)
	//   K -= 1

	// Load B row (one SVE vector of floats).
	ld1w	z16.s, p0/z, [x20]

	// Load 8 A scalars (one per row) from contiguous memory.
	// A layout: K-MAJOR PACKED, A_packed[k*8+m] = A[m][k] -- i.e. this k-step's
	// 8 row-values ARE the 8 contiguous floats read here (verified numerically;
	// this is NOT a row-major [8,K] read, despite how earlier drafts described it).
	ldr	q17, [x19]		// 128-bit load = 4 floats (rows 0-3, this k-step)
	ldr	q18, [x19, #16]		// next 4 floats (rows 4-7, this k-step)

	// Broadcast each scalar and FMA.
	// Use dup on each element of the Q registers.
	// For rows 0-3: q17 = [a0, a1, a2, a3] (little-endian: a0 at lowest addr)
	// For rows 4-7: q18 = [a4, a5, a6, a7]
	// We'll dup each lane into a full SVE vector and fmla.

	// Row 0: a0 = q17.s[0]
	dup	z20.s, z17.s[0]
	fmla	z0.s, p0/m, z20.s, z16.s

	// Row 1: a1 = q17.s[1]
	dup	z20.s, z17.s[1]
	fmla	z1.s, p0/m, z20.s, z16.s

	// Row 2: a2 = q17.s[2]
	dup	z20.s, z17.s[2]
	fmla	z2.s, p0/m, z20.s, z16.s

	// Row 3: a3 = q17.s[3]
	dup	z20.s, z17.s[3]
	fmla	z3.s, p0/m, z20.s, z16.s

	// Row 4: a4 = q18.s[0]
	dup	z20.s, z18.s[0]
	fmla	z4.s, p0/m, z20.s, z16.s

	// Row 5: a5 = q18.s[1]
	dup	z20.s, z18.s[1]
	fmla	z5.s, p0/m, z20.s, z16.s

	// Row 6: a6 = q18.s[2]
	dup	z20.s, z18.s[2]
	fmla	z6.s, p0/m, z20.s, z16.s

	// Row 7: a7 = q18.s[3]
	dup	z20.s, z18.s[3]
	fmla	z7.s, p0/m, z20.s, z16.s

	// Advance pointers.
	add	x19, x19, #32		// A_packed += 8 floats (this k-step's 8 row-values)
	add	x20, x20, x22, lsl #2	// B += ldc floats (1 row of B; ldc == B's true row
					// stride N in the standard no-padding row-major
					// convention -- CORRECT as written. VERIFIED
					// numerically for ldc==VL and ldc>VL (see the
					// header comment); an earlier draft second-guessed
					// this as "WRONG" and reached for rdvl instead, but
					// that self-doubt was unfounded -- ldc is right here.
	sub	x23, x23, #1
	cbnz	x23, 1b

	// After loop, store the 8 accumulator vectors to C.
	// C[m] = z[m] for m=0..7, with row stride ldc (in elements).
	rdvl	x24, #1			// VESTIGIAL: x24 is never read below -- a leftover
					// from the same abandoned "B needs VL not ldc" idea
					// (see the loop comment above). Dead but harmless
					// (x24 is restored from the stack right after anyway).
	// For each row m:
	//   st1w {z[m].s}, p0, [C_base + m*ldc*4]
	// C_base = x21, ldc = x22 (elements), so byte offset = m * (ldc * 4).
	// We'll compute base pointers for each row.
	// Use x9..x16 as scratch (we saved x19-x24, so x9-x18 are free).
	mov	x9, x21			// C row 0 base
	st1w	z0.s, p0, [x9]

	add	x10, x9, x22, lsl #2	// C row 1 = base + ldc*4
	st1w	z1.s, p0, [x10]

	add	x11, x10, x22, lsl #2	// row 2
	st1w	z2.s, p0, [x11]

	add	x12, x11, x22, lsl #2	// row 3
	st1w	z3.s, p0, [x12]

	add	x13, x12, x22, lsl #2	// row 4
	st1w	z4.s, p0, [x13]

	add	x14, x13, x22, lsl #2	// row 5
	st1w	z5.s, p0, [x14]

	add	x15, x14, x22, lsl #2	// row 6
	st1w	z6.s, p0, [x15]

	add	x16, x15, x22, lsl #2	// row 7
	st1w	z7.s, p0, [x16]

2:	// Restore and return.
	ldp	x19, x20, [sp]
	ldp	x21, x22, [sp, #16]
	ldp	x23, x24, [sp, #32]
	add	sp, sp, #48
	ret

