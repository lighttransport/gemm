// A64FX SVE1.0 (Armv8.2-A+SVE, no SVE2) matrix multiplication micro-kernel.
// Computes an 8-row x 1-SVE-vector-column block of C += A @ B in FP32:
//   C[m][0:VL) += sum_k A[m][k] * B[k][0:VL)   for m = 0..7
// A is row-major [8, K] (K contiguous per row, row stride K floats).
// B is row-major [K, N]; only the first VL columns of each row are read here
// (a caller loops this kernel over N in VL-wide column blocks). C is
// row-major [8, N] with row stride ldc (in FLOATS, not bytes).
//
// AAPCS64 calling convention:
//   x0 = A       (const float*, 8 rows x K contiguous)
//   x1 = B       (const float*, K rows x N contiguous, only first VL cols read)
//   x2 = C       (float*, 8 rows x ldc)
//   x3 = ldc     (int64, C row stride in elements)
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
//   void matmul_kernel_8x1_f32_sve(const float *A, const float *B, float *C,
//                                    long ldc, long K);

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
<<<COMPLETION>>>
	ld1w	z8.s, p0/z, [x20]	// z8 = B[k][0:VL)
	add	x20, x20, x22, lsl #2	// B += ldc (skip to next B row; note: B
					// is row-major with stride N, but we only
					// read the first VL columns; the caller
					// ensures B is laid out so that B[k] is
					// contiguous for VL floats; here we just
					// advance by ldc*4 bytes to the next row.
					// Actually B stride is N, not ldc; but
					// the caller passes B as a pointer to the
					// current VL-wide column block, and the
					// next B row is at B + N. Since we don't
					// have N, we rely on the caller to set
					// ldc = N for this kernel. So we use
					// x22 (ldc) as B stride too.
					// If B stride != ldc, we'd need a
					// separate argument; for simplicity we
					// assume ldc == N here.

	// Load 8 A scalars from the 8 rows at column k.
	ld1rsw	z9.s, p0/z, [x19]		// z9 = A[0][k] broadcast
	add	x19, x19, #4			// A row 0 advances by 1 float
	fmla	z0.s, z8.s, z9.s		// C[0] += A[0][k] * B[k]

	ld1rsw	z9.s, p0/z, [x19, x23, lsl #2]	// A[1][k] (row 1 is at A + K)
	fmla	z1.s, z8.s, z9.s

	add	x19, x19, x23, lsl #2		// move to row 2 (skip K floats)
	ld1rsw	z9.s, p0/z, [x19, x23, lsl #2]	// A[2][k]
	fmla	z2.s, z8.s, z9.s

	add	x19, x19, x23, lsl #2		// row 3
	ld1rsw	z9.s, p0/z, [x19, x23, lsl #2]	// A[3][k]
	fmla	z3.s, z8.s, z9.s

	add	x19, x19, x23, lsl #2		// row 4
	ld1rsw	z9.s, p0/z, [x19, x23, lsl #2]	// A[4][k]
	fmla	z4.s, z8.s, z9.s

	add	x19, x19, x23, lsl #2		// row 5
	ld1rsw	z9.s, p0/z, [x19, x23, lsl #2]	// A[5][k]
	fmla	z5.s, z8.s, z9.s

	add	x19, x19, x23, lsl #2		// row 6
	ld1rsw	z9.s, p0/z, [x19, x23, lsl #2]	// A[6][k]
