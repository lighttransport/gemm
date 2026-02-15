	// FP16 5x4 micro-kernel for A64FX (SVE 512-bit)
	// Layout:
	//   Ap: K panels of 5 half elements (stride 10 bytes per k)
	//   Bp: K panels of 4 SVE vectors (stride 256 bytes per k, assuming VL=64B)
	//   C : row-major, ldc is in elements
	.text
	.align	2
	.global	micro_kernel_5x4_f16_sve
	.type	micro_kernel_5x4_f16_sve, %function
micro_kernel_5x4_f16_sve:
	// Save callee-saved GPRs used in the kernel
	sub	sp, sp, #48
	stp	x19, x20, [sp]
	stp	x21, x22, [sp, #16]
	stp	x23, x24, [sp, #32]

	ptrue	p0.h

	// Zero accumulators z0-z19 (5 rows x 4 cols)
	dup	z0.h, #0
	dup	z1.h, #0
	dup	z2.h, #0
	dup	z3.h, #0
	dup	z4.h, #0
	dup	z5.h, #0
	dup	z6.h, #0
	dup	z7.h, #0
	dup	z8.h, #0
	dup	z9.h, #0
	dup	z10.h, #0
	dup	z11.h, #0
	dup	z12.h, #0
	dup	z13.h, #0
	dup	z14.h, #0
	dup	z15.h, #0
	dup	z16.h, #0
	dup	z17.h, #0
	dup	z18.h, #0
	dup	z19.h, #0

	// Move args into callee-saved registers
	mov	x19, x0		// Ap
	mov	x20, x1		// Bp
	mov	x21, x2		// C
	mov	x22, x3		// ldc (elements)
	mov	x23, x4		// K
	cbz	x23, 2f

	.align	6
1:	// K loop
	ld1h	{z24.h}, p0/z, [x20, #0, mul vl]
	ld1h	{z25.h}, p0/z, [x20, #1, mul vl]
	ld1h	{z26.h}, p0/z, [x20, #2, mul vl]
	ld1h	{z27.h}, p0/z, [x20, #3, mul vl]

	ld1rh	{z28.h}, p0/z, [x19, #0]
	fmla	z0.h, p0/m, z28.h, z24.h
	fmla	z1.h, p0/m, z28.h, z25.h
	fmla	z2.h, p0/m, z28.h, z26.h
	fmla	z3.h, p0/m, z28.h, z27.h

	ld1rh	{z29.h}, p0/z, [x19, #2]
	fmla	z4.h, p0/m, z29.h, z24.h
	fmla	z5.h, p0/m, z29.h, z25.h
	fmla	z6.h, p0/m, z29.h, z26.h
	fmla	z7.h, p0/m, z29.h, z27.h

	ld1rh	{z30.h}, p0/z, [x19, #4]
	fmla	z8.h, p0/m, z30.h, z24.h
	fmla	z9.h, p0/m, z30.h, z25.h
	fmla	z10.h, p0/m, z30.h, z26.h
	fmla	z11.h, p0/m, z30.h, z27.h

	ld1rh	{z31.h}, p0/z, [x19, #6]
	fmla	z12.h, p0/m, z31.h, z24.h
	fmla	z13.h, p0/m, z31.h, z25.h
	fmla	z14.h, p0/m, z31.h, z26.h
	fmla	z15.h, p0/m, z31.h, z27.h

	ld1rh	{z28.h}, p0/z, [x19, #8]
	fmla	z16.h, p0/m, z28.h, z24.h
	fmla	z17.h, p0/m, z28.h, z25.h
	fmla	z18.h, p0/m, z28.h, z26.h
	fmla	z19.h, p0/m, z28.h, z27.h

	add	x19, x19, #10		// advance Ap (5 halves)
	add	x20, x20, #256		// advance Bp (4 vectors)
	subs	x23, x23, #1
	b.ne	1b

2:
	lsl	x22, x22, #1		// ldc in bytes

	st1h	{z0.h}, p0, [x21, #0, mul vl]
	st1h	{z1.h}, p0, [x21, #1, mul vl]
	st1h	{z2.h}, p0, [x21, #2, mul vl]
	st1h	{z3.h}, p0, [x21, #3, mul vl]
	add	x21, x21, x22

	st1h	{z4.h}, p0, [x21, #0, mul vl]
	st1h	{z5.h}, p0, [x21, #1, mul vl]
	st1h	{z6.h}, p0, [x21, #2, mul vl]
	st1h	{z7.h}, p0, [x21, #3, mul vl]
	add	x21, x21, x22

	st1h	{z8.h}, p0, [x21, #0, mul vl]
	st1h	{z9.h}, p0, [x21, #1, mul vl]
	st1h	{z10.h}, p0, [x21, #2, mul vl]
	st1h	{z11.h}, p0, [x21, #3, mul vl]
	add	x21, x21, x22

	st1h	{z12.h}, p0, [x21, #0, mul vl]
	st1h	{z13.h}, p0, [x21, #1, mul vl]
	st1h	{z14.h}, p0, [x21, #2, mul vl]
	st1h	{z15.h}, p0, [x21, #3, mul vl]
	add	x21, x21, x22

	st1h	{z16.h}, p0, [x21, #0, mul vl]
	st1h	{z17.h}, p0, [x21, #1, mul vl]
	st1h	{z18.h}, p0, [x21, #2, mul vl]
	st1h	{z19.h}, p0, [x21, #3, mul vl]

	// Restore callee-saved registers and return
	ldp	x19, x20, [sp]
	ldp	x21, x22, [sp, #16]
	ldp	x23, x24, [sp, #32]
	add	sp, sp, #48
	ret
	.size	micro_kernel_5x4_f16_sve, .-micro_kernel_5x4_f16_sve
