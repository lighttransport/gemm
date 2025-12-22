	.text
	.file	"exp_intrinsics_opt.c"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               // -- Begin function exp_f32_poly5_intrin_opt
.LCPI0_0:
	.word	1069066811              // float 1.44269502
.LCPI0_1:
	.word	1060205080              // float 0.693147182
.LCPI0_2:
	.word	1026206379              // float 0.0416666679
.LCPI0_3:
	.word	1007192201              // float 0.00833333377
.LCPI0_4:
	.word	1042983595              // float 0.166666672
	.text
	.globl	exp_f32_poly5_intrin_opt
	.p2align	3
	.type	exp_f32_poly5_intrin_opt,@function
exp_f32_poly5_intrin_opt:               // @exp_f32_poly5_intrin_opt
.Lfunc_begin0:
	.file	1 "/home/u14346/work/gemm/fat" "exp_intrinsics_opt.c"
	.loc	1 17 0                  // exp_intrinsics_opt.c:17:0
	.cfi_startproc
// %bb.0:
	.loc	1 35 20 prologue_end    // exp_intrinsics_opt.c:35:20
	rdvl	x8, #1
	.loc	1 35 5 is_stmt 0        // exp_intrinsics_opt.c:35:5
	cmp	x2, x8
	b.hs	.LBB0_2
// %bb.1:
	.loc	1 0 0                   // exp_intrinsics_opt.c:0:0
	mov	x8, xzr
	.loc	1 105 5 is_stmt 1       // exp_intrinsics_opt.c:105:5
	cmp	x8, x2
	b.lo	.LBB0_5
	b	.LBB0_7
.LBB0_2:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics_opt.c:0:5
	adrp	x9, .LCPI0_0
	ldr	s0, [x9, :lo12:.LCPI0_0]
	adrp	x9, .LCPI0_1
	ldr	s1, [x9, :lo12:.LCPI0_1]
	adrp	x9, .LCPI0_2
	ldr	s2, [x9, :lo12:.LCPI0_2]
	adrp	x9, .LCPI0_3
	ldr	s4, [x9, :lo12:.LCPI0_3]
	adrp	x9, .LCPI0_4
	ldr	s5, [x9, :lo12:.LCPI0_4]
	mov	z0.s, s0
	mov	z3.s, #127              // =0x7f
	fmov	z6.s, #0.50000000
	fmov	z7.s, #1.00000000
	.loc	1 31 19 is_stmt 1       // exp_intrinsics_opt.c:31:19
	ptrue	p0.s
	mov	x8, xzr
	mov	z1.s, s1
	mov	z2.s, s2
	mov	z4.s, s4
	mov	z5.s, s5
	.p2align	2
.LBB0_3:                                // =>This Inner Loop Header: Depth=1
	.loc	1 0 19 is_stmt 0        // exp_intrinsics_opt.c:0:19
	lsl	x9, x8, #2
	add	x10, x0, x9
	.loc	1 66 26 is_stmt 1       // exp_intrinsics_opt.c:66:26
	mov	z24.d, z4.d
	.loc	1 67 26                 // exp_intrinsics_opt.c:67:26
	mov	z25.d, z4.d
	.loc	1 68 26                 // exp_intrinsics_opt.c:68:26
	mov	z26.d, z4.d
	.loc	1 69 26                 // exp_intrinsics_opt.c:69:26
	mov	z27.d, z4.d
	add	x9, x1, x9
	.loc	1 37 26                 // exp_intrinsics_opt.c:37:26
	ld1w	{ z19.s }, p0/z, [x10]
	.loc	1 38 26                 // exp_intrinsics_opt.c:38:26
	ld1w	{ z18.s }, p0/z, [x10, #1, mul vl]
	.loc	1 39 26                 // exp_intrinsics_opt.c:39:26
	ld1w	{ z17.s }, p0/z, [x10, #2, mul vl]
	.loc	1 40 26                 // exp_intrinsics_opt.c:40:26
	ld1w	{ z16.s }, p0/z, [x10, #3, mul vl]
	.loc	1 46 44                 // exp_intrinsics_opt.c:46:44
	mov	z20.d, z16.d
	.loc	1 43 44                 // exp_intrinsics_opt.c:43:44
	mov	z21.d, z19.d
	.loc	1 44 44                 // exp_intrinsics_opt.c:44:44
	mov	z22.d, z18.d
	.loc	1 45 44                 // exp_intrinsics_opt.c:45:44
	mov	z23.d, z17.d
	.loc	1 46 44                 // exp_intrinsics_opt.c:46:44
	fmul	z20.s, p0/m, z20.s, z0.s
	.loc	1 43 44                 // exp_intrinsics_opt.c:43:44
	fmul	z21.s, p0/m, z21.s, z0.s
	.loc	1 44 44                 // exp_intrinsics_opt.c:44:44
	fmul	z22.s, p0/m, z22.s, z0.s
	.loc	1 45 44                 // exp_intrinsics_opt.c:45:44
	fmul	z23.s, p0/m, z23.s, z0.s
	.loc	1 43 26                 // exp_intrinsics_opt.c:43:26
	frintn	z21.s, p0/m, z21.s
	.loc	1 44 26                 // exp_intrinsics_opt.c:44:26
	frintn	z22.s, p0/m, z22.s
	.loc	1 45 26                 // exp_intrinsics_opt.c:45:26
	frintn	z23.s, p0/m, z23.s
	.loc	1 46 26                 // exp_intrinsics_opt.c:46:26
	frintn	z20.s, p0/m, z20.s
	.loc	1 49 26                 // exp_intrinsics_opt.c:49:26
	fmls	z19.s, p0/m, z21.s, z1.s
	.loc	1 50 26                 // exp_intrinsics_opt.c:50:26
	fmls	z18.s, p0/m, z22.s, z1.s
	.loc	1 51 26                 // exp_intrinsics_opt.c:51:26
	fmls	z17.s, p0/m, z23.s, z1.s
	.loc	1 52 26                 // exp_intrinsics_opt.c:52:26
	fmls	z16.s, p0/m, z20.s, z1.s
	.loc	1 58 59                 // exp_intrinsics_opt.c:58:59
	fcvtzs	z20.s, p0/m, z20.s
	.loc	1 66 26                 // exp_intrinsics_opt.c:66:26
	fmad	z24.s, p0/m, z19.s, z2.s
	.loc	1 67 26                 // exp_intrinsics_opt.c:67:26
	fmad	z25.s, p0/m, z18.s, z2.s
	.loc	1 68 26                 // exp_intrinsics_opt.c:68:26
	fmad	z26.s, p0/m, z17.s, z2.s
	.loc	1 69 26                 // exp_intrinsics_opt.c:69:26
	fmad	z27.s, p0/m, z16.s, z2.s
	.loc	1 71 14                 // exp_intrinsics_opt.c:71:14
	fmad	z24.s, p0/m, z19.s, z5.s
	.loc	1 72 14                 // exp_intrinsics_opt.c:72:14
	fmad	z25.s, p0/m, z18.s, z5.s
	.loc	1 73 14                 // exp_intrinsics_opt.c:73:14
	fmad	z26.s, p0/m, z17.s, z5.s
	.loc	1 74 14                 // exp_intrinsics_opt.c:74:14
	fmad	z27.s, p0/m, z16.s, z5.s
	.loc	1 76 14                 // exp_intrinsics_opt.c:76:14
	fmad	z24.s, p0/m, z19.s, z6.s
	.loc	1 77 14                 // exp_intrinsics_opt.c:77:14
	fmad	z25.s, p0/m, z18.s, z6.s
	.loc	1 78 14                 // exp_intrinsics_opt.c:78:14
	fmad	z26.s, p0/m, z17.s, z6.s
	.loc	1 79 14                 // exp_intrinsics_opt.c:79:14
	fmad	z27.s, p0/m, z16.s, z6.s
	.loc	1 81 14                 // exp_intrinsics_opt.c:81:14
	fmad	z24.s, p0/m, z19.s, z7.s
	.loc	1 82 14                 // exp_intrinsics_opt.c:82:14
	fmad	z25.s, p0/m, z18.s, z7.s
	.loc	1 83 14                 // exp_intrinsics_opt.c:83:14
	fmad	z26.s, p0/m, z17.s, z7.s
	.loc	1 84 14                 // exp_intrinsics_opt.c:84:14
	fmad	z27.s, p0/m, z16.s, z7.s
	.loc	1 55 59                 // exp_intrinsics_opt.c:55:59
	fcvtzs	z21.s, p0/m, z21.s
	.loc	1 56 59                 // exp_intrinsics_opt.c:56:59
	fcvtzs	z22.s, p0/m, z22.s
	.loc	1 57 59                 // exp_intrinsics_opt.c:57:59
	fcvtzs	z23.s, p0/m, z23.s
	.loc	1 57 43 is_stmt 0       // exp_intrinsics_opt.c:57:43
	add	z23.s, p0/m, z23.s, z3.s
	.loc	1 57 25                 // exp_intrinsics_opt.c:57:25
	lsl	z23.s, p0/m, z23.s, #23
	.loc	1 86 14 is_stmt 1       // exp_intrinsics_opt.c:86:14
	fmad	z24.s, p0/m, z19.s, z7.s
	.loc	1 87 14                 // exp_intrinsics_opt.c:87:14
	fmad	z25.s, p0/m, z18.s, z7.s
	.loc	1 88 14                 // exp_intrinsics_opt.c:88:14
	fmad	z26.s, p0/m, z17.s, z7.s
	.loc	1 89 14                 // exp_intrinsics_opt.c:89:14
	fmad	z27.s, p0/m, z16.s, z7.s
	.loc	1 55 43                 // exp_intrinsics_opt.c:55:43
	add	z21.s, p0/m, z21.s, z3.s
	.loc	1 56 43                 // exp_intrinsics_opt.c:56:43
	add	z22.s, p0/m, z22.s, z3.s
	.loc	1 58 43                 // exp_intrinsics_opt.c:58:43
	add	z20.s, p0/m, z20.s, z3.s
	.loc	1 55 25                 // exp_intrinsics_opt.c:55:25
	lsl	z21.s, p0/m, z21.s, #23
	.loc	1 56 25                 // exp_intrinsics_opt.c:56:25
	lsl	z22.s, p0/m, z22.s, #23
	.loc	1 58 25                 // exp_intrinsics_opt.c:58:25
	lsl	z20.s, p0/m, z20.s, #23
	.loc	1 94 28                 // exp_intrinsics_opt.c:94:28
	fmul	z26.s, p0/m, z26.s, z23.s
	.loc	1 92 28                 // exp_intrinsics_opt.c:92:28
	fmul	z24.s, p0/m, z24.s, z21.s
	.loc	1 93 28                 // exp_intrinsics_opt.c:93:28
	fmul	z25.s, p0/m, z25.s, z22.s
	.loc	1 95 28                 // exp_intrinsics_opt.c:95:28
	fmul	z27.s, p0/m, z27.s, z20.s
	.loc	1 98 9                  // exp_intrinsics_opt.c:98:9
	st1w	{ z24.s }, p0, [x9]
	.loc	1 99 9                  // exp_intrinsics_opt.c:99:9
	st1w	{ z25.s }, p0, [x9, #1, mul vl]
	.loc	1 100 9                 // exp_intrinsics_opt.c:100:9
	st1w	{ z26.s }, p0, [x9, #2, mul vl]
	.loc	1 101 9                 // exp_intrinsics_opt.c:101:9
	st1w	{ z27.s }, p0, [x9, #3, mul vl]
	.loc	1 35 20                 // exp_intrinsics_opt.c:35:20
	addvl	x9, x8, #2
	addvl	x8, x8, #1
	.loc	1 35 5 is_stmt 0        // exp_intrinsics_opt.c:35:5
	cmp	x9, x2
	b.ls	.LBB0_3
// %bb.4:
	.loc	1 105 5 is_stmt 1       // exp_intrinsics_opt.c:105:5
	cmp	x8, x2
	b.hs	.LBB0_7
.LBB0_5:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics_opt.c:0:5
	adrp	x9, .LCPI0_0
	adrp	x10, .LCPI0_1
	ldr	s1, [x10, :lo12:.LCPI0_1]
	ldr	s0, [x9, :lo12:.LCPI0_0]
	adrp	x9, .LCPI0_2
	ldr	s2, [x9, :lo12:.LCPI0_2]
	adrp	x9, .LCPI0_3
	adrp	x10, .LCPI0_4
	fmov	z7.s, #1.00000000
	mov	z0.s, s0
	ldr	s4, [x9, :lo12:.LCPI0_3]
	ldr	s5, [x10, :lo12:.LCPI0_4]
	mov	z1.s, s1
	mov	z2.s, s2
	mov	z4.s, s4
	mov	z5.s, s5
	mov	z3.s, #127              // =0x7f
	fmov	z6.s, #0.50000000
	.loc	1 106 28 is_stmt 1      // exp_intrinsics_opt.c:106:28
	lsl	x9, x8, #2
	.p2align	2
.LBB0_6:                                // =>This Inner Loop Header: Depth=1
	.loc	1 108 25                // exp_intrinsics_opt.c:108:25
	add	x10, x0, x9
	.loc	1 115 25                // exp_intrinsics_opt.c:115:25
	mov	z18.d, z4.d
	.loc	1 106 28                // exp_intrinsics_opt.c:106:28
	whilelo	p0.s, x8, x2
	.loc	1 105 25                // exp_intrinsics_opt.c:105:25
	incw	x8
	.loc	1 108 25                // exp_intrinsics_opt.c:108:25
	ld1w	{ z16.s }, p0/z, [x10]
	.loc	1 122 9                 // exp_intrinsics_opt.c:122:9
	add	x10, x1, x9
	.loc	1 105 14                // exp_intrinsics_opt.c:105:14
	addvl	x9, x9, #1
	.loc	1 109 48                // exp_intrinsics_opt.c:109:48
	mov	z17.d, z16.d
	fmul	z17.s, p0/m, z17.s, z0.s
	.loc	1 109 25 is_stmt 0      // exp_intrinsics_opt.c:109:25
	frintn	z17.s, p0/m, z17.s
	.loc	1 110 25 is_stmt 1      // exp_intrinsics_opt.c:110:25
	fmls	z16.s, p0/m, z17.s, z1.s
	.loc	1 112 68                // exp_intrinsics_opt.c:112:68
	fcvtzs	z17.s, p0/m, z17.s
	.loc	1 115 25                // exp_intrinsics_opt.c:115:25
	fmad	z18.s, p0/m, z16.s, z2.s
	.loc	1 116 13                // exp_intrinsics_opt.c:116:13
	fmad	z18.s, p0/m, z16.s, z5.s
	.loc	1 117 13                // exp_intrinsics_opt.c:117:13
	fmad	z18.s, p0/m, z16.s, z6.s
	.loc	1 118 13                // exp_intrinsics_opt.c:118:13
	fmad	z18.s, p0/m, z16.s, z7.s
	.loc	1 119 13                // exp_intrinsics_opt.c:119:13
	fmad	z18.s, p0/m, z16.s, z7.s
	.loc	1 112 47                // exp_intrinsics_opt.c:112:47
	add	z17.s, p0/m, z17.s, z3.s
	.loc	1 112 24 is_stmt 0      // exp_intrinsics_opt.c:112:24
	lsl	z17.s, p0/m, z17.s, #23
	.loc	1 121 30 is_stmt 1      // exp_intrinsics_opt.c:121:30
	fmul	z18.s, p0/m, z18.s, z17.s
	.loc	1 122 9                 // exp_intrinsics_opt.c:122:9
	st1w	{ z18.s }, p0, [x10]
	.loc	1 105 5                 // exp_intrinsics_opt.c:105:5
	cmp	x8, x2
	b.lo	.LBB0_6
.LBB0_7:
	.loc	1 124 1                 // exp_intrinsics_opt.c:124:1
	ret
.Ltmp0:
.Lfunc_end0:
	.size	exp_f32_poly5_intrin_opt, .Lfunc_end0-exp_f32_poly5_intrin_opt
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               // -- Begin function exp_f64_poly5_intrin_opt
.LCPI1_0:
	.xword	4609176140021203710     // double 1.4426950408889634
.LCPI1_1:
	.xword	4604418534313441775     // double 0.69314718055994529
.LCPI1_2:
	.xword	4586165620538955093     // double 0.041666666666666664
.LCPI1_3:
	.xword	4575957461383581969     // double 0.0083333333333333332
.LCPI1_4:
	.xword	4595172819793696085     // double 0.16666666666666666
	.text
	.globl	exp_f64_poly5_intrin_opt
	.p2align	3
	.type	exp_f64_poly5_intrin_opt,@function
exp_f64_poly5_intrin_opt:               // @exp_f64_poly5_intrin_opt
.Lfunc_begin1:
	.loc	1 129 0                 // exp_intrinsics_opt.c:129:0
	.cfi_startproc
// %bb.0:
	.loc	1 145 20 prologue_end   // exp_intrinsics_opt.c:145:20
	cnth	x8
	.loc	1 145 5 is_stmt 0       // exp_intrinsics_opt.c:145:5
	cmp	x2, x8
	b.hs	.LBB1_2
// %bb.1:
	.loc	1 0 0                   // exp_intrinsics_opt.c:0:0
	mov	x8, xzr
	.loc	1 207 5 is_stmt 1       // exp_intrinsics_opt.c:207:5
	cmp	x8, x2
	b.lo	.LBB1_5
	b	.LBB1_7
.LBB1_2:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics_opt.c:0:5
	adrp	x9, .LCPI1_0
	ldr	d0, [x9, :lo12:.LCPI1_0]
	adrp	x9, .LCPI1_1
	ldr	d1, [x9, :lo12:.LCPI1_1]
	adrp	x9, .LCPI1_2
	ldr	d2, [x9, :lo12:.LCPI1_2]
	adrp	x9, .LCPI1_3
	ldr	d4, [x9, :lo12:.LCPI1_3]
	adrp	x9, .LCPI1_4
	ldr	d5, [x9, :lo12:.LCPI1_4]
	mov	z0.d, d0
	mov	z3.d, #1023             // =0x3ff
	fmov	z6.d, #0.50000000
	fmov	z7.d, #1.00000000
	.loc	1 142 19 is_stmt 1      // exp_intrinsics_opt.c:142:19
	ptrue	p0.d
	mov	x8, xzr
	mov	z1.d, d1
	mov	z2.d, d2
	mov	z4.d, d4
	mov	z5.d, d5
	.p2align	2
.LBB1_3:                                // =>This Inner Loop Header: Depth=1
	.loc	1 0 19 is_stmt 0        // exp_intrinsics_opt.c:0:19
	lsl	x9, x8, #3
	add	x10, x0, x9
	.loc	1 171 26 is_stmt 1      // exp_intrinsics_opt.c:171:26
	mov	z24.d, z4.d
	.loc	1 172 26                // exp_intrinsics_opt.c:172:26
	mov	z25.d, z4.d
	.loc	1 173 26                // exp_intrinsics_opt.c:173:26
	mov	z26.d, z4.d
	.loc	1 174 26                // exp_intrinsics_opt.c:174:26
	mov	z27.d, z4.d
	add	x9, x1, x9
	.loc	1 146 26                // exp_intrinsics_opt.c:146:26
	ld1d	{ z19.d }, p0/z, [x10]
	.loc	1 147 26                // exp_intrinsics_opt.c:147:26
	ld1d	{ z18.d }, p0/z, [x10, #1, mul vl]
	.loc	1 148 26                // exp_intrinsics_opt.c:148:26
	ld1d	{ z17.d }, p0/z, [x10, #2, mul vl]
	.loc	1 149 26                // exp_intrinsics_opt.c:149:26
	ld1d	{ z16.d }, p0/z, [x10, #3, mul vl]
	.loc	1 154 44                // exp_intrinsics_opt.c:154:44
	mov	z20.d, z16.d
	.loc	1 151 44                // exp_intrinsics_opt.c:151:44
	mov	z21.d, z19.d
	.loc	1 152 44                // exp_intrinsics_opt.c:152:44
	mov	z22.d, z18.d
	.loc	1 153 44                // exp_intrinsics_opt.c:153:44
	mov	z23.d, z17.d
	.loc	1 154 44                // exp_intrinsics_opt.c:154:44
	fmul	z20.d, p0/m, z20.d, z0.d
	.loc	1 151 44                // exp_intrinsics_opt.c:151:44
	fmul	z21.d, p0/m, z21.d, z0.d
	.loc	1 152 44                // exp_intrinsics_opt.c:152:44
	fmul	z22.d, p0/m, z22.d, z0.d
	.loc	1 153 44                // exp_intrinsics_opt.c:153:44
	fmul	z23.d, p0/m, z23.d, z0.d
	.loc	1 151 26                // exp_intrinsics_opt.c:151:26
	frintn	z21.d, p0/m, z21.d
	.loc	1 152 26                // exp_intrinsics_opt.c:152:26
	frintn	z22.d, p0/m, z22.d
	.loc	1 153 26                // exp_intrinsics_opt.c:153:26
	frintn	z23.d, p0/m, z23.d
	.loc	1 154 26                // exp_intrinsics_opt.c:154:26
	frintn	z20.d, p0/m, z20.d
	.loc	1 156 26                // exp_intrinsics_opt.c:156:26
	fmls	z19.d, p0/m, z21.d, z1.d
	.loc	1 157 26                // exp_intrinsics_opt.c:157:26
	fmls	z18.d, p0/m, z22.d, z1.d
	.loc	1 158 26                // exp_intrinsics_opt.c:158:26
	fmls	z17.d, p0/m, z23.d, z1.d
	.loc	1 159 26                // exp_intrinsics_opt.c:159:26
	fmls	z16.d, p0/m, z20.d, z1.d
	.loc	1 164 59                // exp_intrinsics_opt.c:164:59
	fcvtzs	z20.d, p0/m, z20.d
	.loc	1 171 26                // exp_intrinsics_opt.c:171:26
	fmad	z24.d, p0/m, z19.d, z2.d
	.loc	1 172 26                // exp_intrinsics_opt.c:172:26
	fmad	z25.d, p0/m, z18.d, z2.d
	.loc	1 173 26                // exp_intrinsics_opt.c:173:26
	fmad	z26.d, p0/m, z17.d, z2.d
	.loc	1 174 26                // exp_intrinsics_opt.c:174:26
	fmad	z27.d, p0/m, z16.d, z2.d
	.loc	1 176 14                // exp_intrinsics_opt.c:176:14
	fmad	z24.d, p0/m, z19.d, z5.d
	.loc	1 177 14                // exp_intrinsics_opt.c:177:14
	fmad	z25.d, p0/m, z18.d, z5.d
	.loc	1 178 14                // exp_intrinsics_opt.c:178:14
	fmad	z26.d, p0/m, z17.d, z5.d
	.loc	1 179 14                // exp_intrinsics_opt.c:179:14
	fmad	z27.d, p0/m, z16.d, z5.d
	.loc	1 181 14                // exp_intrinsics_opt.c:181:14
	fmad	z24.d, p0/m, z19.d, z6.d
	.loc	1 182 14                // exp_intrinsics_opt.c:182:14
	fmad	z25.d, p0/m, z18.d, z6.d
	.loc	1 183 14                // exp_intrinsics_opt.c:183:14
	fmad	z26.d, p0/m, z17.d, z6.d
	.loc	1 184 14                // exp_intrinsics_opt.c:184:14
	fmad	z27.d, p0/m, z16.d, z6.d
	.loc	1 186 14                // exp_intrinsics_opt.c:186:14
	fmad	z24.d, p0/m, z19.d, z7.d
	.loc	1 187 14                // exp_intrinsics_opt.c:187:14
	fmad	z25.d, p0/m, z18.d, z7.d
	.loc	1 188 14                // exp_intrinsics_opt.c:188:14
	fmad	z26.d, p0/m, z17.d, z7.d
	.loc	1 189 14                // exp_intrinsics_opt.c:189:14
	fmad	z27.d, p0/m, z16.d, z7.d
	.loc	1 161 59                // exp_intrinsics_opt.c:161:59
	fcvtzs	z21.d, p0/m, z21.d
	.loc	1 162 59                // exp_intrinsics_opt.c:162:59
	fcvtzs	z22.d, p0/m, z22.d
	.loc	1 163 59                // exp_intrinsics_opt.c:163:59
	fcvtzs	z23.d, p0/m, z23.d
	.loc	1 163 43 is_stmt 0      // exp_intrinsics_opt.c:163:43
	add	z23.d, p0/m, z23.d, z3.d
	.loc	1 163 25                // exp_intrinsics_opt.c:163:25
	lsl	z23.d, p0/m, z23.d, #52
	.loc	1 191 14 is_stmt 1      // exp_intrinsics_opt.c:191:14
	fmad	z24.d, p0/m, z19.d, z7.d
	.loc	1 192 14                // exp_intrinsics_opt.c:192:14
	fmad	z25.d, p0/m, z18.d, z7.d
	.loc	1 193 14                // exp_intrinsics_opt.c:193:14
	fmad	z26.d, p0/m, z17.d, z7.d
	.loc	1 194 14                // exp_intrinsics_opt.c:194:14
	fmad	z27.d, p0/m, z16.d, z7.d
	.loc	1 161 43                // exp_intrinsics_opt.c:161:43
	add	z21.d, p0/m, z21.d, z3.d
	.loc	1 162 43                // exp_intrinsics_opt.c:162:43
	add	z22.d, p0/m, z22.d, z3.d
	.loc	1 164 43                // exp_intrinsics_opt.c:164:43
	add	z20.d, p0/m, z20.d, z3.d
	.loc	1 161 25                // exp_intrinsics_opt.c:161:25
	lsl	z21.d, p0/m, z21.d, #52
	.loc	1 162 25                // exp_intrinsics_opt.c:162:25
	lsl	z22.d, p0/m, z22.d, #52
	.loc	1 164 25                // exp_intrinsics_opt.c:164:25
	lsl	z20.d, p0/m, z20.d, #52
	.loc	1 198 28                // exp_intrinsics_opt.c:198:28
	fmul	z26.d, p0/m, z26.d, z23.d
	.loc	1 196 28                // exp_intrinsics_opt.c:196:28
	fmul	z24.d, p0/m, z24.d, z21.d
	.loc	1 197 28                // exp_intrinsics_opt.c:197:28
	fmul	z25.d, p0/m, z25.d, z22.d
	.loc	1 199 28                // exp_intrinsics_opt.c:199:28
	fmul	z27.d, p0/m, z27.d, z20.d
	.loc	1 201 9                 // exp_intrinsics_opt.c:201:9
	st1d	{ z24.d }, p0, [x9]
	.loc	1 202 9                 // exp_intrinsics_opt.c:202:9
	st1d	{ z25.d }, p0, [x9, #1, mul vl]
	.loc	1 203 9                 // exp_intrinsics_opt.c:203:9
	st1d	{ z26.d }, p0, [x9, #2, mul vl]
	.loc	1 204 9                 // exp_intrinsics_opt.c:204:9
	st1d	{ z27.d }, p0, [x9, #3, mul vl]
	.loc	1 145 20                // exp_intrinsics_opt.c:145:20
	addvl	x9, x8, #1
	inch	x8
	.loc	1 145 5 is_stmt 0       // exp_intrinsics_opt.c:145:5
	cmp	x9, x2
	b.ls	.LBB1_3
// %bb.4:
	.loc	1 207 5 is_stmt 1       // exp_intrinsics_opt.c:207:5
	cmp	x8, x2
	b.hs	.LBB1_7
.LBB1_5:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics_opt.c:0:5
	adrp	x9, .LCPI1_0
	adrp	x10, .LCPI1_1
	ldr	d1, [x10, :lo12:.LCPI1_1]
	ldr	d0, [x9, :lo12:.LCPI1_0]
	adrp	x9, .LCPI1_2
	ldr	d2, [x9, :lo12:.LCPI1_2]
	adrp	x9, .LCPI1_3
	adrp	x10, .LCPI1_4
	fmov	z7.d, #1.00000000
	mov	z0.d, d0
	ldr	d4, [x9, :lo12:.LCPI1_3]
	ldr	d5, [x10, :lo12:.LCPI1_4]
	mov	z1.d, d1
	mov	z2.d, d2
	mov	z4.d, d4
	mov	z5.d, d5
	mov	z3.d, #1023             // =0x3ff
	fmov	z6.d, #0.50000000
	.loc	1 208 28 is_stmt 1      // exp_intrinsics_opt.c:208:28
	lsl	x9, x8, #3
	.p2align	2
.LBB1_6:                                // =>This Inner Loop Header: Depth=1
	.loc	1 210 25                // exp_intrinsics_opt.c:210:25
	add	x10, x0, x9
	.loc	1 217 25                // exp_intrinsics_opt.c:217:25
	mov	z18.d, z4.d
	.loc	1 208 28                // exp_intrinsics_opt.c:208:28
	whilelo	p0.d, x8, x2
	.loc	1 207 25                // exp_intrinsics_opt.c:207:25
	incd	x8
	.loc	1 210 25                // exp_intrinsics_opt.c:210:25
	ld1d	{ z16.d }, p0/z, [x10]
	.loc	1 224 9                 // exp_intrinsics_opt.c:224:9
	add	x10, x1, x9
	.loc	1 207 14                // exp_intrinsics_opt.c:207:14
	addvl	x9, x9, #1
	.loc	1 211 48                // exp_intrinsics_opt.c:211:48
	mov	z17.d, z16.d
	fmul	z17.d, p0/m, z17.d, z0.d
	.loc	1 211 25 is_stmt 0      // exp_intrinsics_opt.c:211:25
	frintn	z17.d, p0/m, z17.d
	.loc	1 212 25 is_stmt 1      // exp_intrinsics_opt.c:212:25
	fmls	z16.d, p0/m, z17.d, z1.d
	.loc	1 214 68                // exp_intrinsics_opt.c:214:68
	fcvtzs	z17.d, p0/m, z17.d
	.loc	1 217 25                // exp_intrinsics_opt.c:217:25
	fmad	z18.d, p0/m, z16.d, z2.d
	.loc	1 218 13                // exp_intrinsics_opt.c:218:13
	fmad	z18.d, p0/m, z16.d, z5.d
	.loc	1 219 13                // exp_intrinsics_opt.c:219:13
	fmad	z18.d, p0/m, z16.d, z6.d
	.loc	1 220 13                // exp_intrinsics_opt.c:220:13
	fmad	z18.d, p0/m, z16.d, z7.d
	.loc	1 221 13                // exp_intrinsics_opt.c:221:13
	fmad	z18.d, p0/m, z16.d, z7.d
	.loc	1 214 47                // exp_intrinsics_opt.c:214:47
	add	z17.d, p0/m, z17.d, z3.d
	.loc	1 214 24 is_stmt 0      // exp_intrinsics_opt.c:214:24
	lsl	z17.d, p0/m, z17.d, #52
	.loc	1 223 30 is_stmt 1      // exp_intrinsics_opt.c:223:30
	fmul	z18.d, p0/m, z18.d, z17.d
	.loc	1 224 9                 // exp_intrinsics_opt.c:224:9
	st1d	{ z18.d }, p0, [x10]
	.loc	1 207 5                 // exp_intrinsics_opt.c:207:5
	cmp	x8, x2
	b.lo	.LBB1_6
.LBB1_7:
	.loc	1 226 1                 // exp_intrinsics_opt.c:226:1
	ret
.Ltmp1:
.Lfunc_end1:
	.size	exp_f64_poly5_intrin_opt, .Lfunc_end1-exp_f64_poly5_intrin_opt
	.cfi_endproc
                                        // -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang: Fujitsu C/C++ Compiler 4.12.1 (May 27 2025 19:39:57) (based on LLVM 7.1.0)" // string offset=0
.Linfo_string1:
	.asciz	"exp_intrinsics_opt.c"  // string offset=82
.Linfo_string2:
	.asciz	"/home/u14346/work/gemm/fat" // string offset=103
.Linfo_string3:
	.asciz	"exp_f32_poly5_intrin_opt" // string offset=130
.Linfo_string4:
	.asciz	"exp_f64_poly5_intrin_opt" // string offset=155
	.section	.debug_abbrev,"",@progbits
	.byte	1                       // Abbreviation Code
	.byte	17                      // DW_TAG_compile_unit
	.byte	1                       // DW_CHILDREN_yes
	.byte	37                      // DW_AT_producer
	.byte	14                      // DW_FORM_strp
	.byte	19                      // DW_AT_language
	.byte	5                       // DW_FORM_data2
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	16                      // DW_AT_stmt_list
	.byte	23                      // DW_FORM_sec_offset
	.byte	27                      // DW_AT_comp_dir
	.byte	14                      // DW_FORM_strp
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	2                       // Abbreviation Code
	.byte	46                      // DW_TAG_subprogram
	.byte	1                       // DW_CHILDREN_yes
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	3                       // Abbreviation Code
	.ascii	"\200\340\003"          // DW_TAG_FJ_loop
	.byte	0                       // DW_CHILDREN_no
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.ascii	"\200f"                 // DW_AT_FJ_loop_start_line
	.byte	11                      // DW_FORM_data1
	.ascii	"\201f"                 // DW_AT_FJ_loop_end_line
	.byte	11                      // DW_FORM_data1
	.ascii	"\202f"                 // DW_AT_FJ_loop_nest_level
	.byte	11                      // DW_FORM_data1
	.ascii	"\203f"                 // DW_AT_FJ_loop_type
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	0                       // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	103                     // Length of Unit
	.hword	4                       // DWARF version number
	.word	.debug_abbrev           // Offset Into Abbrev. Section
	.byte	8                       // Address Size (in bytes)
	.byte	1                       // Abbrev [1] 0xb:0x60 DW_TAG_compile_unit
	.word	.Linfo_string0          // DW_AT_producer
	.hword	12                      // DW_AT_language
	.word	.Linfo_string1          // DW_AT_name
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.word	.Linfo_string2          // DW_AT_comp_dir
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end1-.Lfunc_begin0 // DW_AT_high_pc
	.byte	2                       // Abbrev [2] 0x2a:0x20 DW_TAG_subprogram
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.word	.Linfo_string3          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	17                      // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0x3d:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	35                      // DW_AT_FJ_loop_start_line
	.byte	102                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	3                       // Abbrev [3] 0x43:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	105                     // DW_AT_FJ_loop_start_line
	.byte	123                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	2                       // Abbrev [2] 0x4a:0x20 DW_TAG_subprogram
	.xword	.Lfunc_begin1           // DW_AT_low_pc
	.word	.Lfunc_end1-.Lfunc_begin1 // DW_AT_high_pc
	.word	.Linfo_string4          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	129                     // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0x5d:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	145                     // DW_AT_FJ_loop_start_line
	.byte	205                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	3                       // Abbrev [3] 0x63:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	207                     // DW_AT_FJ_loop_start_line
	.byte	225                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
	.section	.debug_macinfo,"",@progbits
	.byte	0                       // End Of Macro List Mark

	.ident	"clang: Fujitsu C/C++ Compiler 4.12.1 (May 27 2025 19:39:57) (based on LLVM 7.1.0)"
	.section	.fj.compile_info, "e"
	.ascii	"C::clang"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
