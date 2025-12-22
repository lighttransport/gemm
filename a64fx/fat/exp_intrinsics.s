	.text
	.file	"exp_intrinsics.c"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               // -- Begin function exp_f32_poly1_intrin
.LCPI0_0:
	.word	1069066811              // float 1.44269502
.LCPI0_1:
	.word	1060205080              // float 0.693147182
	.text
	.globl	exp_f32_poly1_intrin
	.p2align	3
	.type	exp_f32_poly1_intrin,@function
exp_f32_poly1_intrin:                   // @exp_f32_poly1_intrin
.Lfunc_begin0:
	.file	1 "/home/u14346/work/gemm/fat" "exp_intrinsics.c"
	.loc	1 18 0                  // exp_intrinsics.c:18:0
	.cfi_startproc
// %bb.0:
	.loc	1 21 5 prologue_end     // exp_intrinsics.c:21:5
	cbz	x2, .LBB0_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI0_0
	ldr	s0, [x9, :lo12:.LCPI0_0]
	adrp	x9, .LCPI0_1
	ldr	s1, [x9, :lo12:.LCPI0_1]
	mov	z0.s, s0
	mov	z1.s, s1
	mov	z2.s, #127              // =0x7f
	fmov	z3.s, #1.00000000
	mov	x8, xzr
	.p2align	2
.LBB0_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 41 25 is_stmt 1       // exp_intrinsics.c:41:25
	mov	z4.d, z3.d
	.loc	1 22 23                 // exp_intrinsics.c:22:23
	whilelo	p0.s, x8, x2
	.loc	1 25 25                 // exp_intrinsics.c:25:25
	ld1w	{ z5.s }, p0/z, [x0, x8, lsl #2]
	.loc	1 28 25                 // exp_intrinsics.c:28:25
	mov	z6.d, z5.d
	fmul	z6.s, p0/m, z6.s, z0.s
	.loc	1 29 13                 // exp_intrinsics.c:29:13
	frintn	z6.s, p0/m, z6.s
	.loc	1 32 25                 // exp_intrinsics.c:32:25
	fmls	z5.s, p0/m, z6.s, z1.s
	.loc	1 41 25                 // exp_intrinsics.c:41:25
	fadd	z4.s, p0/m, z4.s, z5.s
	.loc	1 35 24                 // exp_intrinsics.c:35:24
	fcvtzs	z5.s, p0/m, z6.s
	.loc	1 36 14                 // exp_intrinsics.c:36:14
	add	z5.s, p0/m, z5.s, z2.s
	.loc	1 37 14                 // exp_intrinsics.c:37:14
	lsl	z5.s, p0/m, z5.s, #23
	.loc	1 44 30                 // exp_intrinsics.c:44:30
	fmul	z4.s, p0/m, z4.s, z5.s
	.loc	1 46 9                  // exp_intrinsics.c:46:9
	st1w	{ z4.s }, p0, [x1, x8, lsl #2]
	.loc	1 21 37                 // exp_intrinsics.c:21:37
	incw	x8
	.loc	1 21 5 is_stmt 0        // exp_intrinsics.c:21:5
	cmp	x8, x2
	b.lo	.LBB0_2
.LBB0_3:
	.loc	1 48 1 is_stmt 1        // exp_intrinsics.c:48:1
	ret
.Ltmp0:
.Lfunc_end0:
	.size	exp_f32_poly1_intrin, .Lfunc_end0-exp_f32_poly1_intrin
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               // -- Begin function exp_f32_poly2_intrin
.LCPI1_0:
	.word	1069066811              // float 1.44269502
.LCPI1_1:
	.word	1060205080              // float 0.693147182
	.text
	.globl	exp_f32_poly2_intrin
	.p2align	3
	.type	exp_f32_poly2_intrin,@function
exp_f32_poly2_intrin:                   // @exp_f32_poly2_intrin
.Lfunc_begin1:
	.loc	1 53 0                  // exp_intrinsics.c:53:0
	.cfi_startproc
// %bb.0:
	.loc	1 56 5 prologue_end     // exp_intrinsics.c:56:5
	cbz	x2, .LBB1_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI1_0
	adrp	x10, .LCPI1_1
	ldr	s1, [x10, :lo12:.LCPI1_1]
	ldr	s0, [x9, :lo12:.LCPI1_0]
	mov	z0.s, s0
	mov	z1.s, s1
	mov	z2.s, #127              // =0x7f
	fmov	z3.s, #1.00000000
	fmov	z4.s, #0.50000000
	mov	x8, xzr
	.p2align	2
.LBB1_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 72 25 is_stmt 1       // exp_intrinsics.c:72:25
	mov	z5.d, z4.d
	.loc	1 57 23                 // exp_intrinsics.c:57:23
	whilelo	p0.s, x8, x2
	.loc	1 59 25                 // exp_intrinsics.c:59:25
	ld1w	{ z6.s }, p0/z, [x0, x8, lsl #2]
	.loc	1 61 25                 // exp_intrinsics.c:61:25
	mov	z7.d, z6.d
	fmul	z7.s, p0/m, z7.s, z0.s
	.loc	1 62 13                 // exp_intrinsics.c:62:13
	frintn	z7.s, p0/m, z7.s
	.loc	1 64 25                 // exp_intrinsics.c:64:25
	fmls	z6.s, p0/m, z7.s, z1.s
	.loc	1 66 24                 // exp_intrinsics.c:66:24
	fcvtzs	z7.s, p0/m, z7.s
	.loc	1 72 25                 // exp_intrinsics.c:72:25
	fmad	z5.s, p0/m, z6.s, z3.s
	.loc	1 73 13                 // exp_intrinsics.c:73:13
	fmad	z5.s, p0/m, z6.s, z3.s
	.loc	1 67 14                 // exp_intrinsics.c:67:14
	add	z7.s, p0/m, z7.s, z2.s
	.loc	1 68 14                 // exp_intrinsics.c:68:14
	lsl	z7.s, p0/m, z7.s, #23
	.loc	1 75 30                 // exp_intrinsics.c:75:30
	fmul	z5.s, p0/m, z5.s, z7.s
	.loc	1 76 9                  // exp_intrinsics.c:76:9
	st1w	{ z5.s }, p0, [x1, x8, lsl #2]
	.loc	1 56 37                 // exp_intrinsics.c:56:37
	incw	x8
	.loc	1 56 5 is_stmt 0        // exp_intrinsics.c:56:5
	cmp	x8, x2
	b.lo	.LBB1_2
.LBB1_3:
	.loc	1 78 1 is_stmt 1        // exp_intrinsics.c:78:1
	ret
.Ltmp1:
.Lfunc_end1:
	.size	exp_f32_poly2_intrin, .Lfunc_end1-exp_f32_poly2_intrin
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               // -- Begin function exp_f32_poly3_intrin
.LCPI2_0:
	.word	1069066811              // float 1.44269502
.LCPI2_1:
	.word	1060205080              // float 0.693147182
.LCPI2_2:
	.word	1042983595              // float 0.166666672
	.text
	.globl	exp_f32_poly3_intrin
	.p2align	3
	.type	exp_f32_poly3_intrin,@function
exp_f32_poly3_intrin:                   // @exp_f32_poly3_intrin
.Lfunc_begin2:
	.loc	1 83 0                  // exp_intrinsics.c:83:0
	.cfi_startproc
// %bb.0:
	.loc	1 87 5 prologue_end     // exp_intrinsics.c:87:5
	cbz	x2, .LBB2_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI2_0
	adrp	x10, .LCPI2_1
	ldr	s2, [x10, :lo12:.LCPI2_1]
	ldr	s1, [x9, :lo12:.LCPI2_0]
	adrp	x9, .LCPI2_2
	ldr	s3, [x9, :lo12:.LCPI2_2]
	mov	z1.s, s1
	mov	z2.s, s2
	mov	z3.s, s3
	mov	z0.s, #127              // =0x7f
	fmov	z4.s, #0.50000000
	fmov	z5.s, #1.00000000
	mov	x8, xzr
	.p2align	2
.LBB2_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 103 25 is_stmt 1      // exp_intrinsics.c:103:25
	mov	z6.d, z3.d
	.loc	1 88 23                 // exp_intrinsics.c:88:23
	whilelo	p0.s, x8, x2
	.loc	1 90 25                 // exp_intrinsics.c:90:25
	ld1w	{ z7.s }, p0/z, [x0, x8, lsl #2]
	.loc	1 92 25                 // exp_intrinsics.c:92:25
	mov	z16.d, z7.d
	fmul	z16.s, p0/m, z16.s, z1.s
	.loc	1 93 13                 // exp_intrinsics.c:93:13
	frintn	z16.s, p0/m, z16.s
	.loc	1 95 25                 // exp_intrinsics.c:95:25
	fmls	z7.s, p0/m, z16.s, z2.s
	.loc	1 97 24                 // exp_intrinsics.c:97:24
	fcvtzs	z16.s, p0/m, z16.s
	.loc	1 103 25                // exp_intrinsics.c:103:25
	fmad	z6.s, p0/m, z7.s, z4.s
	.loc	1 104 13                // exp_intrinsics.c:104:13
	fmad	z6.s, p0/m, z7.s, z5.s
	.loc	1 105 13                // exp_intrinsics.c:105:13
	fmad	z6.s, p0/m, z7.s, z5.s
	.loc	1 98 14                 // exp_intrinsics.c:98:14
	add	z16.s, p0/m, z16.s, z0.s
	.loc	1 99 14                 // exp_intrinsics.c:99:14
	lsl	z16.s, p0/m, z16.s, #23
	.loc	1 107 30                // exp_intrinsics.c:107:30
	fmul	z6.s, p0/m, z6.s, z16.s
	.loc	1 108 9                 // exp_intrinsics.c:108:9
	st1w	{ z6.s }, p0, [x1, x8, lsl #2]
	.loc	1 87 37                 // exp_intrinsics.c:87:37
	incw	x8
	.loc	1 87 5 is_stmt 0        // exp_intrinsics.c:87:5
	cmp	x8, x2
	b.lo	.LBB2_2
.LBB2_3:
	.loc	1 110 1 is_stmt 1       // exp_intrinsics.c:110:1
	ret
.Ltmp2:
.Lfunc_end2:
	.size	exp_f32_poly3_intrin, .Lfunc_end2-exp_f32_poly3_intrin
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               // -- Begin function exp_f32_poly4_intrin
.LCPI3_0:
	.word	1069066811              // float 1.44269502
.LCPI3_1:
	.word	1060205080              // float 0.693147182
.LCPI3_2:
	.word	1042983595              // float 0.166666672
.LCPI3_3:
	.word	1026206379              // float 0.0416666679
	.text
	.globl	exp_f32_poly4_intrin
	.p2align	3
	.type	exp_f32_poly4_intrin,@function
exp_f32_poly4_intrin:                   // @exp_f32_poly4_intrin
.Lfunc_begin3:
	.loc	1 115 0                 // exp_intrinsics.c:115:0
	.cfi_startproc
// %bb.0:
	.loc	1 120 5 prologue_end    // exp_intrinsics.c:120:5
	cbz	x2, .LBB3_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI3_0
	ldr	s0, [x9, :lo12:.LCPI3_0]
	adrp	x9, .LCPI3_1
	ldr	s1, [x9, :lo12:.LCPI3_1]
	adrp	x9, .LCPI3_2
	ldr	s3, [x9, :lo12:.LCPI3_2]
	adrp	x9, .LCPI3_3
	ldr	s4, [x9, :lo12:.LCPI3_3]
	mov	z0.s, s0
	mov	z1.s, s1
	mov	z3.s, s3
	mov	z4.s, s4
	mov	z2.s, #127              // =0x7f
	fmov	z5.s, #0.50000000
	fmov	z6.s, #1.00000000
	mov	x8, xzr
	.p2align	2
.LBB3_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 136 25 is_stmt 1      // exp_intrinsics.c:136:25
	mov	z7.d, z4.d
	.loc	1 121 23                // exp_intrinsics.c:121:23
	whilelo	p0.s, x8, x2
	.loc	1 123 25                // exp_intrinsics.c:123:25
	ld1w	{ z16.s }, p0/z, [x0, x8, lsl #2]
	.loc	1 125 25                // exp_intrinsics.c:125:25
	mov	z17.d, z16.d
	fmul	z17.s, p0/m, z17.s, z0.s
	.loc	1 126 13                // exp_intrinsics.c:126:13
	frintn	z17.s, p0/m, z17.s
	.loc	1 128 25                // exp_intrinsics.c:128:25
	fmls	z16.s, p0/m, z17.s, z1.s
	.loc	1 130 24                // exp_intrinsics.c:130:24
	fcvtzs	z17.s, p0/m, z17.s
	.loc	1 136 25                // exp_intrinsics.c:136:25
	fmad	z7.s, p0/m, z16.s, z3.s
	.loc	1 137 13                // exp_intrinsics.c:137:13
	fmad	z7.s, p0/m, z16.s, z5.s
	.loc	1 138 13                // exp_intrinsics.c:138:13
	fmad	z7.s, p0/m, z16.s, z6.s
	.loc	1 139 13                // exp_intrinsics.c:139:13
	fmad	z7.s, p0/m, z16.s, z6.s
	.loc	1 131 14                // exp_intrinsics.c:131:14
	add	z17.s, p0/m, z17.s, z2.s
	.loc	1 132 14                // exp_intrinsics.c:132:14
	lsl	z17.s, p0/m, z17.s, #23
	.loc	1 141 30                // exp_intrinsics.c:141:30
	fmul	z7.s, p0/m, z7.s, z17.s
	.loc	1 142 9                 // exp_intrinsics.c:142:9
	st1w	{ z7.s }, p0, [x1, x8, lsl #2]
	.loc	1 120 37                // exp_intrinsics.c:120:37
	incw	x8
	.loc	1 120 5 is_stmt 0       // exp_intrinsics.c:120:5
	cmp	x8, x2
	b.lo	.LBB3_2
.LBB3_3:
	.loc	1 144 1 is_stmt 1       // exp_intrinsics.c:144:1
	ret
.Ltmp3:
.Lfunc_end3:
	.size	exp_f32_poly4_intrin, .Lfunc_end3-exp_f32_poly4_intrin
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               // -- Begin function exp_f32_poly5_intrin
.LCPI4_0:
	.word	1069066811              // float 1.44269502
.LCPI4_1:
	.word	1060205080              // float 0.693147182
.LCPI4_2:
	.word	1026206379              // float 0.0416666679
.LCPI4_3:
	.word	1007192201              // float 0.00833333377
.LCPI4_4:
	.word	1042983595              // float 0.166666672
	.text
	.globl	exp_f32_poly5_intrin
	.p2align	3
	.type	exp_f32_poly5_intrin,@function
exp_f32_poly5_intrin:                   // @exp_f32_poly5_intrin
.Lfunc_begin4:
	.loc	1 149 0                 // exp_intrinsics.c:149:0
	.cfi_startproc
// %bb.0:
	.loc	1 155 5 prologue_end    // exp_intrinsics.c:155:5
	cbz	x2, .LBB4_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI4_0
	ldr	s0, [x9, :lo12:.LCPI4_0]
	adrp	x9, .LCPI4_1
	ldr	s1, [x9, :lo12:.LCPI4_1]
	adrp	x9, .LCPI4_2
	ldr	s2, [x9, :lo12:.LCPI4_2]
	adrp	x9, .LCPI4_3
	ldr	s4, [x9, :lo12:.LCPI4_3]
	adrp	x9, .LCPI4_4
	ldr	s5, [x9, :lo12:.LCPI4_4]
	mov	z0.s, s0
	mov	z3.s, #127              // =0x7f
	fmov	z6.s, #0.50000000
	fmov	z7.s, #1.00000000
	mov	x8, xzr
	mov	z1.s, s1
	mov	z2.s, s2
	mov	z4.s, s4
	mov	z5.s, s5
	.p2align	2
.LBB4_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 171 25 is_stmt 1      // exp_intrinsics.c:171:25
	mov	z16.d, z4.d
	.loc	1 156 23                // exp_intrinsics.c:156:23
	whilelo	p0.s, x8, x2
	.loc	1 158 25                // exp_intrinsics.c:158:25
	ld1w	{ z17.s }, p0/z, [x0, x8, lsl #2]
	.loc	1 160 25                // exp_intrinsics.c:160:25
	mov	z18.d, z17.d
	fmul	z18.s, p0/m, z18.s, z0.s
	.loc	1 161 13                // exp_intrinsics.c:161:13
	frintn	z18.s, p0/m, z18.s
	.loc	1 163 25                // exp_intrinsics.c:163:25
	fmls	z17.s, p0/m, z18.s, z1.s
	.loc	1 165 24                // exp_intrinsics.c:165:24
	fcvtzs	z18.s, p0/m, z18.s
	.loc	1 171 25                // exp_intrinsics.c:171:25
	fmad	z16.s, p0/m, z17.s, z2.s
	.loc	1 172 13                // exp_intrinsics.c:172:13
	fmad	z16.s, p0/m, z17.s, z5.s
	.loc	1 173 13                // exp_intrinsics.c:173:13
	fmad	z16.s, p0/m, z17.s, z6.s
	.loc	1 174 13                // exp_intrinsics.c:174:13
	fmad	z16.s, p0/m, z17.s, z7.s
	.loc	1 175 13                // exp_intrinsics.c:175:13
	fmad	z16.s, p0/m, z17.s, z7.s
	.loc	1 166 14                // exp_intrinsics.c:166:14
	add	z18.s, p0/m, z18.s, z3.s
	.loc	1 167 14                // exp_intrinsics.c:167:14
	lsl	z18.s, p0/m, z18.s, #23
	.loc	1 177 30                // exp_intrinsics.c:177:30
	fmul	z16.s, p0/m, z16.s, z18.s
	.loc	1 178 9                 // exp_intrinsics.c:178:9
	st1w	{ z16.s }, p0, [x1, x8, lsl #2]
	.loc	1 155 37                // exp_intrinsics.c:155:37
	incw	x8
	.loc	1 155 5 is_stmt 0       // exp_intrinsics.c:155:5
	cmp	x8, x2
	b.lo	.LBB4_2
.LBB4_3:
	.loc	1 180 1 is_stmt 1       // exp_intrinsics.c:180:1
	ret
.Ltmp4:
.Lfunc_end4:
	.size	exp_f32_poly5_intrin, .Lfunc_end4-exp_f32_poly5_intrin
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               // -- Begin function exp_f64_poly1_intrin
.LCPI5_0:
	.xword	4609176140021203710     // double 1.4426950408889634
.LCPI5_1:
	.xword	4604418534313441775     // double 0.69314718055994529
	.text
	.globl	exp_f64_poly1_intrin
	.p2align	3
	.type	exp_f64_poly1_intrin,@function
exp_f64_poly1_intrin:                   // @exp_f64_poly1_intrin
.Lfunc_begin5:
	.loc	1 185 0                 // exp_intrinsics.c:185:0
	.cfi_startproc
// %bb.0:
	.loc	1 188 5 prologue_end    // exp_intrinsics.c:188:5
	cbz	x2, .LBB5_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI5_0
	ldr	d0, [x9, :lo12:.LCPI5_0]
	adrp	x9, .LCPI5_1
	ldr	d1, [x9, :lo12:.LCPI5_1]
	mov	z0.d, d0
	mov	z1.d, d1
	mov	z2.d, #1023             // =0x3ff
	fmov	z3.d, #1.00000000
	mov	x8, xzr
	.p2align	2
.LBB5_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 203 25 is_stmt 1      // exp_intrinsics.c:203:25
	mov	z4.d, z3.d
	.loc	1 189 23                // exp_intrinsics.c:189:23
	whilelo	p0.d, x8, x2
	.loc	1 191 25                // exp_intrinsics.c:191:25
	ld1d	{ z5.d }, p0/z, [x0, x8, lsl #3]
	.loc	1 193 25                // exp_intrinsics.c:193:25
	mov	z6.d, z5.d
	fmul	z6.d, p0/m, z6.d, z0.d
	.loc	1 194 13                // exp_intrinsics.c:194:13
	frintn	z6.d, p0/m, z6.d
	.loc	1 196 25                // exp_intrinsics.c:196:25
	fmls	z5.d, p0/m, z6.d, z1.d
	.loc	1 203 25                // exp_intrinsics.c:203:25
	fadd	z4.d, p0/m, z4.d, z5.d
	.loc	1 198 24                // exp_intrinsics.c:198:24
	fcvtzs	z5.d, p0/m, z6.d
	.loc	1 199 14                // exp_intrinsics.c:199:14
	add	z5.d, p0/m, z5.d, z2.d
	.loc	1 200 14                // exp_intrinsics.c:200:14
	lsl	z5.d, p0/m, z5.d, #52
	.loc	1 205 30                // exp_intrinsics.c:205:30
	fmul	z4.d, p0/m, z4.d, z5.d
	.loc	1 206 9                 // exp_intrinsics.c:206:9
	st1d	{ z4.d }, p0, [x1, x8, lsl #3]
	.loc	1 188 37                // exp_intrinsics.c:188:37
	incd	x8
	.loc	1 188 5 is_stmt 0       // exp_intrinsics.c:188:5
	cmp	x8, x2
	b.lo	.LBB5_2
.LBB5_3:
	.loc	1 208 1 is_stmt 1       // exp_intrinsics.c:208:1
	ret
.Ltmp5:
.Lfunc_end5:
	.size	exp_f64_poly1_intrin, .Lfunc_end5-exp_f64_poly1_intrin
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               // -- Begin function exp_f64_poly2_intrin
.LCPI6_0:
	.xword	4609176140021203710     // double 1.4426950408889634
.LCPI6_1:
	.xword	4604418534313441775     // double 0.69314718055994529
	.text
	.globl	exp_f64_poly2_intrin
	.p2align	3
	.type	exp_f64_poly2_intrin,@function
exp_f64_poly2_intrin:                   // @exp_f64_poly2_intrin
.Lfunc_begin6:
	.loc	1 213 0                 // exp_intrinsics.c:213:0
	.cfi_startproc
// %bb.0:
	.loc	1 216 5 prologue_end    // exp_intrinsics.c:216:5
	cbz	x2, .LBB6_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI6_0
	adrp	x10, .LCPI6_1
	ldr	d1, [x10, :lo12:.LCPI6_1]
	ldr	d0, [x9, :lo12:.LCPI6_0]
	mov	z0.d, d0
	mov	z1.d, d1
	mov	z2.d, #1023             // =0x3ff
	fmov	z3.d, #1.00000000
	fmov	z4.d, #0.50000000
	mov	x8, xzr
	.p2align	2
.LBB6_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 231 25 is_stmt 1      // exp_intrinsics.c:231:25
	mov	z5.d, z4.d
	.loc	1 217 23                // exp_intrinsics.c:217:23
	whilelo	p0.d, x8, x2
	.loc	1 219 25                // exp_intrinsics.c:219:25
	ld1d	{ z6.d }, p0/z, [x0, x8, lsl #3]
	.loc	1 221 25                // exp_intrinsics.c:221:25
	mov	z7.d, z6.d
	fmul	z7.d, p0/m, z7.d, z0.d
	.loc	1 222 13                // exp_intrinsics.c:222:13
	frintn	z7.d, p0/m, z7.d
	.loc	1 224 25                // exp_intrinsics.c:224:25
	fmls	z6.d, p0/m, z7.d, z1.d
	.loc	1 226 24                // exp_intrinsics.c:226:24
	fcvtzs	z7.d, p0/m, z7.d
	.loc	1 231 25                // exp_intrinsics.c:231:25
	fmad	z5.d, p0/m, z6.d, z3.d
	.loc	1 232 13                // exp_intrinsics.c:232:13
	fmad	z5.d, p0/m, z6.d, z3.d
	.loc	1 227 14                // exp_intrinsics.c:227:14
	add	z7.d, p0/m, z7.d, z2.d
	.loc	1 228 14                // exp_intrinsics.c:228:14
	lsl	z7.d, p0/m, z7.d, #52
	.loc	1 234 30                // exp_intrinsics.c:234:30
	fmul	z5.d, p0/m, z5.d, z7.d
	.loc	1 235 9                 // exp_intrinsics.c:235:9
	st1d	{ z5.d }, p0, [x1, x8, lsl #3]
	.loc	1 216 37                // exp_intrinsics.c:216:37
	incd	x8
	.loc	1 216 5 is_stmt 0       // exp_intrinsics.c:216:5
	cmp	x8, x2
	b.lo	.LBB6_2
.LBB6_3:
	.loc	1 237 1 is_stmt 1       // exp_intrinsics.c:237:1
	ret
.Ltmp6:
.Lfunc_end6:
	.size	exp_f64_poly2_intrin, .Lfunc_end6-exp_f64_poly2_intrin
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               // -- Begin function exp_f64_poly3_intrin
.LCPI7_0:
	.xword	4609176140021203710     // double 1.4426950408889634
.LCPI7_1:
	.xword	4604418534313441775     // double 0.69314718055994529
.LCPI7_2:
	.xword	4595172819793696085     // double 0.16666666666666666
	.text
	.globl	exp_f64_poly3_intrin
	.p2align	3
	.type	exp_f64_poly3_intrin,@function
exp_f64_poly3_intrin:                   // @exp_f64_poly3_intrin
.Lfunc_begin7:
	.loc	1 242 0                 // exp_intrinsics.c:242:0
	.cfi_startproc
// %bb.0:
	.loc	1 246 5 prologue_end    // exp_intrinsics.c:246:5
	cbz	x2, .LBB7_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI7_0
	adrp	x10, .LCPI7_1
	ldr	d2, [x10, :lo12:.LCPI7_1]
	ldr	d1, [x9, :lo12:.LCPI7_0]
	adrp	x9, .LCPI7_2
	ldr	d3, [x9, :lo12:.LCPI7_2]
	mov	z1.d, d1
	mov	z2.d, d2
	mov	z3.d, d3
	mov	z0.d, #1023             // =0x3ff
	fmov	z4.d, #0.50000000
	fmov	z5.d, #1.00000000
	mov	x8, xzr
	.p2align	2
.LBB7_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 261 25 is_stmt 1      // exp_intrinsics.c:261:25
	mov	z6.d, z3.d
	.loc	1 247 23                // exp_intrinsics.c:247:23
	whilelo	p0.d, x8, x2
	.loc	1 249 25                // exp_intrinsics.c:249:25
	ld1d	{ z7.d }, p0/z, [x0, x8, lsl #3]
	.loc	1 251 25                // exp_intrinsics.c:251:25
	mov	z16.d, z7.d
	fmul	z16.d, p0/m, z16.d, z1.d
	.loc	1 252 13                // exp_intrinsics.c:252:13
	frintn	z16.d, p0/m, z16.d
	.loc	1 254 25                // exp_intrinsics.c:254:25
	fmls	z7.d, p0/m, z16.d, z2.d
	.loc	1 256 24                // exp_intrinsics.c:256:24
	fcvtzs	z16.d, p0/m, z16.d
	.loc	1 261 25                // exp_intrinsics.c:261:25
	fmad	z6.d, p0/m, z7.d, z4.d
	.loc	1 262 13                // exp_intrinsics.c:262:13
	fmad	z6.d, p0/m, z7.d, z5.d
	.loc	1 263 13                // exp_intrinsics.c:263:13
	fmad	z6.d, p0/m, z7.d, z5.d
	.loc	1 257 14                // exp_intrinsics.c:257:14
	add	z16.d, p0/m, z16.d, z0.d
	.loc	1 258 14                // exp_intrinsics.c:258:14
	lsl	z16.d, p0/m, z16.d, #52
	.loc	1 265 30                // exp_intrinsics.c:265:30
	fmul	z6.d, p0/m, z6.d, z16.d
	.loc	1 266 9                 // exp_intrinsics.c:266:9
	st1d	{ z6.d }, p0, [x1, x8, lsl #3]
	.loc	1 246 37                // exp_intrinsics.c:246:37
	incd	x8
	.loc	1 246 5 is_stmt 0       // exp_intrinsics.c:246:5
	cmp	x8, x2
	b.lo	.LBB7_2
.LBB7_3:
	.loc	1 268 1 is_stmt 1       // exp_intrinsics.c:268:1
	ret
.Ltmp7:
.Lfunc_end7:
	.size	exp_f64_poly3_intrin, .Lfunc_end7-exp_f64_poly3_intrin
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               // -- Begin function exp_f64_poly4_intrin
.LCPI8_0:
	.xword	4609176140021203710     // double 1.4426950408889634
.LCPI8_1:
	.xword	4604418534313441775     // double 0.69314718055994529
.LCPI8_2:
	.xword	4595172819793696085     // double 0.16666666666666666
.LCPI8_3:
	.xword	4586165620538955093     // double 0.041666666666666664
	.text
	.globl	exp_f64_poly4_intrin
	.p2align	3
	.type	exp_f64_poly4_intrin,@function
exp_f64_poly4_intrin:                   // @exp_f64_poly4_intrin
.Lfunc_begin8:
	.loc	1 273 0                 // exp_intrinsics.c:273:0
	.cfi_startproc
// %bb.0:
	.loc	1 278 5 prologue_end    // exp_intrinsics.c:278:5
	cbz	x2, .LBB8_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI8_0
	ldr	d0, [x9, :lo12:.LCPI8_0]
	adrp	x9, .LCPI8_1
	ldr	d1, [x9, :lo12:.LCPI8_1]
	adrp	x9, .LCPI8_2
	ldr	d3, [x9, :lo12:.LCPI8_2]
	adrp	x9, .LCPI8_3
	ldr	d4, [x9, :lo12:.LCPI8_3]
	mov	z0.d, d0
	mov	z1.d, d1
	mov	z3.d, d3
	mov	z4.d, d4
	mov	z2.d, #1023             // =0x3ff
	fmov	z5.d, #0.50000000
	fmov	z6.d, #1.00000000
	mov	x8, xzr
	.p2align	2
.LBB8_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 293 25 is_stmt 1      // exp_intrinsics.c:293:25
	mov	z7.d, z4.d
	.loc	1 279 23                // exp_intrinsics.c:279:23
	whilelo	p0.d, x8, x2
	.loc	1 281 25                // exp_intrinsics.c:281:25
	ld1d	{ z16.d }, p0/z, [x0, x8, lsl #3]
	.loc	1 283 25                // exp_intrinsics.c:283:25
	mov	z17.d, z16.d
	fmul	z17.d, p0/m, z17.d, z0.d
	.loc	1 284 13                // exp_intrinsics.c:284:13
	frintn	z17.d, p0/m, z17.d
	.loc	1 286 25                // exp_intrinsics.c:286:25
	fmls	z16.d, p0/m, z17.d, z1.d
	.loc	1 288 24                // exp_intrinsics.c:288:24
	fcvtzs	z17.d, p0/m, z17.d
	.loc	1 293 25                // exp_intrinsics.c:293:25
	fmad	z7.d, p0/m, z16.d, z3.d
	.loc	1 294 13                // exp_intrinsics.c:294:13
	fmad	z7.d, p0/m, z16.d, z5.d
	.loc	1 295 13                // exp_intrinsics.c:295:13
	fmad	z7.d, p0/m, z16.d, z6.d
	.loc	1 296 13                // exp_intrinsics.c:296:13
	fmad	z7.d, p0/m, z16.d, z6.d
	.loc	1 289 14                // exp_intrinsics.c:289:14
	add	z17.d, p0/m, z17.d, z2.d
	.loc	1 290 14                // exp_intrinsics.c:290:14
	lsl	z17.d, p0/m, z17.d, #52
	.loc	1 298 30                // exp_intrinsics.c:298:30
	fmul	z7.d, p0/m, z7.d, z17.d
	.loc	1 299 9                 // exp_intrinsics.c:299:9
	st1d	{ z7.d }, p0, [x1, x8, lsl #3]
	.loc	1 278 37                // exp_intrinsics.c:278:37
	incd	x8
	.loc	1 278 5 is_stmt 0       // exp_intrinsics.c:278:5
	cmp	x8, x2
	b.lo	.LBB8_2
.LBB8_3:
	.loc	1 301 1 is_stmt 1       // exp_intrinsics.c:301:1
	ret
.Ltmp8:
.Lfunc_end8:
	.size	exp_f64_poly4_intrin, .Lfunc_end8-exp_f64_poly4_intrin
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               // -- Begin function exp_f64_poly5_intrin
.LCPI9_0:
	.xword	4609176140021203710     // double 1.4426950408889634
.LCPI9_1:
	.xword	4604418534313441775     // double 0.69314718055994529
.LCPI9_2:
	.xword	4586165620538955093     // double 0.041666666666666664
.LCPI9_3:
	.xword	4575957461383581969     // double 0.0083333333333333332
.LCPI9_4:
	.xword	4595172819793696085     // double 0.16666666666666666
	.text
	.globl	exp_f64_poly5_intrin
	.p2align	3
	.type	exp_f64_poly5_intrin,@function
exp_f64_poly5_intrin:                   // @exp_f64_poly5_intrin
.Lfunc_begin9:
	.loc	1 306 0                 // exp_intrinsics.c:306:0
	.cfi_startproc
// %bb.0:
	.loc	1 312 5 prologue_end    // exp_intrinsics.c:312:5
	cbz	x2, .LBB9_3
// %bb.1:
	.loc	1 0 5 is_stmt 0         // exp_intrinsics.c:0:5
	adrp	x9, .LCPI9_0
	ldr	d0, [x9, :lo12:.LCPI9_0]
	adrp	x9, .LCPI9_1
	ldr	d1, [x9, :lo12:.LCPI9_1]
	adrp	x9, .LCPI9_2
	ldr	d2, [x9, :lo12:.LCPI9_2]
	adrp	x9, .LCPI9_3
	ldr	d4, [x9, :lo12:.LCPI9_3]
	adrp	x9, .LCPI9_4
	ldr	d5, [x9, :lo12:.LCPI9_4]
	mov	z0.d, d0
	mov	z3.d, #1023             // =0x3ff
	fmov	z6.d, #0.50000000
	fmov	z7.d, #1.00000000
	mov	x8, xzr
	mov	z1.d, d1
	mov	z2.d, d2
	mov	z4.d, d4
	mov	z5.d, d5
	.p2align	2
.LBB9_2:                                // =>This Inner Loop Header: Depth=1
	.loc	1 327 25 is_stmt 1      // exp_intrinsics.c:327:25
	mov	z16.d, z4.d
	.loc	1 313 23                // exp_intrinsics.c:313:23
	whilelo	p0.d, x8, x2
	.loc	1 315 25                // exp_intrinsics.c:315:25
	ld1d	{ z17.d }, p0/z, [x0, x8, lsl #3]
	.loc	1 317 25                // exp_intrinsics.c:317:25
	mov	z18.d, z17.d
	fmul	z18.d, p0/m, z18.d, z0.d
	.loc	1 318 13                // exp_intrinsics.c:318:13
	frintn	z18.d, p0/m, z18.d
	.loc	1 320 25                // exp_intrinsics.c:320:25
	fmls	z17.d, p0/m, z18.d, z1.d
	.loc	1 322 24                // exp_intrinsics.c:322:24
	fcvtzs	z18.d, p0/m, z18.d
	.loc	1 327 25                // exp_intrinsics.c:327:25
	fmad	z16.d, p0/m, z17.d, z2.d
	.loc	1 328 13                // exp_intrinsics.c:328:13
	fmad	z16.d, p0/m, z17.d, z5.d
	.loc	1 329 13                // exp_intrinsics.c:329:13
	fmad	z16.d, p0/m, z17.d, z6.d
	.loc	1 330 13                // exp_intrinsics.c:330:13
	fmad	z16.d, p0/m, z17.d, z7.d
	.loc	1 331 13                // exp_intrinsics.c:331:13
	fmad	z16.d, p0/m, z17.d, z7.d
	.loc	1 323 14                // exp_intrinsics.c:323:14
	add	z18.d, p0/m, z18.d, z3.d
	.loc	1 324 14                // exp_intrinsics.c:324:14
	lsl	z18.d, p0/m, z18.d, #52
	.loc	1 333 30                // exp_intrinsics.c:333:30
	fmul	z16.d, p0/m, z16.d, z18.d
	.loc	1 334 9                 // exp_intrinsics.c:334:9
	st1d	{ z16.d }, p0, [x1, x8, lsl #3]
	.loc	1 312 37                // exp_intrinsics.c:312:37
	incd	x8
	.loc	1 312 5 is_stmt 0       // exp_intrinsics.c:312:5
	cmp	x8, x2
	b.lo	.LBB9_2
.LBB9_3:
	.loc	1 336 1 is_stmt 1       // exp_intrinsics.c:336:1
	ret
.Ltmp9:
.Lfunc_end9:
	.size	exp_f64_poly5_intrin, .Lfunc_end9-exp_f64_poly5_intrin
	.cfi_endproc
                                        // -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang: Fujitsu C/C++ Compiler 4.12.1 (May 27 2025 19:39:57) (based on LLVM 7.1.0)" // string offset=0
.Linfo_string1:
	.asciz	"exp_intrinsics.c"      // string offset=82
.Linfo_string2:
	.asciz	"/home/u14346/work/gemm/fat" // string offset=99
.Linfo_string3:
	.asciz	"exp_f32_poly1_intrin"  // string offset=126
.Linfo_string4:
	.asciz	"exp_f32_poly2_intrin"  // string offset=147
.Linfo_string5:
	.asciz	"exp_f32_poly3_intrin"  // string offset=168
.Linfo_string6:
	.asciz	"exp_f32_poly4_intrin"  // string offset=189
.Linfo_string7:
	.asciz	"exp_f32_poly5_intrin"  // string offset=210
.Linfo_string8:
	.asciz	"exp_f64_poly1_intrin"  // string offset=231
.Linfo_string9:
	.asciz	"exp_f64_poly2_intrin"  // string offset=252
.Linfo_string10:
	.asciz	"exp_f64_poly3_intrin"  // string offset=273
.Linfo_string11:
	.asciz	"exp_f64_poly4_intrin"  // string offset=294
.Linfo_string12:
	.asciz	"exp_f64_poly5_intrin"  // string offset=315
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
	.byte	4                       // Abbreviation Code
	.ascii	"\200\340\003"          // DW_TAG_FJ_loop
	.byte	0                       // DW_CHILDREN_no
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.ascii	"\200f"                 // DW_AT_FJ_loop_start_line
	.byte	11                      // DW_FORM_data1
	.ascii	"\201f"                 // DW_AT_FJ_loop_end_line
	.byte	5                       // DW_FORM_data2
	.ascii	"\202f"                 // DW_AT_FJ_loop_nest_level
	.byte	11                      // DW_FORM_data1
	.ascii	"\203f"                 // DW_AT_FJ_loop_type
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	5                       // Abbreviation Code
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
	.byte	5                       // DW_FORM_data2
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	6                       // Abbreviation Code
	.ascii	"\200\340\003"          // DW_TAG_FJ_loop
	.byte	0                       // DW_CHILDREN_no
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.ascii	"\200f"                 // DW_AT_FJ_loop_start_line
	.byte	5                       // DW_FORM_data2
	.ascii	"\201f"                 // DW_AT_FJ_loop_end_line
	.byte	5                       // DW_FORM_data2
	.ascii	"\202f"                 // DW_AT_FJ_loop_nest_level
	.byte	11                      // DW_FORM_data1
	.ascii	"\203f"                 // DW_AT_FJ_loop_type
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	0                       // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	306                     // Length of Unit
	.hword	4                       // DWARF version number
	.word	.debug_abbrev           // Offset Into Abbrev. Section
	.byte	8                       // Address Size (in bytes)
	.byte	1                       // Abbrev [1] 0xb:0x12b DW_TAG_compile_unit
	.word	.Linfo_string0          // DW_AT_producer
	.hword	12                      // DW_AT_language
	.word	.Linfo_string1          // DW_AT_name
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.word	.Linfo_string2          // DW_AT_comp_dir
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end9-.Lfunc_begin0 // DW_AT_high_pc
	.byte	2                       // Abbrev [2] 0x2a:0x1a DW_TAG_subprogram
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.word	.Linfo_string3          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	18                      // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0x3d:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	21                      // DW_AT_FJ_loop_start_line
	.byte	47                      // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	2                       // Abbrev [2] 0x44:0x1a DW_TAG_subprogram
	.xword	.Lfunc_begin1           // DW_AT_low_pc
	.word	.Lfunc_end1-.Lfunc_begin1 // DW_AT_high_pc
	.word	.Linfo_string4          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	53                      // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0x57:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	56                      // DW_AT_FJ_loop_start_line
	.byte	77                      // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	2                       // Abbrev [2] 0x5e:0x1a DW_TAG_subprogram
	.xword	.Lfunc_begin2           // DW_AT_low_pc
	.word	.Lfunc_end2-.Lfunc_begin2 // DW_AT_high_pc
	.word	.Linfo_string5          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	83                      // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0x71:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	87                      // DW_AT_FJ_loop_start_line
	.byte	109                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	2                       // Abbrev [2] 0x78:0x1a DW_TAG_subprogram
	.xword	.Lfunc_begin3           // DW_AT_low_pc
	.word	.Lfunc_end3-.Lfunc_begin3 // DW_AT_high_pc
	.word	.Linfo_string6          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	115                     // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0x8b:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	120                     // DW_AT_FJ_loop_start_line
	.byte	143                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	2                       // Abbrev [2] 0x92:0x1a DW_TAG_subprogram
	.xword	.Lfunc_begin4           // DW_AT_low_pc
	.word	.Lfunc_end4-.Lfunc_begin4 // DW_AT_high_pc
	.word	.Linfo_string7          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	149                     // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0xa5:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	155                     // DW_AT_FJ_loop_start_line
	.byte	179                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	2                       // Abbrev [2] 0xac:0x1a DW_TAG_subprogram
	.xword	.Lfunc_begin5           // DW_AT_low_pc
	.word	.Lfunc_end5-.Lfunc_begin5 // DW_AT_high_pc
	.word	.Linfo_string8          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	185                     // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0xbf:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	188                     // DW_AT_FJ_loop_start_line
	.byte	207                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	2                       // Abbrev [2] 0xc6:0x1a DW_TAG_subprogram
	.xword	.Lfunc_begin6           // DW_AT_low_pc
	.word	.Lfunc_end6-.Lfunc_begin6 // DW_AT_high_pc
	.word	.Linfo_string9          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	213                     // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0xd9:0x6 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	216                     // DW_AT_FJ_loop_start_line
	.byte	236                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	2                       // Abbrev [2] 0xe0:0x1b DW_TAG_subprogram
	.xword	.Lfunc_begin7           // DW_AT_low_pc
	.word	.Lfunc_end7-.Lfunc_begin7 // DW_AT_high_pc
	.word	.Linfo_string10         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	242                     // DW_AT_decl_line
	.byte	4                       // Abbrev [4] 0xf3:0x7 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.byte	246                     // DW_AT_FJ_loop_start_line
	.hword	267                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	5                       // Abbrev [5] 0xfb:0x1d DW_TAG_subprogram
	.xword	.Lfunc_begin8           // DW_AT_low_pc
	.word	.Lfunc_end8-.Lfunc_begin8 // DW_AT_high_pc
	.word	.Linfo_string11         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.hword	273                     // DW_AT_decl_line
	.byte	6                       // Abbrev [6] 0x10f:0x8 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.hword	278                     // DW_AT_FJ_loop_start_line
	.hword	300                     // DW_AT_FJ_loop_end_line
	.byte	1                       // DW_AT_FJ_loop_nest_level
	.byte	5                       // DW_AT_FJ_loop_type
	.byte	0                       // End Of Children Mark
	.byte	5                       // Abbrev [5] 0x118:0x1d DW_TAG_subprogram
	.xword	.Lfunc_begin9           // DW_AT_low_pc
	.word	.Lfunc_end9-.Lfunc_begin9 // DW_AT_high_pc
	.word	.Linfo_string12         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.hword	306                     // DW_AT_decl_line
	.byte	6                       // Abbrev [6] 0x12c:0x8 DW_TAG_FJ_loop
	.byte	1                       // DW_AT_decl_file
	.hword	312                     // DW_AT_FJ_loop_start_line
	.hword	335                     // DW_AT_FJ_loop_end_line
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
