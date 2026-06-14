	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gemm_mm0_bf16_asm       ; -- Begin function gemm_mm0_bf16_asm
	.globl	gemm_mm0_bf16_asm
	.p2align	8
	.type	gemm_mm0_bf16_asm,@function
gemm_mm0_bf16_asm:                      ; @gemm_mm0_bf16_asm
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	s_lshl_b32 s2, ttmp7, 7
	s_lshl_b32 s3, ttmp9, 7
	v_or_b32_e32 v1, s2, v0
	v_or_b32_e32 v2, s3, v0
	v_dual_mov_b32 v121, 0 :: v_dual_lshlrev_b32 v166, 4, v0
	v_lshlrev_b32_e32 v5, 1, v0
	v_and_b32_e32 v165, 15, v0
	v_lshrrev_b32_e32 v8, 1, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_mov_b32_e32 v124, v121
	v_dual_mov_b32 v123, v121 :: v_dual_and_b32 v170, 0x4f, v0
	v_dual_mov_b32 v6, v121 :: v_dual_and_b32 v167, 64, v5
	v_bfe_u32 v172, v0, 4, 1
	v_dual_mov_b32 v122, v121 :: v_dual_mov_b32 v5, v121
	v_and_b32_e32 v168, 8, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v8, v167, v165
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[161:162], null, 0x2400, v1, s[8:9]
	v_mad_co_i64_i32 v[163:164], null, 0x2400, v2, s[6:7]
	v_dual_mov_b32 v125, v121 :: v_dual_mov_b32 v126, v121
	v_dual_mov_b32 v127, v121 :: v_dual_mov_b32 v128, v121
	; === DTVA frag bases (FB[0..3] in v[238:245]) ===
	v_or_b32_e32 v129, s2, v8
	v_mad_co_i64_i32 v[238:239], null, 0x2400, v129, s[8:9]
	v_lshl_or_b32 v129, v172, 4, 0
	v_add_co_u32 v238, vcc_lo, v129, v238
	v_add_co_ci_u32_e64 v239, null, 0, v239, vcc_lo
	v_mov_b32_e32 v129, 0x24000
	v_add_co_u32 v240, vcc_lo, v129, v238
	v_add_co_ci_u32_e64 v241, null, 0, v239, vcc_lo
	v_add_co_u32 v242, vcc_lo, v129, v240
	v_add_co_ci_u32_e64 v243, null, 0, v241, vcc_lo
	v_add_co_u32 v244, vcc_lo, v129, v242
	v_add_co_ci_u32_e64 v245, null, 0, v243, vcc_lo
	; === DTVA prologue X loads (8 b128, 4 frags x 2 K-halves) ===
	s_clause 0xf
	global_load_b128 v[174:177], v[238:239], off
	global_load_b128 v[206:209], v[238:239], off offset:32
	global_load_b128 v[194:197], v[240:241], off
	global_load_b128 v[210:213], v[240:241], off offset:32
	global_load_b128 v[198:201], v[242:243], off
	global_load_b128 v[214:217], v[242:243], off offset:32
	global_load_b128 v[202:205], v[244:245], off
	global_load_b128 v[218:221], v[244:245], off offset:32
	global_load_b128 v[246:249], v[238:239], off offset:64
	global_load_b128 v[262:265], v[238:239], off offset:96
	global_load_b128 v[250:253], v[240:241], off offset:64
	global_load_b128 v[266:269], v[240:241], off offset:96
	global_load_b128 v[254:257], v[242:243], off offset:64
	global_load_b128 v[270:273], v[242:243], off offset:96
	global_load_b128 v[258:261], v[244:245], off offset:64
	global_load_b128 v[274:277], v[244:245], off offset:96
	s_clause 0x3
	global_load_b128 v[145:148], v[163:164], off
	global_load_b128 v[149:152], v[163:164], off offset:16
	global_load_b128 v[153:156], v[163:164], off offset:32
	global_load_b128 v[157:160], v[163:164], off offset:48
	v_dual_mov_b32 v113, v121 :: v_dual_mov_b32 v114, v121
	v_dual_mov_b32 v115, v121 :: v_dual_mov_b32 v116, v121
	v_dual_mov_b32 v117, v121 :: v_dual_mov_b32 v118, v121
	v_dual_mov_b32 v119, v121 :: v_dual_mov_b32 v120, v121
	v_dual_mov_b32 v105, v121 :: v_dual_mov_b32 v106, v121
	v_dual_mov_b32 v107, v121 :: v_dual_mov_b32 v108, v121
	v_dual_mov_b32 v109, v121 :: v_dual_mov_b32 v110, v121
	v_dual_mov_b32 v111, v121 :: v_dual_mov_b32 v112, v121
	v_dual_mov_b32 v97, v121 :: v_dual_mov_b32 v98, v121
	v_dual_mov_b32 v99, v121 :: v_dual_mov_b32 v100, v121
	v_dual_mov_b32 v101, v121 :: v_dual_mov_b32 v102, v121
	v_dual_mov_b32 v103, v121 :: v_dual_mov_b32 v104, v121
	v_dual_mov_b32 v89, v121 :: v_dual_mov_b32 v90, v121
	v_dual_mov_b32 v91, v121 :: v_dual_mov_b32 v92, v121
	v_dual_mov_b32 v93, v121 :: v_dual_mov_b32 v94, v121
	v_dual_mov_b32 v95, v121 :: v_dual_mov_b32 v96, v121
	v_dual_mov_b32 v81, v121 :: v_dual_mov_b32 v82, v121
	v_dual_mov_b32 v83, v121 :: v_dual_mov_b32 v84, v121
	v_dual_mov_b32 v85, v121 :: v_dual_mov_b32 v86, v121
	v_dual_mov_b32 v87, v121 :: v_dual_mov_b32 v88, v121
	v_dual_mov_b32 v73, v121 :: v_dual_mov_b32 v74, v121
	v_dual_mov_b32 v75, v121 :: v_dual_mov_b32 v76, v121
	v_dual_mov_b32 v77, v121 :: v_dual_mov_b32 v78, v121
	v_dual_mov_b32 v79, v121 :: v_dual_mov_b32 v80, v121
	v_dual_mov_b32 v65, v121 :: v_dual_mov_b32 v66, v121
	v_dual_mov_b32 v67, v121 :: v_dual_mov_b32 v68, v121
	v_dual_mov_b32 v69, v121 :: v_dual_mov_b32 v70, v121
	v_dual_mov_b32 v71, v121 :: v_dual_mov_b32 v72, v121
	v_dual_mov_b32 v57, v121 :: v_dual_mov_b32 v58, v121
	v_dual_mov_b32 v59, v121 :: v_dual_mov_b32 v60, v121
	v_dual_mov_b32 v61, v121 :: v_dual_mov_b32 v62, v121
	v_dual_mov_b32 v63, v121 :: v_dual_mov_b32 v64, v121
	v_dual_mov_b32 v49, v121 :: v_dual_mov_b32 v50, v121
	v_dual_mov_b32 v51, v121 :: v_dual_mov_b32 v52, v121
	v_dual_mov_b32 v53, v121 :: v_dual_mov_b32 v54, v121
	v_dual_mov_b32 v55, v121 :: v_dual_mov_b32 v56, v121
	v_dual_mov_b32 v41, v121 :: v_dual_mov_b32 v42, v121
	v_dual_mov_b32 v43, v121 :: v_dual_mov_b32 v44, v121
	v_dual_mov_b32 v45, v121 :: v_dual_mov_b32 v46, v121
	v_dual_mov_b32 v47, v121 :: v_dual_mov_b32 v48, v121
	v_dual_mov_b32 v33, v121 :: v_dual_mov_b32 v34, v121
	v_dual_mov_b32 v35, v121 :: v_dual_mov_b32 v36, v121
	v_dual_mov_b32 v37, v121 :: v_dual_mov_b32 v38, v121
	v_dual_mov_b32 v39, v121 :: v_dual_mov_b32 v40, v121
	v_dual_mov_b32 v25, v121 :: v_dual_mov_b32 v26, v121
	v_dual_mov_b32 v27, v121 :: v_dual_mov_b32 v28, v121
	v_dual_mov_b32 v29, v121 :: v_dual_mov_b32 v30, v121
	v_dual_mov_b32 v31, v121 :: v_dual_mov_b32 v32, v121
	v_dual_mov_b32 v17, v121 :: v_dual_mov_b32 v18, v121
	v_dual_mov_b32 v19, v121 :: v_dual_mov_b32 v20, v121
	v_dual_mov_b32 v21, v121 :: v_dual_mov_b32 v22, v121
	v_dual_mov_b32 v23, v121 :: v_dual_mov_b32 v24, v121
	v_dual_mov_b32 v9, v121 :: v_dual_mov_b32 v10, v121
	v_dual_mov_b32 v11, v121 :: v_dual_mov_b32 v12, v121
	v_dual_mov_b32 v13, v121 :: v_dual_mov_b32 v14, v121
	v_dual_mov_b32 v15, v121 :: v_dual_mov_b32 v16, v121
	v_dual_mov_b32 v1, v121 :: v_dual_mov_b32 v2, v121
	v_dual_mov_b32 v3, v121 :: v_dual_mov_b32 v4, v121
	v_mov_b32_e32 v7, v121
	v_or_b32_e32 v169, 0x4800, v166
	v_lshl_or_b32 v170, v170, 4, 0x4800
	v_or_b32_e32 v171, 0x4b00, v166
	v_mul_u32_u24_e32 v172, 0x90, v172
	v_dual_mov_b32 v8, v121 :: v_dual_lshlrev_b32 v173, 4, v8
	s_mov_b32 s6, 0
	s_mov_b64 s[0:1], 0
	s_mov_b32 s7, 0
	s_wait_loadcnt 0x3
	ds_store_b128 v166, v[145:148] offset:18432
	s_wait_loadcnt 0x2
	ds_store_b128 v166, v[149:152] offset:20736
	s_wait_loadcnt 0x1
	ds_store_b128 v166, v[153:156] offset:23040
	s_wait_loadcnt 0x0
	ds_store_b128 v166, v[157:160] offset:25344
	;;#ASMSTART
	s_barrier_signal -1
	;;#ASMEND
	;;#ASMSTART
	s_barrier_wait 0xffff
	;;#ASMEND
                                        ; implicit-def: $vgpr129_vgpr130_vgpr131_vgpr132
                                        ; implicit-def: $vgpr133_vgpr134_vgpr135_vgpr136
                                        ; implicit-def: $vgpr145_vgpr146_vgpr147_vgpr148
                                        ; implicit-def: $vgpr149_vgpr150_vgpr151_vgpr152
                                        ; implicit-def: $vgpr153_vgpr154_vgpr155_vgpr156
                                        ; implicit-def: $vgpr157_vgpr158_vgpr159_vgpr160
                                        ; implicit-def: $vgpr137_vgpr138_vgpr139_vgpr140
                                        ; implicit-def: $vgpr141_vgpr142_vgpr143_vgpr144
	s_branch .LBB0_2
.LBB0_1:                                ;   in Loop: Header=BB0_2 Depth=1
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB0_6
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s7, 0x11e0
	s_cselect_b32 s8, -1, 0
	s_cmp_gt_u32 s7, 0x11df
	s_cbranch_scc1 .LBB0_4
; %bb.3:                                ;   in Loop: Header=BB0_2 Depth=1
	s_wait_loadcnt 0x0
	v_add_co_u32 v145, vcc_lo, v163, s0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v146, null, s1, v164, vcc_lo
	s_clause 0x3
	global_load_b128 v[157:160], v[145:146], off offset:64
	global_load_b128 v[153:156], v[145:146], off offset:80
	global_load_b128 v[149:152], v[145:146], off offset:96
	global_load_b128 v[145:148], v[145:146], off offset:112
.LBB0_4:                                ;   in Loop: Header=BB0_2 Depth=1
	s_mul_i32 s9, s6, 0x240
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s8
	v_add_lshl_u32 v141, v172, s9, 4
	s_mov_b32 s8, -1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v230, v170, v141
	v_add_nc_u32_e32 v234, v171, v141
	; === DTVA X loads for THIS iter (FB[0..3]+s[0:1] -> v[130:137]) ===
	; Save loop-branch vcc_lo before v_add_co_u32 sequence clobbers it.
	s_mov_b32 s12, vcc_lo
	v_add_co_u32 v130, vcc_lo, v238, s0
	v_add_co_ci_u32_e64 v131, null, s1, v239, vcc_lo
	v_add_co_u32 v132, vcc_lo, v240, s0
	v_add_co_ci_u32_e64 v133, null, s1, v241, vcc_lo
	v_add_co_u32 v134, vcc_lo, v242, s0
	v_add_co_ci_u32_e64 v135, null, s1, v243, vcc_lo
	v_add_co_u32 v136, vcc_lo, v244, s0
	v_add_co_ci_u32_e64 v137, null, s1, v245, vcc_lo
	; Restore loop-branch vcc_lo so s_cbranch_vccnz at bb.4 tail works.
	s_mov_b32 vcc_lo, s12
	s_clause 0x7
	global_load_b128 v[174:177], v[130:131], off
	global_load_b128 v[206:209], v[130:131], off offset:32
	global_load_b128 v[194:197], v[132:133], off
	global_load_b128 v[210:213], v[132:133], off offset:32
	global_load_b128 v[198:201], v[134:135], off
	global_load_b128 v[214:217], v[134:135], off offset:32
	global_load_b128 v[202:205], v[136:137], off
	global_load_b128 v[218:221], v[136:137], off offset:32
	ds_load_b128 v[178:181], v230
	ds_load_b128 v[182:185], v230 offset:256
	ds_load_b128 v[186:189], v230 offset:512
	ds_load_b128 v[190:193], v234
	ds_load_b128 v[222:225], v230 offset:4608
	ds_load_b128 v[226:229], v230 offset:4864
	ds_load_b128 v[230:233], v230 offset:5120
	ds_load_b128 v[234:237], v234 offset:4608
	s_wait_loadcnt 0x0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_bf16 v[89:96], v[194:197], v[178:181], v[89:96]
	v_wmma_f32_16x16x16_bf16 v[81:88], v[194:197], v[182:185], v[81:88]
	v_wmma_f32_16x16x16_bf16 v[73:80], v[194:197], v[186:189], v[73:80]
	v_wmma_f32_16x16x16_bf16 v[65:72], v[194:197], v[190:193], v[65:72]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_bf16 v[57:64], v[198:201], v[178:181], v[57:64]
	v_wmma_f32_16x16x16_bf16 v[49:56], v[198:201], v[182:185], v[49:56]
	v_wmma_f32_16x16x16_bf16 v[41:48], v[198:201], v[186:189], v[41:48]
	v_wmma_f32_16x16x16_bf16 v[33:40], v[198:201], v[190:193], v[33:40]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x16_bf16 v[25:32], v[202:205], v[178:181], v[25:32]
	v_wmma_f32_16x16x16_bf16 v[17:24], v[202:205], v[182:185], v[17:24]
	v_wmma_f32_16x16x16_bf16 v[9:16], v[202:205], v[186:189], v[9:16]
	v_wmma_f32_16x16x16_bf16 v[1:8], v[202:205], v[190:193], v[1:8]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[89:96], v[210:213], v[222:225], v[89:96]
	v_wmma_f32_16x16x16_bf16 v[81:88], v[210:213], v[226:229], v[81:88]
	v_wmma_f32_16x16x16_bf16 v[73:80], v[210:213], v[230:233], v[73:80]
	v_wmma_f32_16x16x16_bf16 v[65:72], v[210:213], v[234:237], v[65:72]
	v_wmma_f32_16x16x16_bf16 v[57:64], v[214:217], v[222:225], v[57:64]
	v_wmma_f32_16x16x16_bf16 v[49:56], v[214:217], v[226:229], v[49:56]
	v_wmma_f32_16x16x16_bf16 v[41:48], v[214:217], v[230:233], v[41:48]
	v_wmma_f32_16x16x16_bf16 v[33:40], v[214:217], v[234:237], v[33:40]
	v_wmma_f32_16x16x16_bf16 v[25:32], v[218:221], v[222:225], v[25:32]
	v_wmma_f32_16x16x16_bf16 v[17:24], v[218:221], v[226:229], v[17:24]
	v_wmma_f32_16x16x16_bf16 v[9:16], v[218:221], v[230:233], v[9:16]
	v_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
	;;#ASMSTART
	s_barrier_signal -1
	;;#ASMEND
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_1
; %bb.5:                                ;   in Loop: Header=BB0_2 Depth=1
	s_xor_b32 s6, s6, 1
	s_add_nc_u64 s[0:1], s[0:1], 64
	s_wait_alu 0xfffe
	s_mul_i32 s8, s6, 0x2400
	s_add_co_i32 s7, s7, 32
	s_wait_alu 0xfffe
	v_add_nc_u32_e32 v174, s8, v166
	v_add_nc_u32_e32 v175, s8, v169
	s_mov_b32 s8, 0
	s_wait_loadcnt 0x3
	ds_store_b128 v175, v[157:160]
	s_wait_loadcnt 0x2
	ds_store_b128 v175, v[153:156] offset:2304
	s_wait_loadcnt 0x1
	ds_store_b128 v175, v[149:152] offset:4608
	s_wait_loadcnt 0x0
	ds_store_b128 v175, v[145:148] offset:6912
	;;#ASMSTART
	s_barrier_wait 0xffff
	;;#ASMEND
	s_branch .LBB0_1
.LBB0_6:
	s_wait_loadcnt 0x4
	v_dual_mov_b32 v139, 0 :: v_dual_and_b32 v0, 64, v0
	v_mov_b32_e32 v140, 0
	s_cmp_lg_u64 s[10:11], 0
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v0, s3, v0
	s_cmp_eq_u64 s[10:11], 0
	v_or_b32_e32 v129, v0, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v130, 31, v129
	v_lshlrev_b64_e32 v[129:130], 2, v[129:130]
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v131, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v132, null, s11, v130, vcc_lo
	global_load_b32 v140, v[131:132], off
.LBB0_8:
	v_or3_b32 v0, v167, s2, v168
	s_wait_loadcnt 0x0
	v_add_f32_e32 v143, v140, v121
	v_add_f32_e32 v144, v140, v122
	v_add_f32_e32 v145, v140, v123
	v_add_f32_e32 v147, v140, v125
	v_or_b32_e32 v131, 1, v0
	v_or_b32_e32 v133, 2, v0
	v_mad_co_i64_i32 v[121:122], null, 0x4800, v0, s[4:5]
	v_or_b32_e32 v135, 3, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_i64_i32 v[131:132], null, 0x4800, v131, s[4:5]
	v_or_b32_e32 v123, 4, v0
	v_mad_co_i64_i32 v[133:134], null, 0x4800, v133, s[4:5]
	v_or_b32_e32 v141, 5, v0
	v_mad_co_i64_i32 v[135:136], null, 0x4800, v135, s[4:5]
	v_add_co_u32 v121, vcc_lo, v121, v129
	v_mad_co_i64_i32 v[137:138], null, 0x4800, v123, s[4:5]
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v122, null, v122, v130, vcc_lo
	v_add_co_u32 v131, vcc_lo, v131, v129
	v_mad_co_i64_i32 v[141:142], null, 0x4800, v141, s[4:5]
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v132, null, v132, v130, vcc_lo
	v_add_co_u32 v133, vcc_lo, v133, v129
	v_or_b32_e32 v125, 6, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v134, null, v134, v130, vcc_lo
	v_add_co_u32 v123, vcc_lo, v135, v129
	v_or_b32_e32 v149, 7, v0
	v_add_f32_e32 v146, v140, v124
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v124, null, v136, v130, vcc_lo
	v_add_co_u32 v135, vcc_lo, v137, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v136, null, v138, v130, vcc_lo
	v_mad_co_i64_i32 v[137:138], null, 0x4800, v125, s[4:5]
	v_add_co_u32 v125, vcc_lo, v141, v129
	v_add_f32_e32 v148, v140, v126
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v126, null, v142, v130, vcc_lo
	v_mad_co_i64_i32 v[141:142], null, 0x4800, v149, s[4:5]
	v_add_f32_e32 v150, v140, v127
	s_wait_alu 0xfffe
	v_cndmask_b32_e64 v127, 0, 1, s1
	v_add_co_u32 v137, vcc_lo, v137, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v138, null, v138, v130, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ne_u32_e64 s0, 1, v127
	v_add_co_u32 v127, vcc_lo, v141, v129
	v_add_f32_e32 v140, v140, v128
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v128, null, v142, v130, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_clause 0x7
	global_store_b32 v[121:122], v143, off
	global_store_b32 v[131:132], v144, off
	global_store_b32 v[133:134], v145, off
	global_store_b32 v[123:124], v146, off
	global_store_b32 v[135:136], v147, off
	global_store_b32 v[125:126], v148, off
	global_store_b32 v[137:138], v150, off
	global_store_b32 v[127:128], v140, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_10
; %bb.9:
	v_add_co_u32 v139, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v140, null, s11, v130, vcc_lo
	global_load_b32 v139, v[139:140], off offset:64
.LBB0_10:
	s_wait_loadcnt 0x0
	v_dual_add_f32 v140, v139, v113 :: v_dual_mov_b32 v113, 0
	v_dual_add_f32 v141, v139, v114 :: v_dual_mov_b32 v114, 0
	v_add_f32_e32 v115, v139, v115
	v_add_f32_e32 v116, v139, v116
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v117, v139, v117
	v_add_f32_e32 v118, v139, v118
	v_add_f32_e32 v119, v139, v119
	v_add_f32_e32 v120, v139, v120
	s_clause 0x7
	global_store_b32 v[121:122], v140, off offset:64
	global_store_b32 v[131:132], v141, off offset:64
	global_store_b32 v[133:134], v115, off offset:64
	global_store_b32 v[123:124], v116, off offset:64
	global_store_b32 v[135:136], v117, off offset:64
	global_store_b32 v[125:126], v118, off offset:64
	global_store_b32 v[137:138], v119, off offset:64
	global_store_b32 v[127:128], v120, off offset:64
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_12
; %bb.11:
	v_add_co_u32 v114, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v115, null, s11, v130, vcc_lo
	global_load_b32 v114, v[114:115], off offset:128
.LBB0_12:
	s_wait_loadcnt 0x0
	v_add_f32_e32 v105, v114, v105
	v_add_f32_e32 v106, v114, v106
	v_add_f32_e32 v107, v114, v107
	v_add_f32_e32 v108, v114, v108
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v109, v114, v109
	v_add_f32_e32 v110, v114, v110
	v_add_f32_e32 v111, v114, v111
	v_add_f32_e32 v112, v114, v112
	s_clause 0x7
	global_store_b32 v[121:122], v105, off offset:128
	global_store_b32 v[131:132], v106, off offset:128
	global_store_b32 v[133:134], v107, off offset:128
	global_store_b32 v[123:124], v108, off offset:128
	global_store_b32 v[135:136], v109, off offset:128
	global_store_b32 v[125:126], v110, off offset:128
	global_store_b32 v[137:138], v111, off offset:128
	global_store_b32 v[127:128], v112, off offset:128
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_14
; %bb.13:
	v_add_co_u32 v105, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v106, null, s11, v130, vcc_lo
	global_load_b32 v113, v[105:106], off offset:192
.LBB0_14:
	s_wait_loadcnt 0x0
	v_dual_add_f32 v97, v113, v97 :: v_dual_mov_b32 v106, 0
	v_dual_add_f32 v98, v113, v98 :: v_dual_mov_b32 v105, 0
	v_add_f32_e32 v99, v113, v99
	v_add_f32_e32 v100, v113, v100
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v101, v113, v101
	v_add_f32_e32 v102, v113, v102
	v_add_f32_e32 v103, v113, v103
	v_add_f32_e32 v104, v113, v104
	s_clause 0x7
	global_store_b32 v[121:122], v97, off offset:192
	global_store_b32 v[131:132], v98, off offset:192
	global_store_b32 v[133:134], v99, off offset:192
	global_store_b32 v[123:124], v100, off offset:192
	global_store_b32 v[135:136], v101, off offset:192
	global_store_b32 v[125:126], v102, off offset:192
	global_store_b32 v[137:138], v103, off offset:192
	global_store_b32 v[127:128], v104, off offset:192
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_16
; %bb.15:
	v_add_co_u32 v97, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v98, null, s11, v130, vcc_lo
	global_load_b32 v106, v[97:98], off
.LBB0_16:
	v_or_b32_e32 v97, 16, v0
	v_or_b32_e32 v99, 17, v0
	v_or_b32_e32 v101, 18, v0
	v_or_b32_e32 v103, 19, v0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v111, v106, v89
	v_mad_co_i64_i32 v[97:98], null, 0x4800, v97, s[4:5]
	v_mad_co_i64_i32 v[99:100], null, 0x4800, v99, s[4:5]
	v_mad_co_i64_i32 v[101:102], null, 0x4800, v101, s[4:5]
	v_mad_co_i64_i32 v[103:104], null, 0x4800, v103, s[4:5]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v89, vcc_lo, v97, v129
	v_add_f32_e32 v113, v106, v91
	v_or_b32_e32 v91, 20, v0
	v_add_f32_e32 v112, v106, v90
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v90, null, v98, v130, vcc_lo
	v_add_co_u32 v97, vcc_lo, v99, v129
	v_or_b32_e32 v107, 21, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v98, null, v100, v130, vcc_lo
	v_add_co_u32 v99, vcc_lo, v101, v129
	v_add_f32_e32 v115, v106, v93
	v_or_b32_e32 v93, 22, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v100, null, v102, v130, vcc_lo
	v_mad_co_i64_i32 v[101:102], null, 0x4800, v91, s[4:5]
	v_add_co_u32 v91, vcc_lo, v103, v129
	v_or_b32_e32 v109, 23, v0
	v_add_f32_e32 v114, v106, v92
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v92, null, v104, v130, vcc_lo
	v_mad_co_i64_i32 v[103:104], null, 0x4800, v107, s[4:5]
	v_mad_co_i64_i32 v[107:108], null, 0x4800, v93, s[4:5]
	v_mad_co_i64_i32 v[109:110], null, 0x4800, v109, s[4:5]
	v_add_co_u32 v101, vcc_lo, v101, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v102, null, v102, v130, vcc_lo
	v_add_co_u32 v93, vcc_lo, v103, v129
	v_add_f32_e32 v116, v106, v94
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v94, null, v104, v130, vcc_lo
	v_add_co_u32 v103, vcc_lo, v107, v129
	v_add_f32_e32 v117, v106, v95
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v104, null, v108, v130, vcc_lo
	v_add_co_u32 v95, vcc_lo, v109, v129
	v_add_f32_e32 v106, v106, v96
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v96, null, v110, v130, vcc_lo
	s_and_b32 vcc_lo, exec_lo, s0
	s_clause 0x7
	global_store_b32 v[89:90], v111, off
	global_store_b32 v[97:98], v112, off
	global_store_b32 v[99:100], v113, off
	global_store_b32 v[91:92], v114, off
	global_store_b32 v[101:102], v115, off
	global_store_b32 v[93:94], v116, off
	global_store_b32 v[103:104], v117, off
	global_store_b32 v[95:96], v106, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_18
; %bb.17:
	v_add_co_u32 v105, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v106, null, s11, v130, vcc_lo
	global_load_b32 v105, v[105:106], off offset:64
.LBB0_18:
	s_wait_loadcnt 0x0
	v_dual_add_f32 v106, v105, v81 :: v_dual_mov_b32 v81, 0
	v_dual_add_f32 v107, v105, v82 :: v_dual_mov_b32 v82, 0
	v_add_f32_e32 v83, v105, v83
	v_add_f32_e32 v84, v105, v84
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v85, v105, v85
	v_add_f32_e32 v86, v105, v86
	v_add_f32_e32 v87, v105, v87
	v_add_f32_e32 v88, v105, v88
	s_clause 0x7
	global_store_b32 v[89:90], v106, off offset:64
	global_store_b32 v[97:98], v107, off offset:64
	global_store_b32 v[99:100], v83, off offset:64
	global_store_b32 v[91:92], v84, off offset:64
	global_store_b32 v[101:102], v85, off offset:64
	global_store_b32 v[93:94], v86, off offset:64
	global_store_b32 v[103:104], v87, off offset:64
	global_store_b32 v[95:96], v88, off offset:64
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_20
; %bb.19:
	v_add_co_u32 v82, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v83, null, s11, v130, vcc_lo
	global_load_b32 v82, v[82:83], off offset:128
.LBB0_20:
	s_wait_loadcnt 0x0
	v_add_f32_e32 v73, v82, v73
	v_add_f32_e32 v74, v82, v74
	v_add_f32_e32 v75, v82, v75
	v_add_f32_e32 v76, v82, v76
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v77, v82, v77
	v_add_f32_e32 v78, v82, v78
	v_add_f32_e32 v79, v82, v79
	v_add_f32_e32 v80, v82, v80
	s_clause 0x7
	global_store_b32 v[89:90], v73, off offset:128
	global_store_b32 v[97:98], v74, off offset:128
	global_store_b32 v[99:100], v75, off offset:128
	global_store_b32 v[91:92], v76, off offset:128
	global_store_b32 v[101:102], v77, off offset:128
	global_store_b32 v[93:94], v78, off offset:128
	global_store_b32 v[103:104], v79, off offset:128
	global_store_b32 v[95:96], v80, off offset:128
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_22
; %bb.21:
	v_add_co_u32 v73, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v74, null, s11, v130, vcc_lo
	global_load_b32 v81, v[73:74], off offset:192
.LBB0_22:
	s_wait_loadcnt 0x0
	v_dual_add_f32 v65, v81, v65 :: v_dual_mov_b32 v74, 0
	v_dual_add_f32 v66, v81, v66 :: v_dual_mov_b32 v73, 0
	v_add_f32_e32 v67, v81, v67
	v_add_f32_e32 v68, v81, v68
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v69, v81, v69
	v_add_f32_e32 v70, v81, v70
	v_add_f32_e32 v71, v81, v71
	v_add_f32_e32 v72, v81, v72
	s_clause 0x7
	global_store_b32 v[89:90], v65, off offset:192
	global_store_b32 v[97:98], v66, off offset:192
	global_store_b32 v[99:100], v67, off offset:192
	global_store_b32 v[91:92], v68, off offset:192
	global_store_b32 v[101:102], v69, off offset:192
	global_store_b32 v[93:94], v70, off offset:192
	global_store_b32 v[103:104], v71, off offset:192
	global_store_b32 v[95:96], v72, off offset:192
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_24
; %bb.23:
	v_add_co_u32 v65, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v66, null, s11, v130, vcc_lo
	global_load_b32 v74, v[65:66], off
.LBB0_24:
	v_or_b32_e32 v65, 32, v0
	v_or_b32_e32 v67, 33, v0
	v_or_b32_e32 v69, 34, v0
	v_or_b32_e32 v71, 35, v0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v79, v74, v57
	v_mad_co_i64_i32 v[65:66], null, 0x4800, v65, s[4:5]
	v_mad_co_i64_i32 v[67:68], null, 0x4800, v67, s[4:5]
	v_mad_co_i64_i32 v[69:70], null, 0x4800, v69, s[4:5]
	v_mad_co_i64_i32 v[71:72], null, 0x4800, v71, s[4:5]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v57, vcc_lo, v65, v129
	v_add_f32_e32 v81, v74, v59
	v_or_b32_e32 v59, 36, v0
	v_add_f32_e32 v80, v74, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v58, null, v66, v130, vcc_lo
	v_add_co_u32 v65, vcc_lo, v67, v129
	v_or_b32_e32 v75, 37, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v66, null, v68, v130, vcc_lo
	v_add_co_u32 v67, vcc_lo, v69, v129
	v_add_f32_e32 v83, v74, v61
	v_or_b32_e32 v61, 38, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v68, null, v70, v130, vcc_lo
	v_mad_co_i64_i32 v[69:70], null, 0x4800, v59, s[4:5]
	v_add_co_u32 v59, vcc_lo, v71, v129
	v_or_b32_e32 v77, 39, v0
	v_add_f32_e32 v82, v74, v60
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v60, null, v72, v130, vcc_lo
	v_mad_co_i64_i32 v[71:72], null, 0x4800, v75, s[4:5]
	v_mad_co_i64_i32 v[75:76], null, 0x4800, v61, s[4:5]
	v_mad_co_i64_i32 v[77:78], null, 0x4800, v77, s[4:5]
	v_add_co_u32 v69, vcc_lo, v69, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v70, null, v70, v130, vcc_lo
	v_add_co_u32 v61, vcc_lo, v71, v129
	v_add_f32_e32 v84, v74, v62
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v62, null, v72, v130, vcc_lo
	v_add_co_u32 v71, vcc_lo, v75, v129
	v_add_f32_e32 v85, v74, v63
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v72, null, v76, v130, vcc_lo
	v_add_co_u32 v63, vcc_lo, v77, v129
	v_add_f32_e32 v74, v74, v64
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v64, null, v78, v130, vcc_lo
	s_and_b32 vcc_lo, exec_lo, s0
	s_clause 0x7
	global_store_b32 v[57:58], v79, off
	global_store_b32 v[65:66], v80, off
	global_store_b32 v[67:68], v81, off
	global_store_b32 v[59:60], v82, off
	global_store_b32 v[69:70], v83, off
	global_store_b32 v[61:62], v84, off
	global_store_b32 v[71:72], v85, off
	global_store_b32 v[63:64], v74, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_26
; %bb.25:
	v_add_co_u32 v73, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v74, null, s11, v130, vcc_lo
	global_load_b32 v73, v[73:74], off offset:64
.LBB0_26:
	s_wait_loadcnt 0x0
	v_dual_add_f32 v74, v73, v49 :: v_dual_mov_b32 v49, 0
	v_dual_add_f32 v75, v73, v50 :: v_dual_mov_b32 v50, 0
	v_add_f32_e32 v51, v73, v51
	v_add_f32_e32 v52, v73, v52
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v53, v73, v53
	v_add_f32_e32 v54, v73, v54
	v_add_f32_e32 v55, v73, v55
	v_add_f32_e32 v56, v73, v56
	s_clause 0x7
	global_store_b32 v[57:58], v74, off offset:64
	global_store_b32 v[65:66], v75, off offset:64
	global_store_b32 v[67:68], v51, off offset:64
	global_store_b32 v[59:60], v52, off offset:64
	global_store_b32 v[69:70], v53, off offset:64
	global_store_b32 v[61:62], v54, off offset:64
	global_store_b32 v[71:72], v55, off offset:64
	global_store_b32 v[63:64], v56, off offset:64
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_28
; %bb.27:
	v_add_co_u32 v50, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v51, null, s11, v130, vcc_lo
	global_load_b32 v50, v[50:51], off offset:128
.LBB0_28:
	s_wait_loadcnt 0x0
	v_add_f32_e32 v41, v50, v41
	v_add_f32_e32 v42, v50, v42
	v_add_f32_e32 v43, v50, v43
	v_add_f32_e32 v44, v50, v44
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v45, v50, v45
	v_add_f32_e32 v46, v50, v46
	v_add_f32_e32 v47, v50, v47
	v_add_f32_e32 v48, v50, v48
	s_clause 0x7
	global_store_b32 v[57:58], v41, off offset:128
	global_store_b32 v[65:66], v42, off offset:128
	global_store_b32 v[67:68], v43, off offset:128
	global_store_b32 v[59:60], v44, off offset:128
	global_store_b32 v[69:70], v45, off offset:128
	global_store_b32 v[61:62], v46, off offset:128
	global_store_b32 v[71:72], v47, off offset:128
	global_store_b32 v[63:64], v48, off offset:128
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_30
; %bb.29:
	v_add_co_u32 v41, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v42, null, s11, v130, vcc_lo
	global_load_b32 v49, v[41:42], off offset:192
.LBB0_30:
	s_wait_loadcnt 0x0
	v_dual_add_f32 v33, v49, v33 :: v_dual_mov_b32 v42, 0
	v_dual_add_f32 v34, v49, v34 :: v_dual_mov_b32 v41, 0
	v_add_f32_e32 v35, v49, v35
	v_add_f32_e32 v36, v49, v36
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v37, v49, v37
	v_add_f32_e32 v38, v49, v38
	v_add_f32_e32 v39, v49, v39
	v_add_f32_e32 v40, v49, v40
	s_clause 0x7
	global_store_b32 v[57:58], v33, off offset:192
	global_store_b32 v[65:66], v34, off offset:192
	global_store_b32 v[67:68], v35, off offset:192
	global_store_b32 v[59:60], v36, off offset:192
	global_store_b32 v[69:70], v37, off offset:192
	global_store_b32 v[61:62], v38, off offset:192
	global_store_b32 v[71:72], v39, off offset:192
	global_store_b32 v[63:64], v40, off offset:192
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_32
; %bb.31:
	v_add_co_u32 v33, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v34, null, s11, v130, vcc_lo
	global_load_b32 v42, v[33:34], off
.LBB0_32:
	v_or_b32_e32 v33, 48, v0
	v_or_b32_e32 v35, 49, v0
	v_or_b32_e32 v37, 50, v0
	v_or_b32_e32 v39, 51, v0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v47, v42, v25
	v_mad_co_i64_i32 v[33:34], null, 0x4800, v33, s[4:5]
	v_mad_co_i64_i32 v[35:36], null, 0x4800, v35, s[4:5]
	v_mad_co_i64_i32 v[37:38], null, 0x4800, v37, s[4:5]
	v_mad_co_i64_i32 v[39:40], null, 0x4800, v39, s[4:5]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v25, vcc_lo, v33, v129
	v_add_f32_e32 v49, v42, v27
	v_or_b32_e32 v27, 52, v0
	v_add_f32_e32 v48, v42, v26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v26, null, v34, v130, vcc_lo
	v_add_co_u32 v33, vcc_lo, v35, v129
	v_or_b32_e32 v43, 53, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v34, null, v36, v130, vcc_lo
	v_add_co_u32 v35, vcc_lo, v37, v129
	v_add_f32_e32 v51, v42, v29
	v_or_b32_e32 v29, 54, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v36, null, v38, v130, vcc_lo
	v_mad_co_i64_i32 v[37:38], null, 0x4800, v27, s[4:5]
	v_add_co_u32 v27, vcc_lo, v39, v129
	v_or_b32_e32 v0, 55, v0
	v_add_f32_e32 v50, v42, v28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v28, null, v40, v130, vcc_lo
	v_mad_co_i64_i32 v[39:40], null, 0x4800, v43, s[4:5]
	v_mad_co_i64_i32 v[43:44], null, 0x4800, v29, s[4:5]
	v_mad_co_i64_i32 v[45:46], null, 0x4800, v0, s[4:5]
	v_add_co_u32 v37, vcc_lo, v37, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v38, null, v38, v130, vcc_lo
	v_add_co_u32 v29, vcc_lo, v39, v129
	v_add_f32_e32 v52, v42, v30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v30, null, v40, v130, vcc_lo
	v_add_co_u32 v39, vcc_lo, v43, v129
	v_add_f32_e32 v0, v42, v31
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v40, null, v44, v130, vcc_lo
	v_add_co_u32 v31, vcc_lo, v45, v129
	v_add_f32_e32 v42, v42, v32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v32, null, v46, v130, vcc_lo
	s_and_b32 vcc_lo, exec_lo, s0
	s_clause 0x7
	global_store_b32 v[25:26], v47, off
	global_store_b32 v[33:34], v48, off
	global_store_b32 v[35:36], v49, off
	global_store_b32 v[27:28], v50, off
	global_store_b32 v[37:38], v51, off
	global_store_b32 v[29:30], v52, off
	global_store_b32 v[39:40], v0, off
	global_store_b32 v[31:32], v42, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_34
; %bb.33:
	v_add_co_u32 v41, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v42, null, s11, v130, vcc_lo
	global_load_b32 v41, v[41:42], off offset:64
.LBB0_34:
	s_wait_loadcnt 0x0
	v_dual_add_f32 v42, v41, v17 :: v_dual_mov_b32 v17, 0
	v_dual_add_f32 v19, v41, v19 :: v_dual_mov_b32 v0, 0
	v_add_f32_e32 v18, v41, v18
	v_add_f32_e32 v20, v41, v20
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v21, v41, v21
	v_add_f32_e32 v22, v41, v22
	v_add_f32_e32 v23, v41, v23
	v_add_f32_e32 v24, v41, v24
	s_clause 0x7
	global_store_b32 v[25:26], v42, off offset:64
	global_store_b32 v[33:34], v18, off offset:64
	global_store_b32 v[35:36], v19, off offset:64
	global_store_b32 v[27:28], v20, off offset:64
	global_store_b32 v[37:38], v21, off offset:64
	global_store_b32 v[29:30], v22, off offset:64
	global_store_b32 v[39:40], v23, off offset:64
	global_store_b32 v[31:32], v24, off offset:64
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_36
; %bb.35:
	v_add_co_u32 v17, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v18, null, s11, v130, vcc_lo
	global_load_b32 v17, v[17:18], off offset:128
.LBB0_36:
	s_wait_loadcnt 0x0
	v_add_f32_e32 v9, v17, v9
	v_add_f32_e32 v10, v17, v10
	v_add_f32_e32 v11, v17, v11
	v_add_f32_e32 v12, v17, v12
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v13, v17, v13
	v_add_f32_e32 v14, v17, v14
	v_add_f32_e32 v15, v17, v15
	v_add_f32_e32 v16, v17, v16
	s_clause 0x7
	global_store_b32 v[25:26], v9, off offset:128
	global_store_b32 v[33:34], v10, off offset:128
	global_store_b32 v[35:36], v11, off offset:128
	global_store_b32 v[27:28], v12, off offset:128
	global_store_b32 v[37:38], v13, off offset:128
	global_store_b32 v[29:30], v14, off offset:128
	global_store_b32 v[39:40], v15, off offset:128
	global_store_b32 v[31:32], v16, off offset:128
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_38
; %bb.37:
	v_add_co_u32 v9, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, s11, v130, vcc_lo
	global_load_b32 v0, v[9:10], off offset:192
.LBB0_38:
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v0, v1
	v_add_f32_e32 v2, v0, v2
	v_add_f32_e32 v3, v0, v3
	v_add_f32_e32 v4, v0, v4
	v_add_f32_e32 v5, v0, v5
	v_add_f32_e32 v6, v0, v6
	v_add_f32_e32 v7, v0, v7
	v_add_f32_e32 v0, v0, v8
	s_clause 0x7
	global_store_b32 v[25:26], v1, off offset:192
	global_store_b32 v[33:34], v2, off offset:192
	global_store_b32 v[35:36], v3, off offset:192
	global_store_b32 v[27:28], v4, off offset:192
	global_store_b32 v[37:38], v5, off offset:192
	global_store_b32 v[29:30], v6, off offset:192
	global_store_b32 v[39:40], v7, off offset:192
	global_store_b32 v[31:32], v0, off offset:192
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mm0_bf16_asm
		.amdhsa_group_segment_fixed_size 36864
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 278
		.amdhsa_next_free_sgpr 13
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 46
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	gemm_mm0_bf16_asm, .Lfunc_end0-gemm_mm0_bf16_asm
                                        ; -- End function
	.set gemm_mm0_bf16_asm.num_vgpr, 238
	.set gemm_mm0_bf16_asm.num_agpr, 0
	.set gemm_mm0_bf16_asm.numbered_sgpr, 12
	.set gemm_mm0_bf16_asm.num_named_barrier, 0
	.set gemm_mm0_bf16_asm.private_seg_size, 0
	.set gemm_mm0_bf16_asm.uses_vcc, 1
	.set gemm_mm0_bf16_asm.uses_flat_scratch, 0
	.set gemm_mm0_bf16_asm.has_dyn_sized_stack, 0
	.set gemm_mm0_bf16_asm.has_recursion, 0
	.set gemm_mm0_bf16_asm.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5880
; TotalNumSgprs: 14
; NumVgprs: 238
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 36864 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 30
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 241
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_4ddd41ce571b465b,@object ; @__hip_cuid_4ddd41ce571b465b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_4ddd41ce571b465b
__hip_cuid_4ddd41ce571b465b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_4ddd41ce571b465b, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.1 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_4ddd41ce571b465b
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 36864
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           gemm_mm0_bf16_asm
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         gemm_mm0_bf16_asm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     238
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
