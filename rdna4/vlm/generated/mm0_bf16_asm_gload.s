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
	v_or_b32_e32 v1, 0x80, v0
	v_lshrrev_b32_e32 v161, 2, v0
	v_or_b32_e32 v2, 0x100, v0
	s_lshl_b32 s0, ttmp7, 7
	v_dual_mov_b32 v105, 0 :: v_dual_and_b32 v168, 3, v0
	v_lshrrev_b32_e32 v162, 2, v1
	v_or_b32_e32 v1, 0x180, v0
	v_or_b32_e32 v3, s0, v161
	v_lshrrev_b32_e32 v163, 2, v2
	v_dual_mov_b32 v106, v105 :: v_dual_lshlrev_b32 v165, 4, v168
	v_or_b32_e32 v2, s0, v162
	v_lshrrev_b32_e32 v164, 2, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_or_b32_e32 v6, s0, v163
	s_lshl_b32 s1, ttmp9, 7
	v_dual_mov_b32 v108, v105 :: v_dual_lshlrev_b32 v169, 1, v0
	v_or_b32_e32 v7, s0, v164
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[188:189], null, 0x2400, v3, s[8:9]
	v_mad_co_i64_i32 v[186:187], null, 0x2400, v2, s[8:9]
	s_wait_alu 0xfffe
	v_or_b32_e32 v5, s1, v161
	v_mad_co_i64_i32 v[184:185], null, 0x2400, v6, s[8:9]
	v_mad_co_i64_i32 v[182:183], null, 0x2400, v7, s[8:9]
	v_add_co_u32 v1, vcc_lo, v188, v165
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v189, vcc_lo
	v_add_co_u32 v3, vcc_lo, v186, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, 0, v187, vcc_lo
	v_mad_co_i64_i32 v[180:181], null, 0x2400, v5, s[6:7]
	v_or_b32_e32 v7, s1, v162
	s_clause 0x1
	global_load_b128 v[129:132], v[1:2], off
	global_load_b128 v[133:136], v[3:4], off
	v_add_co_u32 v1, vcc_lo, v184, v165
	v_or_b32_e32 v8, s1, v163
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, 0, v185, vcc_lo
	v_add_co_u32 v3, vcc_lo, v182, v165
	v_mad_co_i64_i32 v[178:179], null, 0x2400, v7, s[6:7]
	v_or_b32_e32 v7, s1, v164
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, 0, v183, vcc_lo
	v_add_co_u32 v5, vcc_lo, v180, v165
	v_mad_co_i64_i32 v[176:177], null, 0x2400, v8, s[6:7]
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, 0, v181, vcc_lo
	v_mad_co_i64_i32 v[174:175], null, 0x2400, v7, s[6:7]
	v_lshlrev_b32_e32 v170, 7, v168
	global_load_b128 v[137:140], v[5:6], off
	v_add_co_u32 v5, vcc_lo, v178, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, 0, v179, vcc_lo
	v_add_co_u32 v7, vcc_lo, v176, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, 0, v177, vcc_lo
	v_add_co_u32 v9, vcc_lo, v174, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, 0, v175, vcc_lo
	global_load_b128 v[141:144], v[5:6], off
	global_load_b128 v[145:148], v[1:2], off
	global_load_b128 v[149:152], v[7:8], off
	global_load_b128 v[153:156], v[3:4], off
	global_load_b128 v[157:160], v[9:10], off
	v_lshrrev_b32_e32 v8, 1, v0
	v_mov_b32_e32 v5, v105
	v_add_co_u32 v174, vcc_lo, v174, 64
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v7, v105 :: v_dual_and_b32 v168, 8, v8
	v_or_b32_e32 v8, v170, v161
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v175, null, 0, v175, vcc_lo
	v_add_co_u32 v176, vcc_lo, v176, 64
	v_or_b32_e32 v172, v170, v162
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v177, null, 0, v177, vcc_lo
	v_add_co_u32 v178, vcc_lo, v178, 64
	v_or_b32_e32 v173, v170, v163
	v_dual_mov_b32 v107, v105 :: v_dual_and_b32 v166, 15, v0
	v_and_b32_e32 v169, 64, v169
	v_or_b32_e32 v190, v170, v164
	v_lshlrev_b32_e32 v8, 4, v8
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v179, null, 0, v179, vcc_lo
	v_add_co_u32 v180, vcc_lo, v180, 64
	v_lshlrev_b32_e32 v193, 4, v172
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v181, null, 0, v181, vcc_lo
	v_add_co_u32 v182, vcc_lo, v182, 64
	v_lshlrev_b32_e32 v194, 4, v173
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v183, null, 0, v183, vcc_lo
	v_add_co_u32 v184, vcc_lo, v184, 64
	v_dual_mov_b32 v110, v105 :: v_dual_and_b32 v171, 0x4f, v0
	v_lshlrev_b32_e32 v192, 4, v168
	v_lshlrev_b32_e32 v190, 4, v190
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v185, null, 0, v185, vcc_lo
	v_add_co_u32 v186, vcc_lo, v186, 64
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v187, null, 0, v187, vcc_lo
	v_add_co_u32 v188, vcc_lo, v188, 64
	v_lshl_or_b32 v167, v0, 4, 0x4300
	v_dual_mov_b32 v109, v105 :: v_dual_mov_b32 v112, v105
	v_dual_mov_b32 v111, v105 :: v_dual_mov_b32 v122, v105
	v_dual_mov_b32 v121, v105 :: v_dual_mov_b32 v124, v105
	v_dual_mov_b32 v123, v105 :: v_dual_mov_b32 v126, v105
	v_dual_mov_b32 v125, v105 :: v_dual_mov_b32 v128, v105
	v_dual_mov_b32 v127, v105 :: v_dual_mov_b32 v114, v105
	v_dual_mov_b32 v113, v105 :: v_dual_mov_b32 v116, v105
	v_dual_mov_b32 v115, v105 :: v_dual_mov_b32 v118, v105
	v_dual_mov_b32 v117, v105 :: v_dual_mov_b32 v120, v105
	v_dual_mov_b32 v119, v105 :: v_dual_mov_b32 v98, v105
	v_dual_mov_b32 v97, v105 :: v_dual_mov_b32 v100, v105
	v_dual_mov_b32 v99, v105 :: v_dual_mov_b32 v102, v105
	v_dual_mov_b32 v101, v105 :: v_dual_mov_b32 v104, v105
	v_dual_mov_b32 v103, v105 :: v_dual_mov_b32 v90, v105
	v_dual_mov_b32 v89, v105 :: v_dual_mov_b32 v92, v105
	v_dual_mov_b32 v91, v105 :: v_dual_mov_b32 v94, v105
	v_dual_mov_b32 v93, v105 :: v_dual_mov_b32 v96, v105
	v_dual_mov_b32 v95, v105 :: v_dual_mov_b32 v82, v105
	v_dual_mov_b32 v81, v105 :: v_dual_mov_b32 v84, v105
	v_dual_mov_b32 v83, v105 :: v_dual_mov_b32 v86, v105
	v_dual_mov_b32 v85, v105 :: v_dual_mov_b32 v88, v105
	v_dual_mov_b32 v87, v105 :: v_dual_mov_b32 v74, v105
	v_dual_mov_b32 v73, v105 :: v_dual_mov_b32 v76, v105
	v_dual_mov_b32 v75, v105 :: v_dual_mov_b32 v78, v105
	v_dual_mov_b32 v77, v105 :: v_dual_mov_b32 v80, v105
	v_dual_mov_b32 v79, v105 :: v_dual_mov_b32 v66, v105
	v_dual_mov_b32 v65, v105 :: v_dual_mov_b32 v68, v105
	v_dual_mov_b32 v67, v105 :: v_dual_mov_b32 v70, v105
	v_dual_mov_b32 v69, v105 :: v_dual_mov_b32 v72, v105
	v_dual_mov_b32 v71, v105 :: v_dual_mov_b32 v58, v105
	v_dual_mov_b32 v57, v105 :: v_dual_mov_b32 v60, v105
	v_dual_mov_b32 v59, v105 :: v_dual_mov_b32 v62, v105
	v_dual_mov_b32 v61, v105 :: v_dual_mov_b32 v64, v105
	v_dual_mov_b32 v63, v105 :: v_dual_mov_b32 v50, v105
	v_dual_mov_b32 v49, v105 :: v_dual_mov_b32 v52, v105
	v_dual_mov_b32 v51, v105 :: v_dual_mov_b32 v54, v105
	v_dual_mov_b32 v53, v105 :: v_dual_mov_b32 v56, v105
	v_dual_mov_b32 v55, v105 :: v_dual_mov_b32 v42, v105
	v_dual_mov_b32 v41, v105 :: v_dual_mov_b32 v44, v105
	v_dual_mov_b32 v43, v105 :: v_dual_mov_b32 v46, v105
	v_dual_mov_b32 v45, v105 :: v_dual_mov_b32 v48, v105
	v_dual_mov_b32 v47, v105 :: v_dual_mov_b32 v34, v105
	v_dual_mov_b32 v33, v105 :: v_dual_mov_b32 v36, v105
	v_dual_mov_b32 v35, v105 :: v_dual_mov_b32 v38, v105
	v_dual_mov_b32 v37, v105 :: v_dual_mov_b32 v40, v105
	v_dual_mov_b32 v39, v105 :: v_dual_mov_b32 v26, v105
	v_dual_mov_b32 v25, v105 :: v_dual_mov_b32 v28, v105
	v_dual_mov_b32 v27, v105 :: v_dual_mov_b32 v30, v105
	v_dual_mov_b32 v29, v105 :: v_dual_mov_b32 v32, v105
	v_dual_mov_b32 v31, v105 :: v_dual_mov_b32 v18, v105
	v_dual_mov_b32 v17, v105 :: v_dual_mov_b32 v20, v105
	v_dual_mov_b32 v19, v105 :: v_dual_mov_b32 v22, v105
	v_dual_mov_b32 v21, v105 :: v_dual_mov_b32 v24, v105
	v_dual_mov_b32 v23, v105 :: v_dual_mov_b32 v10, v105
	v_dual_mov_b32 v9, v105 :: v_dual_mov_b32 v12, v105
	v_dual_mov_b32 v11, v105 :: v_dual_mov_b32 v14, v105
	v_dual_mov_b32 v13, v105 :: v_dual_mov_b32 v16, v105
	v_dual_mov_b32 v15, v105 :: v_dual_mov_b32 v2, v105
	v_dual_mov_b32 v1, v105 :: v_dual_mov_b32 v4, v105
	v_dual_mov_b32 v3, v105 :: v_dual_mov_b32 v6, v105
	v_lshl_or_b32 v171, v171, 4, 0x4000
	v_lshlrev_b32_e32 v173, 4, v192
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v189, null, 0, v189, vcc_lo
	s_mov_b32 s2, 0
	s_mov_b32 s3, 0
	s_wait_loadcnt 0x7
	ds_store_b128 v8, v[129:132]
	s_wait_loadcnt 0x5
	ds_store_b128 v8, v[137:140] offset:16384
	ds_store_b128 v193, v[133:136]
	s_wait_loadcnt 0x4
	ds_store_b128 v193, v[141:144] offset:16384
	s_wait_loadcnt 0x3
	ds_store_b128 v194, v[145:148]
	s_wait_loadcnt 0x2
	ds_store_b128 v194, v[149:152] offset:16384
	s_wait_loadcnt 0x1
	ds_store_b128 v190, v[153:156]
	s_wait_loadcnt 0x0
	ds_store_b128 v190, v[157:160] offset:16384
	v_mov_b32_e32 v8, v105
	v_or_b32_e32 v191, v169, v166
	;;#ASMSTART
	s_barrier_signal -1
	;;#ASMEND
	;;#ASMSTART
	s_barrier_wait 0xffff
	;;#ASMEND
                                        ; implicit-def: $vgpr157_vgpr158_vgpr159_vgpr160
                                        ; implicit-def: $vgpr141_vgpr142_vgpr143_vgpr144
                                        ; implicit-def: $vgpr145_vgpr146_vgpr147_vgpr148
                                        ; implicit-def: $vgpr149_vgpr150_vgpr151_vgpr152
                                        ; implicit-def: $vgpr153_vgpr154_vgpr155_vgpr156
                                        ; implicit-def: $vgpr137_vgpr138_vgpr139_vgpr140
                                        ; implicit-def: $vgpr129_vgpr130_vgpr131_vgpr132
                                        ; implicit-def: $vgpr133_vgpr134_vgpr135_vgpr136
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v172, 4, v191
	s_branch .LBB0_2
.LBB0_1:                                ;   in Loop: Header=BB0_2 Depth=1
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB0_6
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_cmp_lt_u32 s3, 0x11e0
	s_cselect_b32 s6, -1, 0
	s_cmp_gt_u32 s3, 0x11df
	s_cbranch_scc1 .LBB0_4
; %bb.3:                                ;   in Loop: Header=BB0_2 Depth=1
	s_wait_loadcnt 0x6
	v_add_co_u32 v129, vcc_lo, v188, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v130, null, 0, v189, vcc_lo
	v_add_co_u32 v131, vcc_lo, v186, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v132, null, 0, v187, vcc_lo
	s_wait_loadcnt 0x5
	v_add_co_u32 v137, vcc_lo, v184, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v138, null, 0, v185, vcc_lo
	s_wait_loadcnt 0x1
	v_add_co_u32 v141, vcc_lo, v182, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v142, null, 0, v183, vcc_lo
	v_add_co_u32 v143, vcc_lo, v180, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v144, null, 0, v181, vcc_lo
	v_add_co_u32 v145, vcc_lo, v178, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v146, null, 0, v179, vcc_lo
	s_wait_loadcnt 0x0
	v_add_co_u32 v157, vcc_lo, v176, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v158, null, 0, v177, vcc_lo
	v_add_co_u32 v159, vcc_lo, v174, v165
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v160, null, 0, v175, vcc_lo
	global_load_b128 v[133:136], v[129:130], off
	global_load_b128 v[129:132], v[131:132], off
	global_load_b128 v[137:140], v[137:138], off
	global_load_b128 v[153:156], v[141:142], off
	global_load_b128 v[149:152], v[143:144], off
	global_load_b128 v[145:148], v[145:146], off
	global_load_b128 v[141:144], v[157:158], off
	global_load_b128 v[157:160], v[159:160], off
.LBB0_4:                                ;   in Loop: Header=BB0_2 Depth=1
	v_lshl_or_b32 v190, s2, 13, v173
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s6, -1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v234, v172, v190
	v_add_nc_u32_e32 v246, v171, v190
	v_add_nc_u32_e32 v250, v167, v190
	ds_load_b128 v[190:193], v234
	ds_load_b128 v[194:197], v246
	ds_load_b128 v[198:201], v246 offset:256
	ds_load_b128 v[202:205], v246 offset:512
	ds_load_b128 v[206:209], v250
	ds_load_b128 v[210:213], v234 offset:256
	ds_load_b128 v[214:217], v234 offset:512
	ds_load_b128 v[218:221], v234 offset:768
	ds_load_b128 v[222:225], v234 offset:4096
	ds_load_b128 v[226:229], v234 offset:4352
	ds_load_b128 v[230:233], v234 offset:4608
	ds_load_b128 v[234:237], v234 offset:4864
	ds_load_b128 v[238:241], v246 offset:4096
	ds_load_b128 v[242:245], v246 offset:4352
	ds_load_b128 v[246:249], v246 offset:4608
	ds_load_b128 v[250:253], v250 offset:4096
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x16_bf16 v[105:112], v[190:193], v[194:197], v[105:112]
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x16_bf16 v[121:128], v[190:193], v[198:201], v[121:128]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x16_bf16 v[113:120], v[190:193], v[202:205], v[113:120]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x16_bf16 v[97:104], v[190:193], v[206:209], v[97:104]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_bf16 v[89:96], v[210:213], v[194:197], v[89:96]
	v_wmma_f32_16x16x16_bf16 v[81:88], v[210:213], v[198:201], v[81:88]
	v_wmma_f32_16x16x16_bf16 v[73:80], v[210:213], v[202:205], v[73:80]
	v_wmma_f32_16x16x16_bf16 v[65:72], v[210:213], v[206:209], v[65:72]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_bf16 v[57:64], v[214:217], v[194:197], v[57:64]
	v_wmma_f32_16x16x16_bf16 v[49:56], v[214:217], v[198:201], v[49:56]
	v_wmma_f32_16x16x16_bf16 v[41:48], v[214:217], v[202:205], v[41:48]
	v_wmma_f32_16x16x16_bf16 v[33:40], v[214:217], v[206:209], v[33:40]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x16_bf16 v[25:32], v[218:221], v[194:197], v[25:32]
	v_wmma_f32_16x16x16_bf16 v[17:24], v[218:221], v[198:201], v[17:24]
	v_wmma_f32_16x16x16_bf16 v[9:16], v[218:221], v[202:205], v[9:16]
	v_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[206:209], v[1:8]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_bf16 v[105:112], v[222:225], v[238:241], v[105:112]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_bf16 v[121:128], v[222:225], v[242:245], v[121:128]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_bf16 v[113:120], v[222:225], v[246:249], v[113:120]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_bf16 v[97:104], v[222:225], v[250:253], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[89:96], v[226:229], v[238:241], v[89:96]
	v_wmma_f32_16x16x16_bf16 v[81:88], v[226:229], v[242:245], v[81:88]
	v_wmma_f32_16x16x16_bf16 v[73:80], v[226:229], v[246:249], v[73:80]
	v_wmma_f32_16x16x16_bf16 v[65:72], v[226:229], v[250:253], v[65:72]
	v_wmma_f32_16x16x16_bf16 v[57:64], v[230:233], v[238:241], v[57:64]
	v_wmma_f32_16x16x16_bf16 v[49:56], v[230:233], v[242:245], v[49:56]
	v_wmma_f32_16x16x16_bf16 v[41:48], v[230:233], v[246:249], v[41:48]
	v_wmma_f32_16x16x16_bf16 v[33:40], v[230:233], v[250:253], v[33:40]
	v_wmma_f32_16x16x16_bf16 v[25:32], v[234:237], v[238:241], v[25:32]
	v_wmma_f32_16x16x16_bf16 v[17:24], v[234:237], v[242:245], v[17:24]
	v_wmma_f32_16x16x16_bf16 v[9:16], v[234:237], v[246:249], v[9:16]
	v_wmma_f32_16x16x16_bf16 v[1:8], v[234:237], v[250:253], v[1:8]
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_1
; %bb.5:                                ;   in Loop: Header=BB0_2 Depth=1
	s_xor_b32 s2, s2, 1
	v_add_co_u32 v174, vcc_lo, v174, 64
	s_wait_alu 0xfffe
	v_lshl_or_b32 v190, s2, 9, v170
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v175, null, 0, v175, vcc_lo
	v_add_co_u32 v176, vcc_lo, v176, 64
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v177, null, 0, v177, vcc_lo
	v_add_co_u32 v178, vcc_lo, v178, 64
	v_add_lshl_u32 v191, v190, v161, 4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v179, null, 0, v179, vcc_lo
	v_add_co_u32 v180, vcc_lo, v180, 64
	v_add_lshl_u32 v192, v190, v162, 4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v181, null, 0, v181, vcc_lo
	v_add_co_u32 v182, vcc_lo, v182, 64
	v_add_lshl_u32 v193, v190, v163, 4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v183, null, 0, v183, vcc_lo
	v_add_co_u32 v184, vcc_lo, v184, 64
	v_add_lshl_u32 v190, v190, v164, 4
	s_wait_loadcnt 0x7
	ds_store_b128 v191, v[133:136]
	s_wait_loadcnt 0x3
	ds_store_b128 v191, v[149:152] offset:16384
	ds_store_b128 v192, v[129:132]
	s_wait_loadcnt 0x2
	ds_store_b128 v192, v[145:148] offset:16384
	ds_store_b128 v193, v[137:140]
	s_wait_loadcnt 0x1
	ds_store_b128 v193, v[141:144] offset:16384
	ds_store_b128 v190, v[153:156]
	s_wait_loadcnt 0x0
	ds_store_b128 v190, v[157:160] offset:16384
	;;#ASMSTART
	s_barrier_signal -1
	;;#ASMEND
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v185, null, 0, v185, vcc_lo
	v_add_co_u32 v186, vcc_lo, v186, 64
	;;#ASMSTART
	s_barrier_wait 0xffff
	;;#ASMEND
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v187, null, 0, v187, vcc_lo
	v_add_co_u32 v188, vcc_lo, v188, 64
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v189, null, 0, v189, vcc_lo
	s_add_co_i32 s3, s3, 32
	s_mov_b32 s6, 0
	s_branch .LBB0_1
.LBB0_6:
	s_wait_loadcnt 0x5
	v_dual_mov_b32 v139, 0 :: v_dual_and_b32 v0, 64, v0
	v_mov_b32_e32 v140, 0
	s_cmp_lg_u64 s[10:11], 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v0, s1, v0
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u64 s[10:11], 0
	v_or_b32_e32 v129, v0, v166
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
	v_or3_b32 v0, v169, s0, v168
	s_wait_loadcnt 0x0
	v_add_f32_e32 v143, v140, v105
	v_add_f32_e32 v144, v140, v106
	v_add_f32_e32 v145, v140, v107
	v_add_f32_e32 v147, v140, v109
	v_or_b32_e32 v131, 1, v0
	v_or_b32_e32 v133, 2, v0
	v_mad_co_i64_i32 v[105:106], null, 0x4800, v0, s[4:5]
	v_or_b32_e32 v135, 3, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_i64_i32 v[131:132], null, 0x4800, v131, s[4:5]
	v_or_b32_e32 v107, 4, v0
	v_mad_co_i64_i32 v[133:134], null, 0x4800, v133, s[4:5]
	v_or_b32_e32 v141, 5, v0
	v_mad_co_i64_i32 v[135:136], null, 0x4800, v135, s[4:5]
	v_add_co_u32 v105, vcc_lo, v105, v129
	v_mad_co_i64_i32 v[137:138], null, 0x4800, v107, s[4:5]
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v106, null, v106, v130, vcc_lo
	v_add_co_u32 v131, vcc_lo, v131, v129
	v_mad_co_i64_i32 v[141:142], null, 0x4800, v141, s[4:5]
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v132, null, v132, v130, vcc_lo
	v_add_co_u32 v133, vcc_lo, v133, v129
	v_or_b32_e32 v109, 6, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v134, null, v134, v130, vcc_lo
	v_add_co_u32 v107, vcc_lo, v135, v129
	v_or_b32_e32 v149, 7, v0
	v_add_f32_e32 v146, v140, v108
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v108, null, v136, v130, vcc_lo
	v_add_co_u32 v135, vcc_lo, v137, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v136, null, v138, v130, vcc_lo
	v_mad_co_i64_i32 v[137:138], null, 0x4800, v109, s[4:5]
	v_add_co_u32 v109, vcc_lo, v141, v129
	v_add_f32_e32 v148, v140, v110
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v110, null, v142, v130, vcc_lo
	v_mad_co_i64_i32 v[141:142], null, 0x4800, v149, s[4:5]
	v_add_f32_e32 v150, v140, v111
	s_wait_alu 0xfffe
	v_cndmask_b32_e64 v111, 0, 1, s1
	v_add_co_u32 v137, vcc_lo, v137, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v138, null, v138, v130, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ne_u32_e64 s0, 1, v111
	v_add_co_u32 v111, vcc_lo, v141, v129
	v_add_f32_e32 v140, v140, v112
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v112, null, v142, v130, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_clause 0x7
	global_store_b32 v[105:106], v143, off
	global_store_b32 v[131:132], v144, off
	global_store_b32 v[133:134], v145, off
	global_store_b32 v[107:108], v146, off
	global_store_b32 v[135:136], v147, off
	global_store_b32 v[109:110], v148, off
	global_store_b32 v[137:138], v150, off
	global_store_b32 v[111:112], v140, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_10
; %bb.9:
	v_add_co_u32 v139, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v140, null, s11, v130, vcc_lo
	global_load_b32 v139, v[139:140], off offset:64
.LBB0_10:
	s_wait_loadcnt 0x0
	v_dual_add_f32 v140, v139, v121 :: v_dual_mov_b32 v121, 0
	v_dual_add_f32 v141, v139, v122 :: v_dual_mov_b32 v122, 0
	v_add_f32_e32 v123, v139, v123
	v_add_f32_e32 v124, v139, v124
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v125, v139, v125
	v_add_f32_e32 v126, v139, v126
	v_add_f32_e32 v127, v139, v127
	v_add_f32_e32 v128, v139, v128
	s_clause 0x7
	global_store_b32 v[105:106], v140, off offset:64
	global_store_b32 v[131:132], v141, off offset:64
	global_store_b32 v[133:134], v123, off offset:64
	global_store_b32 v[107:108], v124, off offset:64
	global_store_b32 v[135:136], v125, off offset:64
	global_store_b32 v[109:110], v126, off offset:64
	global_store_b32 v[137:138], v127, off offset:64
	global_store_b32 v[111:112], v128, off offset:64
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_12
; %bb.11:
	v_add_co_u32 v122, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v123, null, s11, v130, vcc_lo
	global_load_b32 v122, v[122:123], off offset:128
.LBB0_12:
	s_wait_loadcnt 0x0
	v_add_f32_e32 v113, v122, v113
	v_add_f32_e32 v114, v122, v114
	v_add_f32_e32 v115, v122, v115
	v_add_f32_e32 v116, v122, v116
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v117, v122, v117
	v_add_f32_e32 v118, v122, v118
	v_add_f32_e32 v119, v122, v119
	v_add_f32_e32 v120, v122, v120
	s_clause 0x7
	global_store_b32 v[105:106], v113, off offset:128
	global_store_b32 v[131:132], v114, off offset:128
	global_store_b32 v[133:134], v115, off offset:128
	global_store_b32 v[107:108], v116, off offset:128
	global_store_b32 v[135:136], v117, off offset:128
	global_store_b32 v[109:110], v118, off offset:128
	global_store_b32 v[137:138], v119, off offset:128
	global_store_b32 v[111:112], v120, off offset:128
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_14
; %bb.13:
	v_add_co_u32 v113, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v114, null, s11, v130, vcc_lo
	global_load_b32 v121, v[113:114], off offset:192
.LBB0_14:
	s_wait_loadcnt 0x0
	v_dual_add_f32 v97, v121, v97 :: v_dual_mov_b32 v114, 0
	v_dual_add_f32 v98, v121, v98 :: v_dual_mov_b32 v113, 0
	v_add_f32_e32 v99, v121, v99
	v_add_f32_e32 v100, v121, v100
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v101, v121, v101
	v_add_f32_e32 v102, v121, v102
	v_add_f32_e32 v103, v121, v103
	v_add_f32_e32 v104, v121, v104
	s_clause 0x7
	global_store_b32 v[105:106], v97, off offset:192
	global_store_b32 v[131:132], v98, off offset:192
	global_store_b32 v[133:134], v99, off offset:192
	global_store_b32 v[107:108], v100, off offset:192
	global_store_b32 v[135:136], v101, off offset:192
	global_store_b32 v[109:110], v102, off offset:192
	global_store_b32 v[137:138], v103, off offset:192
	global_store_b32 v[111:112], v104, off offset:192
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_16
; %bb.15:
	v_add_co_u32 v97, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v98, null, s11, v130, vcc_lo
	global_load_b32 v114, v[97:98], off
.LBB0_16:
	v_or_b32_e32 v97, 16, v0
	v_or_b32_e32 v99, 17, v0
	v_or_b32_e32 v101, 18, v0
	v_or_b32_e32 v103, 19, v0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v109, v114, v89
	v_mad_co_i64_i32 v[97:98], null, 0x4800, v97, s[4:5]
	v_mad_co_i64_i32 v[99:100], null, 0x4800, v99, s[4:5]
	v_mad_co_i64_i32 v[101:102], null, 0x4800, v101, s[4:5]
	v_mad_co_i64_i32 v[103:104], null, 0x4800, v103, s[4:5]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v89, vcc_lo, v97, v129
	v_add_f32_e32 v111, v114, v91
	v_or_b32_e32 v91, 20, v0
	v_add_f32_e32 v110, v114, v90
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v90, null, v98, v130, vcc_lo
	v_add_co_u32 v97, vcc_lo, v99, v129
	v_or_b32_e32 v105, 21, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v98, null, v100, v130, vcc_lo
	v_add_co_u32 v99, vcc_lo, v101, v129
	v_add_f32_e32 v115, v114, v93
	v_or_b32_e32 v93, 22, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v100, null, v102, v130, vcc_lo
	v_mad_co_i64_i32 v[101:102], null, 0x4800, v91, s[4:5]
	v_add_co_u32 v91, vcc_lo, v103, v129
	v_or_b32_e32 v107, 23, v0
	v_add_f32_e32 v112, v114, v92
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v92, null, v104, v130, vcc_lo
	v_mad_co_i64_i32 v[103:104], null, 0x4800, v105, s[4:5]
	v_mad_co_i64_i32 v[105:106], null, 0x4800, v93, s[4:5]
	v_mad_co_i64_i32 v[107:108], null, 0x4800, v107, s[4:5]
	v_add_co_u32 v101, vcc_lo, v101, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v102, null, v102, v130, vcc_lo
	v_add_co_u32 v93, vcc_lo, v103, v129
	v_add_f32_e32 v116, v114, v94
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v94, null, v104, v130, vcc_lo
	v_add_co_u32 v103, vcc_lo, v105, v129
	v_add_f32_e32 v117, v114, v95
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v104, null, v106, v130, vcc_lo
	v_add_co_u32 v95, vcc_lo, v107, v129
	v_add_f32_e32 v105, v114, v96
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v96, null, v108, v130, vcc_lo
	s_and_b32 vcc_lo, exec_lo, s0
	s_clause 0x7
	global_store_b32 v[89:90], v109, off
	global_store_b32 v[97:98], v110, off
	global_store_b32 v[99:100], v111, off
	global_store_b32 v[91:92], v112, off
	global_store_b32 v[101:102], v115, off
	global_store_b32 v[93:94], v116, off
	global_store_b32 v[103:104], v117, off
	global_store_b32 v[95:96], v105, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_18
; %bb.17:
	v_add_co_u32 v105, vcc_lo, s10, v129
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v106, null, s11, v130, vcc_lo
	global_load_b32 v113, v[105:106], off offset:64
.LBB0_18:
	s_wait_loadcnt 0x0
	v_add_f32_e32 v105, v113, v81
	v_dual_add_f32 v106, v113, v82 :: v_dual_mov_b32 v81, 0
	v_dual_mov_b32 v82, 0 :: v_dual_add_f32 v83, v113, v83
	v_add_f32_e32 v84, v113, v84
	s_and_b32 vcc_lo, exec_lo, s0
	v_add_f32_e32 v85, v113, v85
	v_add_f32_e32 v86, v113, v86
	v_add_f32_e32 v87, v113, v87
	v_add_f32_e32 v88, v113, v88
	s_clause 0x7
	global_store_b32 v[89:90], v105, off offset:64
	global_store_b32 v[97:98], v106, off offset:64
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
		.amdhsa_group_segment_fixed_size 32768
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
		.amdhsa_next_free_vgpr 254
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 52
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
	.set gemm_mm0_bf16_asm.num_vgpr, 254
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
; codeLenInByte = 6632
; TotalNumSgprs: 14
; NumVgprs: 254
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 32768 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 254
; Occupancy: 4
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
	.type	__hip_cuid_993b09ce7bf574ca,@object ; @__hip_cuid_993b09ce7bf574ca
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_993b09ce7bf574ca
__hip_cuid_993b09ce7bf574ca:
	.byte	0                               ; 0x0
	.size	__hip_cuid_993b09ce7bf574ca, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.1 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_993b09ce7bf574ca
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
    .group_segment_fixed_size: 32768
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
    .vgpr_count:     254
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
