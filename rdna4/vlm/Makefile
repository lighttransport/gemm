# Makefile for HIP/ROCm VLM vision encoder (RDNA4)
#
# No hipcc required - kernels compiled at runtime via HIPRTC.
# Only needs gcc + ROCm runtime libraries (libamdhip64.so, libhiprtc.so) on the system.

CC = gcc
CXX = g++
HIPCC = hipcc
ROCM_LLVM ?= /opt/rocm/llvm/bin
CFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

# Sources
ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_vision_encoder.c
LLM_RUNNER_SRC = ../llm/hip_llm_runner.c

# Targets
VISION = test_hip_vision
VLM = test_hip_vlm
BENCH_GEMM = bench_vlm_gemm
BENCH_GEMM_BLASLT = bench_vlm_gemm_blaslt
BENCH_ROCBLAS = bench_vlm_rocblas
BENCH_HIPBLASLT = bench_vlm_hipblaslt
MM0_BLASLT_BRIDGE_O = mm0_hipblaslt_bridge.o
ROCM_PATH ?= /opt/rocm
GEN_DIR = generated
MM0_ASM_LDS_STRIDE ?= 144
MM0_ASM_HIP = $(GEN_DIR)/mm0_bf16_asm.hip
MM0_ASM_S = $(GEN_DIR)/mm0_bf16_asm.s
MM0_ASM_CO = $(GEN_DIR)/mm0_bf16_asm.co
MM0_ASM_GLOAD_HIP = $(GEN_DIR)/mm0_bf16_asm_gload.hip
MM0_ASM_GLOAD_S = $(GEN_DIR)/mm0_bf16_asm_gload.s
MM0_ASM_GLOAD_CO = $(GEN_DIR)/mm0_bf16_asm_gload.co
MM0_ASM_FROM_S_O = $(GEN_DIR)/mm0_bf16_asm_from_s.o
MM0_ASM_FROM_S_CO = $(GEN_DIR)/mm0_bf16_asm_from_s.co
MM0_ASM_HALFSPLIT_S = $(GEN_DIR)/mm0_bf16_asm_halfsplit.s
MM0_ASM_HALFSPLIT_O = $(GEN_DIR)/mm0_bf16_asm_halfsplit.o
MM0_ASM_HALFSPLIT_CO = $(GEN_DIR)/mm0_bf16_asm_halfsplit.co
MM0_ASM_STOREEARLY20_S = $(GEN_DIR)/mm0_bf16_asm_storeearly20.s
MM0_ASM_STOREEARLY20_O = $(GEN_DIR)/mm0_bf16_asm_storeearly20.o
MM0_ASM_STOREEARLY20_CO = $(GEN_DIR)/mm0_bf16_asm_storeearly20.co
MM0_ASM_BFIRST2_S = $(GEN_DIR)/mm0_bf16_asm_bfirst2.s
MM0_ASM_BFIRST2_O = $(GEN_DIR)/mm0_bf16_asm_bfirst2.o
MM0_ASM_BFIRST2_CO = $(GEN_DIR)/mm0_bf16_asm_bfirst2.co
MM0_ASM_PGR1MID_S = $(GEN_DIR)/mm0_bf16_asm_pgr1mid.s
MM0_ASM_PGR1MID_O = $(GEN_DIR)/mm0_bf16_asm_pgr1mid.o
MM0_ASM_PGR1MID_CO = $(GEN_DIR)/mm0_bf16_asm_pgr1mid.co
MM0_ASM_TOPNOWAIT_S = $(GEN_DIR)/mm0_bf16_asm_topnowait.s
MM0_ASM_TOPNOWAIT_O = $(GEN_DIR)/mm0_bf16_asm_topnowait.o
MM0_ASM_TOPNOWAIT_CO = $(GEN_DIR)/mm0_bf16_asm_topnowait.co
MM0_ASM_GLINLINE_S = $(GEN_DIR)/mm0_bf16_asm_glinline.s
MM0_ASM_GLINLINE_O = $(GEN_DIR)/mm0_bf16_asm_glinline.o
MM0_ASM_GLINLINE_CO = $(GEN_DIR)/mm0_bf16_asm_glinline.co
MM0_ASM_DSMOVE_S = $(GEN_DIR)/mm0_bf16_asm_dsmove.s
MM0_ASM_DSMOVE_O = $(GEN_DIR)/mm0_bf16_asm_dsmove.o
MM0_ASM_DSMOVE_CO = $(GEN_DIR)/mm0_bf16_asm_dsmove.co
MM0_ASM_STOREWAIT0_S = $(GEN_DIR)/mm0_bf16_asm_storewait0.s
MM0_ASM_STOREWAIT0_O = $(GEN_DIR)/mm0_bf16_asm_storewait0.o
MM0_ASM_STOREWAIT0_CO = $(GEN_DIR)/mm0_bf16_asm_storewait0.co
MM0_ASM_BARRIERSIG_EARLY_S = $(GEN_DIR)/mm0_bf16_asm_barriersig_early.s
MM0_ASM_BARRIERSIG_EARLY_O = $(GEN_DIR)/mm0_bf16_asm_barriersig_early.o
MM0_ASM_BARRIERSIG_EARLY_CO = $(GEN_DIR)/mm0_bf16_asm_barriersig_early.co
MM0_ASM_TNWBSE_S = $(GEN_DIR)/mm0_bf16_asm_tnw_bse.s
MM0_ASM_TNWBSE_O = $(GEN_DIR)/mm0_bf16_asm_tnw_bse.o
MM0_ASM_TNWBSE_CO = $(GEN_DIR)/mm0_bf16_asm_tnw_bse.co
MM0_ASM_TNWSWBSE_S = $(GEN_DIR)/mm0_bf16_asm_tnw_sw_bse.s
MM0_ASM_TNWSWBSE_O = $(GEN_DIR)/mm0_bf16_asm_tnw_sw_bse.o
MM0_ASM_TNWSWBSE_CO = $(GEN_DIR)/mm0_bf16_asm_tnw_sw_bse.co
MM0_ASM_PGR2LITE_S = $(GEN_DIR)/mm0_bf16_asm_pgr2lite.s
MM0_ASM_PGR2LITE_O = $(GEN_DIR)/mm0_bf16_asm_pgr2lite.o
MM0_ASM_PGR2LITE_CO = $(GEN_DIR)/mm0_bf16_asm_pgr2lite.co
MM0_ASM_PGR2LITE_BSE_S = $(GEN_DIR)/mm0_bf16_asm_pgr2lite_bse.s
MM0_ASM_PGR2LITE_BSE_O = $(GEN_DIR)/mm0_bf16_asm_pgr2lite_bse.o
MM0_ASM_PGR2LITE_BSE_CO = $(GEN_DIR)/mm0_bf16_asm_pgr2lite_bse.co
MM0_ASM_PGR2_BUFLOAD_BSE_S = $(GEN_DIR)/mm0_bf16_asm_pgr2_bufload_bse.s
MM0_ASM_PGR2_BUFLOAD_BSE_O = $(GEN_DIR)/mm0_bf16_asm_pgr2_bufload_bse.o
MM0_ASM_PGR2_BUFLOAD_BSE_CO = $(GEN_DIR)/mm0_bf16_asm_pgr2_bufload_bse.co
MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_S = $(GEN_DIR)/mm0_bf16_asm_pgr2_bufload_noguard_bse.s
MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_O = $(GEN_DIR)/mm0_bf16_asm_pgr2_bufload_noguard_bse.o
MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_CO = $(GEN_DIR)/mm0_bf16_asm_pgr2_bufload_noguard_bse.co
MM0_ASM_PGR2_DISTRIBUTE_BSE_S = $(GEN_DIR)/mm0_bf16_asm_pgr2_distribute_bse.s
MM0_ASM_PGR2_DISTRIBUTE_BSE_O = $(GEN_DIR)/mm0_bf16_asm_pgr2_distribute_bse.o
MM0_ASM_PGR2_DISTRIBUTE_BSE_CO = $(GEN_DIR)/mm0_bf16_asm_pgr2_distribute_bse.co
MM0_ASM_DTVA_S = $(GEN_DIR)/mm0_bf16_asm_dtva.s
MM0_ASM_DTVA_O = $(GEN_DIR)/mm0_bf16_asm_dtva.o
MM0_ASM_DTVA_CO = $(GEN_DIR)/mm0_bf16_asm_dtva.co
MM0_ASM_DTVA_C1_S = $(GEN_DIR)/mm0_bf16_asm_dtva_c1.s
MM0_ASM_DTVA_C1_O = $(GEN_DIR)/mm0_bf16_asm_dtva_c1.o
MM0_ASM_DTVA_C1_CO = $(GEN_DIR)/mm0_bf16_asm_dtva_c1.co
MM0_ASM_DTVA_C3A_S = $(GEN_DIR)/mm0_bf16_asm_dtva_c3a.s
MM0_ASM_DTVA_C3A_O = $(GEN_DIR)/mm0_bf16_asm_dtva_c3a.o
MM0_ASM_DTVA_C3A_CO = $(GEN_DIR)/mm0_bf16_asm_dtva_c3a.co
MM0_ASM_LDS160_S = $(GEN_DIR)/mm0_bf16_asm_lds160.s
MM0_ASM_LDS160_O = $(GEN_DIR)/mm0_bf16_asm_lds160.o
MM0_ASM_LDS160_CO = $(GEN_DIR)/mm0_bf16_asm_lds160.co
MM0_ASM_PGR2_S = $(GEN_DIR)/mm0_bf16_asm_pgr2.s
MM0_ASM_PGR2_O = $(GEN_DIR)/mm0_bf16_asm_pgr2.o
MM0_ASM_PGR2_CO = $(GEN_DIR)/mm0_bf16_asm_pgr2.co
MM0_ASM_DTVB_BASELINE_S = $(GEN_DIR)/mm0_bf16_asm_dtvb_baseline.s
MM0_ASM_DTVB_BASELINE_O = $(GEN_DIR)/mm0_bf16_asm_dtvb_baseline.o
MM0_ASM_DTVB_BASELINE_CO = $(GEN_DIR)/mm0_bf16_asm_dtvb_baseline.co
MM0_DIRECTA_HIP = $(GEN_DIR)/mm0_bf16_directa.hip
MM0_DIRECTA_CO = $(GEN_DIR)/mm0_bf16_directa.co
MM0_DIRECTA_PF_HIP = $(GEN_DIR)/mm0_bf16_directa_pf.hip
MM0_DIRECTA_PF_S = $(GEN_DIR)/mm0_bf16_directa_pf.s
MM0_DIRECTA_PF_CO = $(GEN_DIR)/mm0_bf16_directa_pf.co
MM0_DIRECTA_PF2_HIP = $(GEN_DIR)/mm0_bf16_directa_pf2.hip
MM0_DIRECTA_PF2_CO = $(GEN_DIR)/mm0_bf16_directa_pf2.co
MM0_DIRECTA_PF_NOSWAP_S = $(GEN_DIR)/mm0_bf16_directa_pf_noswap.s
MM0_DIRECTA_PF_NOSWAP_O = $(GEN_DIR)/mm0_bf16_directa_pf_noswap.o
MM0_DIRECTA_PF_NOSWAP_CO = $(GEN_DIR)/mm0_bf16_directa_pf_noswap.co

.PHONY: all clean run help gen-mm0-asm bench-mm0-asm bench-mm0-asm-gload bench-mm0-asm-s bench-mm0-asm-halfsplit bench-mm0-asm-storeearly20 bench-mm0-asm-bfirst2 bench-mm0-asm-pgr1mid bench-mm0-asm-dsmove bench-mm0-asm-storewait0 bench-mm0-asm-barriersig-early bench-mm0-asm-tnwbse bench-mm0-asm-tnwswbse bench-mm0-asm-pgr2lite bench-mm0-asm-pgr2lite-bse bench-mm0-asm-pgr2-bufload-bse bench-mm0-asm-pgr2-bufload-noguard-bse bench-mm0-asm-pgr2-distribute-bse build-mm0-asm-dtva bench-mm0-asm-dtva build-mm0-asm-dtva-c1 bench-mm0-asm-dtva-c1 build-mm0-asm-dtva-c3a bench-mm0-asm-dtva-c3a build-mm0-asm-lds160 bench-mm0-asm-lds160 build-mm0-asm-pgr2 bench-mm0-asm-pgr2 bench-mm0-directa bench-mm0-blaslt

all: $(VISION) $(VLM) $(BENCH_GEMM)

MM0_EXTRACTED_O = mm0_extracted_launcher.o

$(MM0_EXTRACTED_O): mm0_extracted_launcher.cpp
	$(HIPCC) -O3 -fPIE --offload-arch=gfx1201 -c -o $@ $<

$(VISION): test_hip_vision.c $(RUNNER_SRC) $(ROCEW_SRC) $(MM0_EXTRACTED_O) hip_vision_encoder.h ../rocew.h
	$(CC) $(CFLAGS) -o $@ test_hip_vision.c $(RUNNER_SRC) $(ROCEW_SRC) $(MM0_EXTRACTED_O) $(LDFLAGS) -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lstdc++

$(VLM): test_hip_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(ROCEW_SRC) $(MM0_EXTRACTED_O) hip_vision_encoder.h ../llm/hip_llm_runner.h ../rocew.h
	$(CC) $(CFLAGS) -o $@ test_hip_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(ROCEW_SRC) $(MM0_EXTRACTED_O) $(LDFLAGS) -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lstdc++

$(BENCH_GEMM): bench_vlm_gemm.c $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h $(MM0_EXTRACTED_O)
	$(CC) $(CFLAGS) -o $@ bench_vlm_gemm.c $(ROCEW_SRC) $(MM0_EXTRACTED_O) $(LDFLAGS) -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lstdc++

$(MM0_BLASLT_BRIDGE_O): mm0_hipblaslt_bridge.cpp
	$(HIPCC) -O3 -fPIE --offload-arch=gfx1201 -c -o $@ $<

$(BENCH_GEMM_BLASLT): bench_vlm_gemm.c $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h $(MM0_BLASLT_BRIDGE_O) $(MM0_EXTRACTED_O)
	$(CC) $(CFLAGS) -DVLM_HIPBLASLT_ENABLED -o $@ bench_vlm_gemm.c $(ROCEW_SRC) $(MM0_BLASLT_BRIDGE_O) $(MM0_EXTRACTED_O) $(LDFLAGS) -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lhipblaslt -lamdhip64 -lstdc++

bench-mm0-blaslt: $(BENCH_GEMM_BLASLT)
	./$(BENCH_GEMM_BLASLT) --dtype bf16 --shape mm0 --mode mm0blaslt --iters 200 --check

bench-mm0-extract: $(BENCH_GEMM)
	./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0extract --iters 200 --check

$(BENCH_ROCBLAS): bench_vlm_rocblas.cpp
	$(HIPCC) -O3 --offload-arch=gfx1201 -o $@ $< -lrocblas

$(BENCH_HIPBLASLT): bench_vlm_hipblaslt.cpp
	$(HIPCC) -O3 --offload-arch=gfx1201 -o $@ $< -lhipblaslt

gen-mm0-asm $(MM0_ASM_HIP): gen_mm0_bf16_asm.py
	python3 gen_mm0_bf16_asm.py --out-dir $(GEN_DIR) --lds-stride $(MM0_ASM_LDS_STRIDE)

$(MM0_ASM_GLOAD_HIP): gen_mm0_bf16_asm.py
	python3 gen_mm0_bf16_asm.py --out-dir $(GEN_DIR) --basename mm0_bf16_asm_gload --global-layout coalesced

$(MM0_ASM_S): $(MM0_ASM_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --cuda-device-only -S -o $@ $<

$(MM0_ASM_CO): $(MM0_ASM_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --genco -o $@ $<

$(MM0_ASM_GLOAD_S): $(MM0_ASM_GLOAD_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --cuda-device-only -S -o $@ $<

$(MM0_ASM_GLOAD_CO): $(MM0_ASM_GLOAD_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --genco -o $@ $<

$(MM0_ASM_FROM_S_O): $(MM0_ASM_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_FROM_S_CO): $(MM0_ASM_FROM_S_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_HALFSPLIT_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant halfsplit $< $@

$(MM0_ASM_HALFSPLIT_O): $(MM0_ASM_HALFSPLIT_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_HALFSPLIT_CO): $(MM0_ASM_HALFSPLIT_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_STOREEARLY20_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant storeearly20 $< $@

$(MM0_ASM_STOREEARLY20_O): $(MM0_ASM_STOREEARLY20_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_STOREEARLY20_CO): $(MM0_ASM_STOREEARLY20_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_BFIRST2_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant bfirst2 $< $@

$(MM0_ASM_BFIRST2_O): $(MM0_ASM_BFIRST2_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_BFIRST2_CO): $(MM0_ASM_BFIRST2_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_PGR1MID_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant pgr1mid $< $@

$(MM0_ASM_PGR1MID_O): $(MM0_ASM_PGR1MID_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_PGR1MID_CO): $(MM0_ASM_PGR1MID_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_TOPNOWAIT_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant topnowait $< $@

$(MM0_ASM_TOPNOWAIT_O): $(MM0_ASM_TOPNOWAIT_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_TOPNOWAIT_CO): $(MM0_ASM_TOPNOWAIT_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

# NOTE: glinline variant is broken — see patch_mm0_asm_schedule.patch_glinline:
# moving global_loads out of bb.3 (where they're scc-guarded) into .LBB0_4
# causes the last iteration to load with garbage v137/v138 base → kernel hang.
# At the .s level the bb.3 loads already run in parallel with all 32 WMMAs in
# .LBB0_4, so further interleaving cannot extend overlap under PGR1. Need PGR2
# (loads issued 2 K-stages ahead) for additional gain. Targets kept commented.
#$(MM0_ASM_GLINLINE_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
#	python3 patch_mm0_asm_schedule.py --variant glinline $< $@
#
#$(MM0_ASM_GLINLINE_O): $(MM0_ASM_GLINLINE_S)
#	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<
#
#$(MM0_ASM_GLINLINE_CO): $(MM0_ASM_GLINLINE_O)
#	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_DSMOVE_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant dsmove $< $@

$(MM0_ASM_DSMOVE_O): $(MM0_ASM_DSMOVE_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_DSMOVE_CO): $(MM0_ASM_DSMOVE_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_STOREWAIT0_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant storewait0 $< $@

$(MM0_ASM_STOREWAIT0_O): $(MM0_ASM_STOREWAIT0_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_STOREWAIT0_CO): $(MM0_ASM_STOREWAIT0_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_BARRIERSIG_EARLY_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant barriersig-early $< $@

$(MM0_ASM_BARRIERSIG_EARLY_O): $(MM0_ASM_BARRIERSIG_EARLY_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_BARRIERSIG_EARLY_CO): $(MM0_ASM_BARRIERSIG_EARLY_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_TNWBSE_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant topnowait-barriersig-early $< $@

$(MM0_ASM_TNWBSE_O): $(MM0_ASM_TNWBSE_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_TNWBSE_CO): $(MM0_ASM_TNWBSE_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_TNWSWBSE_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant topnowait-storewait0-barriersig-early $< $@

$(MM0_ASM_TNWSWBSE_O): $(MM0_ASM_TNWSWBSE_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_TNWSWBSE_CO): $(MM0_ASM_TNWSWBSE_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_PGR2LITE_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant pgr2-lite $< $@

$(MM0_ASM_PGR2LITE_O): $(MM0_ASM_PGR2LITE_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_PGR2LITE_CO): $(MM0_ASM_PGR2LITE_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_PGR2LITE_BSE_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant pgr2-lite-bse $< $@

$(MM0_ASM_PGR2LITE_BSE_O): $(MM0_ASM_PGR2LITE_BSE_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_PGR2LITE_BSE_CO): $(MM0_ASM_PGR2LITE_BSE_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_PGR2_BUFLOAD_BSE_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant pgr2-bufload-bse $< $@

$(MM0_ASM_PGR2_BUFLOAD_BSE_O): $(MM0_ASM_PGR2_BUFLOAD_BSE_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_PGR2_BUFLOAD_BSE_CO): $(MM0_ASM_PGR2_BUFLOAD_BSE_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant pgr2-bufload-noguard-bse $< $@

$(MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_O): $(MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_CO): $(MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_PGR2_DISTRIBUTE_BSE_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant pgr2-distribute-bse $< $@

$(MM0_ASM_PGR2_DISTRIBUTE_BSE_O): $(MM0_ASM_PGR2_DISTRIBUTE_BSE_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_PGR2_DISTRIBUTE_BSE_CO): $(MM0_ASM_PGR2_DISTRIBUTE_BSE_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_DTVA_S): $(MM0_ASM_BARRIERSIG_EARLY_S) patch_dtva.py
	python3 patch_dtva.py --in-file $< --out-file $@ --step bc

$(MM0_ASM_DTVA_O): $(MM0_ASM_DTVA_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_DTVA_CO): $(MM0_ASM_DTVA_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_DTVA_C1_S): $(MM0_ASM_BARRIERSIG_EARLY_S) patch_dtva.py
	python3 patch_dtva.py --in-file $< --out-file $@ --step c1

$(MM0_ASM_DTVA_C1_O): $(MM0_ASM_DTVA_C1_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_DTVA_C1_CO): $(MM0_ASM_DTVA_C1_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_DTVA_C3A_S): $(MM0_ASM_BARRIERSIG_EARLY_S) patch_dtva.py
	python3 patch_dtva.py --in-file $< --out-file $@ --step c3a

$(MM0_ASM_DTVA_C3A_O): $(MM0_ASM_DTVA_C3A_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_DTVA_C3A_CO): $(MM0_ASM_DTVA_C3A_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_DIRECTA_HIP): gen_mm0_bf16_directa.py
	python3 gen_mm0_bf16_directa.py --out-dir $(GEN_DIR)

$(MM0_DIRECTA_CO): $(MM0_DIRECTA_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --genco -o $@ $<

$(MM0_DIRECTA_PF_HIP): gen_mm0_bf16_directa_pf.py
	python3 gen_mm0_bf16_directa_pf.py --out-dir $(GEN_DIR)

$(MM0_DIRECTA_PF_CO): $(MM0_DIRECTA_PF_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --genco -o $@ $<

$(MM0_DIRECTA_PF2_HIP): gen_mm0_bf16_directa_pf2.py
	python3 gen_mm0_bf16_directa_pf2.py --out-dir $(GEN_DIR)

$(MM0_DIRECTA_PF2_CO): $(MM0_DIRECTA_PF2_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --genco -o $@ $<

$(MM0_DIRECTA_PF_S): $(MM0_DIRECTA_PF_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --cuda-device-only -S -o $@ $<

$(MM0_DIRECTA_PF_NOSWAP_S): $(MM0_DIRECTA_PF_S) patch_directa_pf_noswap.py
	python3 patch_directa_pf_noswap.py --in-file $< --out-file $@

$(MM0_DIRECTA_PF_NOSWAP_O): $(MM0_DIRECTA_PF_NOSWAP_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_DIRECTA_PF_NOSWAP_CO): $(MM0_DIRECTA_PF_NOSWAP_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

bench-mm0-asm: $(BENCH_GEMM) $(MM0_ASM_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-gload: $(BENCH_GEMM) $(MM0_ASM_GLOAD_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_GLOAD_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-s: $(BENCH_GEMM) $(MM0_ASM_FROM_S_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_FROM_S_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-halfsplit: $(BENCH_GEMM) $(MM0_ASM_HALFSPLIT_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_HALFSPLIT_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-storeearly20: $(BENCH_GEMM) $(MM0_ASM_STOREEARLY20_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_STOREEARLY20_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-bfirst2: $(BENCH_GEMM) $(MM0_ASM_BFIRST2_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_BFIRST2_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-pgr1mid: $(BENCH_GEMM) $(MM0_ASM_PGR1MID_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_PGR1MID_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-dsmove: $(BENCH_GEMM) $(MM0_ASM_DSMOVE_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_DSMOVE_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-storewait0: $(BENCH_GEMM) $(MM0_ASM_STOREWAIT0_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_STOREWAIT0_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-barriersig-early: $(BENCH_GEMM) $(MM0_ASM_BARRIERSIG_EARLY_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_BARRIERSIG_EARLY_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-tnwbse: $(BENCH_GEMM) $(MM0_ASM_TNWBSE_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_TNWBSE_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-tnwswbse: $(BENCH_GEMM) $(MM0_ASM_TNWSWBSE_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_TNWSWBSE_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-pgr2lite: $(BENCH_GEMM) $(MM0_ASM_PGR2LITE_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_PGR2LITE_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-pgr2lite-bse: $(BENCH_GEMM) $(MM0_ASM_PGR2LITE_BSE_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_PGR2LITE_BSE_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-pgr2-bufload-bse: $(BENCH_GEMM) $(MM0_ASM_PGR2_BUFLOAD_BSE_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_PGR2_BUFLOAD_BSE_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 500 --check

bench-mm0-asm-pgr2-bufload-noguard-bse: $(BENCH_GEMM) $(MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_PGR2_BUFLOAD_NOGUARD_BSE_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 500 --check

bench-mm0-asm-pgr2-distribute-bse: $(BENCH_GEMM) $(MM0_ASM_PGR2_DISTRIBUTE_BSE_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_PGR2_DISTRIBUTE_BSE_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 500 --check

# Step A is intentionally output-incorrect (drops ds_load/store, leaves WMMA
# A regs uninitialized). Build it to validate scaffolding only — no --check.
build-mm0-asm-dtva: $(MM0_ASM_DTVA_CO)
bench-mm0-asm-dtva: $(BENCH_GEMM) $(MM0_ASM_DTVA_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_DTVA_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

build-mm0-asm-dtva-c1: $(MM0_ASM_DTVA_C1_CO)
bench-mm0-asm-dtva-c1: $(BENCH_GEMM) $(MM0_ASM_DTVA_C1_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_DTVA_C1_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

build-mm0-asm-dtva-c3a: $(MM0_ASM_DTVA_C3A_CO)
bench-mm0-asm-dtva-c3a: $(BENCH_GEMM) $(MM0_ASM_DTVA_C3A_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_DTVA_C3A_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

$(MM0_ASM_LDS160_S): $(MM0_ASM_BARRIERSIG_EARLY_S) patch_lds_stride.py
	python3 patch_lds_stride.py --in-file $< --out-file $@

$(MM0_ASM_LDS160_O): $(MM0_ASM_LDS160_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_LDS160_CO): $(MM0_ASM_LDS160_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

build-mm0-asm-lds160: $(MM0_ASM_LDS160_CO)
bench-mm0-asm-lds160: $(BENCH_GEMM) $(MM0_ASM_LDS160_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_LDS160_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 500 --check

# Track B home file: gen_mm0_bf16_asm_pgr2.py emits mm0_bf16_asm_pgr2.s
# from the barriersig_early baseline. Slice 1 = buffer_load swap
# (perf-neutral baseline). Future slices add LDS stride 160, per-load
# soffset SGPRs, W-tile partition, VGPR reduction — see file docstring.
$(MM0_ASM_PGR2_S): $(MM0_ASM_BARRIERSIG_EARLY_S) gen_mm0_bf16_asm_pgr2.py patch_mm0_asm_schedule.py
	python3 gen_mm0_bf16_asm_pgr2.py --in-file $< --out-file $@

$(MM0_ASM_PGR2_O): $(MM0_ASM_PGR2_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_PGR2_CO): $(MM0_ASM_PGR2_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

build-mm0-asm-pgr2: $(MM0_ASM_PGR2_CO)
bench-mm0-asm-pgr2: $(BENCH_GEMM) $(MM0_ASM_PGR2_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_PGR2_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 500 --check

# DTVB generator landing pad. `baseline` mode is a verbatim pass-through
# of mm0_bf16_asm_barriersig_early.s — validates the generator + dtvb_emit
# pipeline. Should bench at 142 TFLOP/s with cosine=1.0.
$(MM0_ASM_DTVB_BASELINE_S): $(MM0_ASM_BARRIERSIG_EARLY_S) gen_mm0_dtvb.py dtvb_emit/regalloc.py dtvb_emit/isa.py dtvb_emit/scheduler.py
	python3 gen_mm0_dtvb.py --mode baseline --baseline-s $< --out $@

$(MM0_ASM_DTVB_BASELINE_O): $(MM0_ASM_DTVB_BASELINE_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_DTVB_BASELINE_CO): $(MM0_ASM_DTVB_BASELINE_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

build-mm0-asm-dtvb-baseline: $(MM0_ASM_DTVB_BASELINE_CO)
bench-mm0-asm-dtvb-baseline: $(BENCH_GEMM) $(MM0_ASM_DTVB_BASELINE_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_DTVB_BASELINE_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 500 --check

bench-mm0-directa: $(BENCH_GEMM) $(MM0_DIRECTA_CO)
	VLM_GEMM_ASM_CO=$(MM0_DIRECTA_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-directa-pf: $(BENCH_GEMM) $(MM0_DIRECTA_PF_CO)
	VLM_GEMM_ASM_CO=$(MM0_DIRECTA_PF_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-directa-pf2: $(BENCH_GEMM) $(MM0_DIRECTA_PF2_CO)
	VLM_GEMM_ASM_CO=$(MM0_DIRECTA_PF2_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-directa-pf-noswap: $(BENCH_GEMM) $(MM0_DIRECTA_PF_NOSWAP_CO)
	VLM_GEMM_ASM_CO=$(MM0_DIRECTA_PF_NOSWAP_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 500 --check

# dump_clip_embd requires a llama.cpp build with libmtmd/libllama.
# Usage: make dump_clip_embd LLAMA_DIR=/home/syoyo/work/llama.cpp
LLAMA_DIR ?= /home/syoyo/work/llama.cpp
dump_clip_embd: dump_clip_embd.cpp
	$(CXX) -O2 -std=c++17 \
		-I../../common \
		-I$(LLAMA_DIR)/vendor/stb \
		-I$(LLAMA_DIR)/tools/mtmd \
		-I$(LLAMA_DIR)/ggml/include \
		-I$(LLAMA_DIR)/include \
		-L$(LLAMA_DIR)/build/bin \
		-Wl,-rpath,$(LLAMA_DIR)/build/bin \
		-o $@ $< -lmtmd -lllama -lggml -lggml-base -lm

run: $(VISION)
	@echo "Usage: ./$(VISION) <mmproj.gguf> [--image path] [--image-size N] [--ref embd.bin] [--warmup N] [--iters N] [--f32|--f16|--bf16]"

clean:
	rm -f $(VISION) $(VLM) $(BENCH_GEMM) $(BENCH_ROCBLAS) $(BENCH_HIPBLASLT) dump_clip_embd $(MM0_ASM_S) $(MM0_ASM_CO) $(MM0_ASM_GLOAD_S) $(MM0_ASM_GLOAD_CO) $(MM0_ASM_FROM_S_O) $(MM0_ASM_FROM_S_CO) $(MM0_ASM_HALFSPLIT_S) $(MM0_ASM_HALFSPLIT_O) $(MM0_ASM_HALFSPLIT_CO) $(MM0_ASM_STOREEARLY20_S) $(MM0_ASM_STOREEARLY20_O) $(MM0_ASM_STOREEARLY20_CO) $(MM0_ASM_BFIRST2_S) $(MM0_ASM_BFIRST2_O) $(MM0_ASM_BFIRST2_CO) $(MM0_DIRECTA_CO) $(MM0_DIRECTA_PF_HIP) $(MM0_DIRECTA_PF_CO) $(MM0_DIRECTA_PF2_HIP) $(MM0_DIRECTA_PF2_CO) $(MM0_ASM_PGR2_S) $(MM0_ASM_PGR2_O) $(MM0_ASM_PGR2_CO)

help:
	@echo "HIP VLM Vision Encoder (HIPRTC-based, no hipcc needed, RDNA4)"
	@echo "============================================================="
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_hip_vision and test_hip_vlm"
	@echo "  make bench-mm0-asm - Generate/build external mm0 BF16 kernel and benchmark it"
	@echo "    override LDS padding with MM0_ASM_LDS_STRIDE=N, e.g. N=128 for the old baseline"
	@echo "  make bench-mm0-asm-gload - Generate/build coalesced-global-load mm0 BF16 kernel and benchmark it"
	@echo "  make bench-mm0-asm-s - Assemble/link generated AMDGPU .s and benchmark it"
	@echo "  make bench-mm0-asm-halfsplit - Assemble/link patched half-split schedule and benchmark it"
	@echo "  make bench-mm0-asm-storeearly20 - Assemble/link patched early-store schedule and benchmark it"
	@echo "  make bench-mm0-asm-bfirst2 - Assemble/link second-half B-first LDS load schedule and benchmark it"
	@echo "  make bench-mm0-directa - Generate direct-A/LDS-B external kernel and benchmark it"
	@echo "  make dump_clip_embd LLAMA_DIR=/home/syoyo/work/llama.cpp"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_hip_vision <mmproj.gguf> [--image path] [--image-size N] [--ref embd.bin] [--f32|--f16|--bf16]"
	@echo "  ./test_hip_vlm <model.gguf> <mmproj.gguf> <image.jpg> [-n max_tokens] [--vision-f32|--vision-f16|--vision-bf16]"
	@echo "  ./bench_vlm_gemm [--dtype f16|bf16] [--shape all|qkv|attn_out|ffn_up|ffn_down|mm0|mm2] [--mode cta|cta64|direct|mm0spec|mm0vec|mm0pipe|mm0asm|mm0pipegl|mm0pipemid|mm0pipe8|mm0pipek64|mm0pipe4n64] [--iters N] [--check] [--no-bias]"
	@echo "    mm0asm loads VLM_GEMM_ASM_CO and expects kernel gemm_mm0_bf16_asm"
	@echo "  make bench_vlm_rocblas && ./bench_vlm_rocblas [--dtype f16|bf16] [--shape all] [--iters N]"
	@echo "  make bench_vlm_hipblaslt && ./bench_vlm_hipblaslt [--dtype f16|bf16] [--shape all] [--epilogue none|bias|gelu_bias] [--iters N] [--algos N] [--workspace-mb N] [--print-algos|--list-only] [--algo-index N]"
	@echo "  ./dump_hipblaslt_kernel_asm.sh <code-object-or-kernel-name> [/tmp/out-prefix]"
	@echo ""
	@echo "Requirements:"
	@echo "  - libamdhip64.so (HIP runtime)"
	@echo "  - libhiprtc.so (HIPRTC runtime compiler)"
	@echo "  - AMD GPU with RDNA4 (gfx1200/gfx1201)"
