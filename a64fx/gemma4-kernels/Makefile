# Standalone Gemma-4 31B A64FX kernel microbenchmarks (no model / no transformer.h).
# Builds:
#   native  : -O3, fast iteration + native run on the A64FX node (rdcyc timing)
#   fj (fapp): same codegen + fapp_start/stop markers (-DUSE_FAPP), for fapp & qlair.
#             The kernels are hand-asm, so the C-frontend choice only affects the
#             driver loop (negligible). _Float16 needs -Nclang (the Fujitsu traditional
#             -Nnoclang frontend rejects it), so all builds use -Nclang; the Fujitsu
#             fapp runtime links fine under -Nclang.
#
# Targets:
#   make            -> all native profilers
#   make fj         -> all fapp/qlair profilers (*_profile_fj, with fapp markers)
#   make run        -> build native + run each (sanity + rough efficiency)
#   make clean
# Profile:  bash profile_fapp.sh q8v2_profile_fj "16 4000 0"   (17 PA groups -> CSV)

FCC      ?= fcc
NATIVE    = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
FJCFLAGS  = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -DUSE_FAPP

KERN_Q8   = kernel_q8v2_3x4.S
KERN_FP16 = micro_kernel_fp16_12x2_swp.S micro_kernel_fp16_12x2_swp_accum.S

NATIVE_BINS = q8v2_profile fp16_profile attn_profile
FJ_BINS     = q8v2_profile_fj fp16_profile_fj attn_profile_fj

all: $(NATIVE_BINS)
fj:  $(FJ_BINS)

# ---- native ----
q8v2_profile: q8v2_profile.c $(KERN_Q8)
	$(FCC) $(NATIVE) -fopenmp $^ -lm -o $@
fp16_profile: fp16_profile.c $(KERN_FP16)
	$(FCC) $(NATIVE) -fopenmp $^ -lm -o $@
attn_profile: attn_profile.c $(KERN_FP16)
	$(FCC) $(NATIVE) -fopenmp $^ -lm -o $@

# ---- fujitsu (fapp/qlair) ----
q8v2_profile_fj: q8v2_profile.c $(KERN_Q8)
	$(FCC) $(FJCFLAGS) $^ -lm -o $@
fp16_profile_fj: fp16_profile.c $(KERN_FP16)
	$(FCC) $(FJCFLAGS) $^ -lm -o $@
attn_profile_fj: attn_profile.c $(KERN_FP16)
	$(FCC) $(FJCFLAGS) $^ -lm -o $@

run: $(NATIVE_BINS)
	@echo "== q8v2 int8 weight GEMM microkernel (per-block / per-row) =="
	@OMP_NUM_THREADS=1 ./q8v2_profile 16 4000 0
	@OMP_NUM_THREADS=1 ./q8v2_profile 16 4000 1
	@echo "== fp16 12x2 GEMM microkernel (WS3 attention QK^T/PV) =="
	@OMP_NUM_THREADS=1 ./fp16_profile 512 4000
	@echo "== fused fp16 attention (SWA hd=256 / full hd=512) =="
	@OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./attn_profile 256 384 512
	@OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./attn_profile 512 384 0

clean:
	rm -f $(NATIVE_BINS) $(FJ_BINS) *.o

.PHONY: all fj run clean
