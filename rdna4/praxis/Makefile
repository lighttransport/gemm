# rdna4/praxis Makefile.
# Run from anywhere; tests chdir to the repo root. Weights default to $(PRAXIS_ROOT) (default ~/praxis), scratch to $(PRAXIS_OUT) (default <repo>/tmp).
# No hipcc needed: all kernels are compiled at run time through HIPRTC (rocew).
REPO   := $(abspath ../..)
PRAXIS_ROOT ?= $(HOME)/praxis
PRAXIS_OUT  ?= $(REPO)/tmp
CC     ?= gcc
ROCEW   = ../rocew.c
COMMON_HDRS = ../rocew.h ../hip_runner_common.h ../../common/safetensors.h

# ======================= BEGIN Wan2.2 VAE (hip_wan22_vae) =======================
#   make -C rdna4/praxis test_hip_wan22_vae
#   make -C rdna4/praxis vae-test      # 5x64x64 oracle check, 18 checks (rel-L2 <= 0.02, cos >= 0.999)
#   make -C rdna4/praxis vae-bench     # 9x128x128 and 21x256x256 timing + checks
#   make -C rdna4/praxis vae-test-tiled / vae-bench-big
VAE_WEIGHTS ?= $(PRAXIS_ROOT)/wan2.2-vae
VAE_REF     ?= $(REPO)/tmp/praxis_ref
VAE_OUT     ?= $(REPO)/tmp/praxis_out_hip
PY          ?= python3
VAE_CMP      = $(PY) $(REPO)/ref/praxis/compare.py --tol 0.02 --cos 0.999
VAE_CFLAGS   = -O2 -fopenmp -Wall -Wno-unused-function -Wno-unknown-pragmas -I../../common -I.. -D_GNU_SOURCE -march=native

.PHONY: vae vae-test vae-bench vae-bench-big vae-test-tiled vae-clean
vae: test_hip_wan22_vae

test_hip_wan22_vae: test_hip_wan22_vae.c hip_wan22_vae.h $(ROCEW) $(COMMON_HDRS)
	$(CC) $(VAE_CFLAGS) -o $@ test_hip_wan22_vae.c $(ROCEW) -lm -ldl -lpthread -fopenmp

vae-test: test_hip_wan22_vae
	mkdir -p $(VAE_OUT)
	cd $(REPO) && rdna4/praxis/test_hip_wan22_vae --weights $(VAE_WEIGHTS) --ref $(VAE_REF)/vae --out $(VAE_OUT)/vae --taps --self-decode
	$(VAE_CMP) --ref-dir $(VAE_REF)/vae --ours-dir $(VAE_OUT)/vae

vae-bench: test_hip_wan22_vae
	mkdir -p $(VAE_OUT)
	cd $(REPO) && rdna4/praxis/test_hip_wan22_vae --weights $(VAE_WEIGHTS) --ref $(VAE_REF)/vae_9x128 --out $(VAE_OUT)/vae_9x128 --taps --repeat 2
	$(VAE_CMP) --ref-dir $(VAE_REF)/vae_9x128 --ours-dir $(VAE_OUT)/vae_9x128
	cd $(REPO) && rdna4/praxis/test_hip_wan22_vae --weights $(VAE_WEIGHTS) --ref $(VAE_REF)/vae_21x256 --out $(VAE_OUT)/vae_21x256 --repeat 2
	$(VAE_CMP) --ref-dir $(VAE_REF)/vae_21x256 --ours-dir $(VAE_OUT)/vae_21x256

vae-test-tiled: test_hip_wan22_vae
	cd $(REPO) && rdna4/praxis/test_hip_wan22_vae --weights $(VAE_WEIGHTS) --synth 5,256,416 --tile 256,96 --compare-untiled
	cd $(REPO) && rdna4/praxis/test_hip_wan22_vae --weights $(VAE_WEIGHTS) --synth 5,250,410 --tile 256,96 --compare-untiled --u8

vae-bench-big: test_hip_wan22_vae
	cd $(REPO) && rdna4/praxis/test_hip_wan22_vae --weights $(VAE_WEIGHTS) --synth 21,448,832
	cd $(REPO) && rdna4/praxis/test_hip_wan22_vae --weights $(VAE_WEIGHTS) --synth 9,720,1280

vae-clean:
	rm -f test_hip_wan22_vae
# ======================== END Wan2.2 VAE (hip_wan22_vae) ========================

# (the praxis policy runner targets are appended below by the other port)

# ======================= BEGIN praxis-runner (hip_praxis_runner.h) =======================
#   make -C rdna4/praxis test_hip_praxis hip_praxis_server
#   make -C rdna4/praxis test_hip_praxis_all     # tiny/mid/mid128 parity: exact fp32 path (0.02) + WMMA bf16 fast path (0.06) + streaming/sampler/ctx-cache selftests
PX_TMP    ?= $(PRAXIS_OUT)
PX_REF    ?= $(PX_TMP)/praxis_ref
PX_OUT    ?= $(PX_TMP)/praxis_out_hip
PX_MODELS ?= $(PRAXIS_ROOT)
PX_CFLAGS  = -O2 -mavx2 -mfma -mf16c -fopenmp -Wall -Wno-unused-function -Wno-unknown-pragmas -I.. -I../../common -D_GNU_SOURCE
PX_LDFLAGS = -lm -ldl -lpthread -fopenmp
PX_CMP     = python3 $(REPO)/ref/praxis/compare.py --tol 0.02 --cos 0.999 --skip-missing
PX_FASTCMP = python3 $(REPO)/ref/praxis/compare.py --tol 0.06 --cos 0.998 --topk-agree 0.85 --skip-missing
PX_HDRS    = hip_praxis_runner.h hip_praxis_kernels.h ../cuda_driver_compat.h ../hip_runner_common.h ../rocew.h ../../common/safetensors.h

.PHONY: test_hip_praxis_all clean_hip_praxis
test_hip_praxis: test_hip_praxis.c $(PX_HDRS) ../rocew.c
	$(CC) $(PX_CFLAGS) -o $@ $< ../rocew.c $(PX_LDFLAGS)

hip_praxis_server: hip_praxis_server.c $(PX_HDRS) ../rocew.c
	$(CC) $(PX_CFLAGS) -o $@ $< ../rocew.c $(PX_LDFLAGS)

test_hip_praxis_all: test_hip_praxis
	cd $(REPO) && PRAXIS_EXACT=1 rdna4/praxis/test_hip_praxis --model $(PX_MODELS)/tiny --ref $(PX_REF)/backbone --out $(PX_OUT)/backbone_exact --selftest-stream --tol 0.02
	$(PX_CMP) --ref-dir $(PX_REF)/backbone --ours-dir $(PX_OUT)/backbone_exact
	cd $(REPO) && rdna4/praxis/test_hip_praxis --model $(PX_MODELS)/tiny --ref $(PX_REF)/backbone --out $(PX_OUT)/backbone_fast --selftest-stream --tol 0.06
	$(PX_FASTCMP) --ref-dir $(PX_REF)/backbone --ours-dir $(PX_OUT)/backbone_fast
	cd $(REPO) && PRAXIS_EXACT=1 rdna4/praxis/test_hip_praxis --model $(PX_MODELS)/mid --ref $(PX_REF)/mid --out $(PX_OUT)/mid_exact --tol 0.02
	$(PX_CMP) --ref-dir $(PX_REF)/mid --ours-dir $(PX_OUT)/mid_exact
	cd $(REPO) && rdna4/praxis/test_hip_praxis --model $(PX_MODELS)/mid --ref $(PX_REF)/mid --out $(PX_OUT)/mid_fast --selftest-stream --selftest-sampler --tol 0.06
	$(PX_FASTCMP) --ref-dir $(PX_REF)/mid --ours-dir $(PX_OUT)/mid_fast
	cd $(REPO) && PRAXIS_EXACT=1 rdna4/praxis/test_hip_praxis --model $(PX_MODELS)/mid128 --ref $(PX_REF)/mid128 --out $(PX_OUT)/mid128_exact --tol 0.02
	$(PX_CMP) --ref-dir $(PX_REF)/mid128 --ours-dir $(PX_OUT)/mid128_exact
	cd $(REPO) && rdna4/praxis/test_hip_praxis --model $(PX_MODELS)/mid128 --ref $(PX_REF)/mid128 --out $(PX_OUT)/mid128_fast --selftest-stream --selftest-sampler --tol 0.06
	$(PX_FASTCMP) --ref-dir $(PX_REF)/mid128 --ours-dir $(PX_OUT)/mid128_fast

clean_hip_praxis:
	rm -f $(CURDIR)/test_hip_praxis $(CURDIR)/hip_praxis_server
# ======================== END praxis-runner (hip_praxis_runner.h) ========================

# ======================= BEGIN full server (hip_praxis_server_full: policy + in-process Qwen + Wan2.2 VAE) =======================
#   make -C rdna4/praxis hip_praxis_server_full      (the plain hip_praxis_server rule above does not link Qwen/tokenizer/VAE; this one does)
FS_HDRS = $(PX_HDRS) hip_wan22_vae.h ../llm/hip_qwen35_st.h ../../cuda/praxis/praxis_qwen.h
hip_praxis_server_full: hip_praxis_server.c praxis_tok.c ../llm/hip_qwen35_st.c $(FS_HDRS) ../rocew.c
	$(CC) $(PX_CFLAGS) -DHQ35_EXTERNAL_SAFETENSORS -o $@ hip_praxis_server.c praxis_tok.c ../llm/hip_qwen35_st.c ../rocew.c $(PX_LDFLAGS)
# ======================== END full server ========================
