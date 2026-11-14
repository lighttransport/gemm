# Standalone CUDA Wan2.2 VAE (NVRTC + cuBLAS via dlopen; no nvcc needed).
#   make -C cuda/praxis test      # 5x64x64 oracle check (rel-L2<=0.02, cos>=0.999)
#   make -C cuda/praxis refs      # (re)generate oracle dumps (needs ~/.venv torch + praxis src)
#   make -C cuda/praxis bench     # 9x128x128 and 21x256x256 timing + checks
REPO   := $(abspath ../..)
PRAXIS_ROOT ?= $(HOME)/praxis
PY     ?= $(HOME)/.venv/bin/python
CC     ?= gcc
CFLAGS  = -O2 -mavx2 -mfma -mf16c -fopenmp -Wall -Wno-unused-function -Wno-unknown-pragmas -I../../common -I.. -D_GNU_SOURCE
LDFLAGS = -lm -ldl -lpthread -fopenmp
WEIGHTS ?= $(PRAXIS_ROOT)/wan2.2-vae
REF     ?= $(REPO)/tmp/praxis_ref
OUT     ?= $(REPO)/tmp/praxis_out
CMP     = $(PY) -I $(REPO)/ref/praxis/compare.py --tol 0.02 --cos 0.999

.PHONY: all test bench bench_big test_tiled refs clean test_praxis refs_praxis refs_praxis_sharded test_praxis_sharded refs_e2e test_praxis_e2e
all: test_cuda_wan22_vae

test_cuda_wan22_vae: test_cuda_wan22_vae.c cuda_wan22_vae.h ../cuew.c ../cublasew.c \
		../cuew.h ../cublasew.h ../cuda_runner_common.h ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ $< ../cuew.c ../cublasew.c $(LDFLAGS)

$(WEIGHTS)/model.safetensors:
	$(PY) -I $(REPO)/ref/praxis/convert_vae.py --out $(WEIGHTS)

test: test_cuda_wan22_vae $(WEIGHTS)/model.safetensors
	cd $(REPO) && cuda/praxis/test_cuda_wan22_vae --weights $(WEIGHTS) --ref $(REF)/vae --out $(OUT)/vae --taps --self-decode
	$(CMP) --ref-dir $(REF)/vae --ours-dir $(OUT)/vae

refs:
	cd $(REPO) && $(PY) -I ref/praxis/gen_vae_ref.py --inter
	cd $(REPO) && $(PY) -I ref/praxis/gen_vae_ref.py --inter --fp32 --out tmp/praxis_ref/vae_fp32
	cd $(REPO) && $(PY) -I ref/praxis/gen_vae_ref.py --inter --frames 9 --size 128 --out tmp/praxis_ref/vae_9x128
	cd $(REPO) && $(PY) -I ref/praxis/gen_vae_ref.py --frames 21 --size 256 --out tmp/praxis_ref/vae_21x256

bench: test_cuda_wan22_vae $(WEIGHTS)/model.safetensors
	cd $(REPO) && cuda/praxis/test_cuda_wan22_vae --weights $(WEIGHTS) --ref $(REF)/vae_9x128 --out $(OUT)/vae_9x128 --taps --repeat 2
	$(CMP) --ref-dir $(REF)/vae_9x128 --ours-dir $(OUT)/vae_9x128
	cd $(REPO) && cuda/praxis/test_cuda_wan22_vae --weights $(WEIGHTS) --ref $(REF)/vae_21x256 --out $(OUT)/vae_21x256 --repeat 2
	$(CMP) --ref-dir $(REF)/vae_21x256 --ours-dir $(OUT)/vae_21x256

# tiled-vs-untiled seam check (synthetic video, no oracle) incl. non-multiple-of-32 size and uint8 input
test_tiled: test_cuda_wan22_vae $(WEIGHTS)/model.safetensors
	cd $(REPO) && cuda/praxis/test_cuda_wan22_vae --weights $(WEIGHTS) --synth 5,256,416 --tile 256,96 --compare-untiled
	cd $(REPO) && cuda/praxis/test_cuda_wan22_vae --weights $(WEIGHTS) --synth 5,250,410 --tile 256,96 --compare-untiled --u8

# memory/time at 480p / 720p class sizes (automatic tiling, 5 GB budget)
bench_big: test_cuda_wan22_vae $(WEIGHTS)/model.safetensors
	cd $(REPO) && cuda/praxis/test_cuda_wan22_vae --weights $(WEIGHTS) --synth 21,448,832
	cd $(REPO) && cuda/praxis/test_cuda_wan22_vae --weights $(WEIGHTS) --synth 9,720,1280

clean:
	rm -f test_cuda_wan22_vae test_cuda_praxis bench_praxis_synth test_praxis_fp4 test_praxis_attn test_praxis_shards praxis_e2e

# ---- praxis-1 policy runner (backbone + MoE + action expert + window_joint sampler) ----
#   make -C cuda/praxis test_praxis          # tiny-model parity (rel-L2 <= 0.02) + expert-streaming self-test
#   make -C cuda/praxis refs_praxis          # regenerate the tiny model + reference dumps (needs ~/.venv torch + praxis src)
TINY    ?= $(PRAXIS_ROOT)/tiny
PRAXIS_HDRS = cuda_praxis_runner.h praxis_kernels.h praxis_fp4_kernels.h ../cublasew.h ../cublasew.c ../cuew.h ../cuda_runner_common.h ../../common/safetensors.h

bench_praxis_synth: bench_praxis_synth.c $(PRAXIS_HDRS) ../cuew.c
	$(CC) $(CFLAGS) -o $@ $< ../cuew.c ../cublasew.c $(LDFLAGS)

test_praxis_fp4: test_praxis_fp4.c $(PRAXIS_HDRS) praxis_fp4_kernels.h ../cuew.c
	$(CC) $(CFLAGS) -o $@ $< ../cuew.c ../cublasew.c $(LDFLAGS)

# kernel exactness of the NVFP4 W4A4 / W4A16 / FP8 W8A16 expert GEMMs + packer round trips (mid model, ~10 s)
.PHONY: check_praxis_fp4
check_praxis_fp4: test_praxis_fp4
	cd $(REPO) && cuda/praxis/test_praxis_fp4 --model $(MID)

test_praxis_attn: test_praxis_attn.c $(PRAXIS_HDRS) ../cuew.c
	$(CC) $(CFLAGS) -o $@ $< ../cuew.c ../cublasew.c $(LDFLAGS)

test_cuda_praxis: test_cuda_praxis.c $(PRAXIS_HDRS) ../cuew.c
	$(CC) $(CFLAGS) -o $@ $< ../cuew.c ../cublasew.c $(LDFLAGS)

$(REF)/backbone/meta.json $(TINY)/model.safetensors:
	cd $(REPO) && $(PY) -I ref/praxis/gen_backbone_ref.py --out tmp/praxis_ref/backbone --model-out $(TINY)

refs_praxis:
	cd $(REPO) && $(PY) -I ref/praxis/gen_backbone_ref.py --out tmp/praxis_ref/backbone --model-out $(TINY)

FASTCMP = $(PY) -I $(REPO)/ref/praxis/compare.py --tol 0.06 --cos 0.998 --topk-agree 0.85 --skip-missing
MID     ?= $(PRAXIS_ROOT)/mid
MID128  ?= $(PRAXIS_ROOT)/mid128

# exact fp32 path (PRAXIS_EXACT=1): fp32 reference parity at 0.02 + streaming self-test; fast bf16/cuBLAS/flash path at the
# bf16-appropriate 0.06 (route flips of near-tied experts dominate; see README)
test_praxis: test_cuda_praxis $(REF)/backbone/meta.json $(TINY)/model.safetensors
	cd $(REPO) && PRAXIS_EXACT=1 cuda/praxis/test_cuda_praxis --model $(TINY) --ref $(REF)/backbone --out $(OUT)/backbone --selftest-stream
	$(PY) -I $(REPO)/ref/praxis/compare.py --tol 0.02 --cos 0.999 --skip-missing --ref-dir $(REF)/backbone --ours-dir $(OUT)/backbone
	$(PY) -I $(REPO)/ref/praxis/compare.py --tol 0.02 --cos 0.999 --skip-missing --ref-dir $(REF)/backbone --ours-dir $(OUT)/backbone/stream2
	cd $(REPO) && cuda/praxis/test_cuda_praxis --model $(TINY) --ref $(REF)/backbone --out $(OUT)/backbone_fast --selftest-stream --tol 0.06
	$(FASTCMP) --ref-dir $(REF)/backbone --ours-dir $(OUT)/backbone_fast
	$(FASTCMP) --ref-dir $(REF)/backbone --ours-dir $(OUT)/backbone_fast/stream2

# mid-size configs (dim 256 / head_dim 64 and dim 384 / head_dim 128, 8 experts top-2, expert_hidden 128), torch fp32 reference
refs_praxis_mid:
	cd $(REPO) && $(PY) -I ref/praxis/gen_backbone_ref.py --preset mid --text-len 16 --out tmp/praxis_ref/mid --model-out $(MID)
	cd $(REPO) && $(PY) -I ref/praxis/gen_backbone_ref.py --preset mid128 --text-len 16 --out tmp/praxis_ref/mid128 --model-out $(MID128)

test_praxis_mid: test_cuda_praxis
	cd $(REPO) && PRAXIS_EXACT=1 cuda/praxis/test_cuda_praxis --model $(MID) --ref $(REF)/mid --out $(OUT)/mid_exact --tol 0.02
	$(PY) -I $(REPO)/ref/praxis/compare.py --tol 0.02 --cos 0.999 --skip-missing --ref-dir $(REF)/mid --ours-dir $(OUT)/mid_exact
	cd $(REPO) && cuda/praxis/test_cuda_praxis --model $(MID) --ref $(REF)/mid --out $(OUT)/mid_fast --selftest-stream --selftest-sampler --tol 0.06
	$(FASTCMP) --ref-dir $(REF)/mid --ours-dir $(OUT)/mid_fast
	cd $(REPO) && PRAXIS_EXACT=1 cuda/praxis/test_cuda_praxis --model $(MID128) --ref $(REF)/mid128 --out $(OUT)/mid128_exact --tol 0.02
	$(PY) -I $(REPO)/ref/praxis/compare.py --tol 0.02 --cos 0.999 --skip-missing --ref-dir $(REF)/mid128 --ours-dir $(OUT)/mid128_exact
	cd $(REPO) && cuda/praxis/test_cuda_praxis --model $(MID128) --ref $(REF)/mid128 --out $(OUT)/mid128_fast --selftest-stream --selftest-sampler --tol 0.06
	$(FASTCMP) --ref-dir $(REF)/mid128 --ours-dir $(OUT)/mid128_fast

# ---- sharded safetensors (model.safetensors.index.json + N shards, real layout) + host RSS + pack cache ----
#   make -C cuda/praxis refs_praxis_sharded test_praxis_sharded
# tiny/mid/mid128 are rewritten as 3/4/5 shards (identical weights, scratch refs); `big` (dim 768, 32 experts, 19 MoE layers, 6 shards,
# 1.6 GB of bf16 on disk) is the host-RSS test model. Sharded exact-path outputs must be BIT-identical to the single-file runs.
TINY_SH ?= $(PRAXIS_ROOT)/tiny_sh
MID_SH  ?= $(PRAXIS_ROOT)/mid_sh
MID128_SH ?= $(PRAXIS_ROOT)/mid128_sh
BIG_SH  ?= $(PRAXIS_ROOT)/big_sh
PCACHE  ?= $(OUT)/mid128.packcache
test_praxis_shards: test_praxis_shards.c $(PRAXIS_HDRS) ../cuew.c
	$(CC) $(CFLAGS) -o $@ test_praxis_shards.c ../cuew.c ../cublasew.c $(LDFLAGS)

refs_praxis_sharded:
	cd $(REPO) && $(PY) -I ref/praxis/gen_backbone_ref.py --preset tiny --shards 3 --out $(OUT)/shref_tiny --model-out $(TINY_SH)
	cd $(REPO) && $(PY) -I ref/praxis/gen_backbone_ref.py --preset mid --text-len 16 --shards 4 --out $(OUT)/shref_mid --model-out $(MID_SH)
	cd $(REPO) && $(PY) -I ref/praxis/gen_backbone_ref.py --preset mid128 --text-len 16 --shards 5 --out $(OUT)/shref_mid128 --model-out $(MID128_SH)
	cd $(REPO) && $(PY) -I ref/praxis/gen_backbone_ref.py --preset big --skip-ref --shards 6 --out $(OUT)/shref_big --model-out $(BIG_SH)

test_praxis_sharded: test_cuda_praxis test_praxis_shards
	cd $(REPO) && PRAXIS_EXACT=1 cuda/praxis/test_cuda_praxis --model $(TINY_SH) --ref $(REF)/backbone --out $(OUT)/tiny_sh_exact --tol 0.02
	cd $(REPO) && PRAXIS_EXACT=1 cuda/praxis/test_cuda_praxis --model $(MID_SH) --ref $(REF)/mid --out $(OUT)/mid_sh_exact --tol 0.02
	cd $(REPO) && PRAXIS_EXACT=1 cuda/praxis/test_cuda_praxis --model $(MID128_SH) --ref $(REF)/mid128 --out $(OUT)/mid128_sh_exact --tol 0.02
	@for m in mid mid128; do for f in enc_b2_out dec_vel smp_out_video smp_out_action; do cmp $(OUT)/$${m}_exact/$$f.npy $(OUT)/$${m}_sh_exact/$$f.npy || exit 1; done; echo "$$m: sharded == single-file (bit-identical)"; done
	cd $(REPO) && PRAXIS_NVFP4=1 cuda/praxis/test_cuda_praxis --model $(MID128_SH) --ref $(REF)/mid128 --out $(OUT)/mid128_sh_fp4 --tol 0.1 > /dev/null
	: > $(PCACHE)
	cd $(REPO) && PRAXIS_NVFP4=1 PRAXIS_PACK_CACHE=$(PCACHE) cuda/praxis/test_cuda_praxis --model $(MID128_SH) --ref $(REF)/mid128 --out $(OUT)/mid128_sh_fp4w --tol 0.1 > /dev/null
	cd $(REPO) && PRAXIS_NVFP4=1 PRAXIS_PACK_CACHE=$(PCACHE) cuda/praxis/test_cuda_praxis --model $(MID128_SH) --ref $(REF)/mid128 --out $(OUT)/mid128_sh_fp4h --tol 0.1 > /dev/null
	@for f in dec_vel smp_out_video smp_out_action; do cmp $(OUT)/mid128_sh_fp4/$$f.npy $(OUT)/mid128_sh_fp4h/$$f.npy || exit 1; done; echo "pack cache: hit == freshly packed (bit-identical)"
	cd $(REPO) && PRAXIS_NVFP4=1 PRAXIS_EXPERT_FMT=nvfp4a16 cuda/praxis/test_cuda_praxis --model $(MID128_SH) --ref $(REF)/mid128 --out $(OUT)/mid128_sh_fp4a16 --tol 0.1 > /dev/null
	rm -f $(PCACHE).mix $(PCACHE).mix.p*
	cd $(REPO) && PRAXIS_NVFP4=1 PRAXIS_EXPERT_FMT=nvfp4a16 PRAXIS_FP8_BANKS=dn PRAXIS_PACK_CACHE=$(PCACHE).mix cuda/praxis/test_cuda_praxis --model $(MID128_SH) --ref $(REF)/mid128 --out $(OUT)/mid128_sh_mixw --tol 0.1 > /dev/null
	cd $(REPO) && PRAXIS_NVFP4=1 PRAXIS_EXPERT_FMT=nvfp4a16 PRAXIS_FP8_BANKS=dn PRAXIS_PACK_CACHE=$(PCACHE).mix cuda/praxis/test_cuda_praxis --model $(MID128_SH) --ref $(REF)/mid128 --out $(OUT)/mid128_sh_mixh --tol 0.1 > /dev/null
	@for f in dec_vel smp_out_video smp_out_action; do cmp $(OUT)/mid128_sh_mixw/$$f.npy $(OUT)/mid128_sh_mixh/$$f.npy || exit 1; done; echo "mixed NVFP4/FP8 plan pack cache: hit == freshly packed (bit-identical)"
	cd $(REPO) && PRAXIS_NVFP4=1 PRAXIS_FP8_BANKS=all cuda/praxis/test_cuda_praxis --model $(MID128_SH) --ref $(REF)/mid128 --out $(OUT)/mid128_sh_fp8 --tol 0.06 > /dev/null
	cd $(REPO) && cuda/praxis/test_praxis_shards --model $(BIG_SH) --nvfp4 --calib $(MID128_SH) --check-bound
	cd $(REPO) && ! PRAXIS_NO_DROP=1 cuda/praxis/test_praxis_shards --model $(BIG_SH) --nvfp4 --calib $(MID128_SH) --check-bound > /dev/null && echo "PRAXIS_NO_DROP=1 (load-everything behaviour) correctly FAILS the bound"

# ---- end-to-end driver ----
#   make -C cuda/praxis refs_e2e test_praxis_e2e    (Qwen3.5-2B + Wan2.2 VAE + mid2048 policy vs the same pipeline in PyTorch)
E2E_REF ?= $(REPO)/tmp/praxis_ref/e2e
refs_e2e:
	cd $(REPO) && $(PY) -I ref/praxis/gen_e2e_ref.py --out tmp/praxis_ref/e2e --model-out $(PRAXIS_ROOT)/mid2048

# Qwen3.5-2B conditioning runs IN-PROCESS: praxis_qwen.c (transformer.h impl + thin API) + cuda/llm/cuda_llm_runner.c are separate objects built with
# the cuda/llm flags (-march=native); the child-process path (--qwen-spawn, cuda/llm/test_qwen35_st) is kept as a fallback.
QOBJ = $(REPO)/tmp/praxis_obj
QCFLAGS = -O3 -Wno-all -D_GNU_SOURCE -I.. -mavx2 -mfma -march=native
praxis_e2e: praxis_e2e.c praxis_qwen.c praxis_qwen.h $(PRAXIS_HDRS) cuda_wan22_vae.h ../cuew.c ../llm/cuda_llm_runner.c ../llm/cuda_llm_runner.h
	mkdir -p $(QOBJ)
	$(CC) $(QCFLAGS) -c praxis_qwen.c -o $(QOBJ)/praxis_qwen.o
	$(CC) $(QCFLAGS) -c ../llm/cuda_llm_runner.c -o $(QOBJ)/cuda_llm_runner.o
	$(CC) $(CFLAGS) -o $@ praxis_e2e.c $(QOBJ)/praxis_qwen.o $(QOBJ)/cuda_llm_runner.o ../cuew.c ../cublasew.c $(LDFLAGS)

praxis_server: praxis_server.c praxis_e2e.c praxis_qwen.c praxis_qwen.h $(PRAXIS_HDRS) cuda_wan22_vae.h ../cuew.c ../llm/cuda_llm_runner.c ../llm/cuda_llm_runner.h
	mkdir -p $(QOBJ)
	$(CC) $(QCFLAGS) -c praxis_qwen.c -o $(QOBJ)/praxis_qwen.o
	$(CC) $(QCFLAGS) -c ../llm/cuda_llm_runner.c -o $(QOBJ)/cuda_llm_runner.o
	$(CC) $(CFLAGS) -o $@ praxis_server.c $(QOBJ)/praxis_qwen.o $(QOBJ)/cuda_llm_runner.o ../cuew.c ../cublasew.c $(LDFLAGS)

test_praxis_e2e: praxis_e2e
	cd $(REPO) && cuda/praxis/praxis_e2e --policy $(PRAXIS_ROOT)/mid2048 --ids $(E2E_REF)/ids.npy --ids-uncond $(E2E_REF)/ids_u.npy \
	  --ext $(E2E_REF)/ext.npy --wrist $(E2E_REF)/wrist.npy --proprio $(E2E_REF)/proprio.npy --noise-v $(E2E_REF)/smp_noise.salt0.0.npy \
	  --noise-a $(E2E_REF)/smp_noise.salt1.0.npy --tile-h 64 --tile-w 128 --K 2 --steps 2 --cfg-video 4 --cfg-action 1.5 --decode \
	  --text-cache $(OUT)/e2e_textcache --out $(OUT)/e2e
	$(PY) -I $(REPO)/ref/praxis/compare_e2e.py --ref $(E2E_REF) --ours $(OUT)/e2e

# --- real praxis-1-droid, K'=1 (first call, depth 0) parity vs the torch fp32 oracle of call 1 (needs the 54 GB checkpoint + tmp/praxis_ref/real_fp32/call1; not part of `make test`) ---
# bf16 experts (parity only, ring 2 x 2.9 GB): dec_vel 1.2e-2, ae_out 4.1e-3, actions 6.2e-2;  mixed: 3.1e-2 / 1.3e-2 / 4.3e-2  (torch bf16 itself: 3.0e-2 / 8.7e-3 / 0.34)
test_praxis_real_k1: test_cuda_praxis
	cd $(REPO) && PRAXIS_EXPERT_FMT=mixed PRAXIS_PACK_CACHE=$(PRAXIS_ROOT)/praxis-1-droid.nvfp4.packcache cuda/praxis/test_cuda_praxis --model $(PRAXIS_ROOT)/praxis-1-droid \
	  --ref tmp/praxis_ref/real_fp32/call1 --out $(OUT)/real_k1_mixed --stream 2 --tol 0.5
