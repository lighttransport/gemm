CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic -std=c11
COMMON = ../../common
A64FX_CC ?= fcc
A64FX_MPICC ?= mpifcc
A64FX_CFLAGS ?= -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast

.PHONY: all a64fx clean test validate
.DEFAULT_GOAL := all

ds41f_run test_attention test_index_candidates test_staged_expert test_ops test_rope_layout \
 test_mhc_mix test_pointwise test_int8_attention ds41f_utofu_test test_selection \
 bench_attention_kernel test_sparse_tiles test_exp2: ds41f_sparse_sve.h ds41f_exp2.h

bench_batch bench_quant ds41f_run test_attention test_index_candidates test_cache test_staged_expert ds41f_kernel_test ds41f_sve_test ds41f_utofu_test: ds41f_quant_sve.h ds41f_kernels.h
ds41f_run test_attention test_index_candidates test_staged_expert ds41f_utofu_test: ds41f_profile.h
ds41f_run test_attention test_index_candidates: ds41f_int8.c ds41f_int8.h ds41f_weights.h
ds41f_run test_attention test_index_candidates test_int8_attention: ds41f_attention.h
ds41f_run test_attention test_index_candidates test_int8_attention test_int8 bench_int8 test_staged_expert ds41f_utofu_test: ds41f_alloc.h ds41f_tensor.h
ds41f_run test_prefetch test_engram: ds41f_engram.h
ds41f_run: ds41f_comm.h ds41f_ops.h
test_exp2: test_exp2.c ds41f_ops.c ds41f_ops.h
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c,$^) -lm -o $@
test_sparse_tiles: test_sparse_tiles.c ds41f_ops.c ds41f_ops.h
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c,$^) -lm -o $@
bench_attention_kernel: bench_attention_kernel.c ds41f_ops.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
bench_batch: bench_batch.c ds41f_batch.c ds41f_kernels.c ../gemma4-kernels/sgemm_bf16_2x12.S
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
bench_hardware: bench_hardware.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
ds41f_run: ds41f_run.c ds41f_comm.c ds41f_attention.c ds41f_weights.c ds41f_tensor.c ds41f_cache.c ds41f_kernels.c ds41f_sve.c ds41f_ops.c ds41f_expert.c ds41f_engram.c ds41f_profile.c ds41f_profile.h ds41f_prefetch.c ds41f_prefetch.h
	$(A64FX_MPICC) $(A64FX_CFLAGS) -DDS41F_ENABLE_PROFILE -fopenmp -pthread -I. $(filter %.c %.S,$^) -ltofucom -lm -o $@
test_attention: test_attention.c ds41f_attention.c ds41f_weights.c ds41f_tensor.c ds41f_cache.c ds41f_kernels.c ds41f_sve.c ds41f_ops.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
test_index_candidates: test_index_candidates.c ds41f_attention.c ds41f_weights.c ds41f_tensor.c ds41f_cache.c ds41f_kernels.c ds41f_sve.c ds41f_ops.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
test_cache: test_cache.c ds41f_cache.c ds41f_kernels.c
	$(A64FX_CC) $(A64FX_CFLAGS) -I. $(filter %.c %.S,$^) -lm -o $@
test_engram: test_engram.c ds41f_engram.c
	$(A64FX_CC) $(A64FX_CFLAGS) -I. $(filter %.c %.S,$^) -lm -o $@
test_staged_expert: test_staged_expert.c ds41f_tensor.c ds41f_sve.c ds41f_kernels.c ds41f_expert.c ds41f_ops.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
bench_quant: bench_quant.c ds41f_sve.c ds41f_kernels.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
test_ops: test_ops.c ds41f_ops.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
test_rope_layout: test_rope_layout.c ds41f_ops.c ds41f_ops.h
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
test_mhc_mix: test_mhc_mix.c ds41f_ops.c ds41f_ops.h
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
test_tensor_local: test_tensor_local.c ds41f_tensor.c ds41f_tensor.h ds41f_alloc.h
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -o $@
test_pointwise: test_pointwise.c ds41f_ops.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
test_prefetch: test_prefetch.c ds41f_prefetch.c ds41f_prefetch.h ds41f_engram.c ds41f_kernels.c ds41f_profile.c ds41f_profile.h
	$(A64FX_CC) $(A64FX_CFLAGS) -DDS41F_ENABLE_PROFILE -pthread -I. $(filter %.c %.S,$^) -lm -o $@
test_int8_attention: test_int8_attention.c ds41f_attention.c ds41f_weights.c ds41f_tensor.c ds41f_cache.c ds41f_kernels.c ds41f_sve.c ds41f_ops.c ds41f_int8.c ds41f_int8.h ds41f_weights.h ds41f_profile.h
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
test_int8: test_int8.c ds41f_int8.c ds41f_int8.h ds41f_kernels.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
bench_int8: bench_int8.c ds41f_int8.c ds41f_int8.h ds41f_sve.c ds41f_kernels.c ds41f_tensor.c
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c %.S,$^) -lm -o $@
all: ds41f_stage ds41f_bench ds41f_kernel_test
test_tp_comm: test_tp_comm.c ds41f_comm.c ds41f_comm.h ../utofu-tests/tp_allreduce.h
	$(A64FX_MPICC) $(A64FX_CFLAGS) -I. $(filter %.c,$^) -ltofucom -o $@
test_broadcast: test_broadcast.c ds41f_comm.c ds41f_comm.h ../utofu-tests/tp_allreduce.h
	$(A64FX_MPICC) $(A64FX_CFLAGS) -I. $(filter %.c,$^) -ltofucom -o $@
test_selection: test_selection.c ds41f_ops.c ds41f_ops.h
	$(A64FX_CC) $(A64FX_CFLAGS) -fopenmp -I. $(filter %.c,$^) -lm -o $@
a64fx: test_exp2 test_tp_comm test_sparse_tiles test_broadcast test_selection test_tensor_local test_rope_layout test_mhc_mix test_int8 test_int8_attention bench_int8 ds41f_run ds41f_sve_test ds41f_utofu_test test_attention test_index_candidates test_cache test_ops test_pointwise test_prefetch test_engram test_staged_expert bench_batch bench_quant bench_attention_kernel bench_hardware

ds41f_stage: ds41f_stage.c
	$(CC) $(CFLAGS) -I$(COMMON) $< -o $@

ds41f_bench: ds41f_bench.c ds41f_engram.c ds41f_engram.h
	$(CC) $(CFLAGS) -I. ds41f_bench.c ds41f_engram.c -lm -pthread -o $@

ds41f_kernel_test: ds41f_kernel_test.c ds41f_kernels.c ds41f_model.c ds41f_moe.c ds41f_runtime.c ds41f_kv.c ds41f_sve.c
	$(CC) $(CFLAGS) -I. $(filter %.c %.S,$^) -lm -o $@

ds41f_sve_test: ds41f_kernel_test.c ds41f_kernels.c ds41f_model.c ds41f_moe.c ds41f_runtime.c ds41f_kv.c ds41f_sve.c
	$(A64FX_CC) $(A64FX_CFLAGS) -I. $(filter %.c %.S,$^) -lm -o $@

ds41f_utofu_test: ds41f_utofu_test.c ../utofu-tests/tp_allreduce.h ds41f_expert.c ds41f_tensor.c ds41f_sve.c ds41f_kernels.c ds41f_ops.c
	$(A64FX_MPICC) $(A64FX_CFLAGS) -fopenmp -I. -I../utofu-tests ds41f_utofu_test.c ds41f_expert.c ds41f_tensor.c ds41f_sve.c ds41f_kernels.c ds41f_ops.c -ltofucom -lm -o $@

test: ds41f_kernel_test
	./ds41f_kernel_test

validate:
	python3 validate_model.py $${DS41F_MODEL_DIR:-/home/u14346/models/ds41f}

ds41f_bench_utofu: ds41f_bench.c ds41f_engram.c ds41f_engram.h ../q38fn/ngram_utofu_source.c
	$(A64FX_CC) $(A64FX_CFLAGS) -DDS41F_USE_UTOFU -DQ38FN_NGRAM_HEAD_DIM=256 \
		-DQ38FN_NGRAM_ROW_BYTES=512 -I. -I$(COMMON) -I../q38fn -I../utofu-tests \
		ds41f_bench.c ds41f_engram.c ../q38fn/ngram_utofu_source.c -lm -lpthread -ltofucom -o $@

clean:
	rm -f ds41f_stage ds41f_bench ds41f_bench_utofu ds41f_kernel_test ds41f_sve_test ds41f_utofu_test bench_quant test_ops test_engram test_staged_expert
	rm -f test_exp2 test_tp_comm test_sparse_tiles test_broadcast test_selection ds41f_run test_attention test_index_candidates test_cache test_pointwise test_rope_layout test_mhc_mix test_tensor_local test_prefetch test_int8 test_int8_attention bench_int8 bench_batch bench_attention_kernel bench_hardware
