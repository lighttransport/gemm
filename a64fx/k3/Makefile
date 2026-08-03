CC = fcc
CFLAGS ?= -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -I../../common
WARN = -Wall -Wextra -Wpedantic -Wno-unused-function

.PHONY: all runner full-runner full-convert full-runner-test full-12n-regression fast-12n-decode test real-test probe-kda probe-kda-mpi probe-moe-mpi probe-expert-tp-mpi sim validate clean
all: k3_kernel_test k3_runtime_test k3_real_mxfp4_test k3_kda_probe k3_moe_probe k3_dense_probe k3_prefill_probe

runner: k3_ep_runner

full-runner: k3_full_runner

full-convert: k3_full_convert

full-runner-test: k3_full_runner
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_k3_full_runner

# Requires an active 12-node PJM allocation; deliberately model-free.
full-12n-regression: k3_full_runner
	./run_k3_12n_regression.sh

# Requires an active 12-node PJM allocation and streamed rank-local staging.
fast-12n-decode: k3_ep_runner
	./run_k3_fast_decode_12n.sh

k3_ep_runner: k3_ep_runner.c k3_moe.h k3_dense.h k3_kernels.h k3_runtime.h ../utofu-tests/tp_allreduce.h ../utofu-tests/tofu_demo.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp -pthread $< -lm -ltofucom -o $@

k3_full_runner: k3_full_runner.c k3_moe.h k3_dense.h k3_kernels.h k3_runtime.h ../utofu-tests/tp_allreduce.h ../utofu-tests/tofu_demo.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp -pthread $< -lm -ltofucom -o $@

k3_full_convert: k3_full_convert.c k3_dense.h k3_kernels.h k3_runtime.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp -pthread $< -lm -o $@

k3_kernel_test: k3_kernel_test.c k3_kernels.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp $< -lm -o $@

k3_runtime_test: k3_runtime_test.c k3_runtime.h
	$(CC) $(CFLAGS) $(WARN) $< -o $@

k3_real_mxfp4_test: k3_real_mxfp4_test.c
	$(CC) $(CFLAGS) $(WARN) $< -lm -o $@

k3_kda_probe: k3_kda_probe.c k3_kernels.h k3_runtime.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp $< -lm -o $@

k3_moe_probe: k3_moe_probe.c k3_moe.h k3_dense.h k3_kernels.h k3_runtime.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp $< -lm -o $@

k3_dense_probe: k3_dense_probe.c k3_dense.h k3_runtime.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp $< -lm -o $@

k3_prefill_probe: k3_prefill_probe.c k3_prefill.h k3_dense.h k3_runtime.h ../vlm/kernels/micro_kernel_bf16B_8x3_pv.S
	$(CC) $(CFLAGS) $(WARN) -fopenmp $< ../vlm/kernels/micro_kernel_bf16B_8x3_pv.S -lm -o $@

test: k3_kernel_test k3_runtime_test full-runner-test
	./k3_kernel_test
	./k3_runtime_test

real-test: all
	./run_partial_test.sh

probe-kda: k3_kda_probe
	./run_kda_probe.sh

probe-kda-mpi: k3_kda_probe
	./run_kda_probe_mpi.sh

probe-moe-mpi: k3_moe_probe
	./run_moe_probe_mpi.sh

probe-expert-tp-mpi: k3_moe_probe
	./run_expert_tp_probe_mpi.sh

sim:
	python3 k3_sim.py

validate:
	PYTHONDONTWRITEBYTECODE=1 python3 k3_stage.py --validate-all

clean:
	rm -f k3_kernel_test k3_runtime_test k3_real_mxfp4_test k3_kda_probe k3_moe_probe k3_dense_probe k3_prefill_probe k3_ep_runner k3_full_runner k3_full_convert
