CC = fcc
CFLAGS ?= -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -I../../common
WARN = -Wall -Wextra -Wpedantic -Wno-unused-function

.PHONY: all test real-test probe-kda probe-kda-mpi sim validate clean
all: k3_kernel_test k3_real_mxfp4_test k3_kda_probe

k3_kernel_test: k3_kernel_test.c k3_kernels.h
	$(CC) $(CFLAGS) $(WARN) $< -lm -o $@

k3_real_mxfp4_test: k3_real_mxfp4_test.c
	$(CC) $(CFLAGS) $(WARN) $< -lm -o $@

k3_kda_probe: k3_kda_probe.c k3_kernels.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp $< -lm -o $@

test: k3_kernel_test
	./k3_kernel_test

real-test: all
	./run_partial_test.sh

probe-kda: k3_kda_probe
	./run_kda_probe.sh

probe-kda-mpi: k3_kda_probe
	./run_kda_probe_mpi.sh

sim:
	python3 k3_sim.py

validate:
	PYTHONDONTWRITEBYTECODE=1 python3 k3_stage.py --validate-all

clean:
	rm -f k3_kernel_test k3_real_mxfp4_test k3_kda_probe
