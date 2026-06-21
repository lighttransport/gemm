# Standalone dev/verify harnesses for the MMQ (prefill) + MMVQ (decode) ports.
NVCC = nvcc
ARCH = sm_120

# Hand-written MMQ kernel harnesses (self-contained).
TARGETS = mmq_iq2xxs_test mmq_iq2xxs_moe_test mmq_iq3xxs_test mmq_q2_K_test mmq_q4_K_test mmq_q4_0_test mmq_iq3_s_test mmq_iq2_s_test

# Vendored-kernel harnesses. Two flavors:
#  - MMQ (prefill) Phase-0 proofs include the UPSTREAM mmq.cuh directly (mul_mat_q is
#    __global__ there, so the test launches it) -> upstream includes only.
#  - MMVQ (decode) tests include the vendored mmq_vendor/mmvq.cuh (mul_mat_vec_q is
#    __device__; the test supplies its own __global__ wrapper) -> need -I../mmq_vendor.
# The rest of the ggml-cuda headers come from the upstream llama.cpp tree.
LLAMA_DIR ?= /home/syoyo/work/llama.cpp
UPSTREAM_INC = -I$(LLAMA_DIR)/ggml/src/ggml-cuda -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/ggml/src
MMQ_FLAGS  = -O3 -std=c++17 -arch=sm_120a $(UPSTREAM_INC)
MMVQ_FLAGS = -O3 -std=c++17 -arch=sm_120a -I../mmq_vendor $(UPSTREAM_INC)
VENDOR_TARGETS = mmq_iq2xxs_vendor_test mmq_q6k_vendor_test mmvq_iq2xxs_vendor_test mmvq_iq3s_vendor_test mmvq_q8_0_test dbg_iq3s

all: $(TARGETS)
vendor: $(VENDOR_TARGETS)

mmq_iq2xxs_test: mmq_iq2xxs_test.cu iq2xxs_tables.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_iq2xxs_moe_test: mmq_iq2xxs_moe_test.cu iq2xxs_tables.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_iq3xxs_test: mmq_iq3xxs_test.cu iq2xxs_tables.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_q2_K_test: mmq_q2_K_test.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_q4_K_test: mmq_q4_K_test.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_q4_0_test: mmq_q4_0_test.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_iq3_s_test: mmq_iq3_s_test.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_iq2_s_test: mmq_iq2_s_test.cu iq2s_grid.inc
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

# MMQ (prefill) Phase-0 proofs: upstream mmq.cuh (__global__) + quantize.cu.
mmq_iq2xxs_vendor_test: mmq_iq2xxs_vendor_test.cu
	$(NVCC) $(MMQ_FLAGS) -o $@ $< $(LLAMA_DIR)/ggml/src/ggml-cuda/quantize.cu

mmq_q6k_vendor_test: mmq_q6k_vendor_test.cu
	$(NVCC) $(MMQ_FLAGS) -o $@ $< $(LLAMA_DIR)/ggml/src/ggml-cuda/quantize.cu

# MMVQ (decode) tests: vendored mmvq.cuh (__device__) + self-supplied __global__ wrapper.
mmvq_iq2xxs_vendor_test: mmvq_iq2xxs_vendor_test.cu ../mmq_vendor/mmvq.cuh
	$(NVCC) $(MMVQ_FLAGS) -o $@ $<

mmvq_iq3s_vendor_test: mmvq_iq3s_vendor_test.cu ../mmq_vendor/mmvq.cuh
	$(NVCC) $(MMVQ_FLAGS) -o $@ $<

mmvq_q8_0_test: mmvq_q8_0_test.cu ../mmq_vendor/mmvq.cuh
	$(NVCC) $(MMVQ_FLAGS) -o $@ $<

dbg_iq3s: dbg_iq3s.cu ../mmq_vendor/mmvq.cuh
	$(NVCC) $(MMVQ_FLAGS) -o $@ $<

run: mmq_iq2xxs_test
	./mmq_iq2xxs_test

clean:
	rm -f $(TARGETS) $(VENDOR_TARGETS) \
	      dbg_iq3s_-O0 dbg_iq3s_-O1 dbg_iq3s_-O2 dbg_iq3s_-O3 mmvq_vendor_test mmvq_iq3s_test

.PHONY: all vendor run clean
