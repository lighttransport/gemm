CC = fcc
CFLAGS ?= -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -I../../common
WARN = -Wall -Wextra -Wpedantic -Wno-unused-function
IQ_ASM = k3_iq1_pair32.S k3_iq1_quad64.S k3_iq2_pair32.S
MODEL_DIR ?= /home/u14346/models/k3/iq1
NODES ?= 12

.PHONY: all runner full-runner full-convert full-runner-test torch-ref-run torch-ref-test torch-ref-self-test torch-cu129-setup iq1-text iq-layer-12n iq-decode-fapp native-plan tp-stage-test test real-test probe-kda probe-kda-mpi probe-moe-mpi probe-expert-tp-mpi sim validate clean
all: k3_kernel_test k3_quant_kernel_test k3_decode_ids k3_prompt_ids k3_gguf_layer_bench k3_gguf_adapter_test k3_gguf_expert_tp_test k3_gguf_iq_tp_probe k3_runtime_test k3_real_mxfp4_test k3_kda_probe k3_moe_probe k3_dense_probe k3_prefill_probe

runner: k3_ep_runner

full-runner: k3_full_runner

# Built with the site Clang/llama.cpp RPC libraries; used only by the 32-node
# IQ1 GGUF text smoke job.
iq1-text:
	CLANG=/vol0004/apps/r/OSS_CN/llvm-21.1.0/own_clangfx/clang-comp/bin/clang++; \
	B=/home/u14346/work/llama.cpp/build-a64fx-rpc-clang; \
	$$CLANG -std=c++17 -O3 -march=armv8.2-a+sve \
		-I/home/u14346/work/llama.cpp/include -I/home/u14346/work/llama.cpp/ggml/include \
		k3_iq1_text_runner.cpp -L$$B/bin -Wl,-rpath,$$B/bin \
		-lllama -lggml -lggml-rpc -lggml-cpu -lggml-base -lomp -lstdc++fs \
		-pthread -ldl -lm -o k3_iq1_text_runner

full-convert: k3_full_convert

full-runner-test: k3_full_runner
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_k3_full_runner

torch-ref-test:
	cd ../../cuda/llm/k3 && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_k3_torch_partial_ref

torch-ref-run:
	../../cuda/llm/k3/run_k3_torch_ref.sh

torch-ref-self-test:
	cd ../../cuda/llm/k3 && PYTHONDONTWRITEBYTECODE=1 python3 k3_torch_partial_ref.py --self-test

torch-cu129-setup:
	../../cuda/llm/k3/k3_setup_torch_cuda129.sh

# Requires an active 12-node PJM allocation; deliberately model-free.
full-12n-regression: k3_full_runner
	./run_k3_12n_regression.sh

# Requires an active 12-node PJM allocation and a GGUF model directory.
iq-layer-12n: k3_gguf_layer_bench
	./run_k3_iq_layer_12n.sh

# Header-only GGUF -> native K3 ownership/layout validation; no payload staging.
native-plan:
	PYTHONDONTWRITEBYTECODE=1 python3 k3_gguf_native_stage.py $(MODEL_DIR) --nodes $(NODES)

tp-stage-test:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
		test_k3_gguf_expert_tp_stage.py test_k3_gguf_native_stage.py \
		test_k3_gguf_mla_materialize.py test_make_k3_prompt_ids.py \
		test_validate_k3_codegen.py

# Requires an active 12-node PJM allocation and streamed rank-local staging.
fast-12n-decode: k3_ep_runner
	./run_k3_fast_decode_12n.sh

k3_ep_runner: k3_ep_runner.c k3_moe.h k3_dense.h k3_kernels.h k3_runtime.h $(IQ_ASM) ../utofu-tests/tp_allreduce.h ../utofu-tests/tofu_demo.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp -pthread $< $(IQ_ASM) -lm -ltofucom -o $@

k3_full_runner: k3_full_runner.c k3_moe.h k3_dense.h k3_kernels.h k3_prefill.h k3_quant.h k3_runtime.h $(IQ_ASM) ../utofu-tests/tp_allreduce.h ../utofu-tests/tofu_demo.h ../vlm/kernels/micro_kernel_bf16B_8x3_pv.S
	$(CC) $(CFLAGS) $(WARN) -fopenmp -pthread $< $(IQ_ASM) ../vlm/kernels/micro_kernel_bf16B_8x3_pv.S -lm -ltofucom -o $@

k3_full_convert: k3_full_convert.c k3_dense.h k3_kernels.h k3_runtime.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp -pthread $< -lm -o $@

k3_kernel_test: k3_kernel_test.c k3_kernels.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp $< -lm -o $@

k3_quant_kernel_test: k3_quant_kernel_test.c k3_quant.h $(IQ_ASM) ../../common/ggml_dequant.h ../../common/gguf_loader.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp $< $(IQ_ASM) -lm -o $@

k3_decode_ids: k3_decode_ids.c ../../common/gguf_loader.h ../../common/bpe_tokenizer.h
	$(CC) $(CFLAGS) $(WARN) $< -lm -o $@

k3_prompt_ids: k3_prompt_ids.c ../../common/gguf_loader.h ../../common/bpe_tokenizer.h
	$(CC) $(CFLAGS) $(WARN) $< -lm -o $@

k3_gguf_layer_bench: k3_gguf_layer_bench.c k3_quant.h $(IQ_ASM) ../../common/ggml_dequant.h ../../common/gguf_loader.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp $< $(IQ_ASM) -lm -o $@

k3_gguf_layer_bench_fapp: k3_gguf_layer_bench.c k3_quant.h $(IQ_ASM) ../../common/ggml_dequant.h ../../common/gguf_loader.h
	$(CC) $(CFLAGS) $(WARN) -fopenmp -DUSE_FAPP $< $(IQ_ASM) -lm -o $@

iq-decode-fapp: k3_gguf_layer_bench_fapp

k3_gguf_adapter_test: k3_gguf_adapter_test.c k3_gguf_graph_adapter.h
	$(CC) $(CFLAGS) $(WARN) $< -o $@

k3_gguf_expert_tp_test: k3_gguf_expert_tp_test.c k3_gguf_expert_tp.h k3_quant.h ../../common/ggml_dequant.h ../../common/gguf_loader.h
	$(CC) $(CFLAGS) $(WARN) $< -o $@

k3_gguf_iq_tp_probe: k3_gguf_iq_tp_probe.c k3_gguf_expert_tp.h k3_quant.h ../../common/ggml_dequant.h ../../common/gguf_loader.h
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

test: k3_kernel_test k3_quant_kernel_test k3_gguf_adapter_test k3_gguf_expert_tp_test k3_runtime_test full-runner-test tp-stage-test
	./k3_kernel_test
	./k3_quant_kernel_test
	./k3_gguf_adapter_test
	./k3_gguf_expert_tp_test
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
	rm -f k3_kernel_test k3_quant_kernel_test k3_decode_ids k3_prompt_ids k3_gguf_layer_bench k3_gguf_layer_bench_fapp k3_gguf_adapter_test k3_gguf_expert_tp_test k3_gguf_iq_tp_probe k3_runtime_test k3_real_mxfp4_test k3_kda_probe k3_moe_probe k3_dense_probe k3_prefill_probe k3_ep_runner k3_full_runner k3_full_convert
