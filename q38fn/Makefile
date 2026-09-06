CC ?= cc
MPICC ?= mpiclang
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -I../common
THREAD_FLAGS = -pthread
FAPP_CPPFLAGS ?= -I/opt/FJSVxtclanga/tcsds-1.2.43/clang-comp/include/external
FAPP_LDFLAGS ?= -Wl,--no-as-needed,--allow-shlib-undefined -L/opt/FJSVxtclanga/tcsds-1.2.43/lib64/fapp -l:libfjprofcore.so.1 -l:libfjprofomp.so.1
A64FX_RUNNER_FLAGS ?= -O3 -ffast-math -fopenmp
# FAPP is provided by TCSDS and needs its Clang frontend for ACLE SVE
# intrinsics.  Keep this separate from the normal runner flags because an
# external LLVM MPI wrapper does not understand -Nclang.
FAPP_A64FX_RUNNER_FLAGS ?= -Nclang -O3 -ffast-math -fopenmp -march=armv8.2-a+sve

.PHONY: all test test_target_code clean utofu_source utofu_probe utofu_probe_mpi utofu_real_probe utofu_pipeline_probe utofu_pipeline_probe_mpi ngram_stage ngram_stage_mpi hbm_probe lowbit_probe lowbit_plan owner_load_probe

all: q38fn_runner q38fn_inspect ngram_pipeline_probe test_ngram_pipeline

compare_f32: compare_f32.c
	$(CC) $(CFLAGS) $< -o $@ -lm

decode_trace: decode_trace.c ../common/glm5_bpe.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

q38fn_pp_runner: q38fn_pp_runner.c ../common/q38fn_runtime.h ../common/glm5_bpe.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm -pthread

q38fn_stage_mpi: q38fn_stage_mpi.c ../common/glm53f_safetensors.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

q38fn_tp_stage_mpi: q38fn_tp_stage_mpi.c ../common/glm53f_safetensors.h \
		../common/q38fn_tp_layout.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_TP_STAGE_DEFAULT_Q5=1 $< -o $@ -lm

q38fn_tp4_stage_mpi: q38fn_tp_stage_mpi.c ../common/glm53f_safetensors.h \
		../common/q38fn_tp_layout.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_TP_RANKS=4 \
		-DQ38FN_TP_STAGE_DEFAULT_Q5=1 $< -o $@ -lm

q38fn_tp6_stage_mpi: q38fn_tp_stage_mpi.c ../common/glm53f_safetensors.h \
		../common/q38fn_tp_layout.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_TP_RANKS=6 \
		-DQ38FN_TP_STAGE_DEFAULT_Q5=1 $< -o $@ -lm

q38fn_tp_reconstruct_mpi: q38fn_tp_reconstruct_mpi.c ../common/q38fn_tp_blob.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

q38fn_tp_trim_mpi: q38fn_tp_trim_mpi.c
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $< -o $@

q38fn_tp_upgrade_mpi: q38fn_tp_upgrade_mpi.c ../common/q38fn_tp_blob.h \
		../common/q38fn_tp_layout.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $< -o $@

q38fn_tp_hc_repack_mpi: q38fn_tp_hc_repack_mpi.c ../common/q38fn_tp_blob.h \
		../common/q38fn_tp_layout.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $< -o $@

q38fn_tp_replicate_hc_mpi: q38fn_tp_replicate_hc_mpi.c ../common/q38fn_tp_blob.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

q38fn_tp_q5_to_q8_mpi: q38fn_tp_q5_to_q8_mpi.c ../common/q38fn_tp_blob.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -fopenmp $< -o $@ -lm

q38fn_tp_q5_to_bf16_mpi: q38fn_tp_q5_to_bf16_mpi.c ../common/q38fn_tp_blob.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -fopenmp $< -o $@ -lm

q38fn_tp_runner: q38fn_tp_runner.c ../common/q38fn_tp_runtime.h ../common/glm5_bpe.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

q38fn_tp4_runner: q38fn_tp_runner.c ../common/q38fn_tp_runtime.h ../common/glm5_bpe.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_TP_RANKS=4 $< -o $@ -lm

q38fn_tp4_blob_probe: q38fn_tp_blob_probe.c ../common/q38fn_tp_blob.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_TP_RANKS=4 $< -o $@ -lm

# Fugaku-only runner: MPI launches ranks and broadcasts prompt tokens; all hot
# tensor-parallel reductions use the uTofu one-sided transport.
q38fn_tp_runner_utofu: q38fn_tp_runner.c q38fn_utofu_transport.c \
		q38fn_utofu_transport.h ../common/q38fn_tp_runtime.h \
		../common/glm5_bpe.h ../a64fx/utofu-tests/tp_allreduce.h
	$(MPICC) $(CFLAGS) $(A64FX_RUNNER_FLAGS) $(CPPFLAGS) -DQ38FN_USE_UTOFU \
		-I../a64fx/utofu-tests q38fn_tp_runner.c q38fn_utofu_transport.c \
		-o $@ -lm -ltofucom -pthread

q38fn_tp_runner_utofu_fapp: q38fn_tp_runner.c q38fn_utofu_transport.c \
		q38fn_utofu_transport.h ../common/q38fn_tp_runtime.h \
		../common/glm5_bpe.h ../a64fx/utofu-tests/tp_allreduce.h
	$(MPICC) $(CFLAGS) $(FAPP_A64FX_RUNNER_FLAGS) $(CPPFLAGS) $(FAPP_CPPFLAGS) -DQ38FN_HAVE_FAPP -DQ38FN_USE_UTOFU \
		-I../a64fx/utofu-tests q38fn_tp_runner.c q38fn_utofu_transport.c \
		-o $@ -no-pie $(FAPP_LDFLAGS) -lm -ltofucom -pthread

q38fn_tp4_runner_utofu: q38fn_tp_runner.c q38fn_utofu_transport.c \
		q38fn_utofu_transport.h ../common/q38fn_tp_runtime.h \
		../common/glm5_bpe.h ../a64fx/utofu-tests/tp_allreduce.h
	$(MPICC) $(CFLAGS) $(A64FX_RUNNER_FLAGS) $(CPPFLAGS) -DQ38FN_USE_UTOFU -DQ38FN_TP_RANKS=4 \
		-I../a64fx/utofu-tests q38fn_tp_runner.c q38fn_utofu_transport.c \
		-o $@ -lm -ltofucom -pthread

q38fn_tp4_runner_utofu_fapp: q38fn_tp_runner.c q38fn_utofu_transport.c \
		q38fn_utofu_transport.h ../common/q38fn_tp_runtime.h \
		../common/glm5_bpe.h ../a64fx/utofu-tests/tp_allreduce.h
	$(MPICC) $(CFLAGS) $(FAPP_A64FX_RUNNER_FLAGS) $(CPPFLAGS) $(FAPP_CPPFLAGS) -DQ38FN_HAVE_FAPP -DQ38FN_USE_UTOFU \
		-DQ38FN_TP_RANKS=4 -I../a64fx/utofu-tests \
		q38fn_tp_runner.c q38fn_utofu_transport.c \
		-o $@ -no-pie $(FAPP_LDFLAGS) -lm -ltofucom -pthread

q38fn_tp6_runner_utofu: q38fn_tp_runner.c q38fn_utofu_transport.c \
		q38fn_utofu_transport.h ../common/q38fn_tp_runtime.h \
		../common/glm5_bpe.h ../a64fx/utofu-tests/tp_allreduce.h
	$(MPICC) $(CFLAGS) $(A64FX_RUNNER_FLAGS) $(CPPFLAGS) -DQ38FN_USE_UTOFU -DQ38FN_TP_RANKS=6 \
		-I../a64fx/utofu-tests q38fn_tp_runner.c q38fn_utofu_transport.c \
		-o $@ -lm -ltofucom -pthread

q38fn_runner: q38fn_runner.c ../common/glm5_bpe.h ../common/glm53f_safetensors.h \
		../common/q38fn_arch.h ../common/q38fn_runtime.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

q38fn_inspect: q38fn_inspect.c ../common/glm53f_safetensors.h \
		../common/q38fn_arch.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

ngram_pipeline_probe: ngram_pipeline_probe.c ../common/q38fn_ngram_pipeline.c \
		../common/q38fn_ngram_pipeline.h ../common/q38fn_arch.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(THREAD_FLAGS) ngram_pipeline_probe.c \
		../common/q38fn_ngram_pipeline.c -o $@

test_ngram_pipeline: test_ngram_pipeline.c ../common/q38fn_ngram_pipeline.c \
		../common/q38fn_ngram_pipeline.h ../common/q38fn_ngram_transport.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(THREAD_FLAGS) test_ngram_pipeline.c \
		../common/q38fn_ngram_pipeline.c -o $@

test_q38fn_runtime: test_q38fn_runtime.c ../common/q38fn_runtime.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

test_q38fn_lowbit: test_q38fn_lowbit.c ../common/q38fn_lowbit.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

test_q38fn_tp_layout: test_q38fn_tp_layout.c ../common/q38fn_tp_layout.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

test_q38fn_tp_blob: test_q38fn_tp_blob.c ../common/q38fn_tp_blob.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

test_q38fn_tp_runtime: test_q38fn_tp_runtime.c ../common/q38fn_tp_runtime.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

test_q38fn_spec: test_q38fn_spec.c ../common/q38fn_spec.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

test_q38fn_tp_runtime_4: test_q38fn_tp_runtime.c ../common/q38fn_tp_runtime.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_TP_RANKS=4 $< -o $@ -lm

test_q38fn_tp_layout_4: test_q38fn_tp_layout.c ../common/q38fn_tp_layout.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_TP_RANKS=4 $< -o $@

test_q38fn_tp_layout_6: test_q38fn_tp_layout.c ../common/q38fn_tp_layout.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_TP_RANKS=6 $< -o $@

test: test_ngram_pipeline test_q38fn_runtime test_q38fn_lowbit test_q38fn_tp_layout test_q38fn_tp_layout_4 test_q38fn_tp_blob test_q38fn_tp_runtime test_q38fn_tp_runtime_4 test_q38fn_spec
	./test_ngram_pipeline
	./test_q38fn_runtime
	./test_q38fn_lowbit
	./test_q38fn_tp_layout
	./test_q38fn_tp_layout_4
	./test_q38fn_tp_blob
	./test_q38fn_tp_runtime
	./test_q38fn_tp_runtime_4
	./test_q38fn_spec

test_target_code: test_target_fibonacci.c
	$(CC) $(CFLAGS) $< -o test_target_fibonacci
	@test "$$($(CURDIR)/test_target_fibonacci)" = "0 1 1 2 3 5 8 13 21 34"

# Fugaku-only compile check; load lang/tcsds-1.2.43 before invoking this.
utofu_source: ../a64fx/q38fn/ngram_utofu_source.c \
		../a64fx/q38fn/ngram_utofu_source.h ../common/q38fn_ngram_transport.h
	$(CC) -Nclang -O2 -Wall -Wextra -Wpedantic -march=armv8.2-a+sve \
		-I../common -I../a64fx/utofu-tests -pthread -c \
		../a64fx/q38fn/ngram_utofu_source.c -o ngram_utofu_source.o

utofu_probe: ../a64fx/q38fn/ngram_utofu_probe.c \
		../a64fx/q38fn/ngram_utofu_source.c \
		../a64fx/q38fn/ngram_utofu_source.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -I../a64fx/q38fn -pthread \
		../a64fx/q38fn/ngram_utofu_probe.c \
	../a64fx/q38fn/ngram_utofu_source.c -o ngram_utofu_probe -ltofucom

# MPI-linked variants are needed by Fugaku's mpiexec to start one pure-uTofu
# application process per allocated rank.  The transport itself remains MPI-free.
utofu_probe_mpi: ../a64fx/q38fn/ngram_utofu_probe.c \
		../a64fx/q38fn/ngram_utofu_source.c \
		../a64fx/q38fn/ngram_utofu_source.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_USE_MPI -I../a64fx/q38fn -pthread \
		../a64fx/q38fn/ngram_utofu_probe.c \
		../a64fx/q38fn/ngram_utofu_source.c -o ngram_utofu_probe_mpi -ltofucom

utofu_real_probe: ../a64fx/q38fn/ngram_utofu_real_probe.c \
		../a64fx/q38fn/ngram_utofu_source.c ../a64fx/q38fn/ngram_owner_hbm.c \
		../a64fx/q38fn/ngram_utofu_source.h ../a64fx/q38fn/ngram_owner_hbm.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -I../a64fx/q38fn -pthread \
		../a64fx/q38fn/ngram_utofu_real_probe.c \
		../a64fx/q38fn/ngram_utofu_source.c \
		../a64fx/q38fn/ngram_owner_hbm.c -fopenmp -o ngram_utofu_real_probe -ltofucom

utofu_pipeline_probe: ../a64fx/q38fn/ngram_utofu_pipeline_probe.c \
		../a64fx/q38fn/ngram_utofu_source.c ../a64fx/q38fn/ngram_owner_hbm.c \
		../common/q38fn_ngram_pipeline.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -I../a64fx/q38fn -pthread \
		../a64fx/q38fn/ngram_utofu_pipeline_probe.c \
		../a64fx/q38fn/ngram_utofu_source.c ../a64fx/q38fn/ngram_owner_hbm.c \
	../common/q38fn_ngram_pipeline.c -fopenmp -o ngram_utofu_pipeline_probe -ltofucom

utofu_pipeline_probe_mpi: ../a64fx/q38fn/ngram_utofu_pipeline_probe.c \
		../a64fx/q38fn/ngram_utofu_source.c ../a64fx/q38fn/ngram_owner_hbm.c \
		../common/q38fn_ngram_pipeline.c
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_USE_MPI -I../a64fx/q38fn -pthread \
		../a64fx/q38fn/ngram_utofu_pipeline_probe.c \
		../a64fx/q38fn/ngram_utofu_source.c ../a64fx/q38fn/ngram_owner_hbm.c \
		../common/q38fn_ngram_pipeline.c -fopenmp -o ngram_utofu_pipeline_probe_mpi -ltofucom

ngram_stage:
	$(CC) $(CFLAGS) $(CPPFLAGS) ../a64fx/q38fn/ngram_stage.c \
		-o ngram_stage

ngram_stage_mpi: ../a64fx/q38fn/ngram_stage.c
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -DQ38FN_USE_MPI \
		-DQ38FN_STAGE_PARALLEL -fopenmp ../a64fx/q38fn/ngram_stage.c -o ngram_stage_mpi

# Fugaku-only resident HBM stream/lookup probe. Use the native compiler:
#   make -C q38fn hbm_probe CC='fcc -Nclang'
hbm_probe: ngram_hbm_probe.c ../common/q38fn_arch.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -fopenmp ngram_hbm_probe.c -o ngram_hbm_probe -lm

lowbit_probe: q38fn_lowbit_probe.c ../common/q38fn_lowbit.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -I../a64fx/k3 -fopenmp $< -o q38fn_lowbit_probe -lm

lowbit_plan: q38fn_lowbit_plan.c ../common/q38fn_lowbit.h \
		../common/q38fn_tp_layout.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o q38fn_lowbit_plan -lm

owner_load_probe: ../a64fx/q38fn/ngram_owner_load_probe.c \
		../a64fx/q38fn/ngram_owner_hbm.c ../a64fx/q38fn/ngram_owner_hbm.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -I../a64fx/q38fn -fopenmp \
		../a64fx/q38fn/ngram_owner_load_probe.c \
		../a64fx/q38fn/ngram_owner_hbm.c -o owner_load_probe -lm

clean:
	rm -f q38fn_runner q38fn_pp_runner q38fn_stage_mpi q38fn_tp_stage_mpi q38fn_tp4_stage_mpi q38fn_tp_upgrade_mpi q38fn_tp_hc_repack_mpi q38fn_tp_reconstruct_mpi q38fn_tp_runner q38fn_tp4_runner q38fn_tp4_blob_probe q38fn_tp_runner_utofu q38fn_tp_runner_utofu_fapp q38fn_tp4_runner_utofu q38fn_tp4_runner_utofu_fapp q38fn_inspect ngram_pipeline_probe test_ngram_pipeline test_ngram_pipeline_asan test_q38fn_runtime test_q38fn_lowbit test_q38fn_tp_layout test_q38fn_tp_layout_4 test_q38fn_tp_blob test_q38fn_tp_runtime test_q38fn_tp_runtime_4 ngram_utofu_source.o ngram_utofu_probe ngram_utofu_probe_mpi ngram_utofu_real_probe ngram_utofu_pipeline_probe ngram_utofu_pipeline_probe_mpi ngram_stage ngram_stage_mpi owner_load_probe
	rm -f compare_f32 decode_trace ngram_hbm_probe q38fn_lowbit_probe tsan_probe
	rm -f q38fn_lowbit_plan test_q38fn_spec
	rm -f test_ngram_pipeline_dbg test_ngram_pipeline_san \
		test_ngram_pipeline_san_check test_ngram_pipeline_san_run \
		test_ngram_pipeline_tsan test_ngram_pipeline_tsan_check \
		test_ngram_pipeline_tsan_new test_target_fibonacci
