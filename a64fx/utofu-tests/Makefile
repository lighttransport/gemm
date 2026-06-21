# MPI-free uTofu multi-node Put demo (A64FX / Fugaku)
#
#   tofu_topo_helper : MPI + uTofu, writes tofu_topo.txt (the only MPI program)
#   tofu_put_demo    : pure uTofu, NO MPI -- reads the file and communicates
#
# Native A64FX node toolchain (this Fugaku node, modules lang/tcsds + LLVM):
#   pure uTofu app -> fcc -Nclang   (no MPI)
#   MPI helper     -> mpiclang      (OSS-CN LLVM MPI wrapper; mpifcc is broken
#                                    here because OPAL_PREFIX points at OSS-CN)
# On a login/cross node use the cross compilers instead (fccpx / mpifccpx) and
# add -lmpi to the helper.

CC        = fcc
CFLAGS    = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall
MPICC     = mpiclang
MPICFLAGS = -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall
TOFU      = -ltofucom

all: tofu_topo_helper tofu_put_demo utofu_init_probe ring_attn_bench tni_stripe_bench ring_attn_overlap ring_attn_async moe_dispatch_bench moe_async_bench allgather_bench reducescatter_bench pp_handoff_bench p2p_latbw_bench ep_prefill_synth

# helper uses MPI + uTofu
tofu_topo_helper: tofu_topo_helper.c tofu_demo.h
	$(MPICC) $(MPICFLAGS) -o $@ $< $(TOFU)

# pure uTofu app: NO MPI
tofu_put_demo: tofu_put_demo.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# pure uTofu init probe: NO MPI, logs each rank to UTOFU_PROBE_DIR
utofu_init_probe: utofu_init_probe.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# pure uTofu ring-attention decode cost benchmark: NO MPI, OpenMP for mem stream
ring_attn_bench: ring_attn_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(TOFU)

# multi-TNI striping microbench: pure uTofu, NO MPI
tni_stripe_bench: tni_stripe_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# MoE expert-dispatch all-to-all benchmark: pure uTofu, NO MPI
moe_dispatch_bench: moe_dispatch_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# MoE dispatch+combine all-to-all || expert SwiGLU compute OVERLAP: NO MPI, OpenMP, libhwb
moe_async_bench: moe_async_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(TOFU) -lhwb

# all-gather benchmark (naive / multi-TNI / ring / recursive-doubling): pure uTofu, NO MPI
allgather_bench: allgather_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# reduce-scatter benchmark (naive / multi-TNI / ring + AG composite vs fused tree): NO MPI
reducescatter_bench: reducescatter_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# pipeline-parallel handoff benchmark (point-to-point stage send/recv chain): NO MPI
pp_handoff_bench: pp_handoff_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# point-to-point latency/bandwidth bench (async put/get/armw, torus-distance sweep): NO MPI
p2p_latbw_bench: p2p_latbw_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# sum-all-reduce primitive (recursive doubling) correctness + latency: NO MPI
tp_allreduce_test: tp_allreduce_test.c tofu_demo.h tp_allreduce.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# communication-only EP prefill roofline: allreduce and synthetic token dispatch, NO MPI
ep_prefill_synth: ep_prefill_synth.c tofu_demo.h tp_allreduce.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU) -lm

# pure HBM bandwidth + measured-comm floor for MoE EP decode (no gemm): OpenMP, libnuma, NO MPI
# comm term is folded in from tp_allreduce_test (run that separately under mpiexec)
ep_bw_bench: ep_bw_bench.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $< -lm -lnuma

# ring-attention comm/compute OVERLAP validation (real math): NO MPI, OpenMP
ring_attn_overlap: ring_attn_overlap.c tofu_demo.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(TOFU)

# ring-attention async across-step driver + HW barrier: NO MPI, OpenMP, libhwb
ring_attn_async: ring_attn_async.c tofu_demo.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(TOFU) -lhwb

clean:
	rm -f tofu_topo_helper tofu_put_demo utofu_init_probe ring_attn_bench tni_stripe_bench ring_attn_overlap ring_attn_async moe_dispatch_bench moe_async_bench allgather_bench reducescatter_bench pp_handoff_bench p2p_latbw_bench ep_prefill_synth tofu_topo.txt

.PHONY: all clean
