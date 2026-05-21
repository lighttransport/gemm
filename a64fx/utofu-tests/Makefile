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

all: tofu_topo_helper tofu_put_demo ring_attn_bench tni_stripe_bench ring_attn_overlap ring_attn_async moe_dispatch_bench allgather_bench reducescatter_bench

# helper uses MPI + uTofu
tofu_topo_helper: tofu_topo_helper.c tofu_demo.h
	$(MPICC) $(MPICFLAGS) -o $@ $< $(TOFU)

# pure uTofu app: NO MPI
tofu_put_demo: tofu_put_demo.c tofu_demo.h
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

# all-gather benchmark (naive / multi-TNI / ring / recursive-doubling): pure uTofu, NO MPI
allgather_bench: allgather_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# reduce-scatter benchmark (naive / multi-TNI / ring + AG composite vs fused tree): NO MPI
reducescatter_bench: reducescatter_bench.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

# ring-attention comm/compute OVERLAP validation (real math): NO MPI, OpenMP
ring_attn_overlap: ring_attn_overlap.c tofu_demo.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(TOFU)

# ring-attention async across-step driver + HW barrier: NO MPI, OpenMP, libhwb
ring_attn_async: ring_attn_async.c tofu_demo.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(TOFU) -lhwb

clean:
	rm -f tofu_topo_helper tofu_put_demo ring_attn_bench tni_stripe_bench ring_attn_overlap ring_attn_async moe_dispatch_bench allgather_bench reducescatter_bench tofu_topo.txt

.PHONY: all clean
