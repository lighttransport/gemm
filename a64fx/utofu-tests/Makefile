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

all: tofu_topo_helper tofu_put_demo

# helper uses MPI + uTofu
tofu_topo_helper: tofu_topo_helper.c tofu_demo.h
	$(MPICC) $(MPICFLAGS) -o $@ $< $(TOFU)

# pure uTofu app: NO MPI
tofu_put_demo: tofu_put_demo.c tofu_demo.h
	$(CC) $(CFLAGS) -o $@ $< $(TOFU)

clean:
	rm -f tofu_topo_helper tofu_put_demo tofu_topo.txt

.PHONY: all clean
