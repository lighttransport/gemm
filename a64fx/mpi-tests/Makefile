MPICC ?= mpifccpx
CFLAGS ?= -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall -Wextra -std=c11
LDFLAGS ?=

all: mpi_noncontig_probe

mpi_noncontig_probe: mpi_noncontig_probe.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f mpi_noncontig_probe

.PHONY: all clean
