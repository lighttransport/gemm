# cpu/svdquant — portable CPU SVDQuant forward validation.
#
# Validates svdquant_cpu.h against the PyTorch reference dumped by
# ref/svdquant/gen_svdquant_ref.py. No external deps beyond libm.

CC      ?= gcc
COMMON   = ../../common
DUMPS    = ../../ref/svdquant/dumps
CFLAGS   = -O2 -std=c11 -Wall -Wextra -I$(COMMON)
LDFLAGS  = -lm

.PHONY: all test clean

all: test_svdquant

test_svdquant: test_svdquant.c svdquant_cpu.h $(COMMON)/npy_io.h
	$(CC) $(CFLAGS) -o $@ test_svdquant.c $(LDFLAGS)

# Generate the reference dumps if missing, then run the test.
$(DUMPS)/manifest.json:
	cd ../../ref/svdquant && python3 gen_svdquant_ref.py --out dumps

test: test_svdquant $(DUMPS)/manifest.json
	./test_svdquant $(DUMPS)

clean:
	rm -f test_svdquant
