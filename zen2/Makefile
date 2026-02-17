CC      = clang
CFLAGS  = -O3 -march=znver2 -mavx2 -mfma -ffp-contract=fast -Wall
ASFLAGS = -c
TARGET  = bench

OBJS = bench.o gemm_driver.o gemm_pack.o gemm_kernel.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

bench.o: bench.c gemm.h gemm_pack.h
	$(CC) $(CFLAGS) -c -o $@ $<

gemm_driver.o: gemm_driver.c gemm.h gemm_pack.h
	$(CC) $(CFLAGS) -c -o $@ $<

gemm_pack.o: gemm_pack.c gemm_pack.h gemm.h
	$(CC) $(CFLAGS) -c -o $@ $<

gemm_kernel.o: gemm_kernel.S
	$(CC) $(ASFLAGS) -o $@ $<

clean:
	rm -f *.o $(TARGET)

.PHONY: all clean
