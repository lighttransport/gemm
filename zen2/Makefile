CC      = clang
CFLAGS  = -O3 -march=znver2 -mavx2 -mfma -ffp-contract=fast -Wall
ASFLAGS = -c
TARGET  = bench

OBJS = bench.o gemm_driver.o gemm_pack.o gemm_kernel.o

FLASH_OBJS = bench_flash.o flash_attention.o gemm_pack.o gemm_kernel.o

all: $(TARGET) bench_flash

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

bench_flash: $(FLASH_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

bench.o: bench.c gemm.h gemm_pack.h
	$(CC) $(CFLAGS) -c -o $@ $<

bench_flash.o: bench_flash.c flash_attention.h
	$(CC) $(CFLAGS) -c -o $@ $<

flash_attention.o: flash_attention.c flash_attention.h gemm.h gemm_pack.h
	$(CC) $(CFLAGS) -c -o $@ $<

gemm_driver.o: gemm_driver.c gemm.h gemm_pack.h
	$(CC) $(CFLAGS) -c -o $@ $<

gemm_pack.o: gemm_pack.c gemm_pack.h gemm.h
	$(CC) $(CFLAGS) -c -o $@ $<

gemm_kernel.o: gemm_kernel.S
	$(CC) $(ASFLAGS) -o $@ $<

clean:
	rm -f *.o $(TARGET) bench_flash

.PHONY: all clean
