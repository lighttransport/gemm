CC  = fcc
AS  = fcc
CFLAGS  = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
ASFLAGS = -Nclang -O3 -march=armv8.2-a+sve
LDFLAGS = -lm

OBJS = sincos_poly.o sincos_ftmad.o rope.o
FUSED_OBJS = micro_fused_proj_rope.o micro_fused_proj_rope_f16.o micro_fused_proj_rope_f16_6x2.o fused_rope_qk.o

all: bench bench_fused

bench: bench.c $(OBJS) sincos.h
	$(CC) $(CFLAGS) -o $@ bench.c $(OBJS) $(LDFLAGS)

bench_fused: bench_fused.c $(FUSED_OBJS) fused_rope_qk.h
	$(CC) $(CFLAGS) -o $@ bench_fused.c $(FUSED_OBJS) $(LDFLAGS)

sincos_poly.o: sincos_poly.S
	$(AS) $(ASFLAGS) -c -o $@ $<

sincos_ftmad.o: sincos_ftmad.S
	$(AS) $(ASFLAGS) -c -o $@ $<

rope.o: rope.S
	$(AS) $(ASFLAGS) -c -o $@ $<

micro_fused_proj_rope.o: micro_fused_proj_rope.S
	$(AS) $(ASFLAGS) -c -o $@ $<

micro_fused_proj_rope_f16.o: micro_fused_proj_rope_f16.S
	$(AS) $(ASFLAGS) -c -o $@ $<

micro_fused_proj_rope_f16_6x2.o: micro_fused_proj_rope_f16_6x2.S
	$(AS) $(ASFLAGS) -c -o $@ $<

fused_rope_qk.o: fused_rope_qk.c fused_rope_qk.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f bench bench_fused *.o

.PHONY: all clean
