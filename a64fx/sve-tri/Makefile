CC  = fcc
AS  = fcc
CFLAGS  = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
ASFLAGS = -Nclang -O3 -march=armv8.2-a+sve
LDFLAGS = -lm

OBJS = sincos_poly.o sincos_ftmad.o rope.o

all: bench

bench: bench.c $(OBJS) sincos.h
	$(CC) $(CFLAGS) -o $@ bench.c $(OBJS) $(LDFLAGS)

sincos_poly.o: sincos_poly.S
	$(AS) $(ASFLAGS) -c -o $@ $<

sincos_ftmad.o: sincos_ftmad.S
	$(AS) $(ASFLAGS) -c -o $@ $<

rope.o: rope.S
	$(AS) $(ASFLAGS) -c -o $@ $<

clean:
	rm -f bench *.o

.PHONY: all clean
