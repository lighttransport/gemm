CC = fcc
CFLAGS = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
LDFLAGS = -lm

all: bench_norm

bench_norm: bench_norm.c norm_sve.c sve_math.h norm_sve.h
	$(CC) $(CFLAGS) -o $@ bench_norm.c norm_sve.c $(LDFLAGS)

bench_norm_omp: bench_norm.c norm_sve.c sve_math.h norm_sve.h
	$(CC) $(CFLAGS) -Kopenmp -o $@ bench_norm.c norm_sve.c $(LDFLAGS)

clean:
	rm -f bench_norm bench_norm_omp
