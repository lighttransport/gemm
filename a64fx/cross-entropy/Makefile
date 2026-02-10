CC = fcc
CFLAGS = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
LDFLAGS = -lm

all: bench_cross_entropy

bench_cross_entropy: bench_cross_entropy.c cross_entropy_sve.c sve_math.h cross_entropy_sve.h
	$(CC) $(CFLAGS) -o $@ bench_cross_entropy.c cross_entropy_sve.c $(LDFLAGS)

bench_cross_entropy_omp: bench_cross_entropy.c cross_entropy_sve.c sve_math.h cross_entropy_sve.h
	$(CC) $(CFLAGS) -Kopenmp -o $@ bench_cross_entropy.c cross_entropy_sve.c $(LDFLAGS)

clean:
	rm -f bench_cross_entropy bench_cross_entropy_omp
