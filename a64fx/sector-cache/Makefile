# Makefile for A64FX Sector Cache Test

# Use fcc for native A64FX execution
CC = fcc
CFLAGS = -Nclang -Ofast -march=armv8.2-a+sve -ffj-ocl -ffj-no-largepage
CFLAGS_PMU = -Nclang -O2 -march=armv8.2-a+sve -ffj-no-largepage
CFLAGS_PRAGMA = -Nnoclang -O2 -Kocl,hpctag
LDFLAGS = -lm

TARGET = bench_sector_cache

SRCS = bench_sector_cache.c

.PHONY: all clean pmu pmu_fapp l1_reuse l1_reuse_fapp l1_pragma l1_pragma_fapp l2_conflict l2_conflict_fapp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# PMU verification (no fapp, cycle count only)
pmu: test_sector_tag_pmu.c
	$(CC) $(CFLAGS_PMU) -o test_sector_tag_pmu $< $(LDFLAGS)

# PMU verification (with fapp instrumentation)
pmu_fapp: test_sector_tag_pmu.c
	$(CC) $(CFLAGS_PMU) -DUSE_FAPP -o test_sector_tag_pmu_fapp $< $(LDFLAGS)

# L1 reuse test (no fapp)
l1_reuse: test_sector_l1_reuse.c
	$(CC) $(CFLAGS_PMU) -o test_sector_l1_reuse $< $(LDFLAGS)

# L1 reuse test (with fapp)
l1_reuse_fapp: test_sector_l1_reuse.c
	$(CC) $(CFLAGS_PMU) -DUSE_FAPP -o test_sector_l1_reuse_fapp $< $(LDFLAGS)

# L1 reuse test with FCC pragmas (no fapp)
l1_pragma: test_sector_l1_pragma.c
	$(CC) $(CFLAGS_PRAGMA) -o test_sector_l1_pragma $< $(LDFLAGS)

# L1 reuse test with FCC pragmas (with fapp)
l1_pragma_fapp: test_sector_l1_pragma.c
	$(CC) $(CFLAGS_PRAGMA) -DUSE_FAPP -o test_sector_l1_pragma_fapp $< $(LDFLAGS)

# L2 way-conflict test (no fapp)
l2_conflict: test_sector_l2_conflict.c
	$(CC) $(CFLAGS_PRAGMA) -o test_sector_l2_conflict $< $(LDFLAGS)

# L2 way-conflict test (with fapp)
l2_conflict_fapp: test_sector_l2_conflict.c
	$(CC) $(CFLAGS_PRAGMA) -DUSE_FAPP -o test_sector_l2_conflict_fapp $< $(LDFLAGS)

# Assembly output for pragma version (inspect __jwe_xset_sccr calls)
l1_pragma_asm: test_sector_l1_pragma.c
	$(CC) $(CFLAGS_PRAGMA) -S -o test_sector_l1_pragma.s $<

clean:
	rm -f $(TARGET) test_sector_tag_pmu test_sector_tag_pmu_fapp test_sector_l1_reuse test_sector_l1_reuse_fapp
	rm -f test_sector_l1_pragma test_sector_l1_pragma_fapp test_sector_l1_pragma.s
	rm -rf fapp_output fapp_pa1 fapp_pa2 prof_sc1 prof_sc3 prof_l1 prof_l1_sc1
