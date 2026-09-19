CC ?= cc
CFLAGS ?= -O3 -std=c11
WARN = -Wall -Wextra -Wpedantic
OMP = -fopenmp
LDLIBS = -lm
BUILD_TMP ?= $(CURDIR)/tmp
export TMPDIR = $(BUILD_TMP)

OBJS = dspark.o dspark_kernels.o

.PHONY: all test validate clean

all: test_dspark validate_dspark bench_dspark

%.o: %.c dspark.h dspark_internal.h
	mkdir -p $(BUILD_TMP)
	$(CC) $(CFLAGS) $(WARN) $(OMP) -I../.. -c $< -o $@

test_dspark: test_dspark.c $(OBJS)
	$(CC) $(CFLAGS) $(WARN) $(OMP) -I../.. $^ $(LDLIBS) -o $@

validate_dspark: validate_dspark.c $(OBJS)
	$(CC) $(CFLAGS) $(WARN) $(OMP) -I../.. $^ $(LDLIBS) -o $@

bench_dspark: bench_dspark.c $(OBJS)
	$(CC) $(CFLAGS) $(WARN) $(OMP) -I../.. $^ $(LDLIBS) -o $@

test: test_dspark
	./test_dspark

validate: validate_dspark
	@test -n "$(DRAFT_DIR)" -a -n "$(TARGET_DIR)" || \
	  (echo 'set DRAFT_DIR=... TARGET_DIR=...' >&2; exit 2)
	./validate_dspark --headers "$(DRAFT_DIR)" "$(TARGET_DIR)"

clean:
	rm -f $(OBJS) test_dspark validate_dspark bench_dspark
