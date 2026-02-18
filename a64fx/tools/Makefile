# Makefile for PMU Counter Library
# Lightweight hardware counter access for A64FX via perf_event_open()

COMPILER ?= fcc
CC = $(COMPILER)

ifeq ($(COMPILER),fcc)
    CFLAGS = -Nclang -O2 -march=armv8.2-a+sve
    LDFLAGS =
else ifeq ($(COMPILER),fccpx)
    CFLAGS = -Nclang -O2 -march=armv8.2-a+sve
    LDFLAGS =
else
    CFLAGS = -O2 -march=armv8.2-a+sve -Wall
    LDFLAGS =
endif

AR = ar
ARFLAGS = rcs

# Sources
PMU_SRC = pmu.c
PMU_HDR = pmu.h pmu_events.h
PMU_OBJ = pmu.o

# Targets
LIB     = libpmu.a
EXAMPLE = pmu_example

.PHONY: all lib example clean help

all: lib example

help:
	@echo "PMU Counter Library for A64FX"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build library and example (default)"
	@echo "  lib      - Build libpmu.a static library"
	@echo "  example  - Build pmu_example test binary"
	@echo "  clean    - Remove build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make COMPILER=fcc         # Native compile (default)"
	@echo "  make COMPILER=fccpx       # Cross-compile for compute nodes"
	@echo "  make COMPILER=gcc         # GCC"
	@echo ""
	@echo "Link against libpmu.a in your project:"
	@echo "  $(CC) $(CFLAGS) -I. -o bench bench.c -L. -lpmu"

lib: $(LIB)

example: $(EXAMPLE)

$(LIB): $(PMU_OBJ)
	$(AR) $(ARFLAGS) $@ $^

$(EXAMPLE): pmu_example.c $(LIB)
	$(CC) $(CFLAGS) -o $@ pmu_example.c -L. -lpmu $(LDFLAGS) -lm

$(PMU_OBJ): $(PMU_SRC) $(PMU_HDR)
	$(CC) $(CFLAGS) -c -o $@ $(PMU_SRC)

clean:
	rm -f $(PMU_OBJ) $(LIB) $(EXAMPLE)
