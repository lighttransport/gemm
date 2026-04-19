COMPILER ?= gcc
ARCH ?= zen2
COMMON = ../../common

ifeq ($(ARCH),zen2)
    ARCH_FLAGS = -march=znver2 -mtune=znver2
else ifeq ($(ARCH),zen3)
    ARCH_FLAGS = -march=znver3 -mtune=znver3
else ifeq ($(ARCH),zen4)
    ARCH_FLAGS = -march=znver4 -mtune=znver4
else ifeq ($(ARCH),native)
    ARCH_FLAGS = -march=native
else
    ARCH_FLAGS = -march=x86-64-v3
endif

ifeq ($(COMPILER),gcc)
    CC = gcc
else ifeq ($(COMPILER),clang)
    CC = clang
else
    $(error Unknown COMPILER=$(COMPILER). Use gcc or clang)
endif

CFLAGS = -O3 $(ARCH_FLAGS) -mavx2 -mfma -mf16c -ffast-math
CFLAGS += -funroll-loops -fomit-frame-pointer -Wall -Wextra -fopenmp
LDFLAGS = -lm -lpthread -fopenmp

ifdef DEBUG
    CFLAGS = -O0 -g $(ARCH_FLAGS) -mavx2 -mfma -mf16c -Wall -Wextra
    CFLAGS += -fsanitize=address -fsanitize=undefined
    LDFLAGS += -fsanitize=address -fsanitize=undefined
endif

HEADERS = $(wildcard $(COMMON)/*.h) sam3_runner.h

.PHONY: all clean
all: test_sam3 verify_patch_embed verify_vit verify_fpn verify_text verify_detr verify_detr_dec verify_dot_score verify_mask verify_final verify_bpe

test_sam3: test_sam3.c sam3_runner.c sam3_clip_bpe.c $(HEADERS) sam3_clip_bpe.h
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ test_sam3.c sam3_runner.c sam3_clip_bpe.c $(LDFLAGS)

verify_patch_embed: verify_patch_embed.c sam3_runner.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_patch_embed.c sam3_runner.c $(LDFLAGS)

verify_vit: verify_vit.c sam3_runner.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_vit.c sam3_runner.c $(LDFLAGS)

verify_fpn: verify_fpn.c sam3_runner.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_fpn.c sam3_runner.c $(LDFLAGS)

verify_text: verify_text.c sam3_runner.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_text.c sam3_runner.c $(LDFLAGS)

verify_detr: verify_detr.c sam3_runner.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_detr.c sam3_runner.c $(LDFLAGS)

verify_detr_dec: verify_detr_dec.c sam3_runner.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_detr_dec.c sam3_runner.c $(LDFLAGS)

verify_dot_score: verify_dot_score.c sam3_runner.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_dot_score.c sam3_runner.c $(LDFLAGS)

verify_mask: verify_mask.c sam3_runner.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_mask.c sam3_runner.c $(LDFLAGS)

verify_final: verify_final.c sam3_runner.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_final.c sam3_runner.c $(LDFLAGS)

verify_bpe: verify_bpe.c sam3_clip_bpe.c sam3_clip_bpe.h
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_bpe.c sam3_clip_bpe.c $(LDFLAGS)

clean:
	rm -f test_sam3 verify_patch_embed verify_vit verify_fpn \
	      verify_text verify_detr verify_detr_dec \
	      verify_dot_score verify_mask verify_final verify_bpe
