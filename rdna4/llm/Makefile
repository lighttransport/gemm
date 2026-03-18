# Makefile for HIP LLM runner (RDNA4)
#
# No hipcc required - kernels compiled at runtime via HIPRTC.
# Only needs gcc + ROCm runtime libraries (libamdhip64.so, libhiprtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

# Sources
ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_llm_runner.c
TEST_SRC = test_hip_llm.c

# Target
TARGET = test_hip_llm

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) hip_llm_runner.h ../rocew.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]"

clean:
	rm -f $(TARGET)

help:
	@echo "HIP LLM Runner (HIPRTC-based, no hipcc needed)"
	@echo "================================================"
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_hip_llm"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_hip_llm <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]"
	@echo ""
	@echo "Example:"
	@echo "  ./test_hip_llm Qwen3-Embedding-0.6B-f16.gguf -t \"Hello world\" -n 4"
	@echo ""
	@echo "Requirements:"
	@echo "  - libamdhip64.so (HIP runtime)"
	@echo "  - libhiprtc.so (HIPRTC runtime compiler)"
	@echo "  - AMD GPU with RDNA4 (gfx1200/gfx1201)"
