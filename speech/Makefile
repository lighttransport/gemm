# speech/: Qwen3-TTS runner (CPU + CUDA) and the ja_align Japanese CTC aligner.
#   make            -> CPU tools in build/
#   make cuda       -> CUDA runner (NVRTC via cuew; no CUDA toolkit link)
#   make test       -> unit tests (no model weights needed)
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall -Wextra
BUILD   ?= build
COMMON  := ../common
CUDADIR := ../cuda
QT      := qwen3_tts
JA      := ja_align

QT_HDRS := $(QT)/qtts_ops.h $(QT)/qtts_codec.h $(QT)/qtts_talker.h $(QT)/qtts_tokenizer.h $(QT)/wav_io.h
JA_HDRS := $(wildcard $(JA)/*.h)

all: $(BUILD)/qwen3_tts $(BUILD)/test_codec $(BUILD)/test_tokenizer

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/qwen3_tts: $(QT)/qwen3_tts_main.c $(QT_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -I$(COMMON) -I$(QT) -o $@ $< -lm

$(BUILD)/test_codec: tests/test_codec.c $(QT_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -I$(COMMON) -I$(QT) -o $@ $< -lm

$(BUILD)/test_tokenizer: tests/test_tokenizer.c $(QT)/qtts_tokenizer.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(COMMON) -I$(QT) -o $@ $< -lm

# ja_align is dependency-free: only its own directory is on the include path.
$(BUILD)/ja_align: $(JA)/ja_align_main.c $(JA_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -I$(JA) -o $@ $< -lm

$(BUILD)/test_ja_ctc: tests/test_ja_ctc.c $(JA_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -I$(JA) -o $@ $< -lm

$(BUILD)/test_ja_dsp: tests/test_ja_dsp.c $(JA_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -I$(JA) -o $@ $< -lm

test: $(BUILD)/test_ja_ctc $(BUILD)/test_ja_dsp
	$(BUILD)/test_ja_ctc
	$(BUILD)/test_ja_dsp

cuda: $(BUILD)/qwen3_tts_cuda

$(BUILD)/qwen3_tts_cuda: $(QT)/qwen3_tts_cuda_main.c $(QT)/qtts_cuda.c $(QT)/qtts_cuda.h $(QT_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -I$(COMMON) -I$(CUDADIR) -I$(QT) -o $@ $(QT)/qwen3_tts_cuda_main.c $(QT)/qtts_cuda.c \
	    $(CUDADIR)/cuew.c -ldl -lm -lpthread

clean:
	rm -rf $(BUILD)

.PHONY: all test cuda clean
