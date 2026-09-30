export TMPDIR := $(abspath ../tmp/speech-compiler)
$(shell mkdir -p $(TMPDIR))
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

QT_HDRS := $(QT)/qtts_clone.h $(QT)/qtts_spk.h $(QT)/qtts_codec_enc.h $(QT)/qtts_ops.h $(QT)/qtts_codec.h $(QT)/qtts_talker.h $(QT)/qtts_tokenizer.h $(QT)/wav_io.h
JA_HDRS := $(wildcard $(JA)/*.h)

all: $(BUILD)/qwen3_tts $(BUILD)/test_codec $(BUILD)/test_tokenizer $(BUILD)/ja_align $(BUILD)/test_w2v2

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

$(BUILD)/test_w2v2: tests/test_w2v2.c $(JA_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -I$(JA) -o $@ $< -lm

$(BUILD)/test_ja_dsp: tests/test_ja_dsp.c $(JA_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -I$(JA) -o $@ $< -lm

test: $(BUILD)/test_ja_ctc $(BUILD)/test_ja_dsp
	$(BUILD)/test_ja_ctc
	$(BUILD)/test_ja_dsp

cuda: $(BUILD)/qwen3_tts_cuda $(BUILD)/tts_ja_cuda $(BUILD)/ja_align_cuda

# CUDA aligner: still self-contained (libcuda / libnvrtc are dlopen'ed), only -ldl
$(BUILD)/ja_align_cuda: $(JA)/ja_align_main.c $(JA_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -DJA_WITH_CUDA -I$(JA) -o $@ $< -ldl -lm

$(BUILD)/qwen3_tts_cuda: $(QT)/qwen3_tts_main.c $(QT)/qtts_cuda.h $(QT)/qtts_cuda_kernels.h $(QT_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -DQTTS_WITH_CUDA -I$(COMMON) -I$(CUDADIR) -I$(QT) -o $@ $(QT)/qwen3_tts_main.c \
	    $(CUDADIR)/cuew.c -ldl -lm -lpthread

$(BUILD)/tts_ja: tts_ja_main.c $(QT_HDRS) $(JA_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -I$(COMMON) -I$(QT) -I$(JA) -o $@ $< -lm

$(BUILD)/tts_ja_cuda: tts_ja_main.c $(QT)/qtts_cuda.h $(QT)/qtts_cuda_kernels.h $(QT_HDRS) $(JA_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -DQTTS_WITH_CUDA -I$(COMMON) -I$(CUDADIR) -I$(QT) -I$(JA) -o $@ $< $(CUDADIR)/cuew.c -ldl -lm -lpthread

clean:
	rm -rf $(BUILD)

.PHONY: all test cuda clean

# Portable graphs compiled against the dynamic HIP driver boundary.
$(BUILD)/qwen3_tts_rocm $(BUILD)/tts_ja_rocm $(BUILD)/ja_align_rocm: ../rdna4/rocew.c ../rdna4/cuda_driver_compat.h

rocm: $(BUILD)/qwen3_tts_rocm $(BUILD)/tts_ja_rocm $(BUILD)/ja_align_rocm
HIP_DEPS = ../rdna4/cuda_driver_compat.h ../rdna4/rocew.h ../rdna4/hip_runner_common.h
$(BUILD)/qwen3_tts_rocm: $(QT)/qwen3_tts_main.c $(QT)/qtts_cuda.h $(QT)/qtts_cuda_kernels.h $(QT_HDRS) $(HIP_DEPS) | $(BUILD)
	$(CC) $(CFLAGS) -DQTTS_WITH_CUDA -DQTTS_WITH_HIP -I$(COMMON) -I$(QT) -o $@ $< ../rdna4/rocew.c -ldl -lm -lpthread
$(BUILD)/tts_ja_rocm: tts_ja_main.c $(QT)/qtts_cuda.h $(QT)/qtts_cuda_kernels.h $(QT_HDRS) $(JA_HDRS) $(HIP_DEPS) | $(BUILD)
	$(CC) $(CFLAGS) -DQTTS_WITH_CUDA -DQTTS_WITH_HIP -I$(COMMON) -I$(QT) -I$(JA) -o $@ $< ../rdna4/rocew.c -ldl -lm -lpthread
$(BUILD)/ja_align_rocm: $(JA)/ja_align_main.c $(JA_HDRS) $(HIP_DEPS) | $(BUILD)
	$(CC) $(CFLAGS) -DJA_WITH_CUDA -DJA_WITH_HIP -I$(JA) -o $@ $< ../rdna4/rocew.c -ldl -lm
.PHONY: rocm
