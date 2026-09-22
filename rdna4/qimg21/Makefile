# Native Qwen Image 2.1 denoiser for RDNA4.
# Host code is plain C; HIP and HIPRTC are loaded dynamically through rocew.
CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -Wpedantic -I.. -I../../common
LDFLAGS ?= -ldl -lm -lpthread

TARGET = test_hip_qimg21_native test_hip_qimg21_text test_hip_qimg21_vision test_hip_qimg21_vae test_hip_qimg21_vae_encode test_hip_qimg21_wmma
SRC = test_hip_qimg21_native.c ../rocew.c

.PHONY: all clean probe rocew-probe

all: $(TARGET)

test_hip_qimg21_native: $(SRC) hip_qimg21_runner.h edit_runtime.h edit_kernels.h mma64_kernels.h \
           norm_vector_kernels.h rope_table_kernels.h quant_weights.h \
           ../../common/safetensors.h ../rocew.h ../hip_runner_common.h \
           ../hip_kernels_common.h ../qimg/hip_qimg_kernels.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

test_hip_qimg21_text: test_hip_qimg21_text.c test_hip_qimg21_native.c \
                      text_kernels.h qwen_tokenizer_json.h hip_qimg21_runner.h \
                      ../../common/gguf_loader.h ../../common/bpe_tokenizer.h \
                      ../hip_kernels_common.h ../qimg/hip_qimg_kernels.h
	$(CC) $(CFLAGS) -o $@ test_hip_qimg21_text.c ../rocew.c $(LDFLAGS)

test_hip_qimg21_vision: test_hip_qimg21_vision.c test_hip_qimg21_native.c \
                        hip_qimg21_runner.h joint_layout.h edit_kernels.h \
                        ../hip_kernels_common.h ../qimg/hip_qimg_kernels.h
	$(CC) $(CFLAGS) -o $@ test_hip_qimg21_vision.c ../rocew.c $(LDFLAGS)

test_hip_qimg21_vae: test_hip_qimg21_vae.c test_hip_qimg21_native.c \
                     hip_qimg21_runner.h ../hip_kernels_common.h ../qimg/hip_qimg_kernels.h
	$(CC) $(CFLAGS) -o $@ test_hip_qimg21_vae.c ../rocew.c $(LDFLAGS)

test_hip_qimg21_vae_encode: test_hip_qimg21_vae_encode.c test_hip_qimg21_vae.c \
                            hip_qimg21_runner.h ../hip_kernels_common.h \
                            ../qimg/hip_qimg_kernels.h ../../common/stb_image.h
	$(CC) $(CFLAGS) -o $@ test_hip_qimg21_vae_encode.c ../rocew.c $(LDFLAGS)

test_hip_qimg21_wmma: test_hip_qimg21_wmma.c hip_qimg21_runner.h ../rocew.c ../rocew.h
	$(CC) $(CFLAGS) -o $@ test_hip_qimg21_wmma.c ../rocew.c $(LDFLAGS)

probe: $(TARGET)
	./$(TARGET) --help

rocew-probe: ../rocew_probe.c ../rocew.c ../rocew.h
	$(CC) $(CFLAGS) -o ../rocew_probe ../rocew_probe.c ../rocew.c $(LDFLAGS)

clean:
	rm -f $(TARGET)
