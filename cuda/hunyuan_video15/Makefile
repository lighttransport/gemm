ROOT := ../..
NATIVE ?= $(ROOT)/tmp/hunyuan-video15-native
BUILD ?= $(NATIVE)/build
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic
CPPFLAGS += -I$(NATIVE)/include
LDLIBS += -L$(BUILD)/bin -Wl,-rpath,'$$ORIGIN/../../tmp/hunyuan-video15-native/build/bin' -lstable-diffusion -lm
.PHONY: all setup test clean
all: test_cuda_hunyuan_video15
setup:
	python3 setup_native.py
test_cuda_hunyuan_video15: test_cuda_hunyuan_video15.c hunyuan_video15.c hunyuan_video15.h progress.h
	$(CC) $(CPPFLAGS) $(CFLAGS) test_cuda_hunyuan_video15.c hunyuan_video15.c $(LDLIBS) -o $@
test: all progress-probe
	./test_cuda_hunyuan_video15 --validate --task i2v --image portrait.png --prompt smile
	! ./test_cuda_hunyuan_video15 --validate --task t2v --preset fast12 --prompt smile
	! ./test_cuda_hunyuan_video15 --validate --task i2v --prompt smile
	! ./test_cuda_hunyuan_video15 --validate --task t2v --frames 80 --prompt smile
	! ./test_cuda_hunyuan_video15 --validate --task t2v --seed 12junk --prompt smile
	./test_cuda_hunyuan_video15_progress
clean:
	rm -f test_cuda_hunyuan_video15 test_cuda_hunyuan_video15_siglip test_cuda_hunyuan_video15_vae test_cuda_hunyuan_video15_qwen test_cuda_hunyuan_video15_vae_decode test_cuda_hunyuan_video15_dit
	rm -f test_cuda_hunyuan_video15_attention
	rm -f test_cuda_hunyuan_video15_progress
	rm -f test_cuda_hunyuan_video15_vae_tiling

.PHONY: spatial-tiling-probe
spatial-tiling-probe: test_cuda_hunyuan_video15_vae_tiling
test_cuda_hunyuan_video15_vae_tiling: vae_tiling_probe.cpp vae_spatial_tiling.hpp
	$(CXX) -DGGML_MAX_NAME=160 -std=c++17 -O2 -Wall -Wextra -Wpedantic -I$(NATIVE)/src -I$(NATIVE)/ggml/include $(CPPFLAGS) $< $(LDLIBS) -o $@

.PHONY: progress-probe
progress-probe: test_cuda_hunyuan_video15_progress
test_cuda_hunyuan_video15_progress: progress_probe.cpp progress.h
	$(CXX) -DGGML_MAX_NAME=160 -std=c++17 -O2 -Wall -Wextra -Wpedantic -I$(NATIVE)/src -I$(NATIVE)/ggml/include $(CPPFLAGS) $< $(LDLIBS) -o $@

.PHONY: attention-probe
attention-probe: test_cuda_hunyuan_video15_attention
test_cuda_hunyuan_video15_attention: attention_probe.cpp cuda_attention.h
	$(CXX) -DGGML_MAX_NAME=160 -std=c++17 -O2 -Wall -Wextra -Wpedantic -I$(NATIVE)/ggml/include $(CPPFLAGS) $< $(LDLIBS) -o $@

.PHONY: probe
probe: test_cuda_hunyuan_video15_siglip
test_cuda_hunyuan_video15_siglip: siglip_probe.cpp siglip.hpp $(NATIVE)/src/model/te/hv15_siglip.hpp
	$(CXX) -DGGML_MAX_NAME=160 -std=c++17 -O2 -Wall -Wextra -I$(NATIVE)/src -I$(NATIVE)/thirdparty -I$(NATIVE)/ggml/include -I$(NATIVE)/ggml/src $(CPPFLAGS) $< $(LDLIBS) -o $@

.PHONY: vae-probe
vae-probe: test_cuda_hunyuan_video15_vae
test_cuda_hunyuan_video15_vae: vae_probe.cpp $(NATIVE)/src/model/vae/hunyuan_vae.hpp
	$(CXX) -DGGML_MAX_NAME=160 -std=c++17 -O2 -Wall -Wextra -I$(NATIVE)/src -I$(NATIVE)/thirdparty -I$(NATIVE)/ggml/include -I$(NATIVE)/ggml/src $(CPPFLAGS) $< $(LDLIBS) -o $@

.PHONY: qwen-probe
qwen-probe: test_cuda_hunyuan_video15_qwen
test_cuda_hunyuan_video15_qwen: qwen_probe.cpp $(NATIVE)/src/conditioning/conditioner.hpp
	$(CXX) -DGGML_MAX_NAME=160 -std=c++17 -O2 -Wall -Wextra -I$(NATIVE)/src -I$(NATIVE)/thirdparty -I$(NATIVE)/ggml/include -I$(NATIVE)/ggml/src $(CPPFLAGS) $< $(LDLIBS) -o $@

.PHONY: vae-decode-probe
vae-decode-probe: test_cuda_hunyuan_video15_vae_decode
test_cuda_hunyuan_video15_vae_decode: vae_decode_probe.cpp $(NATIVE)/src/model/vae/hunyuan_vae.hpp
	$(CXX) -DGGML_MAX_NAME=160 -std=c++17 -O2 -Wall -Wextra -I$(NATIVE)/src -I$(NATIVE)/thirdparty -I$(NATIVE)/ggml/include -I$(NATIVE)/ggml/src $(CPPFLAGS) $< $(LDLIBS) -o $@

.PHONY: dit-probe
dit-probe: test_cuda_hunyuan_video15_dit
test_cuda_hunyuan_video15_dit: dit_probe.cpp vision_projection.hpp $(NATIVE)/src/model/diffusion/hunyuan.hpp
	$(CXX) -DGGML_MAX_NAME=160 -std=c++17 -O2 -Wall -Wextra -I$(NATIVE)/src -I$(NATIVE)/thirdparty -I$(NATIVE)/ggml/include -I$(NATIVE)/ggml/src $(CPPFLAGS) $< $(LDLIBS) -o $@
