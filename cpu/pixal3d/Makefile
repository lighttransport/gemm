CC ?= cc
CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp -fPIC
CFLAGS ?= -O3 -Wall -Wextra -fPIC
CFLAGS += -MMD -MP
LDLIBS = -lopenblas -ldl -lm -lpthread -fopenmp
OPENCV_ROOT ?= $(abspath ../../ref/pixal3d/deps/opencv/usr)
CXXFLAGS += -MMD -MP -I$(OPENCV_ROOT)/include/opencv4
LDLIBS += -L$(OPENCV_ROOT)/lib/x86_64-linux-gnu -Wl,-rpath,$(OPENCV_ROOT)/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc
export TMPDIR := $(abspath ../../tmp/pixal3d)
OBJECTS = io.o math.o engine.o pointwise.o flow.o conditioning.o decoders.o preprocess.o json.o pipeline.o geometry.o remesh.o simplify.o unwrap.o postprocess.o inpaint_opencv.o glb.o lightrt.o xatlas.o
all: libpixal3d.so pixal3d
pixal3d: main.o libpixal3d.so
	$(CXX) -o $@.new main.o -L. -lpixal3d -Wl,-rpath,'$$ORIGIN' $(LDLIBS)
	mv $@.new $@
validation: libpixal3d_validation.so
test: test_pixal3d
	./test_pixal3d
test_pixal3d: test_pixal3d.o libpixal3d.so
	$(CC) -o $@.new test_pixal3d.o -L. -lpixal3d -Wl,-rpath,'$$ORIGIN' -lm
	mv $@.new $@
test_pixal3d.o: test_pixal3d.c ../../common/pixal3d.h | $(TMPDIR)
	$(CC) $(CFLAGS) -std=c11 -Wpedantic -c $< -o $@
libpixal3d_validation.so: validation.o libpixal3d.so
	$(CXX) -shared -o $@.new validation.o -L. -lpixal3d -Wl,-rpath,'$$ORIGIN' $(LDLIBS)
	mv $@.new $@
libpixal3d.so: $(OBJECTS)
	$(CXX) -shared -o $@.new $^ $(LDLIBS)
	mv $@.new $@
%.o: %.cc engine.hh ../../common/pixal3d.h | $(TMPDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
io.o: io.c ../../common/safetensors.h | $(TMPDIR)
	$(CC) $(CFLAGS) -c $< -o $@
geometry.o: geometry.c ../../common/trellis2_fdg_mesh.h | $(TMPDIR)
	$(CC) $(CFLAGS) -c $< -o $@
lightrt.o: ../../common/lightrt.cc ../../common/lightrt.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@
xatlas.o: ../../common/xatlas.cc ../../common/xatlas.h
	$(CXX) $(CXXFLAGS) -Wno-type-limits -c $< -o $@
$(TMPDIR):
	mkdir -p $@
-include $(OBJECTS:.o=.d) main.d validation.d test_pixal3d.d
clean:
	rm -f *.o *.d libpixal3d.so libpixal3d_validation.so pixal3d test_pixal3d
.PHONY: all clean validation test
