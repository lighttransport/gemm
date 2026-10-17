CC = gcc
CFLAGS = -O3 -Wall -Wextra -march=native -D_GNU_SOURCE -fopenmp
LDLIBS = -ldl -lm -lpthread
TASK_TMP := $(abspath ../../tmp)
export TMPDIR := $(TASK_TMP)
OBJECTS = test_hip_flux3.o hip_flux3_runner.o hip_flux2_vae.o rocew.o
.PHONY: all clean
all: test_hip_flux3
$(OBJECTS) hip_flux3_kernels.inc: | $(TASK_TMP)
$(TASK_TMP):
	mkdir -p "$@"
hip_flux3_kernels.inc: hip_flux3_kernels.hip
	python3 -c 'import json,pathlib; p=pathlib.Path("$<"); print("static const char flux3_kernels[] ="); [print(json.dumps(s)) for s in p.read_text().splitlines(keepends=True)]; print(";")' > $@
hip_flux3_runner.o: hip_flux3_runner.c hip_flux3_runner.h hip_flux3_kernels.inc ../../common/safetensors.h ../hip_runner_common.h ../rocew.h
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<
test_hip_flux3.o: test_hip_flux3.c hip_flux3_runner.h ../flux2/hip_flux2_runner.h
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<
hip_flux2_vae.o: ../flux2/hip_flux2_runner.c
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<
rocew.o: ../rocew.c ../rocew.h
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<
test_hip_flux3: $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) $(LDLIBS)
-include $(OBJECTS:.o=.d)
clean:
	rm -f test_hip_flux3 hip_flux3_kernels.inc $(OBJECTS) $(OBJECTS:.o=.d)
