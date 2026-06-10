# Standalone dev harness for the IQ2_XXS MMQ port.
NVCC = nvcc
ARCH = sm_120
TARGET = mmq_iq2xxs_test

all: $(TARGET)

$(TARGET): mmq_iq2xxs_test.cu iq2xxs_tables.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ mmq_iq2xxs_test.cu

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
