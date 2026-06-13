# Standalone dev harness for the IQ2_XXS MMQ port.
NVCC = nvcc
ARCH = sm_120
TARGETS = mmq_iq2xxs_test mmq_iq2xxs_moe_test

all: $(TARGETS)

mmq_iq2xxs_test: mmq_iq2xxs_test.cu iq2xxs_tables.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_iq2xxs_moe_test: mmq_iq2xxs_moe_test.cu iq2xxs_tables.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

run: mmq_iq2xxs_test
	./mmq_iq2xxs_test

clean:
	rm -f $(TARGETS)

.PHONY: all run clean
