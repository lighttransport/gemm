# Standalone dev harness for the IQ2_XXS MMQ port.
NVCC = nvcc
ARCH = sm_120
TARGETS = mmq_iq2xxs_test mmq_iq2xxs_moe_test mmq_iq3xxs_test mmq_q2_K_test mmq_iq3_s_test mmq_iq2_s_test

all: $(TARGETS)

mmq_iq2xxs_test: mmq_iq2xxs_test.cu iq2xxs_tables.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_iq2xxs_moe_test: mmq_iq2xxs_moe_test.cu iq2xxs_tables.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_iq3xxs_test: mmq_iq3xxs_test.cu iq2xxs_tables.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_q2_K_test: mmq_q2_K_test.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_iq3_s_test: mmq_iq3_s_test.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

mmq_iq2_s_test: mmq_iq2_s_test.cu iq2s_grid.inc
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

run: mmq_iq2xxs_test
	./mmq_iq2xxs_test

clean:
	rm -f $(TARGETS)

.PHONY: all run clean
