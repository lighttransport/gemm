#===============
# Explicit instantiations: src/herk
#===============

HERK_pair_A = $(word 1,$(subst _, ,$(1)))
HERK_pair_B = $(word 2,$(subst _, ,$(1)))

HERK_TYPE_PAIRS_COMPLEX := c_c c_z z_c z_z
HERK_LT_BACKEND_CODES := i8 f8

HERK_DIR := src/herk
HERK_SRC := $(HERK_DIR)/cu_recipe/herk.cu
HERKLT_SRC := $(HERK_DIR)/cu_recipe/herklt.cu

# herk(): INT8 only.
define ADD_HERK
$(eval $(call ADD_INST_OBJ,    $(HERK_DIR)/herk_i8_$(1)_$(2),    $(HERK_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach pair,$(HERK_TYPE_PAIRS_COMPLEX),\
  $(call ADD_HERK,$(call HERK_pair_A,$(pair)),$(call HERK_pair_B,$(pair))))

# herkLt(): INT8 and FP8.
define ADD_HERKLT
$(eval $(call ADD_INST_OBJ,    $(HERK_DIR)/herklt_$(1)_$(2)_$(3),    $(HERKLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(HERK_LT_BACKEND_CODES),\
  $(foreach pair,$(HERK_TYPE_PAIRS_COMPLEX),\
    $(call ADD_HERKLT,$(b),$(call HERK_pair_A,$(pair)),$(call HERK_pair_B,$(pair)))))

