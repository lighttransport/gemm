#===============
# Explicit instantiations: src/syrk
#===============

SYRK_pair_A = $(word 1,$(subst _, ,$(1)))
SYRK_pair_B = $(word 2,$(subst _, ,$(1)))

SYRK_TYPE_PAIRS_REAL := s_s s_d d_s d_d
SYRK_TYPE_PAIRS_COMPLEX := c_c c_z z_c z_z
SYRK_LT_BACKEND_CODES := i8 f8

SYRK_DIR := src/syrk
SYRK_SRC := $(SYRK_DIR)/cu_recipe/syrk.cu
SYRKLT_SRC := $(SYRK_DIR)/cu_recipe/syrklt.cu

# syrk(): INT8 only.
define ADD_SYRK
$(eval $(call ADD_INST_OBJ,    $(SYRK_DIR)/syrk_i8_$(1)_$(2),    $(SYRK_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach pair,$(SYRK_TYPE_PAIRS_REAL),\
  $(call ADD_SYRK,$(call SYRK_pair_A,$(pair)),$(call SYRK_pair_B,$(pair))))

$(foreach pair,$(SYRK_TYPE_PAIRS_COMPLEX),\
  $(call ADD_SYRK,$(call SYRK_pair_A,$(pair)),$(call SYRK_pair_B,$(pair))))

# syrkLt(): INT8 and FP8.
define ADD_SYRKLT
$(eval $(call ADD_INST_OBJ,    $(SYRK_DIR)/syrklt_$(1)_$(2)_$(3),    $(SYRKLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(SYRK_LT_BACKEND_CODES),\
  $(foreach pair,$(SYRK_TYPE_PAIRS_REAL),\
    $(call ADD_SYRKLT,$(b),$(call SYRK_pair_A,$(pair)),$(call SYRK_pair_B,$(pair)))))

$(foreach b,$(SYRK_LT_BACKEND_CODES),\
  $(foreach pair,$(SYRK_TYPE_PAIRS_COMPLEX),\
    $(call ADD_SYRKLT,$(b),$(call SYRK_pair_A,$(pair)),$(call SYRK_pair_B,$(pair)))))

