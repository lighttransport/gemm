#===============
# Explicit instantiations: src/her2k
#===============

HER2K_TYPE_TRIPLES_COMPLEX := \
    c_c_c c_c_z c_z_c c_z_z \
    z_c_c z_c_z z_z_c z_z_z
HER2K_LT_BACKEND_CODES := i8 f8

HER2K_DIR := src/her2k
HER2K_SRC := $(HER2K_DIR)/cu_recipe/her2k.cu
HER2KLT_SRC := $(HER2K_DIR)/cu_recipe/her2klt.cu

# her2k(): INT8 only.
define ADD_HER2K
$(eval $(call ADD_INST_OBJ,    $(HER2K_DIR)/her2k_i8_$(1)_$(2)_$(3),    $(HER2K_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach trip,$(HER2K_TYPE_TRIPLES_COMPLEX),\
  $(call ADD_HER2K,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

# her2kLt(): INT8 and FP8.
define ADD_HER2KLT
$(eval $(call ADD_INST_OBJ,    $(HER2K_DIR)/her2klt_$(1)_$(2)_$(3)_$(4),    $(HER2KLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(4)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(HER2K_LT_BACKEND_CODES),\
  $(foreach trip,$(HER2K_TYPE_TRIPLES_COMPLEX),\
    $(call ADD_HER2KLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

