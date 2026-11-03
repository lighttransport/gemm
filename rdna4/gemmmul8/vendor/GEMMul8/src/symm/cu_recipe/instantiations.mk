#===============
# Explicit instantiations: src/symm
#===============

SYMM_TYPE_TRIPLES_REAL := \
    s_s_s s_s_d s_d_s s_d_d \
    d_s_s d_s_d d_d_s d_d_d
SYMM_TYPE_TRIPLES_COMPLEX := \
    c_c_c c_c_z c_z_c c_z_z \
    z_c_c z_c_z z_z_c z_z_z
SYMM_LT_BACKEND_CODES := i8 f8

SYMM_DIR := src/symm
SYMM_SRC := $(SYMM_DIR)/cu_recipe/symm.cu
SYMMLT_SRC := $(SYMM_DIR)/cu_recipe/symmlt.cu

# symm(): INT8 only.
define ADD_SYMM
$(eval $(call ADD_INST_OBJ,    $(SYMM_DIR)/symm_i8_$(1)_$(2)_$(3),    $(SYMM_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach trip,$(SYMM_TYPE_TRIPLES_REAL),\
  $(call ADD_SYMM,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

$(foreach trip,$(SYMM_TYPE_TRIPLES_COMPLEX),\
  $(call ADD_SYMM,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

# symmLt(): INT8 and FP8.
define ADD_SYMMLT
$(eval $(call ADD_INST_OBJ,    $(SYMM_DIR)/symmlt_$(1)_$(2)_$(3)_$(4),    $(SYMMLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(4)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(SYMM_LT_BACKEND_CODES),\
  $(foreach trip,$(SYMM_TYPE_TRIPLES_REAL),\
    $(call ADD_SYMMLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

$(foreach b,$(SYMM_LT_BACKEND_CODES),\
  $(foreach trip,$(SYMM_TYPE_TRIPLES_COMPLEX),\
    $(call ADD_SYMMLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

