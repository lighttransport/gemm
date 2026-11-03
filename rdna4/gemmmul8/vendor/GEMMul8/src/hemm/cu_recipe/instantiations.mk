#===============
# Explicit instantiations: src/hemm
#===============

HEMM_TYPE_TRIPLES_COMPLEX := \
    c_c_c c_c_z c_z_c c_z_z \
    z_c_c z_c_z z_z_c z_z_z
HEMM_LT_BACKEND_CODES := i8 f8

HEMM_DIR := src/hemm
HEMM_SRC := $(HEMM_DIR)/cu_recipe/hemm.cu
HEMMLT_SRC := $(HEMM_DIR)/cu_recipe/hemmlt.cu

# hemm(): INT8 only.
define ADD_HEMM
$(eval $(call ADD_INST_OBJ,    $(HEMM_DIR)/hemm_i8_$(1)_$(2)_$(3),    $(HEMM_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach trip,$(HEMM_TYPE_TRIPLES_COMPLEX),\
  $(call ADD_HEMM,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

# hemmLt(): INT8 and FP8.
define ADD_HEMMLT
$(eval $(call ADD_INST_OBJ,    $(HEMM_DIR)/hemmlt_$(1)_$(2)_$(3)_$(4),    $(HEMMLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(4)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(HEMM_LT_BACKEND_CODES),\
  $(foreach trip,$(HEMM_TYPE_TRIPLES_COMPLEX),\
    $(call ADD_HEMMLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

