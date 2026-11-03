#===============
# Explicit instantiations: src/herkx
#===============

HERKX_TYPE_TRIPLES_COMPLEX := \
    c_c_c c_c_z c_z_c c_z_z \
    z_c_c z_c_z z_z_c z_z_z
HERKX_LT_BACKEND_CODES := i8 f8

HERKX_DIR := src/herkx
HERKX_SRC := $(HERKX_DIR)/cu_recipe/herkx.cu
HERKXLT_SRC := $(HERKX_DIR)/cu_recipe/herkxlt.cu

# herkx(): INT8 only.
define ADD_HERKX
$(eval $(call ADD_INST_OBJ,    $(HERKX_DIR)/herkx_i8_$(1)_$(2)_$(3),    $(HERKX_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach trip,$(HERKX_TYPE_TRIPLES_COMPLEX),\
  $(call ADD_HERKX,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

# herkxLt(): INT8 and FP8.
define ADD_HERKXLT
$(eval $(call ADD_INST_OBJ,    $(HERKX_DIR)/herkxlt_$(1)_$(2)_$(3)_$(4),    $(HERKXLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(4)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(HERKX_LT_BACKEND_CODES),\
  $(foreach trip,$(HERKX_TYPE_TRIPLES_COMPLEX),\
    $(call ADD_HERKXLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

