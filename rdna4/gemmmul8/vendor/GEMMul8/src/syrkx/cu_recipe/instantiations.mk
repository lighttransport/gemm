#===============
# Explicit instantiations: src/syrkx
#===============

SYRKX_TYPE_TRIPLES_REAL := \
    s_s_s s_s_d s_d_s s_d_d \
    d_s_s d_s_d d_d_s d_d_d
SYRKX_TYPE_TRIPLES_COMPLEX := \
    c_c_c c_c_z c_z_c c_z_z \
    z_c_c z_c_z z_z_c z_z_z
SYRKX_LT_BACKEND_CODES := i8 f8

SYRKX_DIR := src/syrkx
SYRKX_SRC := $(SYRKX_DIR)/cu_recipe/syrkx.cu
SYRKXLT_SRC := $(SYRKX_DIR)/cu_recipe/syrkxlt.cu

# syrkx(): INT8 only.
define ADD_SYRKX
$(eval $(call ADD_INST_OBJ,    $(SYRKX_DIR)/syrkx_i8_$(1)_$(2)_$(3),    $(SYRKX_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach trip,$(SYRKX_TYPE_TRIPLES_REAL),\
  $(call ADD_SYRKX,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

$(foreach trip,$(SYRKX_TYPE_TRIPLES_COMPLEX),\
  $(call ADD_SYRKX,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

# syrkxLt(): INT8 and FP8.
define ADD_SYRKXLT
$(eval $(call ADD_INST_OBJ,    $(SYRKX_DIR)/syrkxlt_$(1)_$(2)_$(3)_$(4),    $(SYRKXLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(4)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(SYRKX_LT_BACKEND_CODES),\
  $(foreach trip,$(SYRKX_TYPE_TRIPLES_REAL),\
    $(call ADD_SYRKXLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

$(foreach b,$(SYRKX_LT_BACKEND_CODES),\
  $(foreach trip,$(SYRKX_TYPE_TRIPLES_COMPLEX),\
    $(call ADD_SYRKXLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

