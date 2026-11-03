#===============
# Explicit instantiations: src/trmm
#===============

TRMM_TYPE_TRIPLES_REAL := \
    s_s_s s_s_d s_d_s s_d_d \
    d_s_s d_s_d d_d_s d_d_d
TRMM_TYPE_TRIPLES_COMPLEX := \
    c_c_c c_c_z c_z_c c_z_z \
    z_c_c z_c_z z_z_c z_z_z
TRMM_LT_BACKEND_CODES := i8 f8

TRMM_DIR := src/trmm
TRMM_SRC := $(TRMM_DIR)/cu_recipe/trmm.cu
TRMMLT_SRC := $(TRMM_DIR)/cu_recipe/trmmlt.cu

# trmm(): INT8 only.
define ADD_TRMM
$(eval $(call ADD_INST_OBJ,    $(TRMM_DIR)/trmm_i8_$(1)_$(2)_$(3),    $(TRMM_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach trip,$(TRMM_TYPE_TRIPLES_REAL),\
  $(call ADD_TRMM,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

$(foreach trip,$(TRMM_TYPE_TRIPLES_COMPLEX),\
  $(call ADD_TRMM,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

# trmmLt(): INT8 and FP8.
define ADD_TRMMLT
$(eval $(call ADD_INST_OBJ,    $(TRMM_DIR)/trmmlt_$(1)_$(2)_$(3)_$(4),    $(TRMMLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(4)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(TRMM_LT_BACKEND_CODES),\
  $(foreach trip,$(TRMM_TYPE_TRIPLES_REAL),\
    $(call ADD_TRMMLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

$(foreach b,$(TRMM_LT_BACKEND_CODES),\
  $(foreach trip,$(TRMM_TYPE_TRIPLES_COMPLEX),\
    $(call ADD_TRMMLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

