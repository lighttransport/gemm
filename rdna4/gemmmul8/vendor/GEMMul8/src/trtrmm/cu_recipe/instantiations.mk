#===============
# Explicit instantiations: src/trtrmm
#===============

TRTRMM_TYPE_TRIPLES_REAL := \
    s_s_s s_s_d s_d_s s_d_d \
    d_s_s d_s_d d_d_s d_d_d
TRTRMM_TYPE_TRIPLES_COMPLEX := \
    c_c_c c_c_z c_z_c c_z_z \
    z_c_c z_c_z z_z_c z_z_z
TRTRMM_LT_BACKEND_CODES := i8 f8

TRTRMM_DIR := src/trtrmm
TRTRMM_SRC := $(TRTRMM_DIR)/cu_recipe/trtrmm.cu
TRTRMMLT_SRC := $(TRTRMM_DIR)/cu_recipe/trtrmmlt.cu

# trtrmm(): INT8 only.
define ADD_TRTRMM
$(eval $(call ADD_INST_OBJ,    $(TRTRMM_DIR)/trtrmm_i8_$(1)_$(2)_$(3),    $(TRTRMM_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach trip,$(TRTRMM_TYPE_TRIPLES_REAL),\
  $(call ADD_TRTRMM,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

$(foreach trip,$(TRTRMM_TYPE_TRIPLES_COMPLEX),\
  $(call ADD_TRTRMM,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

# trtrmmLt(): INT8 and FP8.
define ADD_TRTRMMLT
$(eval $(call ADD_INST_OBJ,    $(TRTRMM_DIR)/trtrmmlt_$(1)_$(2)_$(3)_$(4),    $(TRTRMMLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(4)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(TRTRMM_LT_BACKEND_CODES),\
  $(foreach trip,$(TRTRMM_TYPE_TRIPLES_REAL),\
    $(call ADD_TRTRMMLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

$(foreach b,$(TRTRMM_LT_BACKEND_CODES),\
  $(foreach trip,$(TRTRMM_TYPE_TRIPLES_COMPLEX),\
    $(call ADD_TRTRMMLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

