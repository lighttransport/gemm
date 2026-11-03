#===============
# Explicit instantiations: src/syr2k
#===============

SYR2K_TYPE_TRIPLES_REAL := \
    s_s_s s_s_d s_d_s s_d_d \
    d_s_s d_s_d d_d_s d_d_d
SYR2K_TYPE_TRIPLES_COMPLEX := \
    c_c_c c_c_z c_z_c c_z_z \
    z_c_c z_c_z z_z_c z_z_z
SYR2K_LT_BACKEND_CODES := i8 f8

SYR2K_DIR := src/syr2k
SYR2K_SRC := $(SYR2K_DIR)/cu_recipe/syr2k.cu
SYR2KLT_SRC := $(SYR2K_DIR)/cu_recipe/syr2klt.cu

# syr2k(): INT8 only.
define ADD_SYR2K
$(eval $(call ADD_INST_OBJ,    $(SYR2K_DIR)/syr2k_i8_$(1)_$(2)_$(3),    $(SYR2K_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach trip,$(SYR2K_TYPE_TRIPLES_REAL),\
  $(call ADD_SYR2K,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

$(foreach trip,$(SYR2K_TYPE_TRIPLES_COMPLEX),\
  $(call ADD_SYR2K,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

# syr2kLt(): INT8 and FP8.
define ADD_SYR2KLT
$(eval $(call ADD_INST_OBJ,    $(SYR2K_DIR)/syr2klt_$(1)_$(2)_$(3)_$(4),    $(SYR2KLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(4)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(SYR2K_LT_BACKEND_CODES),\
  $(foreach trip,$(SYR2K_TYPE_TRIPLES_REAL),\
    $(call ADD_SYR2KLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

$(foreach b,$(SYR2K_LT_BACKEND_CODES),\
  $(foreach trip,$(SYR2K_TYPE_TRIPLES_COMPLEX),\
    $(call ADD_SYR2KLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

