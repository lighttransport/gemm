#===============
# Explicit instantiations: src/trsm
#===============

TRSM_pair_A = $(word 1,$(subst _, ,$(1)))
TRSM_pair_B = $(word 2,$(subst _, ,$(1)))

TRSM_TYPE_PAIRS_REAL := s_s d_d
TRSM_TYPE_PAIRS_COMPLEX := c_c z_z
TRSM_LT_BACKEND_CODES := i8 f8

TRSM_DIR := src/trsm
TRSM_SRC := $(TRSM_DIR)/cu_recipe/trsm.cu
TRSMLT_SRC := $(TRSM_DIR)/cu_recipe/trsmlt.cu
TRSM_BLOCK_SIZE_SRC := $(TRSM_DIR)/cu_recipe/block_size_trsm.cu

# set_block_size_trsm() & get_block_size_trsm()
$(eval $(call ADD_INST_OBJ,\
    $(TRSM_DIR)/block_size_trsm,\
    $(TRSM_BLOCK_SIZE_SRC),\
    ))

# trsm(): INT8 only.
define ADD_TRSM
$(eval $(call ADD_INST_OBJ,    $(TRSM_DIR)/trsm_i8_$(1)_$(2),    $(TRSM_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach pair,$(TRSM_TYPE_PAIRS_REAL),\
  $(call ADD_TRSM,$(call TRSM_pair_A,$(pair)),$(call TRSM_pair_B,$(pair))))

$(foreach pair,$(TRSM_TYPE_PAIRS_COMPLEX),\
  $(call ADD_TRSM,$(call TRSM_pair_A,$(pair)),$(call TRSM_pair_B,$(pair))))

# trsmLt(): INT8 and FP8.
define ADD_TRSMLT
$(eval $(call ADD_INST_OBJ,    $(TRSM_DIR)/trsmlt_$(1)_$(2)_$(3),    $(TRSMLT_SRC),    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(TRSM_LT_BACKEND_CODES),\
  $(foreach pair,$(TRSM_TYPE_PAIRS_REAL),\
    $(call ADD_TRSMLT,$(b),$(call TRSM_pair_A,$(pair)),$(call TRSM_pair_B,$(pair)))))

$(foreach b,$(TRSM_LT_BACKEND_CODES),\
  $(foreach pair,$(TRSM_TYPE_PAIRS_COMPLEX),\
    $(call ADD_TRSMLT,$(b),$(call TRSM_pair_A,$(pair)),$(call TRSM_pair_B,$(pair)))))

