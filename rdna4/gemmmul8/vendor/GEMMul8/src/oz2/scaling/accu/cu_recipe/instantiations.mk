#===============
# Explicit instantiations: src/oz2/scaling/accu
#===============

ACCU_TYPE_CODES_REAL := s d
ACCU_TYPE_CODES_COMPLEX := c z
ACCU_TYPE_CODES_ALL := $(ACCU_TYPE_CODES_REAL) $(ACCU_TYPE_CODES_COMPLEX)
ACCU_BACKEND_CODES := i8 f8
ACCU_TRI_PATTERNS := full_nonunit upper_nonunit upper_unit lower_nonunit lower_unit
ACCU_UPLO_CODES := upper lower
ACCU_OUT_UPLO_CODES := out_full out_upper out_lower
ACCU_OUT_UPLO_CODES_TRI := out_upper out_lower

ACCU_DIR := src/oz2/scaling/accu
ACCU_EXTRACT_SRC      := $(ACCU_DIR)/cu_recipe/extract.cu
ACCU_EXTRACT_SYMM_SRC := $(ACCU_DIR)/cu_recipe/extract_symm.cu
ACCU_EXTRACT_HEMM_SRC := $(ACCU_DIR)/cu_recipe/extract_hemm.cu
ACCU_SCALING_SRC      := $(ACCU_DIR)/cu_recipe/scaling.cu
ACCU_SCALING_SYRK_SRC := $(ACCU_DIR)/cu_recipe/scaling_syrk.cu
ACCU_SCALING_HERK_SRC := $(ACCU_DIR)/cu_recipe/scaling_herk.cu
ACCU_SCALING_SYMM_SRC := $(ACCU_DIR)/cu_recipe/scaling_symm.cu
ACCU_SCALING_HEMM_SRC := $(ACCU_DIR)/cu_recipe/scaling_hemm.cu

# $(1): s|d|c|z, $(2): i8|f8, $(3): triangular pattern
define ADD_ACCU_EXTRACT
$(eval $(call ADD_INST_OBJ,\
    $(ACCU_DIR)/extract_$(1)_$(2)_$(3),\
    $(ACCU_EXTRACT_SRC),\
    -DGEMMUL8_INST_TYPE=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(2)) \
    -DGEMMUL8_INST_FILLMODE=$(call uplo_cpp,$(call tri_uplo,$(3))) \
    -DGEMMUL8_INST_DIAG=$(call diag_cpp,$(call tri_diag,$(3)))))
endef

$(foreach t,$(ACCU_TYPE_CODES_ALL),\
  $(foreach b,$(ACCU_BACKEND_CODES),\
    $(foreach p,$(ACCU_TRI_PATTERNS),\
      $(call ADD_ACCU_EXTRACT,$(t),$(b),$(p)))))

# $(1): s|d|c|z, $(2): i8|f8, $(3): upper|lower
define ADD_ACCU_EXTRACT_SYMM
$(eval $(call ADD_INST_OBJ,\
    $(ACCU_DIR)/extract_symm_$(1)_$(2)_$(3),\
    $(ACCU_EXTRACT_SYMM_SRC),\
    -DGEMMUL8_INST_TYPE=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(2)) \
    -DGEMMUL8_INST_FILLMODE=$(call uplo_cpp,$(3))))
endef

$(foreach t,$(ACCU_TYPE_CODES_ALL),\
  $(foreach b,$(ACCU_BACKEND_CODES),\
    $(foreach u,$(ACCU_UPLO_CODES),\
      $(call ADD_ACCU_EXTRACT_SYMM,$(t),$(b),$(u)))))

# $(1): c|z, $(2): i8|f8, $(3): upper|lower
define ADD_ACCU_EXTRACT_HEMM
$(eval $(call ADD_INST_OBJ,\
    $(ACCU_DIR)/extract_hemm_$(1)_$(2)_$(3),\
    $(ACCU_EXTRACT_HEMM_SRC),\
    -DGEMMUL8_INST_TYPE=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(2)) \
    -DGEMMUL8_INST_FILLMODE=$(call uplo_cpp,$(3))))
endef

$(foreach t,$(ACCU_TYPE_CODES_COMPLEX),\
  $(foreach b,$(ACCU_BACKEND_CODES),\
    $(foreach u,$(ACCU_UPLO_CODES),\
      $(call ADD_ACCU_EXTRACT_HEMM,$(t),$(b),$(u)))))

# $(1): s|d|c|z, $(2): i8|f8, $(3): triangular pattern, $(4): output uplo
define ADD_ACCU_SCALING
$(eval $(call ADD_INST_OBJ,\
    $(ACCU_DIR)/scaling_$(1)_$(2)_$(3)_$(4),\
    $(ACCU_SCALING_SRC),\
    -DGEMMUL8_INST_TYPE=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(2)) \
    -DGEMMUL8_INST_FILLMODE=$(call uplo_cpp,$(call tri_uplo,$(3))) \
    -DGEMMUL8_INST_DIAG=$(call diag_cpp,$(call tri_diag,$(3))) \
    -DGEMMUL8_INST_FILLMODE_C=$(call uplo_cpp,$(call out_uplo,$(4)))))
endef

$(foreach t,$(ACCU_TYPE_CODES_ALL),\
  $(foreach b,$(ACCU_BACKEND_CODES),\
    $(foreach p,$(ACCU_TRI_PATTERNS),\
      $(foreach out,$(ACCU_OUT_UPLO_CODES),\
        $(call ADD_ACCU_SCALING,$(t),$(b),$(p),$(out))))))

# $(1): s|d|c|z, $(2): i8|f8, $(3): output uplo
define ADD_ACCU_SCALING_SYRK
$(eval $(call ADD_INST_OBJ,\
    $(ACCU_DIR)/scaling_syrk_$(1)_$(2)_$(3),\
    $(ACCU_SCALING_SYRK_SRC),\
    -DGEMMUL8_INST_TYPE=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(2)) \
    -DGEMMUL8_INST_FILLMODE=$(call uplo_cpp,$(call out_uplo,$(3)))))
endef

$(foreach t,$(ACCU_TYPE_CODES_ALL),\
  $(foreach b,$(ACCU_BACKEND_CODES),\
    $(foreach out,$(ACCU_OUT_UPLO_CODES_TRI),\
      $(call ADD_ACCU_SCALING_SYRK,$(t),$(b),$(out)))))

# $(1): c|z, $(2): i8|f8, $(3): output uplo
define ADD_ACCU_SCALING_HERK
$(eval $(call ADD_INST_OBJ,\
    $(ACCU_DIR)/scaling_herk_$(1)_$(2)_$(3),\
    $(ACCU_SCALING_HERK_SRC),\
    -DGEMMUL8_INST_TYPE=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(2)) \
    -DGEMMUL8_INST_FILLMODE=$(call uplo_cpp,$(call out_uplo,$(3)))))
endef

$(foreach t,$(ACCU_TYPE_CODES_COMPLEX),\
  $(foreach b,$(ACCU_BACKEND_CODES),\
    $(foreach out,$(ACCU_OUT_UPLO_CODES_TRI),\
      $(call ADD_ACCU_SCALING_HERK,$(t),$(b),$(out)))))

# $(1): s|d|c|z, $(2): i8|f8, $(3): upper|lower
define ADD_ACCU_SCALING_SYMM
$(eval $(call ADD_INST_OBJ,\
    $(ACCU_DIR)/scaling_symm_$(1)_$(2)_$(3),\
    $(ACCU_SCALING_SYMM_SRC),\
    -DGEMMUL8_INST_TYPE=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(2)) \
    -DGEMMUL8_INST_FILLMODE=$(call uplo_cpp,$(3))))
endef

$(foreach t,$(ACCU_TYPE_CODES_ALL),\
  $(foreach b,$(ACCU_BACKEND_CODES),\
    $(foreach u,$(ACCU_UPLO_CODES),\
      $(call ADD_ACCU_SCALING_SYMM,$(t),$(b),$(u)))))

# $(1): c|z, $(2): i8|f8, $(3): upper|lower
define ADD_ACCU_SCALING_HEMM
$(eval $(call ADD_INST_OBJ,\
    $(ACCU_DIR)/scaling_hemm_$(1)_$(2)_$(3),\
    $(ACCU_SCALING_HEMM_SRC),\
    -DGEMMUL8_INST_TYPE=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(2)) \
    -DGEMMUL8_INST_FILLMODE=$(call uplo_cpp,$(3))))
endef

$(foreach t,$(ACCU_TYPE_CODES_COMPLEX),\
  $(foreach b,$(ACCU_BACKEND_CODES),\
    $(foreach u,$(ACCU_UPLO_CODES),\
      $(call ADD_ACCU_SCALING_HEMM,$(t),$(b),$(u)))))
