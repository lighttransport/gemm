#===============
# Instantiation helper functions
#===============

# GEMMul8 type code -> C++ type.
type_cpp = $(strip $(if $(filter s,$(1)),float,\
           $(if $(filter d,$(1)),double,\
           $(if $(filter c,$(1)),cuFloatComplex,\
           $(if $(filter z,$(1)),cuDoubleComplex,\
           $(error Invalid type code: $(1)))))))

# Backend code -> enum member suffix.
# The recipe .cu files use Backend::GEMMUL8_INST_BACKEND.
backend_cpp = $(strip $(if $(filter i8,$(1)),INT8,\
              $(if $(filter f8,$(1)),FP8,\
              $(error Invalid backend code: $(1)))))

# Fill-mode code -> cuBLAS fill-mode constant.
uplo_cpp = $(strip $(if $(filter full,$(1)),CUBLAS_FILL_MODE_FULL,\
           $(if $(filter upper,$(1)),CUBLAS_FILL_MODE_UPPER,\
           $(if $(filter lower,$(1)),CUBLAS_FILL_MODE_LOWER,\
           $(error Invalid uplo code: $(1))))))

# Diagonal-mode code -> cuBLAS diagonal-mode constant.
diag_cpp = $(strip $(if $(filter nonunit,$(1)),CUBLAS_DIAG_NON_UNIT,\
           $(if $(filter unit,$(1)),CUBLAS_DIAG_UNIT,\
           $(error Invalid diag code: $(1)))))

# Conjugation code -> C++ bool literal.
conj_cpp = $(strip $(if $(filter conj0,$(1)),false,\
           $(if $(filter conj1,$(1)),true,\
           $(error Invalid conj code: $(1)))))

# Matrix-complexity code -> C++ bool literal.
complex_cpp = $(strip $(if $(filter real,$(1)),false,\
              $(if $(filter complex,$(1)),true,\
              $(error Invalid complex code: $(1)))))

# Matrix-complexity code -> 0/1.
complex_cpp_01 = $(strip $(if $(filter real,$(1)),0,\
                 $(if $(filter complex,$(1)),1,\
                 $(error Invalid complex code: $(1)))))

# Triangular pattern helpers.
tri_uplo = $(strip $(if $(filter full_nonunit,$(1)),full,\
           $(if $(filter upper_nonunit upper_unit,$(1)),upper,\
           $(if $(filter lower_nonunit lower_unit,$(1)),lower,\
           $(error Invalid triangular pattern: $(1))))))

tri_diag = $(strip $(if $(filter full_nonunit upper_nonunit lower_nonunit,$(1)),nonunit,\
           $(if $(filter upper_unit lower_unit,$(1)),unit,\
           $(error Invalid triangular pattern: $(1)))))

# Output-UPLO helper.
out_uplo = $(strip $(if $(filter out_full,$(1)),full,\
           $(if $(filter out_upper,$(1)),upper,\
           $(if $(filter out_lower,$(1)),lower,\
           $(error Invalid output uplo: $(1))))))

# Triplet code helpers, e.g. c_c_s -> T=c, alpha=c, beta=s.
trip_T  = $(word 1,$(subst _, ,$(1)))
trip_A  = $(word 2,$(subst _, ,$(1)))
trip_B  = $(word 3,$(subst _, ,$(1)))
