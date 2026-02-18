/*
 * pmu_events.h - A64FX PMU Event Definitions
 *
 * Auto-derived from a64fx/doc/a64fx_pmu_events.csv
 * All ~185 hardware performance monitoring events for A64FX.
 */
#ifndef PMU_EVENTS_H
#define PMU_EVENTS_H

#include <stdint.h>

/* ========================================================================
 * ARMv8 Standard Events
 * ======================================================================== */
#define PMU_SW_INCR                     0x0000
#define PMU_L1I_CACHE_REFILL            0x0001
#define PMU_L1I_TLB_REFILL             0x0002
#define PMU_L1D_CACHE_REFILL            0x0003
#define PMU_L1D_CACHE                   0x0004
#define PMU_L1D_TLB_REFILL             0x0005
#define PMU_INST_RETIRED                0x0008
#define PMU_EXC_TAKEN                   0x0009
#define PMU_EXC_RETURN                  0x000a
#define PMU_CID_WRITE_RETIRED           0x000b
#define PMU_BR_MIS_PRED                 0x0010
#define PMU_CPU_CYCLES                  0x0011
#define PMU_BR_PRED                     0x0012
#define PMU_L1I_CACHE                   0x0014
#define PMU_L1D_CACHE_WB               0x0015
#define PMU_L2D_CACHE                   0x0016
#define PMU_L2D_CACHE_REFILL            0x0017
#define PMU_L2D_CACHE_WB               0x0018
#define PMU_INST_SPEC                   0x001b
#define PMU_STALL_FRONTEND              0x0023
#define PMU_STALL_BACKEND               0x0024
#define PMU_L2D_TLB_REFILL             0x002d
#define PMU_L2I_TLB_REFILL             0x002e
#define PMU_L2D_TLB                     0x002f
#define PMU_L2I_TLB                     0x0030
#define PMU_L1D_CACHE_REFILL_PRF        0x0049
#define PMU_L2D_CACHE_REFILL_PRF        0x0059
#define PMU_LDREX_SPEC                  0x006c
#define PMU_STREX_SPEC                  0x006f
#define PMU_LD_SPEC                     0x0070
#define PMU_ST_SPEC                     0x0071
#define PMU_LDST_SPEC                   0x0072
#define PMU_DP_SPEC                     0x0073
#define PMU_ASE_SPEC                    0x0074
#define PMU_VFP_SPEC                    0x0075
#define PMU_PC_WRITE_SPEC               0x0076
#define PMU_CRYPTO_SPEC                 0x0077
#define PMU_BR_IMMED_SPEC               0x0078
#define PMU_BR_RETURN_SPEC              0x0079
#define PMU_BR_INDIRECT_SPEC            0x007a
#define PMU_ISB_SPEC                    0x007c
#define PMU_DSB_SPEC                    0x007d
#define PMU_DMB_SPEC                    0x007e
#define PMU_EXC_UNDEF                   0x0081
#define PMU_EXC_SVC                     0x0082
#define PMU_EXC_PABORT                  0x0083
#define PMU_EXC_DABORT                  0x0084
#define PMU_EXC_IRQ                     0x0086
#define PMU_EXC_FIQ                     0x0087
#define PMU_EXC_SMC                     0x0088
#define PMU_EXC_HVC                     0x008a
#define PMU_DCZVA_SPEC                  0x009f

/* ========================================================================
 * A64FX-specific Events
 * ======================================================================== */
#define PMU_FP_MV_SPEC                  0x0105
#define PMU_PRD_SPEC                    0x0108
#define PMU_IEL_SPEC                    0x0109
#define PMU_IREG_SPEC                   0x010a
#define PMU_FP_LD_SPEC                  0x0112
#define PMU_FP_ST_SPEC                  0x0113
#define PMU_BC_LD_SPEC                  0x011a
#define PMU_EFFECTIVE_INST_SPEC         0x0121
#define PMU_PRE_INDEX_SPEC              0x0123
#define PMU_POST_INDEX_SPEC             0x0124
#define PMU_UOP_SPLIT                   0x0139
#define PMU_LD_COMP_WAIT_L2_MISS        0x0180
#define PMU_LD_COMP_WAIT_L2_MISS_EX     0x0181
#define PMU_LD_COMP_WAIT_L1_MISS        0x0182
#define PMU_LD_COMP_WAIT_L1_MISS_EX     0x0183
#define PMU_LD_COMP_WAIT                0x0184
#define PMU_LD_COMP_WAIT_EX             0x0185
#define PMU_LD_COMP_WAIT_PFP_BUSY       0x0186
#define PMU_LD_COMP_WAIT_PFP_BUSY_EX    0x0187
#define PMU_LD_COMP_WAIT_PFP_BUSY_SWPF  0x0188
#define PMU_EU_COMP_WAIT                0x0189
#define PMU_FL_COMP_WAIT                0x018a
#define PMU_BR_COMP_WAIT                0x018b
#define PMU_ROB_EMPTY                   0x018c
#define PMU_ROB_EMPTY_STQ_BUSY          0x018d
#define PMU_WFE_WFI_CYCLE               0x018e
#define PMU_0INST_COMMIT                0x0190
#define PMU_1INST_COMMIT                0x0191
#define PMU_2INST_COMMIT                0x0192
#define PMU_3INST_COMMIT                0x0193
#define PMU_4INST_COMMIT                0x0194
#define PMU_UOP_ONLY_COMMIT             0x0198
#define PMU_SINGLE_MOVPRFX_COMMIT       0x0199
#define PMU_EAGA_VAL                    0x01a0
#define PMU_EAGB_VAL                    0x01a1
#define PMU_EXA_VAL                     0x01a2
#define PMU_EXB_VAL                     0x01a3
#define PMU_FLA_VAL                     0x01a4
#define PMU_FLB_VAL                     0x01a5
#define PMU_PRX_VAL                     0x01a6
#define PMU_FLA_VAL_PRD_CNT             0x01b4
#define PMU_FLB_VAL_PRD_CNT             0x01b5
#define PMU_EA_CORE                     0x01e0
#define PMU_L1D_CACHE_REFILL_DM         0x0200
#define PMU_L1D_CACHE_REFILL_HWPRF      0x0202
#define PMU_L1_MISS_WAIT                0x0208
#define PMU_L1I_MISS_WAIT               0x0209
#define PMU_L1HWPF_STREAM_PF            0x0230
#define PMU_L1HWPF_INJ_ALLOC_PF         0x0231
#define PMU_L1HWPF_INJ_NOALLOC_PF       0x0232
#define PMU_L2HWPF_STREAM_PF            0x0233
#define PMU_L2HWPF_INJ_ALLOC_PF         0x0234
#define PMU_L2HWPF_INJ_NOALLOC_PF       0x0235
#define PMU_L2HWPF_OTHER                0x0236
#define PMU_L1_PIPE0_VAL                0x0240
#define PMU_L1_PIPE1_VAL                0x0241
#define PMU_L1_PIPE0_VAL_IU_TAG_ADRS_SCE 0x0250
#define PMU_L1_PIPE0_VAL_IU_TAG_ADRS_PFE 0x0251
#define PMU_L1_PIPE1_VAL_IU_TAG_ADRS_SCE 0x0252
#define PMU_L1_PIPE1_VAL_IU_TAG_ADRS_PFE 0x0253
#define PMU_L1_PIPE0_COMP               0x0260
#define PMU_L1_PIPE1_COMP               0x0261
#define PMU_L1I_PIPE_COMP               0x0268
#define PMU_L1I_PIPE_VAL                0x0269
#define PMU_L1_PIPE_ABORT_STLD_INTLK    0x0274
#define PMU_L1_PIPE0_VAL_IU_NOT_SEC0    0x02a0
#define PMU_L1_PIPE1_VAL_IU_NOT_SEC0    0x02a1
#define PMU_L1_PIPE_COMP_GATHER_2FLOW   0x02b0
#define PMU_L1_PIPE_COMP_GATHER_1FLOW   0x02b1
#define PMU_L1_PIPE_COMP_GATHER_0FLOW   0x02b2
#define PMU_L1_PIPE_COMP_SCATTER_1FLOW  0x02b3
#define PMU_L1_PIPE0_COMP_PRD_CNT       0x02b8
#define PMU_L1_PIPE1_COMP_PRD_CNT       0x02b9
#define PMU_L2D_CACHE_REFILL_DM         0x0300
#define PMU_L2D_CACHE_REFILL_HWPRF      0x0302
#define PMU_L2_MISS_WAIT                0x0308
#define PMU_L2_MISS_COUNT               0x0309
#define PMU_BUS_READ_TOTAL_TOFU         0x0314
#define PMU_BUS_READ_TOTAL_PCI          0x0315
#define PMU_BUS_READ_TOTAL_MEM          0x0316
#define PMU_BUS_WRITE_TOTAL_CMG0        0x0318
#define PMU_BUS_WRITE_TOTAL_CMG1        0x0319
#define PMU_BUS_WRITE_TOTAL_CMG2        0x031a
#define PMU_BUS_WRITE_TOTAL_CMG3        0x031b
#define PMU_BUS_WRITE_TOTAL_TOFU        0x031c
#define PMU_BUS_WRITE_TOTAL_PCI         0x031d
#define PMU_BUS_WRITE_TOTAL_MEM         0x031e
#define PMU_L2D_SWAP_DM                 0x0325
#define PMU_L2D_CACHE_MIBMCH_PRF        0x0326
#define PMU_L2_PIPE_VAL                 0x0330
#define PMU_L2_PIPE_COMP_ALL            0x0350
#define PMU_L2_PIPE_COMP_PF_L2MIB_MCH   0x0370
#define PMU_L2D_CACHE_SWAP_LOCAL        0x0396
#define PMU_EA_L2                       0x03e0
#define PMU_EA_MEMORY                   0x03e8

/* ========================================================================
 * SVE Events
 * ======================================================================== */
#define PMU_SIMD_INST_RETIRED           0x8000
#define PMU_SVE_INST_RETIRED            0x8002
#define PMU_UOP_SPEC_SVE                0x8008
#define PMU_SVE_MATH_SPEC               0x800e
#define PMU_FP_SPEC                     0x8010
#define PMU_FP_FMA_SPEC                 0x8028
#define PMU_FP_RECPE_SPEC               0x8034
#define PMU_FP_CVT_SPEC                 0x8038
#define PMU_ASE_SVE_INT_SPEC            0x8043
#define PMU_SVE_PRED_SPEC               0x8074
#define PMU_SVE_MOVPRFX_SPEC            0x807c
#define PMU_SVE_MOVPRFX_U_SPEC          0x807f
#define PMU_ASE_SVE_LD_SPEC             0x8085
#define PMU_ASE_SVE_ST_SPEC             0x8086
#define PMU_PRF_SPEC                    0x8087
#define PMU_BASE_LD_REG_SPEC            0x8089
#define PMU_BASE_ST_REG_SPEC            0x808a
#define PMU_SVE_LDR_REG_SPEC            0x8091
#define PMU_SVE_STR_REG_SPEC            0x8092
#define PMU_SVE_LDR_PREG_SPEC           0x8095
#define PMU_SVE_STR_PREG_SPEC           0x8096
#define PMU_SVE_PRF_CONTIG_SPEC         0x809f
#define PMU_ASE_SVE_LD_MULTI_SPEC       0x80a5
#define PMU_ASE_SVE_ST_MULTI_SPEC       0x80a6
#define PMU_SVE_LD_GATHER_SPEC          0x80ad
#define PMU_SVE_ST_SCATTER_SPEC         0x80ae
#define PMU_SVE_PRF_GATHER_SPEC         0x80af
#define PMU_SVE_LDFF_SPEC               0x80bc
#define PMU_FP_SCALE_OPS_SPEC           0x80c0
#define PMU_FP_FIXED_OPS_SPEC           0x80c1
#define PMU_FP_HP_SCALE_OPS_SPEC        0x80c2
#define PMU_FP_HP_FIXED_OPS_SPEC        0x80c3
#define PMU_FP_SP_SCALE_OPS_SPEC        0x80c4
#define PMU_FP_SP_FIXED_OPS_SPEC        0x80c5
#define PMU_FP_DP_SCALE_OPS_SPEC        0x80c6
#define PMU_FP_DP_FIXED_OPS_SPEC        0x80c7

/* ========================================================================
 * Event name lookup table (sorted by code for binary search)
 * ======================================================================== */
typedef struct {
    uint16_t code;
    const char *name;
} pmu_event_entry_t;

static const pmu_event_entry_t pmu_event_table[] = {
    { 0x0000, "SW_INCR" },
    { 0x0001, "L1I_CACHE_REFILL" },
    { 0x0002, "L1I_TLB_REFILL" },
    { 0x0003, "L1D_CACHE_REFILL" },
    { 0x0004, "L1D_CACHE" },
    { 0x0005, "L1D_TLB_REFILL" },
    { 0x0008, "INST_RETIRED" },
    { 0x0009, "EXC_TAKEN" },
    { 0x000a, "EXC_RETURN" },
    { 0x000b, "CID_WRITE_RETIRED" },
    { 0x0010, "BR_MIS_PRED" },
    { 0x0011, "CPU_CYCLES" },
    { 0x0012, "BR_PRED" },
    { 0x0014, "L1I_CACHE" },
    { 0x0015, "L1D_CACHE_WB" },
    { 0x0016, "L2D_CACHE" },
    { 0x0017, "L2D_CACHE_REFILL" },
    { 0x0018, "L2D_CACHE_WB" },
    { 0x001b, "INST_SPEC" },
    { 0x0023, "STALL_FRONTEND" },
    { 0x0024, "STALL_BACKEND" },
    { 0x002d, "L2D_TLB_REFILL" },
    { 0x002e, "L2I_TLB_REFILL" },
    { 0x002f, "L2D_TLB" },
    { 0x0030, "L2I_TLB" },
    { 0x0049, "L1D_CACHE_REFILL_PRF" },
    { 0x0059, "L2D_CACHE_REFILL_PRF" },
    { 0x006c, "LDREX_SPEC" },
    { 0x006f, "STREX_SPEC" },
    { 0x0070, "LD_SPEC" },
    { 0x0071, "ST_SPEC" },
    { 0x0072, "LDST_SPEC" },
    { 0x0073, "DP_SPEC" },
    { 0x0074, "ASE_SPEC" },
    { 0x0075, "VFP_SPEC" },
    { 0x0076, "PC_WRITE_SPEC" },
    { 0x0077, "CRYPTO_SPEC" },
    { 0x0078, "BR_IMMED_SPEC" },
    { 0x0079, "BR_RETURN_SPEC" },
    { 0x007a, "BR_INDIRECT_SPEC" },
    { 0x007c, "ISB_SPEC" },
    { 0x007d, "DSB_SPEC" },
    { 0x007e, "DMB_SPEC" },
    { 0x0081, "EXC_UNDEF" },
    { 0x0082, "EXC_SVC" },
    { 0x0083, "EXC_PABORT" },
    { 0x0084, "EXC_DABORT" },
    { 0x0086, "EXC_IRQ" },
    { 0x0087, "EXC_FIQ" },
    { 0x0088, "EXC_SMC" },
    { 0x008a, "EXC_HVC" },
    { 0x009f, "DCZVA_SPEC" },
    { 0x0105, "FP_MV_SPEC" },
    { 0x0108, "PRD_SPEC" },
    { 0x0109, "IEL_SPEC" },
    { 0x010a, "IREG_SPEC" },
    { 0x0112, "FP_LD_SPEC" },
    { 0x0113, "FP_ST_SPEC" },
    { 0x011a, "BC_LD_SPEC" },
    { 0x0121, "EFFECTIVE_INST_SPEC" },
    { 0x0123, "PRE_INDEX_SPEC" },
    { 0x0124, "POST_INDEX_SPEC" },
    { 0x0139, "UOP_SPLIT" },
    { 0x0180, "LD_COMP_WAIT_L2_MISS" },
    { 0x0181, "LD_COMP_WAIT_L2_MISS_EX" },
    { 0x0182, "LD_COMP_WAIT_L1_MISS" },
    { 0x0183, "LD_COMP_WAIT_L1_MISS_EX" },
    { 0x0184, "LD_COMP_WAIT" },
    { 0x0185, "LD_COMP_WAIT_EX" },
    { 0x0186, "LD_COMP_WAIT_PFP_BUSY" },
    { 0x0187, "LD_COMP_WAIT_PFP_BUSY_EX" },
    { 0x0188, "LD_COMP_WAIT_PFP_BUSY_SWPF" },
    { 0x0189, "EU_COMP_WAIT" },
    { 0x018a, "FL_COMP_WAIT" },
    { 0x018b, "BR_COMP_WAIT" },
    { 0x018c, "ROB_EMPTY" },
    { 0x018d, "ROB_EMPTY_STQ_BUSY" },
    { 0x018e, "WFE_WFI_CYCLE" },
    { 0x0190, "0INST_COMMIT" },
    { 0x0191, "1INST_COMMIT" },
    { 0x0192, "2INST_COMMIT" },
    { 0x0193, "3INST_COMMIT" },
    { 0x0194, "4INST_COMMIT" },
    { 0x0198, "UOP_ONLY_COMMIT" },
    { 0x0199, "SINGLE_MOVPRFX_COMMIT" },
    { 0x01a0, "EAGA_VAL" },
    { 0x01a1, "EAGB_VAL" },
    { 0x01a2, "EXA_VAL" },
    { 0x01a3, "EXB_VAL" },
    { 0x01a4, "FLA_VAL" },
    { 0x01a5, "FLB_VAL" },
    { 0x01a6, "PRX_VAL" },
    { 0x01b4, "FLA_VAL_PRD_CNT" },
    { 0x01b5, "FLB_VAL_PRD_CNT" },
    { 0x01e0, "EA_CORE" },
    { 0x0200, "L1D_CACHE_REFILL_DM" },
    { 0x0202, "L1D_CACHE_REFILL_HWPRF" },
    { 0x0208, "L1_MISS_WAIT" },
    { 0x0209, "L1I_MISS_WAIT" },
    { 0x0230, "L1HWPF_STREAM_PF" },
    { 0x0231, "L1HWPF_INJ_ALLOC_PF" },
    { 0x0232, "L1HWPF_INJ_NOALLOC_PF" },
    { 0x0233, "L2HWPF_STREAM_PF" },
    { 0x0234, "L2HWPF_INJ_ALLOC_PF" },
    { 0x0235, "L2HWPF_INJ_NOALLOC_PF" },
    { 0x0236, "L2HWPF_OTHER" },
    { 0x0240, "L1_PIPE0_VAL" },
    { 0x0241, "L1_PIPE1_VAL" },
    { 0x0250, "L1_PIPE0_VAL_IU_TAG_ADRS_SCE" },
    { 0x0251, "L1_PIPE0_VAL_IU_TAG_ADRS_PFE" },
    { 0x0252, "L1_PIPE1_VAL_IU_TAG_ADRS_SCE" },
    { 0x0253, "L1_PIPE1_VAL_IU_TAG_ADRS_PFE" },
    { 0x0260, "L1_PIPE0_COMP" },
    { 0x0261, "L1_PIPE1_COMP" },
    { 0x0268, "L1I_PIPE_COMP" },
    { 0x0269, "L1I_PIPE_VAL" },
    { 0x0274, "L1_PIPE_ABORT_STLD_INTLK" },
    { 0x02a0, "L1_PIPE0_VAL_IU_NOT_SEC0" },
    { 0x02a1, "L1_PIPE1_VAL_IU_NOT_SEC0" },
    { 0x02b0, "L1_PIPE_COMP_GATHER_2FLOW" },
    { 0x02b1, "L1_PIPE_COMP_GATHER_1FLOW" },
    { 0x02b2, "L1_PIPE_COMP_GATHER_0FLOW" },
    { 0x02b3, "L1_PIPE_COMP_SCATTER_1FLOW" },
    { 0x02b8, "L1_PIPE0_COMP_PRD_CNT" },
    { 0x02b9, "L1_PIPE1_COMP_PRD_CNT" },
    { 0x0300, "L2D_CACHE_REFILL_DM" },
    { 0x0302, "L2D_CACHE_REFILL_HWPRF" },
    { 0x0308, "L2_MISS_WAIT" },
    { 0x0309, "L2_MISS_COUNT" },
    { 0x0314, "BUS_READ_TOTAL_TOFU" },
    { 0x0315, "BUS_READ_TOTAL_PCI" },
    { 0x0316, "BUS_READ_TOTAL_MEM" },
    { 0x0318, "BUS_WRITE_TOTAL_CMG0" },
    { 0x0319, "BUS_WRITE_TOTAL_CMG1" },
    { 0x031a, "BUS_WRITE_TOTAL_CMG2" },
    { 0x031b, "BUS_WRITE_TOTAL_CMG3" },
    { 0x031c, "BUS_WRITE_TOTAL_TOFU" },
    { 0x031d, "BUS_WRITE_TOTAL_PCI" },
    { 0x031e, "BUS_WRITE_TOTAL_MEM" },
    { 0x0325, "L2D_SWAP_DM" },
    { 0x0326, "L2D_CACHE_MIBMCH_PRF" },
    { 0x0330, "L2_PIPE_VAL" },
    { 0x0350, "L2_PIPE_COMP_ALL" },
    { 0x0370, "L2_PIPE_COMP_PF_L2MIB_MCH" },
    { 0x0396, "L2D_CACHE_SWAP_LOCAL" },
    { 0x03e0, "EA_L2" },
    { 0x03e8, "EA_MEMORY" },
    { 0x8000, "SIMD_INST_RETIRED" },
    { 0x8002, "SVE_INST_RETIRED" },
    { 0x8008, "UOP_SPEC" },
    { 0x800e, "SVE_MATH_SPEC" },
    { 0x8010, "FP_SPEC" },
    { 0x8028, "FP_FMA_SPEC" },
    { 0x8034, "FP_RECPE_SPEC" },
    { 0x8038, "FP_CVT_SPEC" },
    { 0x8043, "ASE_SVE_INT_SPEC" },
    { 0x8074, "SVE_PRED_SPEC" },
    { 0x807c, "SVE_MOVPRFX_SPEC" },
    { 0x807f, "SVE_MOVPRFX_U_SPEC" },
    { 0x8085, "ASE_SVE_LD_SPEC" },
    { 0x8086, "ASE_SVE_ST_SPEC" },
    { 0x8087, "PRF_SPEC" },
    { 0x8089, "BASE_LD_REG_SPEC" },
    { 0x808a, "BASE_ST_REG_SPEC" },
    { 0x8091, "SVE_LDR_REG_SPEC" },
    { 0x8092, "SVE_STR_REG_SPEC" },
    { 0x8095, "SVE_LDR_PREG_SPEC" },
    { 0x8096, "SVE_STR_PREG_SPEC" },
    { 0x809f, "SVE_PRF_CONTIG_SPEC" },
    { 0x80a5, "ASE_SVE_LD_MULTI_SPEC" },
    { 0x80a6, "ASE_SVE_ST_MULTI_SPEC" },
    { 0x80ad, "SVE_LD_GATHER_SPEC" },
    { 0x80ae, "SVE_ST_SCATTER_SPEC" },
    { 0x80af, "SVE_PRF_GATHER_SPEC" },
    { 0x80bc, "SVE_LDFF_SPEC" },
    { 0x80c0, "FP_SCALE_OPS_SPEC" },
    { 0x80c1, "FP_FIXED_OPS_SPEC" },
    { 0x80c2, "FP_HP_SCALE_OPS_SPEC" },
    { 0x80c3, "FP_HP_FIXED_OPS_SPEC" },
    { 0x80c4, "FP_SP_SCALE_OPS_SPEC" },
    { 0x80c5, "FP_SP_FIXED_OPS_SPEC" },
    { 0x80c6, "FP_DP_SCALE_OPS_SPEC" },
    { 0x80c7, "FP_DP_FIXED_OPS_SPEC" },
};

#define PMU_EVENT_TABLE_SIZE \
    (sizeof(pmu_event_table) / sizeof(pmu_event_table[0]))

#endif /* PMU_EVENTS_H */
