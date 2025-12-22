#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

// =============================================================================
// Section 1: PMU Event Map
// =============================================================================

const EVENT_MAP = {
  // ARMv8 Common Events
  '0x0000': { name: 'SW_INCR', desc: 'Software increment' },
  '0x0001': { name: 'L1I_CACHE_REFILL', desc: 'L1I cache refill' },
  '0x0002': { name: 'L1I_TLB_REFILL', desc: 'L1I TLB refill' },
  '0x0003': { name: 'L1D_CACHE_REFILL', desc: 'L1D cache refill' },
  '0x0004': { name: 'L1D_CACHE', desc: 'L1D cache access' },
  '0x0005': { name: 'L1D_TLB_REFILL', desc: 'L1D TLB refill' },
  '0x0008': { name: 'INST_RETIRED', desc: 'Instructions retired' },
  '0x0009': { name: 'EXC_TAKEN', desc: 'Exceptions taken' },
  '0x000a': { name: 'EXC_RETURN', desc: 'Exception returns' },
  '0x000b': { name: 'CID_WRITE_RETIRED', desc: 'Context ID write retired' },
  '0x0010': { name: 'BR_MIS_PRED', desc: 'Branch mispredictions' },
  '0x0011': { name: 'CPU_CYCLES', desc: 'CPU cycles' },
  '0x0012': { name: 'BR_PRED', desc: 'Branch predictions' },
  '0x0014': { name: 'L1I_CACHE', desc: 'L1I cache access' },
  '0x0015': { name: 'L1D_CACHE_WB', desc: 'L1D cache write-back' },
  '0x0016': { name: 'L2D_CACHE', desc: 'L2 cache access' },
  '0x0017': { name: 'L2D_CACHE_REFILL', desc: 'L2 cache refill' },
  '0x0018': { name: 'L2D_CACHE_WB', desc: 'L2 cache write-back' },
  '0x001b': { name: 'INST_SPEC', desc: 'Instructions speculatively executed' },
  '0x0023': { name: 'STALL_FRONTEND', desc: 'Frontend stall cycles' },
  '0x0024': { name: 'STALL_BACKEND', desc: 'Backend stall cycles' },
  '0x002d': { name: 'L2D_TLB_REFILL', desc: 'L2D TLB refill' },
  '0x002e': { name: 'L2I_TLB_REFILL', desc: 'L2I TLB refill' },
  '0x002f': { name: 'L2D_TLB', desc: 'L2D TLB access' },
  '0x0030': { name: 'L2I_TLB', desc: 'L2I TLB access' },
  '0x0049': { name: 'L1D_CACHE_REFILL_PRF', desc: 'L1D cache refill by prefetch' },
  '0x0059': { name: 'L2D_CACHE_REFILL_PRF', desc: 'L2D cache refill by prefetch' },
  '0x006c': { name: 'LDREX_SPEC', desc: 'Load-exclusive instructions' },
  '0x006f': { name: 'STREX_SPEC', desc: 'Store-exclusive instructions' },
  '0x0070': { name: 'LD_SPEC', desc: 'Load instructions' },
  '0x0071': { name: 'ST_SPEC', desc: 'Store instructions' },
  '0x0072': { name: 'LDST_SPEC', desc: 'Load/store instructions' },
  '0x0073': { name: 'DP_SPEC', desc: 'Integer data-processing instructions' },
  '0x0074': { name: 'ASE_SPEC', desc: 'Advanced SIMD instructions' },
  '0x0075': { name: 'VFP_SPEC', desc: 'Floating-point instructions' },
  '0x0076': { name: 'PC_WRITE_SPEC', desc: 'Software PC change' },
  '0x0077': { name: 'CRYPTO_SPEC', desc: 'Cryptographic instructions' },
  '0x0078': { name: 'BR_IMMED_SPEC', desc: 'Immediate branch instructions' },
  '0x0079': { name: 'BR_RETURN_SPEC', desc: 'Procedure return instructions' },
  '0x007a': { name: 'BR_INDIRECT_SPEC', desc: 'Indirect branch instructions' },
  '0x007c': { name: 'ISB_SPEC', desc: 'ISB instructions' },
  '0x007d': { name: 'DSB_SPEC', desc: 'DSB instructions' },
  '0x007e': { name: 'DMB_SPEC', desc: 'DMB instructions' },
  '0x0081': { name: 'EXC_UNDEF', desc: 'Undefined exceptions' },
  '0x0082': { name: 'EXC_SVC', desc: 'SVC exceptions' },
  '0x0083': { name: 'EXC_PABORT', desc: 'Prefetch abort exceptions' },
  '0x0084': { name: 'EXC_DABORT', desc: 'Data abort exceptions' },
  '0x0086': { name: 'EXC_IRQ', desc: 'IRQ exceptions' },
  '0x0087': { name: 'EXC_FIQ', desc: 'FIQ exceptions' },
  '0x0088': { name: 'EXC_SMC', desc: 'SMC exceptions' },
  '0x008a': { name: 'EXC_HVC', desc: 'HVC exceptions' },
  '0x009f': { name: 'DCZVA_SPEC', desc: 'DC ZVA instructions' },

  // A64FX Specific Events
  '0x0105': { name: 'FP_MV_SPEC', desc: 'FP move operations' },
  '0x0108': { name: 'PRD_SPEC', desc: 'Predicate register operations' },
  '0x0109': { name: 'IEL_SPEC', desc: 'Inter-element operations' },
  '0x010a': { name: 'IREG_SPEC', desc: 'Inter-register operations' },
  '0x0112': { name: 'FP_LD_SPEC', desc: 'FP load operations' },
  '0x0113': { name: 'FP_ST_SPEC', desc: 'FP store operations' },
  '0x011a': { name: 'BC_LD_SPEC', desc: 'Broadcast load operations' },
  '0x0121': { name: 'EFFECTIVE_INST_SPEC', desc: 'Effective instructions (excl. MOVPRFX)' },
  '0x0123': { name: 'PRE_INDEX_SPEC', desc: 'Pre-index operations' },
  '0x0124': { name: 'POST_INDEX_SPEC', desc: 'Post-index operations' },
  '0x0139': { name: 'UOP_SPLIT', desc: 'Micro-operation splits' },
  '0x0180': { name: 'LD_COMP_WAIT_L2_MISS', desc: 'Cycles waiting for memory (L2 miss)' },
  '0x0181': { name: 'LD_COMP_WAIT_L2_MISS_EX', desc: 'Cycles waiting for memory (L2 miss, int load)' },
  '0x0182': { name: 'LD_COMP_WAIT_L1_MISS', desc: 'Cycles waiting for L2 (L1 miss)' },
  '0x0183': { name: 'LD_COMP_WAIT_L1_MISS_EX', desc: 'Cycles waiting for L2 (L1 miss, int load)' },
  '0x0184': { name: 'LD_COMP_WAIT', desc: 'Cycles waiting for cache/memory' },
  '0x0185': { name: 'LD_COMP_WAIT_EX', desc: 'Cycles waiting for cache/memory (int load)' },
  '0x0186': { name: 'LD_COMP_WAIT_PFP_BUSY', desc: 'Cycles waiting for prefetch port' },
  '0x0187': { name: 'LD_COMP_WAIT_PFP_BUSY_EX', desc: 'Cycles waiting for prefetch port (int load)' },
  '0x0188': { name: 'LD_COMP_WAIT_PFP_BUSY_SWPF', desc: 'Cycles waiting for prefetch port (SW prefetch)' },
  '0x0189': { name: 'EU_COMP_WAIT', desc: 'Cycles waiting for int/FP instruction' },
  '0x018a': { name: 'FL_COMP_WAIT', desc: 'Cycles waiting for FP/SIMD instruction' },
  '0x018b': { name: 'BR_COMP_WAIT', desc: 'Cycles waiting for branch instruction' },
  '0x018c': { name: 'ROB_EMPTY', desc: 'Cycles with empty CSE' },
  '0x018d': { name: 'ROB_EMPTY_STQ_BUSY', desc: 'Cycles with empty CSE and full SP' },
  '0x018e': { name: 'WFE_WFI_CYCLE', desc: 'Cycles halted by WFE/WFI' },
  '0x0190': { name: '0INST_COMMIT', desc: 'Cycles with 0 instructions committed' },
  '0x0191': { name: '1INST_COMMIT', desc: 'Cycles with 1 instruction committed' },
  '0x0192': { name: '2INST_COMMIT', desc: 'Cycles with 2 instructions committed' },
  '0x0193': { name: '3INST_COMMIT', desc: 'Cycles with 3 instructions committed' },
  '0x0194': { name: '4INST_COMMIT', desc: 'Cycles with 4 instructions committed' },
  '0x0198': { name: 'UOP_ONLY_COMMIT', desc: 'Cycles with only micro-ops committed' },
  '0x0199': { name: 'SINGLE_MOVPRFX_COMMIT', desc: 'Cycles with only MOVPRFX committed' },
  '0x01a0': { name: 'EAGA_VAL', desc: 'EAGA pipeline valid cycles' },
  '0x01a1': { name: 'EAGB_VAL', desc: 'EAGB pipeline valid cycles' },
  '0x01a2': { name: 'EXA_VAL', desc: 'EXA pipeline valid cycles' },
  '0x01a3': { name: 'EXB_VAL', desc: 'EXB pipeline valid cycles' },
  '0x01a4': { name: 'FLA_VAL', desc: 'FLA pipeline valid cycles' },
  '0x01a5': { name: 'FLB_VAL', desc: 'FLB pipeline valid cycles' },
  '0x01a6': { name: 'PRX_VAL', desc: 'PRX pipeline valid cycles' },
  '0x01b4': { name: 'FLA_VAL_PRD_CNT', desc: 'FLA pipeline predicate count' },
  '0x01b5': { name: 'FLB_VAL_PRD_CNT', desc: 'FLB pipeline predicate count' },
  '0x01e0': { name: 'EA_CORE', desc: 'Core energy consumption (8nJ/count)' },
  '0x0200': { name: 'L1D_CACHE_REFILL_DM', desc: 'L1D cache refill (demand)' },
  '0x0202': { name: 'L1D_CACHE_REFILL_HWPRF', desc: 'L1D cache refill (HW prefetch)' },
  '0x0208': { name: 'L1_MISS_WAIT', desc: 'Outstanding L1D miss requests' },
  '0x0209': { name: 'L1I_MISS_WAIT', desc: 'Outstanding L1I miss requests' },
  '0x0230': { name: 'L1HWPF_STREAM_PF', desc: 'L1 HW streaming prefetch requests' },
  '0x0231': { name: 'L1HWPF_INJ_ALLOC_PF', desc: 'L1 HW prefetch injection (alloc)' },
  '0x0232': { name: 'L1HWPF_INJ_NOALLOC_PF', desc: 'L1 HW prefetch injection (non-alloc)' },
  '0x0233': { name: 'L2HWPF_STREAM_PF', desc: 'L2 HW streaming prefetch requests' },
  '0x0234': { name: 'L2HWPF_INJ_ALLOC_PF', desc: 'L2 HW prefetch injection (alloc)' },
  '0x0235': { name: 'L2HWPF_INJ_NOALLOC_PF', desc: 'L2 HW prefetch injection (non-alloc)' },
  '0x0236': { name: 'L2HWPF_OTHER', desc: 'L2 HW prefetch (other)' },
  '0x0240': { name: 'L1_PIPE0_VAL', desc: 'L1D pipeline#0 valid cycles' },
  '0x0241': { name: 'L1_PIPE1_VAL', desc: 'L1D pipeline#1 valid cycles' },
  '0x0250': { name: 'L1_PIPE0_VAL_IU_TAG_ADRS_SCE', desc: 'L1D pipe#0 requests with sce=1' },
  '0x0251': { name: 'L1_PIPE0_VAL_IU_TAG_ADRS_PFE', desc: 'L1D pipe#0 requests with pfe=1' },
  '0x0252': { name: 'L1_PIPE1_VAL_IU_TAG_ADRS_SCE', desc: 'L1D pipe#1 requests with sce=1' },
  '0x0253': { name: 'L1_PIPE1_VAL_IU_TAG_ADRS_PFE', desc: 'L1D pipe#1 requests with pfe=1' },
  '0x0260': { name: 'L1_PIPE0_COMP', desc: 'L1D pipeline#0 completed requests' },
  '0x0261': { name: 'L1_PIPE1_COMP', desc: 'L1D pipeline#1 completed requests' },
  '0x0268': { name: 'L1I_PIPE_COMP', desc: 'L1I pipeline completed requests' },
  '0x0269': { name: 'L1I_PIPE_VAL', desc: 'L1I pipeline valid cycles' },
  '0x0274': { name: 'L1_PIPE_ABORT_STLD_INTLK', desc: 'L1D pipe abort (store-load interlock)' },
  '0x02a0': { name: 'L1_PIPE0_VAL_IU_NOT_SEC0', desc: 'L1D pipe#0 requests (sector != 0)' },
  '0x02a1': { name: 'L1_PIPE1_VAL_IU_NOT_SEC0', desc: 'L1D pipe#1 requests (sector != 0)' },
  '0x02b0': { name: 'L1_PIPE_COMP_GATHER_2FLOW', desc: 'Gather 2-element 2-flow' },
  '0x02b1': { name: 'L1_PIPE_COMP_GATHER_1FLOW', desc: 'Gather 2-element 1-flow' },
  '0x02b2': { name: 'L1_PIPE_COMP_GATHER_0FLOW', desc: 'Gather 2-element 0-flow' },
  '0x02b3': { name: 'L1_PIPE_COMP_SCATTER_1FLOW', desc: 'Scatter flows' },
  '0x02b8': { name: 'L1_PIPE0_COMP_PRD_CNT', desc: 'L1D pipe#0 predicate count' },
  '0x02b9': { name: 'L1_PIPE1_COMP_PRD_CNT', desc: 'L1D pipe#1 predicate count' },
  '0x0300': { name: 'L2D_CACHE_REFILL_DM', desc: 'L2 cache refill (demand)' },
  '0x0302': { name: 'L2D_CACHE_REFILL_HWPRF', desc: 'L2 cache refill (HW prefetch)' },
  '0x0308': { name: 'L2_MISS_WAIT', desc: 'Outstanding L2 miss requests' },
  '0x0309': { name: 'L2_MISS_COUNT', desc: 'L2 cache misses' },
  '0x0314': { name: 'BUS_READ_TOTAL_TOFU', desc: 'Bus read from Tofu' },
  '0x0315': { name: 'BUS_READ_TOTAL_PCI', desc: 'Bus read from PCI' },
  '0x0316': { name: 'BUS_READ_TOTAL_MEM', desc: 'Bus read from memory' },
  '0x0318': { name: 'BUS_WRITE_TOTAL_CMG0', desc: 'Bus write to CMG0' },
  '0x0319': { name: 'BUS_WRITE_TOTAL_CMG1', desc: 'Bus write to CMG1' },
  '0x031a': { name: 'BUS_WRITE_TOTAL_CMG2', desc: 'Bus write to CMG2' },
  '0x031b': { name: 'BUS_WRITE_TOTAL_CMG3', desc: 'Bus write to CMG3' },
  '0x031c': { name: 'BUS_WRITE_TOTAL_TOFU', desc: 'Bus write to Tofu' },
  '0x031d': { name: 'BUS_WRITE_TOTAL_PCI', desc: 'Bus write to PCI' },
  '0x031e': { name: 'BUS_WRITE_TOTAL_MEM', desc: 'Bus write to memory' },
  '0x0325': { name: 'L2D_SWAP_DM', desc: 'L2 swap (demand hits prefetch buffer)' },
  '0x0326': { name: 'L2D_CACHE_MIBMCH_PRF', desc: 'L2 prefetch hits demand buffer' },
  '0x0330': { name: 'L2_PIPE_VAL', desc: 'L2 pipeline valid cycles' },
  '0x0350': { name: 'L2_PIPE_COMP_ALL', desc: 'L2 pipeline completed requests' },
  '0x0370': { name: 'L2_PIPE_COMP_PF_L2MIB_MCH', desc: 'L2 prefetch hits demand buffer' },
  '0x0391': { name: 'L2_OC_RD_MIB_HIT', desc: 'L2 OC read MIB hit' },
  '0x0396': { name: 'L2D_CACHE_SWAP_LOCAL', desc: 'L2 swap local' },
  '0x03ae': { name: 'L2_OC_WR_MIB_HIT', desc: 'L2 OC write MIB hit' },
  '0x03e0': { name: 'EA_L2', desc: 'L2 energy consumption (32nJ/count)' },
  '0x03e8': { name: 'EA_MEMORY', desc: 'Memory energy consumption (256nJ/count)' },

  // SVE Common Events
  '0x8000': { name: 'SIMD_INST_RETIRED', desc: 'SIMD instructions retired' },
  '0x8002': { name: 'SVE_INST_RETIRED', desc: 'SVE instructions retired' },
  '0x8008': { name: 'UOP_SPEC', desc: 'Micro-operations' },
  '0x800e': { name: 'SVE_MATH_SPEC', desc: 'SVE math operations' },
  '0x8010': { name: 'FP_SPEC', desc: 'Floating-point operations' },
  '0x8028': { name: 'FP_FMA_SPEC', desc: 'FP fused multiply-add operations' },
  '0x8034': { name: 'FP_RECPE_SPEC', desc: 'FP reciprocal estimate operations' },
  '0x8038': { name: 'FP_CVT_SPEC', desc: 'FP convert operations' },
  '0x8043': { name: 'ASE_SVE_INT_SPEC', desc: 'SIMD/SVE integer operations' },
  '0x8074': { name: 'SVE_PRED_SPEC', desc: 'SVE predicated operations' },
  '0x807c': { name: 'SVE_MOVPRFX_SPEC', desc: 'SVE MOVPRFX operations' },
  '0x807f': { name: 'SVE_MOVPRFX_U_SPEC', desc: 'SVE MOVPRFX unfused operations' },
  '0x8085': { name: 'ASE_SVE_LD_SPEC', desc: 'SIMD/SVE load operations' },
  '0x8086': { name: 'ASE_SVE_ST_SPEC', desc: 'SIMD/SVE store operations' },
  '0x8087': { name: 'PRF_SPEC', desc: 'Prefetch operations' },
  '0x8089': { name: 'BASE_LD_REG_SPEC', desc: 'Base register load operations' },
  '0x808a': { name: 'BASE_ST_REG_SPEC', desc: 'Base register store operations' },
  '0x8091': { name: 'SVE_LDR_REG_SPEC', desc: 'SVE LDR operations' },
  '0x8092': { name: 'SVE_STR_REG_SPEC', desc: 'SVE STR operations' },
  '0x8095': { name: 'SVE_LDR_PREG_SPEC', desc: 'SVE LDR predicate operations' },
  '0x8096': { name: 'SVE_STR_PREG_SPEC', desc: 'SVE STR predicate operations' },
  '0x809f': { name: 'SVE_PRF_CONTIG_SPEC', desc: 'SVE contiguous prefetch operations' },
  '0x80a5': { name: 'ASE_SVE_LD_MULTI_SPEC', desc: 'SIMD/SVE multi-vector load' },
  '0x80a6': { name: 'ASE_SVE_ST_MULTI_SPEC', desc: 'SIMD/SVE multi-vector store' },
  '0x80ad': { name: 'SVE_LD_GATHER_SPEC', desc: 'SVE gather load operations' },
  '0x80ae': { name: 'SVE_ST_SCATTER_SPEC', desc: 'SVE scatter store operations' },
  '0x80af': { name: 'SVE_PRF_GATHER_SPEC', desc: 'SVE gather prefetch operations' },
  '0x80bc': { name: 'SVE_LDFF_SPEC', desc: 'SVE first-fault load operations' },
  '0x80c0': { name: 'FP_SCALE_OPS_SPEC', desc: 'SVE FP operations (scaled)' },
  '0x80c1': { name: 'FP_FIXED_OPS_SPEC', desc: 'SIMD FP operations (fixed)' },
  '0x80c2': { name: 'FP_HP_SCALE_OPS_SPEC', desc: 'SVE half-precision FP ops' },
  '0x80c3': { name: 'FP_HP_FIXED_OPS_SPEC', desc: 'SIMD half-precision FP ops' },
  '0x80c4': { name: 'FP_SP_SCALE_OPS_SPEC', desc: 'SVE single-precision FP ops' },
  '0x80c5': { name: 'FP_SP_FIXED_OPS_SPEC', desc: 'SIMD single-precision FP ops' },
  '0x80c6': { name: 'FP_DP_SCALE_OPS_SPEC', desc: 'SVE double-precision FP ops' },
  '0x80c7': { name: 'FP_DP_FIXED_OPS_SPEC', desc: 'SIMD double-precision FP ops' },
};

// =============================================================================
// Section 2: CSV Parser Functions
// =============================================================================

function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current);
  return result;
}

function parseCSV(content) {
  const lines = content.split('\n').filter(l => l.trim());
  const data = {
    info: {},
    cpuFrequency: 0,
    timeStats: {},
    cpupa: {}
  };

  let cpupaHeaders = [];

  for (const line of lines) {
    const fields = parseCSVLine(line);
    const label = fields[0];

    if (label === 'FAPP-info') {
      const name = fields[1];
      const value = fields[2];
      if (name === 'Profiler version') data.info.profilerVersion = value;
      else if (name === 'Measured time') data.info.measuredTime = value;
      else if (name === 'Timer clock frequency') data.info.timerFrequency = parseInt(value);
      else if (name === 'Vector length') data.info.vectorLength = parseInt(value);
    }
    else if (label === 'FAPP-cpu-freq') {
      // fields: FAPP-cpu-freq, Spawn, Process-start, Process-end, Frequency
      data.cpuFrequency = parseInt(fields[4]);
    }
    else if (label === 'FAPP-time-stat') {
      // fields: FAPP-time-stat, Spawn, Process, Name, Number, Elapsed, User, System, Call
      const regionName = fields[3];
      const elapsed = parseFloat(fields[5]);
      data.timeStats[regionName] = elapsed;
    }
    else if (label === 'LABEL-FAPP-cpupa') {
      cpupaHeaders = fields.slice(1);
    }
    else if (label === 'FAPP-cpupa') {
      const level = fields[1];
      if (level !== 'Process') continue;

      const regionName = fields[5];
      if (!data.cpupa[regionName]) {
        data.cpupa[regionName] = { counters: {} };
      }

      for (let i = 9; i < fields.length && i < cpupaHeaders.length + 1; i++) {
        const headerIdx = i - 1;
        const header = cpupaHeaders[headerIdx];
        if (header && header.startsWith('0x')) {
          const value = parseInt(fields[i]);
          if (!isNaN(value)) {
            data.cpupa[regionName].counters[header] = value;
          }
        } else if (header === 'PMCCNTR') {
          data.cpupa[regionName].counters['PMCCNTR'] = parseInt(fields[i]);
        } else if (header === 'CNTVCT') {
          data.cpupa[regionName].counters['CNTVCT'] = parseInt(fields[i]);
        }
      }
    }
  }

  return data;
}

function loadFiles(maxFiles, baseDir) {
  const mergedData = {
    info: {},
    cpuFrequency: 0,
    timeStats: {},
    regions: {}
  };

  for (let i = 1; i <= maxFiles; i++) {
    const filePath = path.join(baseDir, `pa${i}.csv`);
    if (!fs.existsSync(filePath)) {
      console.error(`Warning: File not found: ${filePath}`);
      continue;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const data = parseCSV(content);

    if (i === 1) {
      mergedData.info = data.info;
      mergedData.cpuFrequency = data.cpuFrequency;
      mergedData.timeStats = data.timeStats;
    }

    for (const [regionName, regionData] of Object.entries(data.cpupa)) {
      if (!mergedData.regions[regionName]) {
        mergedData.regions[regionName] = {
          elapsed: data.timeStats[regionName] || 0,
          counters: {}
        };
      }
      Object.assign(mergedData.regions[regionName].counters, regionData.counters);
    }
  }

  return mergedData;
}

// =============================================================================
// Section 3: Metric Calculation Functions
// =============================================================================

function getCounter(counters, code) {
  return counters[code] !== undefined ? counters[code] : null;
}

function formatValue(value, decimals = 2) {
  if (value === null || value === undefined || isNaN(value)) return 'N/A';
  if (!isFinite(value)) return 'N/A';

  if (Math.abs(value) >= 1e12) return (value / 1e12).toFixed(decimals) + ' T';
  if (Math.abs(value) >= 1e9) return (value / 1e9).toFixed(decimals) + ' G';
  if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(decimals) + ' M';
  if (Math.abs(value) >= 1e3) return (value / 1e3).toFixed(decimals) + ' K';
  return value.toFixed(decimals);
}

function calculateMetrics(counters, elapsed, cpuFreq) {
  const metrics = [];
  const cycles = getCounter(counters, 'PMCCNTR');

  // Core Performance
  const effectiveInst = getCounter(counters, '0x0121');
  const instRetired = getCounter(counters, '0x0008');

  if (effectiveInst !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'IPC',
      value: effectiveInst / cycles,
      unit: '',
      category: 'Core Performance'
    });
    metrics.push({
      name: 'CPI',
      value: cycles / effectiveInst,
      unit: '',
      category: 'Core Performance'
    });
  }

  // Cache Performance
  const l1dRefill = getCounter(counters, '0x0003');
  const l1dCache = getCounter(counters, '0x0004');
  const l2dRefillRaw = getCounter(counters, '0x0017'); // L2D_CACHE_REFILL (raw)
  const l2dCache = getCounter(counters, '0x0016');
  const l1MissWait = getCounter(counters, '0x0208');
  const l2MissWait = getCounter(counters, '0x0308');

  // L1D Miss Rate breakdown
  const l1dRefillPrf = getCounter(counters, '0x0049');  // L1D_CACHE_REFILL_PRF (SW + HW prefetch)
  const l1dRefillHwprf = getCounter(counters, '0x0202'); // L1D_CACHE_REFILL_HWPRF
  const ldSpec = getCounter(counters, '0x0070');        // LD_SPEC
  const stSpec = getCounter(counters, '0x0071');        // ST_SPEC

  // L1D miss rate = L1D_CACHE_REFILL / (LD_SPEC + ST_SPEC)
  if (l1dRefill !== null && ldSpec !== null && stSpec !== null) {
    const ldstSpec = ldSpec + stSpec;
    if (ldstSpec > 0) {
      metrics.push({
        name: 'L1D Miss Rate',
        value: (l1dRefill / ldstSpec) * 100,
        unit: '%',
        category: 'Cache Performance'
      });
    }
  }

  // L1D miss demand/HW/SW rates = respective_refill / L1D_CACHE_REFILL
  if (l1dRefill !== null && l1dRefill > 0 && l1dRefillPrf !== null) {
    const demandRefill = l1dRefill - l1dRefillPrf;
    metrics.push({
      name: 'L1D Miss Demand Rate',
      value: (demandRefill / l1dRefill) * 100,
      unit: '%',
      category: 'Cache Performance'
    });
  }

  if (l1dRefill !== null && l1dRefill > 0 && l1dRefillHwprf !== null) {
    metrics.push({
      name: 'L1D Miss HW Prefetch Rate',
      value: (l1dRefillHwprf / l1dRefill) * 100,
      unit: '%',
      category: 'Cache Performance'
    });
  }

  if (l1dRefill !== null && l1dRefill > 0 && l1dRefillPrf !== null && l1dRefillHwprf !== null) {
    const swPrefetchRefill = l1dRefillPrf - l1dRefillHwprf;
    metrics.push({
      name: 'L1D Miss SW Prefetch Rate',
      value: (swPrefetchRefill / l1dRefill) * 100,
      unit: '%',
      category: 'Cache Performance'
    });
  }

  // L2 Miss Rate breakdown with A64FX PMU Errata corrections
  // Raw counters
  const l2dRefillDmRaw = getCounter(counters, '0x0300');   // L2D_CACHE_REFILL_DM
  const l2dSwapDm = getCounter(counters, '0x0325');        // L2D_SWAP_DM (demand hits prefetch buffer)
  const l2dRefillPrfRaw = getCounter(counters, '0x0059'); // L2D_CACHE_REFILL_PRF (SW + HW)
  const l2dRefillHwprf = getCounter(counters, '0x0302');   // L2D_CACHE_REFILL_HWPRF
  const l2dCacheMibmchPrf = getCounter(counters, '0x0326'); // L2D_CACHE_MIBMCH_PRF
  const l2MissCountRaw = getCounter(counters, '0x0309');   // L2_MISS_COUNT
  const l2dCacheSwapLocal = getCounter(counters, '0x0396'); // L2D_CACHE_SWAP_LOCAL
  const l2PipeCompPfL2mibMch = getCounter(counters, '0x0370'); // L2_PIPE_COMP_PF_L2MIB_MCH

  // A64FX PMU Errata Corrections:
  // L2D_CACHE_REFILL (corrected) = L2D_CACHE_REFILL - L2D_SWAP_DM - L2D_CACHE_MIBMCH_PRF
  // L2D_CACHE_REFILL_DM (corrected) = L2D_CACHE_REFILL_DM - L2D_SWAP_DM
  // L2D_CACHE_REFILL_PRF (corrected) = L2D_CACHE_REFILL_PRF - L2D_CACHE_MIBMCH_PRF
  // L2_MISS_COUNT (corrected) = L2_MISS_COUNT - L2D_CACHE_SWAP_LOCAL - L2_PIPE_COMP_PF_L2MIB_MCH

  // Calculate corrected L2 values
  let l2dRefillDm = null;
  if (l2dRefillDmRaw !== null && l2dSwapDm !== null) {
    l2dRefillDm = l2dRefillDmRaw - l2dSwapDm;
  }

  let l2dRefillPrf = null;
  if (l2dRefillPrfRaw !== null && l2dCacheMibmchPrf !== null) {
    l2dRefillPrf = l2dRefillPrfRaw - l2dCacheMibmchPrf;
  }

  let l2MissCount = null;
  if (l2MissCountRaw !== null && l2dCacheSwapLocal !== null && l2PipeCompPfL2mibMch !== null) {
    l2MissCount = l2MissCountRaw - l2dCacheSwapLocal - l2PipeCompPfL2mibMch;
  }

  // L2 miss rate = corrected L2_MISS_COUNT / (LD_SPEC + ST_SPEC)
  if (l2MissCount !== null && ldSpec !== null && stSpec !== null) {
    const ldstSpec = ldSpec + stSpec;
    if (ldstSpec > 0) {
      metrics.push({
        name: 'L2 Miss Rate',
        value: (l2MissCount / ldstSpec) * 100,
        unit: '%',
        category: 'Cache Performance'
      });
    }
  }

  // L2 miss demand rate = corrected L2D_CACHE_REFILL_DM / corrected L2_MISS_COUNT
  if (l2MissCount !== null && l2MissCount > 0 && l2dRefillDm !== null) {
    metrics.push({
      name: 'L2 Miss Demand Rate',
      value: (l2dRefillDm / l2MissCount) * 100,
      unit: '%',
      category: 'Cache Performance'
    });
  }

  if (l2MissCount !== null && l2MissCount > 0 && l2dRefillHwprf !== null) {
    metrics.push({
      name: 'L2 Miss HW Prefetch Rate',
      value: (l2dRefillHwprf / l2MissCount) * 100,
      unit: '%',
      category: 'Cache Performance'
    });
  }

  // L2 miss SW prefetch rate = corrected L2D_CACHE_REFILL_PRF / corrected L2_MISS_COUNT
  // Note: L2D_CACHE_REFILL_PRF includes both SW and HW prefetch, subtract HW to get SW only
  if (l2MissCount !== null && l2MissCount > 0 && l2dRefillPrf !== null && l2dRefillHwprf !== null) {
    const swPrefetchRefill = l2dRefillPrf - l2dRefillHwprf;
    metrics.push({
      name: 'L2 Miss SW Prefetch Rate',
      value: (swPrefetchRefill / l2MissCount) * 100,
      unit: '%',
      category: 'Cache Performance'
    });
  }

  if (l1MissWait !== null && l1dRefill !== null && l1dRefill > 0) {
    metrics.push({
      name: 'L1D Miss Latency',
      value: l1MissWait / l1dRefill,
      unit: 'cycles',
      category: 'Cache Performance'
    });
  }

  // L2D_CACHE_REFILL (corrected) = L2D_CACHE_REFILL - L2D_SWAP_DM - L2D_CACHE_MIBMCH_PRF
  let l2dRefillCorrected = null;
  if (l2dRefillRaw !== null && l2dSwapDm !== null && l2dCacheMibmchPrf !== null) {
    l2dRefillCorrected = l2dRefillRaw - l2dSwapDm - l2dCacheMibmchPrf;
  }

  if (l2MissWait !== null && l2dRefillCorrected !== null && l2dRefillCorrected > 0) {
    metrics.push({
      name: 'L2 Miss Latency',
      value: l2MissWait / l2dRefillCorrected,
      unit: 'cycles',
      category: 'Cache Performance'
    });
  }

  // Floating-Point Performance
  // FP_SCALE_OPS_SPEC (0x80c0) is scaled for 128-bit vectors
  // A64FX has 512-bit SVE vectors, so multiply by 512/128 = 4
  const fpScaleOps = getCounter(counters, '0x80c0');
  const fpFixedOps = getCounter(counters, '0x80c1');
  const sveScaleFactor = 4; // 512-bit / 128-bit

  if (fpScaleOps !== null && elapsed > 0) {
    const gflops = (fpScaleOps * sveScaleFactor) / (elapsed * 1e9);
    metrics.push({
      name: 'GFLOPS (SVE)',
      value: gflops,
      unit: '',
      category: 'Floating-Point Performance'
    });
  }

  if (fpFixedOps !== null && elapsed > 0) {
    const gflops = fpFixedOps / (elapsed * 1e9);
    metrics.push({
      name: 'GFLOPS (SIMD)',
      value: gflops,
      unit: '',
      category: 'Floating-Point Performance'
    });
  }

  if (fpScaleOps !== null || fpFixedOps !== null) {
    const totalFp = (fpScaleOps ? fpScaleOps * sveScaleFactor : 0) + (fpFixedOps || 0);
    if (elapsed > 0) {
      metrics.push({
        name: 'GFLOPS (Total)',
        value: totalFp / (elapsed * 1e9),
        unit: '',
        category: 'Floating-Point Performance'
      });
    }
    metrics.push({
      name: 'FP Operations',
      value: totalFp,
      unit: '',
      category: 'Floating-Point Performance',
      format: 'count'
    });
  }

  // Memory Bandwidth
  const busReadMem = getCounter(counters, '0x0316');
  const busWriteMem = getCounter(counters, '0x031e');

  if (busReadMem !== null && elapsed > 0) {
    const bw = (busReadMem * 256) / (elapsed * 1e9);
    metrics.push({
      name: 'Memory Read BW',
      value: bw,
      unit: 'GB/s',
      category: 'Memory Bandwidth'
    });
  }

  if (busWriteMem !== null && elapsed > 0) {
    const bw = (busWriteMem * 256) / (elapsed * 1e9);
    metrics.push({
      name: 'Memory Write BW',
      value: bw,
      unit: 'GB/s',
      category: 'Memory Bandwidth'
    });
  }

  // Memory Busy Rate
  if (busReadMem !== null && busWriteMem !== null && cycles !== null && cycles > 0) {
    // Memory busy rate: transactions * empirical factor / cycles
    // Factor ~2.17 accounts for memory controller characteristics
    const memBusy = ((busReadMem + busWriteMem) * 2.17 / cycles) * 100;
    metrics.push({
      name: 'Memory Busy',
      value: memBusy,
      unit: '%',
      category: 'Busy Rate'
    });
  }

  // Pipeline Utilization
  const zeroInstCommit = getCounter(counters, '0x0190');
  const ldCompWait = getCounter(counters, '0x0184');
  const flCompWait = getCounter(counters, '0x018a');
  const euCompWait = getCounter(counters, '0x0189');

  if (zeroInstCommit !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Stall Rate (0-commit)',
      value: (zeroInstCommit / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  if (ldCompWait !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Memory Stall',
      value: (ldCompWait / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  if (flCompWait !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'FP/SIMD Stall',
      value: (flCompWait / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  // Pipeline Valid Cycles
  const flaVal = getCounter(counters, '0x01a4');
  const flbVal = getCounter(counters, '0x01a5');

  if (flaVal !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'FLA Utilization',
      value: (flaVal / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  if (flbVal !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'FLB Utilization',
      value: (flbVal / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  // Integer Pipeline Utilization
  const exaVal = getCounter(counters, '0x01a2');
  const exbVal = getCounter(counters, '0x01a3');

  if (exaVal !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Integer A Utilization',
      value: (exaVal / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  if (exbVal !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Integer B Utilization',
      value: (exbVal / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  // L1 Cache Busy Rate (average of two pipelines)
  const l1Pipe0Val = getCounter(counters, '0x0240');
  const l1Pipe1Val = getCounter(counters, '0x0241');

  if (l1Pipe0Val !== null && l1Pipe1Val !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'L1 Busy',
      value: ((l1Pipe0Val + l1Pipe1Val) / (2 * cycles)) * 100,
      unit: '%',
      category: 'Busy Rate'
    });
  }

  // L2 Cache Busy Rate (shared across cores, divide by 2)
  const l2PipeVal = getCounter(counters, '0x0330');

  if (l2PipeVal !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'L2 Busy',
      value: (l2PipeVal / (2 * cycles)) * 100,
      unit: '%',
      category: 'Busy Rate'
    });
  }

  // SVE Utilization
  const sveInstRetired = getCounter(counters, '0x8002');
  const simdInstRetired = getCounter(counters, '0x8000');

  if (sveInstRetired !== null && instRetired !== null && instRetired > 0) {
    metrics.push({
      name: 'SVE Instruction Ratio',
      value: (sveInstRetired / instRetired) * 100,
      unit: '%',
      category: 'SVE Utilization'
    });
  }

  if (simdInstRetired !== null) {
    metrics.push({
      name: 'SIMD Instructions',
      value: simdInstRetired,
      unit: '',
      category: 'SVE Utilization',
      format: 'count'
    });
  }

  if (sveInstRetired !== null) {
    metrics.push({
      name: 'SVE Instructions',
      value: sveInstRetired,
      unit: '',
      category: 'SVE Utilization',
      format: 'count'
    });
  }

  // Energy (if available)
  const eaCore = getCounter(counters, '0x01e0');
  const eaL2 = getCounter(counters, '0x03e0');
  const eaMemory = getCounter(counters, '0x03e8');

  if (eaCore !== null) {
    metrics.push({
      name: 'Core Energy',
      value: eaCore * 8 / 1e9,
      unit: 'J',
      category: 'Energy'
    });
  }

  if (eaL2 !== null) {
    metrics.push({
      name: 'L2 Energy',
      value: eaL2 * 32 / 1e9,
      unit: 'J',
      category: 'Energy'
    });
  }

  if (eaMemory !== null) {
    metrics.push({
      name: 'Memory Energy',
      value: eaMemory * 256 / 1e9,
      unit: 'J',
      category: 'Energy'
    });
  }

  // Branch Prediction
  const brMisPred = getCounter(counters, '0x0010');
  const brPred = getCounter(counters, '0x0012');

  if (brMisPred !== null && brPred !== null && brPred > 0) {
    metrics.push({
      name: 'Branch Mispredict Rate',
      value: (brMisPred / brPred) * 100,
      unit: '%',
      category: 'Branch Prediction'
    });
  }

  if (brMisPred !== null) {
    metrics.push({
      name: 'Branch Mispredictions',
      value: brMisPred,
      unit: '',
      category: 'Branch Prediction',
      format: 'count'
    });
  }

  if (brPred !== null) {
    metrics.push({
      name: 'Branch Predictions',
      value: brPred,
      unit: '',
      category: 'Branch Prediction',
      format: 'count'
    });
  }

  // TLB Performance
  const l1dTlbRefill = getCounter(counters, '0x0005');
  const l2dTlbRefill = getCounter(counters, '0x002d');
  const l1iTlbRefill = getCounter(counters, '0x0002');

  if (l1dTlbRefill !== null && ldSpec !== null && stSpec !== null) {
    const ldstSpec = ldSpec + stSpec;
    if (ldstSpec > 0) {
      metrics.push({
        name: 'L1D TLB Miss Rate',
        value: (l1dTlbRefill / ldstSpec) * 100,
        unit: '%',
        category: 'TLB Performance'
      });
    }
  }

  if (l2dTlbRefill !== null && l1dTlbRefill !== null && l1dTlbRefill > 0) {
    metrics.push({
      name: 'L2D TLB Miss Rate',
      value: (l2dTlbRefill / l1dTlbRefill) * 100,
      unit: '%',
      category: 'TLB Performance'
    });
  }

  if (l1dTlbRefill !== null) {
    metrics.push({
      name: 'L1D TLB Misses',
      value: l1dTlbRefill,
      unit: '',
      category: 'TLB Performance',
      format: 'count'
    });
  }

  if (l2dTlbRefill !== null) {
    metrics.push({
      name: 'L2D TLB Misses',
      value: l2dTlbRefill,
      unit: '',
      category: 'TLB Performance',
      format: 'count'
    });
  }

  // Detailed Memory Stall Breakdown
  const ldCompWaitL2Miss = getCounter(counters, '0x0180');
  const ldCompWaitL1Miss = getCounter(counters, '0x0182');
  const ldCompWaitPfpBusy = getCounter(counters, '0x0186');

  if (ldCompWaitL2Miss !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'L2 Miss Stall',
      value: (ldCompWaitL2Miss / cycles) * 100,
      unit: '%',
      category: 'Memory Stall Breakdown'
    });
  }

  if (ldCompWaitL1Miss !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'L1 Miss Stall',
      value: (ldCompWaitL1Miss / cycles) * 100,
      unit: '%',
      category: 'Memory Stall Breakdown'
    });
  }

  if (ldCompWaitPfpBusy !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Prefetch Port Stall',
      value: (ldCompWaitPfpBusy / cycles) * 100,
      unit: '%',
      category: 'Memory Stall Breakdown'
    });
  }

  // Instruction Commit Distribution
  const oneInstCommit = getCounter(counters, '0x0191');
  const twoInstCommit = getCounter(counters, '0x0192');
  const threeInstCommit = getCounter(counters, '0x0193');
  const fourInstCommit = getCounter(counters, '0x0194');

  if (cycles !== null && cycles > 0) {
    if (oneInstCommit !== null) {
      metrics.push({
        name: '1-Inst Commit Rate',
        value: (oneInstCommit / cycles) * 100,
        unit: '%',
        category: 'Commit Distribution'
      });
    }
    if (twoInstCommit !== null) {
      metrics.push({
        name: '2-Inst Commit Rate',
        value: (twoInstCommit / cycles) * 100,
        unit: '%',
        category: 'Commit Distribution'
      });
    }
    if (threeInstCommit !== null) {
      metrics.push({
        name: '3-Inst Commit Rate',
        value: (threeInstCommit / cycles) * 100,
        unit: '%',
        category: 'Commit Distribution'
      });
    }
    if (fourInstCommit !== null) {
      metrics.push({
        name: '4-Inst Commit Rate',
        value: (fourInstCommit / cycles) * 100,
        unit: '%',
        category: 'Commit Distribution'
      });
    }
  }

  // Address Generation Pipeline Utilization
  const eagaVal = getCounter(counters, '0x01a0');
  const eagbVal = getCounter(counters, '0x01a1');
  const prxVal = getCounter(counters, '0x01a6');

  if (eagaVal !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'EAGA Utilization',
      value: (eagaVal / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  if (eagbVal !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'EAGB Utilization',
      value: (eagbVal / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  if (prxVal !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'PRX Utilization',
      value: (prxVal / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  // FP Precision Breakdown
  const fpSpScaleOps = getCounter(counters, '0x80c4');
  const fpSpFixedOps = getCounter(counters, '0x80c5');
  const fpDpScaleOps = getCounter(counters, '0x80c6');
  const fpDpFixedOps = getCounter(counters, '0x80c7');

  if (fpSpScaleOps !== null && elapsed > 0) {
    const gflops = (fpSpScaleOps * sveScaleFactor) / (elapsed * 1e9);
    metrics.push({
      name: 'GFLOPS SP (SVE)',
      value: gflops,
      unit: '',
      category: 'FP Precision Breakdown'
    });
  }

  if (fpSpFixedOps !== null && elapsed > 0) {
    metrics.push({
      name: 'GFLOPS SP (SIMD)',
      value: fpSpFixedOps / (elapsed * 1e9),
      unit: '',
      category: 'FP Precision Breakdown'
    });
  }

  if (fpDpScaleOps !== null && elapsed > 0) {
    const gflops = (fpDpScaleOps * sveScaleFactor) / (elapsed * 1e9);
    metrics.push({
      name: 'GFLOPS DP (SVE)',
      value: gflops,
      unit: '',
      category: 'FP Precision Breakdown'
    });
  }

  if (fpDpFixedOps !== null && elapsed > 0) {
    metrics.push({
      name: 'GFLOPS DP (SIMD)',
      value: fpDpFixedOps / (elapsed * 1e9),
      unit: '',
      category: 'FP Precision Breakdown'
    });
  }

  // FP Instruction Breakdown
  const fpFmaSpec = getCounter(counters, '0x8028');
  const fpSpec = getCounter(counters, '0x8010');

  if (fpFmaSpec !== null && fpSpec !== null && fpSpec > 0) {
    metrics.push({
      name: 'FMA Ratio',
      value: (fpFmaSpec / fpSpec) * 100,
      unit: '%',
      category: 'FP Instruction Breakdown'
    });
  }

  if (fpFmaSpec !== null) {
    metrics.push({
      name: 'FMA Operations',
      value: fpFmaSpec,
      unit: '',
      category: 'FP Instruction Breakdown',
      format: 'count'
    });
  }

  // SVE/SIMD Detailed Metrics
  const aseSveLdSpec = getCounter(counters, '0x8085');
  const aseSveStSpec = getCounter(counters, '0x8086');
  const sveGatherSpec = getCounter(counters, '0x80ad');
  const sveScatterSpec = getCounter(counters, '0x80ae');
  const sveLdMultiSpec = getCounter(counters, '0x80a5');
  const sveStMultiSpec = getCounter(counters, '0x80a6');
  const prfSpec = getCounter(counters, '0x8087');

  if (aseSveLdSpec !== null) {
    metrics.push({
      name: 'SIMD/SVE Loads',
      value: aseSveLdSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  if (aseSveStSpec !== null) {
    metrics.push({
      name: 'SIMD/SVE Stores',
      value: aseSveStSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  if (sveGatherSpec !== null) {
    metrics.push({
      name: 'SVE Gather Ops',
      value: sveGatherSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  if (sveScatterSpec !== null) {
    metrics.push({
      name: 'SVE Scatter Ops',
      value: sveScatterSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  if (sveLdMultiSpec !== null) {
    metrics.push({
      name: 'SVE Multi-Vec Loads',
      value: sveLdMultiSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  if (sveStMultiSpec !== null) {
    metrics.push({
      name: 'SVE Multi-Vec Stores',
      value: sveStMultiSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  if (prfSpec !== null) {
    metrics.push({
      name: 'Prefetch Ops',
      value: prfSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  // Gather/Scatter Efficiency
  if (sveGatherSpec !== null && aseSveLdSpec !== null && aseSveLdSpec > 0) {
    metrics.push({
      name: 'Gather Ratio',
      value: (sveGatherSpec / aseSveLdSpec) * 100,
      unit: '%',
      category: 'SVE/SIMD Operations'
    });
  }

  if (sveScatterSpec !== null && aseSveStSpec !== null && aseSveStSpec > 0) {
    metrics.push({
      name: 'Scatter Ratio',
      value: (sveScatterSpec / aseSveStSpec) * 100,
      unit: '%',
      category: 'SVE/SIMD Operations'
    });
  }

  // HW Prefetch Activity
  const l1HwpfStreamPf = getCounter(counters, '0x0230');
  const l1HwpfInjAllocPf = getCounter(counters, '0x0231');
  const l1HwpfInjNoallocPf = getCounter(counters, '0x0232');
  const l2HwpfStreamPf = getCounter(counters, '0x0233');
  const l2HwpfInjAllocPf = getCounter(counters, '0x0234');
  const l2HwpfInjNoallocPf = getCounter(counters, '0x0235');

  if (l1HwpfStreamPf !== null) {
    metrics.push({
      name: 'L1 HW Stream Prefetch',
      value: l1HwpfStreamPf,
      unit: '',
      category: 'HW Prefetch',
      format: 'count'
    });
  }

  if (l2HwpfStreamPf !== null) {
    metrics.push({
      name: 'L2 HW Stream Prefetch',
      value: l2HwpfStreamPf,
      unit: '',
      category: 'HW Prefetch',
      format: 'count'
    });
  }

  const l1HwpfTotal = (l1HwpfStreamPf || 0) + (l1HwpfInjAllocPf || 0) + (l1HwpfInjNoallocPf || 0);
  const l2HwpfTotal = (l2HwpfStreamPf || 0) + (l2HwpfInjAllocPf || 0) + (l2HwpfInjNoallocPf || 0);

  if (l1HwpfTotal > 0) {
    metrics.push({
      name: 'L1 HW Prefetch Total',
      value: l1HwpfTotal,
      unit: '',
      category: 'HW Prefetch',
      format: 'count'
    });
  }

  if (l2HwpfTotal > 0) {
    metrics.push({
      name: 'L2 HW Prefetch Total',
      value: l2HwpfTotal,
      unit: '',
      category: 'HW Prefetch',
      format: 'count'
    });
  }

  // Bus Traffic Breakdown
  const busWriteCmg0 = getCounter(counters, '0x0318');
  const busWriteCmg1 = getCounter(counters, '0x0319');
  const busWriteCmg2 = getCounter(counters, '0x031a');
  const busWriteCmg3 = getCounter(counters, '0x031b');
  const busReadTofu = getCounter(counters, '0x0314');
  const busWriteTofu = getCounter(counters, '0x031c');
  const busReadPci = getCounter(counters, '0x0315');
  const busWritePci = getCounter(counters, '0x031d');

  const cmgWriteTotal = (busWriteCmg0 || 0) + (busWriteCmg1 || 0) + (busWriteCmg2 || 0) + (busWriteCmg3 || 0);

  if (cmgWriteTotal > 0 && elapsed > 0) {
    const bw = (cmgWriteTotal * 256) / (elapsed * 1e9);
    metrics.push({
      name: 'CMG Write BW',
      value: bw,
      unit: 'GB/s',
      category: 'Bus Traffic'
    });
  }

  if (busReadTofu !== null && elapsed > 0) {
    const bw = (busReadTofu * 256) / (elapsed * 1e9);
    metrics.push({
      name: 'Tofu Read BW',
      value: bw,
      unit: 'GB/s',
      category: 'Bus Traffic'
    });
  }

  if (busWriteTofu !== null && elapsed > 0) {
    const bw = (busWriteTofu * 256) / (elapsed * 1e9);
    metrics.push({
      name: 'Tofu Write BW',
      value: bw,
      unit: 'GB/s',
      category: 'Bus Traffic'
    });
  }

  // L1 Cache Details
  const l1Pipe0Comp = getCounter(counters, '0x0260');
  const l1Pipe1Comp = getCounter(counters, '0x0261');
  const l1PipeAbort = getCounter(counters, '0x0274');

  if (l1Pipe0Comp !== null && l1Pipe1Comp !== null) {
    metrics.push({
      name: 'L1D Requests',
      value: l1Pipe0Comp + l1Pipe1Comp,
      unit: '',
      category: 'L1 Cache Details',
      format: 'count'
    });
  }

  if (l1PipeAbort !== null && l1Pipe0Comp !== null && l1Pipe1Comp !== null) {
    const totalReq = l1Pipe0Comp + l1Pipe1Comp;
    if (totalReq > 0) {
      metrics.push({
        name: 'L1D Abort Rate',
        value: (l1PipeAbort / totalReq) * 100,
        unit: '%',
        category: 'L1 Cache Details'
      });
    }
  }

  // Micro-op Statistics
  const uopSpec = getCounter(counters, '0x8008');
  const uopSplit = getCounter(counters, '0x0139');

  if (uopSpec !== null && effectiveInst !== null && effectiveInst > 0) {
    metrics.push({
      name: 'Micro-ops per Inst',
      value: uopSpec / effectiveInst,
      unit: '',
      category: 'Micro-op Statistics'
    });
  }

  if (uopSplit !== null) {
    metrics.push({
      name: 'Micro-op Splits',
      value: uopSplit,
      unit: '',
      category: 'Micro-op Statistics',
      format: 'count'
    });
  }

  // ROB/CSE Statistics
  const robEmpty = getCounter(counters, '0x018c');
  const robEmptyStqBusy = getCounter(counters, '0x018d');

  if (robEmpty !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'ROB Empty Rate',
      value: (robEmpty / cycles) * 100,
      unit: '%',
      category: 'ROB Statistics'
    });
  }

  if (robEmptyStqBusy !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'ROB Empty (STQ Busy)',
      value: (robEmptyStqBusy / cycles) * 100,
      unit: '%',
      category: 'ROB Statistics'
    });
  }

  // Integer Stall
  if (euCompWait !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Integer Stall',
      value: (euCompWait / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  // Branch Stall
  const brCompWait = getCounter(counters, '0x018b');
  if (brCompWait !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Branch Stall',
      value: (brCompWait / cycles) * 100,
      unit: '%',
      category: 'Pipeline Utilization'
    });
  }

  // L1I Cache
  const l1iCacheRefill = getCounter(counters, '0x0001');
  if (l1iCacheRefill !== null) {
    metrics.push({
      name: 'L1I Cache Refills',
      value: l1iCacheRefill,
      unit: '',
      category: 'L1I Cache',
      format: 'count'
    });
  }

  // L2 Cache Write-back
  const l2dCacheWb = getCounter(counters, '0x0018');
  if (l2dCacheWb !== null) {
    metrics.push({
      name: 'L2D Cache Write-backs',
      value: l2dCacheWb,
      unit: '',
      category: 'L2 Cache Details',
      format: 'count'
    });
  }

  // Crypto/DCZVA Instructions
  const cryptoSpec = getCounter(counters, '0x0077');
  const dczvaSpec = getCounter(counters, '0x009f');

  if (cryptoSpec !== null) {
    metrics.push({
      name: 'Crypto Instructions',
      value: cryptoSpec,
      unit: '',
      category: 'Special Instructions',
      format: 'count'
    });
  }

  if (dczvaSpec !== null) {
    metrics.push({
      name: 'DC ZVA Instructions',
      value: dczvaSpec,
      unit: '',
      category: 'Special Instructions',
      format: 'count'
    });
  }

  // FP Move/Load/Store Operations
  const fpMvSpec = getCounter(counters, '0x0105');
  const fpLdSpec = getCounter(counters, '0x0112');
  const fpStSpec = getCounter(counters, '0x0113');

  if (fpMvSpec !== null) {
    metrics.push({
      name: 'FP Move Ops',
      value: fpMvSpec,
      unit: '',
      category: 'FP Operations Detail',
      format: 'count'
    });
  }

  if (fpLdSpec !== null) {
    metrics.push({
      name: 'FP Load Ops',
      value: fpLdSpec,
      unit: '',
      category: 'FP Operations Detail',
      format: 'count'
    });
  }

  if (fpStSpec !== null) {
    metrics.push({
      name: 'FP Store Ops',
      value: fpStSpec,
      unit: '',
      category: 'FP Operations Detail',
      format: 'count'
    });
  }

  // Predicate/Inter-element/Inter-register Operations
  const prdSpec = getCounter(counters, '0x0108');
  const ielSpec = getCounter(counters, '0x0109');
  const iregSpec = getCounter(counters, '0x010a');

  if (prdSpec !== null) {
    metrics.push({
      name: 'Predicate Ops',
      value: prdSpec,
      unit: '',
      category: 'SVE Register Ops',
      format: 'count'
    });
  }

  if (ielSpec !== null) {
    metrics.push({
      name: 'Inter-element Ops',
      value: ielSpec,
      unit: '',
      category: 'SVE Register Ops',
      format: 'count'
    });
  }

  if (iregSpec !== null) {
    metrics.push({
      name: 'Inter-register Ops',
      value: iregSpec,
      unit: '',
      category: 'SVE Register Ops',
      format: 'count'
    });
  }

  // Broadcast Load
  const bcLdSpec = getCounter(counters, '0x011a');
  if (bcLdSpec !== null) {
    metrics.push({
      name: 'Broadcast Loads',
      value: bcLdSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  // Extended Memory Stall Breakdown (integer load specific)
  const ldCompWaitL2MissEx = getCounter(counters, '0x0181');
  const ldCompWaitL1MissEx = getCounter(counters, '0x0183');
  const ldCompWaitEx = getCounter(counters, '0x0185');
  const ldCompWaitPfpBusyEx = getCounter(counters, '0x0187');
  const ldCompWaitPfpBusySwpf = getCounter(counters, '0x0188');

  if (ldCompWaitL2MissEx !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'L2 Miss Stall (Int)',
      value: (ldCompWaitL2MissEx / cycles) * 100,
      unit: '%',
      category: 'Memory Stall Breakdown'
    });
  }

  if (ldCompWaitL1MissEx !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'L1 Miss Stall (Int)',
      value: (ldCompWaitL1MissEx / cycles) * 100,
      unit: '%',
      category: 'Memory Stall Breakdown'
    });
  }

  if (ldCompWaitEx !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Memory Stall (Int)',
      value: (ldCompWaitEx / cycles) * 100,
      unit: '%',
      category: 'Memory Stall Breakdown'
    });
  }

  if (ldCompWaitPfpBusyEx !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Prefetch Port Stall (Int)',
      value: (ldCompWaitPfpBusyEx / cycles) * 100,
      unit: '%',
      category: 'Memory Stall Breakdown'
    });
  }

  if (ldCompWaitPfpBusySwpf !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'Prefetch Port Stall (SW PF)',
      value: (ldCompWaitPfpBusySwpf / cycles) * 100,
      unit: '%',
      category: 'Memory Stall Breakdown'
    });
  }

  // WFE/WFI Cycles
  const wfeWfiCycle = getCounter(counters, '0x018e');
  if (wfeWfiCycle !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'WFE/WFI Rate',
      value: (wfeWfiCycle / cycles) * 100,
      unit: '%',
      category: 'Idle Statistics'
    });
  }

  // UOP/MOVPRFX Commit
  const uopOnlyCommit = getCounter(counters, '0x0198');
  const singleMovprfxCommit = getCounter(counters, '0x0199');

  if (uopOnlyCommit !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'UOP Only Commit Rate',
      value: (uopOnlyCommit / cycles) * 100,
      unit: '%',
      category: 'Commit Distribution'
    });
  }

  if (singleMovprfxCommit !== null && cycles !== null && cycles > 0) {
    metrics.push({
      name: 'MOVPRFX Only Commit Rate',
      value: (singleMovprfxCommit / cycles) * 100,
      unit: '%',
      category: 'Commit Distribution'
    });
  }

  // FLA/FLB Predicate Counts
  const flaValPrdCnt = getCounter(counters, '0x01b4');
  const flbValPrdCnt = getCounter(counters, '0x01b5');

  if (flaValPrdCnt !== null) {
    metrics.push({
      name: 'FLA Predicate Count',
      value: flaValPrdCnt,
      unit: '',
      category: 'SVE Predication',
      format: 'count'
    });
  }

  if (flbValPrdCnt !== null) {
    metrics.push({
      name: 'FLB Predicate Count',
      value: flbValPrdCnt,
      unit: '',
      category: 'SVE Predication',
      format: 'count'
    });
  }

  // Calculate predication efficiency
  if (flaValPrdCnt !== null && flaVal !== null && flaVal > 0) {
    // Each SVE operation can process up to 8 DP or 16 SP elements
    // PRD_CNT counts total active predicate elements
    const avgActiveElements = flaValPrdCnt / flaVal;
    metrics.push({
      name: 'FLA Avg Active Elements',
      value: avgActiveElements,
      unit: '',
      category: 'SVE Predication'
    });
  }

  if (flbValPrdCnt !== null && flbVal !== null && flbVal > 0) {
    const avgActiveElements = flbValPrdCnt / flbVal;
    metrics.push({
      name: 'FLB Avg Active Elements',
      value: avgActiveElements,
      unit: '',
      category: 'SVE Predication'
    });
  }

  // L2 HW Prefetch Other
  const l2HwpfOther = getCounter(counters, '0x0236');
  if (l2HwpfOther !== null) {
    metrics.push({
      name: 'L2 HW Prefetch Other',
      value: l2HwpfOther,
      unit: '',
      category: 'HW Prefetch',
      format: 'count'
    });
  }

  // Gather Flow Statistics
  const gatherFlow2 = getCounter(counters, '0x02b0');
  const gatherFlow1 = getCounter(counters, '0x02b1');
  const gatherFlow0 = getCounter(counters, '0x02b2');

  if (gatherFlow2 !== null) {
    metrics.push({
      name: 'Gather 2-Flow',
      value: gatherFlow2,
      unit: '',
      category: 'Gather Statistics',
      format: 'count'
    });
  }

  if (gatherFlow1 !== null) {
    metrics.push({
      name: 'Gather 1-Flow',
      value: gatherFlow1,
      unit: '',
      category: 'Gather Statistics',
      format: 'count'
    });
  }

  if (gatherFlow0 !== null) {
    metrics.push({
      name: 'Gather 0-Flow',
      value: gatherFlow0,
      unit: '',
      category: 'Gather Statistics',
      format: 'count'
    });
  }

  // Gather efficiency (more flows = better)
  const totalGatherFlows = (gatherFlow2 || 0) + (gatherFlow1 || 0) + (gatherFlow0 || 0);
  if (totalGatherFlows > 0) {
    const avgFlows = ((gatherFlow2 || 0) * 2 + (gatherFlow1 || 0) * 1) / totalGatherFlows;
    metrics.push({
      name: 'Gather Avg Flows',
      value: avgFlows,
      unit: '',
      category: 'Gather Statistics'
    });
  }

  // L1 Pipe Predicate Counts
  const l1Pipe0PrdCnt = getCounter(counters, '0x02b8');
  const l1Pipe1PrdCnt = getCounter(counters, '0x02b9');

  if (l1Pipe0PrdCnt !== null) {
    metrics.push({
      name: 'L1 Pipe0 Predicate Count',
      value: l1Pipe0PrdCnt,
      unit: '',
      category: 'L1 Cache Details',
      format: 'count'
    });
  }

  if (l1Pipe1PrdCnt !== null) {
    metrics.push({
      name: 'L1 Pipe1 Predicate Count',
      value: l1Pipe1PrdCnt,
      unit: '',
      category: 'L1 Cache Details',
      format: 'count'
    });
  }

  // PCI Bus Traffic
  if (busReadPci !== null && elapsed > 0) {
    const bw = (busReadPci * 256) / (elapsed * 1e9);
    metrics.push({
      name: 'PCI Read BW',
      value: bw,
      unit: 'GB/s',
      category: 'Bus Traffic'
    });
  }

  if (busWritePci !== null && elapsed > 0) {
    const bw = (busWritePci * 256) / (elapsed * 1e9);
    metrics.push({
      name: 'PCI Write BW',
      value: bw,
      unit: 'GB/s',
      category: 'Bus Traffic'
    });
  }

  // L2 OC MIB Hit
  const l2OcRdMibHit = getCounter(counters, '0x0391');
  const l2OcWrMibHit = getCounter(counters, '0x03ae');

  if (l2OcRdMibHit !== null) {
    metrics.push({
      name: 'L2 OC Read MIB Hits',
      value: l2OcRdMibHit,
      unit: '',
      category: 'L2 Cache Details',
      format: 'count'
    });
  }

  if (l2OcWrMibHit !== null) {
    metrics.push({
      name: 'L2 OC Write MIB Hits',
      value: l2OcWrMibHit,
      unit: '',
      category: 'L2 Cache Details',
      format: 'count'
    });
  }

  // SVE Math Operations
  const sveMathSpec = getCounter(counters, '0x800e');
  if (sveMathSpec !== null) {
    metrics.push({
      name: 'SVE Math Ops',
      value: sveMathSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  // FP Reciprocal/Convert Operations
  const fpRecpeSpec = getCounter(counters, '0x8034');
  const fpCvtSpec = getCounter(counters, '0x8038');

  if (fpRecpeSpec !== null) {
    metrics.push({
      name: 'FP Reciprocal Estimate Ops',
      value: fpRecpeSpec,
      unit: '',
      category: 'FP Operations Detail',
      format: 'count'
    });
  }

  if (fpCvtSpec !== null) {
    metrics.push({
      name: 'FP Convert Ops',
      value: fpCvtSpec,
      unit: '',
      category: 'FP Operations Detail',
      format: 'count'
    });
  }

  // SIMD/SVE Integer Operations
  const aseSveIntSpec = getCounter(counters, '0x8043');
  if (aseSveIntSpec !== null) {
    metrics.push({
      name: 'SIMD/SVE Integer Ops',
      value: aseSveIntSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  // SVE MOVPRFX Operations
  const sveMovprfxSpec = getCounter(counters, '0x807c');
  if (sveMovprfxSpec !== null) {
    metrics.push({
      name: 'SVE MOVPRFX Ops',
      value: sveMovprfxSpec,
      unit: '',
      category: 'SVE/SIMD Operations',
      format: 'count'
    });
  }

  // SVE LDR/STR Register Operations
  const sveLdrRegSpec = getCounter(counters, '0x8091');
  const sveStrRegSpec = getCounter(counters, '0x8092');
  const sveLdrPregSpec = getCounter(counters, '0x8095');
  const sveStrPregSpec = getCounter(counters, '0x8096');

  if (sveLdrRegSpec !== null) {
    metrics.push({
      name: 'SVE LDR Ops',
      value: sveLdrRegSpec,
      unit: '',
      category: 'SVE Load/Store',
      format: 'count'
    });
  }

  if (sveStrRegSpec !== null) {
    metrics.push({
      name: 'SVE STR Ops',
      value: sveStrRegSpec,
      unit: '',
      category: 'SVE Load/Store',
      format: 'count'
    });
  }

  if (sveLdrPregSpec !== null) {
    metrics.push({
      name: 'SVE LDR Predicate Ops',
      value: sveLdrPregSpec,
      unit: '',
      category: 'SVE Load/Store',
      format: 'count'
    });
  }

  if (sveStrPregSpec !== null) {
    metrics.push({
      name: 'SVE STR Predicate Ops',
      value: sveStrPregSpec,
      unit: '',
      category: 'SVE Load/Store',
      format: 'count'
    });
  }

  // SVE Prefetch Operations
  const svePrfContigSpec = getCounter(counters, '0x809f');
  const svePrfGatherSpec = getCounter(counters, '0x80af');

  if (svePrfContigSpec !== null) {
    metrics.push({
      name: 'SVE Contiguous Prefetch',
      value: svePrfContigSpec,
      unit: '',
      category: 'SVE Prefetch',
      format: 'count'
    });
  }

  if (svePrfGatherSpec !== null) {
    metrics.push({
      name: 'SVE Gather Prefetch',
      value: svePrfGatherSpec,
      unit: '',
      category: 'SVE Prefetch',
      format: 'count'
    });
  }

  // SVE First-Fault Load
  const sveLdffSpec = getCounter(counters, '0x80bc');
  if (sveLdffSpec !== null) {
    metrics.push({
      name: 'SVE First-Fault Loads',
      value: sveLdffSpec,
      unit: '',
      category: 'SVE Load/Store',
      format: 'count'
    });
  }

  // Load/Store Counts
  if (ldSpec !== null) {
    metrics.push({
      name: 'Load Instructions',
      value: ldSpec,
      unit: '',
      category: 'Instruction Mix',
      format: 'count'
    });
  }

  if (stSpec !== null) {
    metrics.push({
      name: 'Store Instructions',
      value: stSpec,
      unit: '',
      category: 'Instruction Mix',
      format: 'count'
    });
  }

  if (effectiveInst !== null) {
    metrics.push({
      name: 'Effective Instructions',
      value: effectiveInst,
      unit: '',
      category: 'Instruction Mix',
      format: 'count'
    });
  }

  // Total Energy
  if (eaCore !== null || eaL2 !== null || eaMemory !== null) {
    const totalEnergy = (eaCore ? eaCore * 8 : 0) + (eaL2 ? eaL2 * 32 : 0) + (eaMemory ? eaMemory * 256 : 0);
    metrics.push({
      name: 'Total Energy',
      value: totalEnergy / 1e9,
      unit: 'J',
      category: 'Energy'
    });
    if (elapsed > 0) {
      metrics.push({
        name: 'Average Power',
        value: (totalEnergy / 1e9) / elapsed,
        unit: 'W',
        category: 'Energy'
      });
    }
  }

  return metrics;
}

// =============================================================================
// Section 3.5: Bottleneck Analysis Functions
// =============================================================================

function analyzeBottlenecks(counters, elapsed, cpuFreq) {
  const findings = [];
  const cycles = counters['PMCCNTR'];

  // Helper to get metric value
  const get = (code) => counters[code] !== undefined ? counters[code] : null;

  // Thresholds for bottleneck detection
  const THRESHOLDS = {
    MEMORY_STALL_HIGH: 20,      // % of cycles
    MEMORY_STALL_MODERATE: 10,
    L2_MISS_STALL_HIGH: 15,
    L1_MISS_RATE_HIGH: 20,      // %
    L2_MISS_RATE_HIGH: 5,
    TLB_MISS_RATE_HIGH: 1,
    BRANCH_MISPREDICT_HIGH: 5,  // %
    BRANCH_MISPREDICT_MODERATE: 1,
    FP_STALL_HIGH: 10,
    STALL_RATE_HIGH: 30,
    STALL_RATE_MODERATE: 15,
    IPC_LOW: 1.0,
    IPC_MODERATE: 2.0,
    VECTORIZATION_LOW: 50,      // % SVE utilization
    PREDICATION_LOW: 6,         // avg active elements (out of 8 for DP)
    ROB_EMPTY_HIGH: 10,
    GATHER_RATIO_HIGH: 20,      // % of loads being gathers
    MEMORY_BW_HIGH: 800,        // GB/s (A64FX peak ~1TB/s)
  };

  // === IPC Analysis ===
  const effectiveInst = get('0x0121');
  if (effectiveInst !== null && cycles !== null && cycles > 0) {
    const ipc = effectiveInst / cycles;
    if (ipc < THRESHOLDS.IPC_LOW) {
      findings.push({
        severity: 'high',
        category: 'Core Performance',
        issue: `Very low IPC (${ipc.toFixed(2)})`,
        detail: 'Instructions per cycle is below 1.0, indicating significant stalls or inefficiencies.',
        suggestions: [
          'Check memory access patterns for cache misses',
          'Look for data dependencies causing pipeline stalls',
          'Consider loop unrolling or software pipelining'
        ]
      });
    } else if (ipc < THRESHOLDS.IPC_MODERATE) {
      findings.push({
        severity: 'medium',
        category: 'Core Performance',
        issue: `Moderate IPC (${ipc.toFixed(2)})`,
        detail: 'IPC is below 2.0. A64FX can retire up to 4 instructions/cycle.',
        suggestions: [
          'Analyze stall breakdown to identify limiting factors',
          'Check for instruction-level parallelism opportunities'
        ]
      });
    }
  }

  // === Memory Stall Analysis ===
  const ldCompWait = get('0x0184');
  const ldCompWaitL2Miss = get('0x0180');
  const ldCompWaitL1Miss = get('0x0182');

  if (ldCompWait !== null && cycles !== null && cycles > 0) {
    const memStallPct = (ldCompWait / cycles) * 100;
    if (memStallPct > THRESHOLDS.MEMORY_STALL_HIGH) {
      findings.push({
        severity: 'high',
        category: 'Memory',
        issue: `High memory stall rate (${memStallPct.toFixed(1)}%)`,
        detail: 'Significant time spent waiting for memory operations.',
        suggestions: [
          'Use software prefetching (PRFM instructions)',
          'Improve data locality and cache blocking',
          'Consider data layout changes for better cache utilization',
          'Check for false sharing in multi-threaded code'
        ]
      });
    } else if (memStallPct > THRESHOLDS.MEMORY_STALL_MODERATE) {
      findings.push({
        severity: 'medium',
        category: 'Memory',
        issue: `Moderate memory stall rate (${memStallPct.toFixed(1)}%)`,
        detail: 'Noticeable time spent waiting for memory.',
        suggestions: [
          'Review memory access patterns',
          'Consider prefetching for predictable access patterns'
        ]
      });
    }
  }

  // === L2 Miss Stall ===
  if (ldCompWaitL2Miss !== null && cycles !== null && cycles > 0) {
    const l2StallPct = (ldCompWaitL2Miss / cycles) * 100;
    if (l2StallPct > THRESHOLDS.L2_MISS_STALL_HIGH) {
      findings.push({
        severity: 'high',
        category: 'Memory',
        issue: `High L2 miss stall (${l2StallPct.toFixed(1)}%)`,
        detail: 'Significant time waiting for data from main memory.',
        suggestions: [
          'Working set may exceed L2 cache (8MB per CMG)',
          'Use cache blocking/tiling to fit in L2',
          'Increase prefetch distance for L2 prefetches',
          'Consider NUMA-aware memory allocation'
        ]
      });
    }
  }

  // === Cache Miss Rate Analysis ===
  const l1dRefill = get('0x0003');
  const ldSpec = get('0x0070');
  const stSpec = get('0x0071');

  if (l1dRefill !== null && ldSpec !== null && stSpec !== null) {
    const ldstSpec = ldSpec + stSpec;
    if (ldstSpec > 0) {
      const l1MissRate = (l1dRefill / ldstSpec) * 100;
      if (l1MissRate > THRESHOLDS.L1_MISS_RATE_HIGH) {
        findings.push({
          severity: 'high',
          category: 'Cache',
          issue: `High L1D miss rate (${l1MissRate.toFixed(1)}%)`,
          detail: 'Many memory accesses miss the L1 data cache.',
          suggestions: [
            'Improve spatial locality (sequential access patterns)',
            'Use smaller data types if precision allows',
            'Consider Structure of Arrays (SoA) layout',
            'Check for cache line splitting in unaligned accesses'
          ]
        });
      }
    }
  }

  // === L2 Miss Rate ===
  const l2MissCountRaw = get('0x0309');
  const l2dCacheSwapLocal = get('0x0396');
  const l2PipeCompPfL2mibMch = get('0x0370');

  if (l2MissCountRaw !== null && l2dCacheSwapLocal !== null && l2PipeCompPfL2mibMch !== null) {
    const l2MissCount = l2MissCountRaw - l2dCacheSwapLocal - l2PipeCompPfL2mibMch;
    if (ldSpec !== null && stSpec !== null) {
      const ldstSpec = ldSpec + stSpec;
      if (ldstSpec > 0) {
        const l2MissRate = (l2MissCount / ldstSpec) * 100;
        if (l2MissRate > THRESHOLDS.L2_MISS_RATE_HIGH) {
          findings.push({
            severity: 'high',
            category: 'Cache',
            issue: `High L2 miss rate (${l2MissRate.toFixed(1)}%)`,
            detail: 'Significant traffic going to main memory.',
            suggestions: [
              'Apply cache blocking to fit working set in L2 (8MB)',
              'Use L2 prefetching (PRFM PLDL2KEEP/STRM)',
              'Consider data compression to reduce memory footprint'
            ]
          });
        }
      }
    }
  }

  // === TLB Analysis ===
  const l1dTlbRefill = get('0x0005');
  const l2dTlbRefill = get('0x002d');

  if (l1dTlbRefill !== null && ldSpec !== null && stSpec !== null) {
    const ldstSpec = ldSpec + stSpec;
    if (ldstSpec > 0) {
      const tlbMissRate = (l1dTlbRefill / ldstSpec) * 100;
      if (tlbMissRate > THRESHOLDS.TLB_MISS_RATE_HIGH) {
        findings.push({
          severity: 'medium',
          category: 'TLB',
          issue: `High TLB miss rate (${tlbMissRate.toFixed(2)}%)`,
          detail: 'Address translation overhead is significant.',
          suggestions: [
            'Use huge pages (2MB or 1GB) to reduce TLB pressure',
            'Improve memory access locality',
            'Consider reducing working set size'
          ]
        });
      }
    }
  }

  // === Branch Prediction ===
  const brMisPred = get('0x0010');
  const brPred = get('0x0012');

  if (brMisPred !== null && brPred !== null && brPred > 0) {
    const misPredRate = (brMisPred / brPred) * 100;
    if (misPredRate > THRESHOLDS.BRANCH_MISPREDICT_HIGH) {
      findings.push({
        severity: 'high',
        category: 'Branch',
        issue: `High branch misprediction rate (${misPredRate.toFixed(2)}%)`,
        detail: 'Branch mispredictions cause pipeline flushes.',
        suggestions: [
          'Use branchless code where possible (CSEL, conditional moves)',
          'Sort data to make branches more predictable',
          'Use SVE predication instead of branches',
          'Consider loop unswitching'
        ]
      });
    } else if (misPredRate > THRESHOLDS.BRANCH_MISPREDICT_MODERATE) {
      findings.push({
        severity: 'low',
        category: 'Branch',
        issue: `Moderate branch misprediction rate (${misPredRate.toFixed(2)}%)`,
        detail: 'Some branch mispredictions detected.',
        suggestions: [
          'Review branch-heavy code sections',
          'Consider predicated instructions'
        ]
      });
    }
  }

  // === FP/SIMD Stall ===
  const flCompWait = get('0x018a');
  if (flCompWait !== null && cycles !== null && cycles > 0) {
    const fpStallPct = (flCompWait / cycles) * 100;
    if (fpStallPct > THRESHOLDS.FP_STALL_HIGH) {
      findings.push({
        severity: 'medium',
        category: 'Compute',
        issue: `High FP/SIMD stall (${fpStallPct.toFixed(1)}%)`,
        detail: 'Waiting for floating-point or SIMD operations.',
        suggestions: [
          'Check for long-latency FP operations (divisions, sqrt)',
          'Increase instruction-level parallelism',
          'Use FMA instructions to combine multiply-add',
          'Consider reciprocal approximations instead of divisions'
        ]
      });
    }
  }

  // === Overall Stall Rate ===
  const zeroInstCommit = get('0x0190');
  if (zeroInstCommit !== null && cycles !== null && cycles > 0) {
    const stallRate = (zeroInstCommit / cycles) * 100;
    if (stallRate > THRESHOLDS.STALL_RATE_HIGH) {
      findings.push({
        severity: 'high',
        category: 'Core Performance',
        issue: `Very high stall rate (${stallRate.toFixed(1)}%)`,
        detail: 'CPU spends significant time without committing instructions.',
        suggestions: [
          'Analyze memory stall breakdown for root cause',
          'Check for resource conflicts (registers, execution units)'
        ]
      });
    }
  }

  // === SVE Vectorization Efficiency ===
  const flaValPrdCnt = get('0x01b4');
  const flaVal = get('0x01a4');
  const flbValPrdCnt = get('0x01b5');
  const flbVal = get('0x01a5');

  if (flaValPrdCnt !== null && flaVal !== null && flaVal > 0) {
    const avgActiveElements = flaValPrdCnt / flaVal;
    // For DP, max is 8 elements; for SP, max is 16
    if (avgActiveElements < THRESHOLDS.PREDICATION_LOW) {
      findings.push({
        severity: 'medium',
        category: 'Vectorization',
        issue: `Low vector utilization (${avgActiveElements.toFixed(1)} avg elements)`,
        detail: 'SVE vectors are not fully utilized (max 8 for DP, 16 for SP).',
        suggestions: [
          'Check for loop remainder handling',
          'Ensure data is aligned to vector boundaries',
          'Consider padding arrays to vector length multiples',
          'Review predication patterns for partial vector operations'
        ]
      });
    }
  }

  // === Gather/Scatter Analysis ===
  const sveGatherSpec = get('0x80ad');
  const aseSveLdSpec = get('0x8085');

  if (sveGatherSpec !== null && aseSveLdSpec !== null && aseSveLdSpec > 0) {
    const gatherRatio = (sveGatherSpec / aseSveLdSpec) * 100;
    if (gatherRatio > THRESHOLDS.GATHER_RATIO_HIGH) {
      findings.push({
        severity: 'medium',
        category: 'Vectorization',
        issue: `High gather load ratio (${gatherRatio.toFixed(1)}%)`,
        detail: 'Many loads use gather operations which are slower than contiguous loads.',
        suggestions: [
          'Restructure data for contiguous access',
          'Consider data layout transformations (AoS to SoA)',
          'Pre-gather data into temporary contiguous buffers'
        ]
      });
    }
  }

  // === ROB Empty Analysis ===
  const robEmpty = get('0x018c');
  if (robEmpty !== null && cycles !== null && cycles > 0) {
    const robEmptyRate = (robEmpty / cycles) * 100;
    if (robEmptyRate > THRESHOLDS.ROB_EMPTY_HIGH) {
      findings.push({
        severity: 'medium',
        category: 'Frontend',
        issue: `High ROB empty rate (${robEmptyRate.toFixed(1)}%)`,
        detail: 'Instruction supply is not keeping up with execution.',
        suggestions: [
          'Check for L1I cache misses',
          'Look for instruction fetch bottlenecks',
          'Consider code layout optimization'
        ]
      });
    }
  }

  // === Memory Bandwidth Analysis ===
  const busReadMem = get('0x0316');
  const busWriteMem = get('0x031e');

  if (busReadMem !== null && busWriteMem !== null && elapsed > 0) {
    const readBw = (busReadMem * 256) / (elapsed * 1e9);
    const writeBw = (busWriteMem * 256) / (elapsed * 1e9);
    const totalBw = readBw + writeBw;

    // A64FX theoretical peak: ~1TB/s (all channels)
    // Practical sustained: ~800 GB/s
    if (totalBw > THRESHOLDS.MEMORY_BW_HIGH) {
      findings.push({
        severity: 'info',
        category: 'Memory',
        issue: `High memory bandwidth utilization (${totalBw.toFixed(1)} GB/s)`,
        detail: 'Application is memory bandwidth intensive.',
        suggestions: [
          'Workload is memory-bound; optimize for memory efficiency',
          'Consider cache blocking to reduce memory traffic',
          'Use non-temporal stores for write-only data'
        ]
      });
    }
  }

  // === FMA Utilization ===
  const fpFmaSpec = get('0x8028');
  const fpSpec = get('0x8010');

  if (fpFmaSpec !== null && fpSpec !== null && fpSpec > 0) {
    const fmaRatio = (fpFmaSpec / fpSpec) * 100;
    if (fmaRatio < 50) {
      findings.push({
        severity: 'low',
        category: 'Compute',
        issue: `Low FMA utilization (${fmaRatio.toFixed(1)}%)`,
        detail: 'Not all FP operations use fused multiply-add.',
        suggestions: [
          'Use -ffp-contract=fast compiler flag',
          'Restructure computations to enable FMA fusion',
          'Consider using FMA intrinsics explicitly'
        ]
      });
    }
  }

  // === Commit Distribution Analysis ===
  const fourInstCommit = get('0x0194');
  if (fourInstCommit !== null && cycles !== null && cycles > 0) {
    const fourCommitRate = (fourInstCommit / cycles) * 100;
    if (fourCommitRate < 30) {
      findings.push({
        severity: 'low',
        category: 'Core Performance',
        issue: `Low 4-instruction commit rate (${fourCommitRate.toFixed(1)}%)`,
        detail: 'CPU rarely commits maximum 4 instructions per cycle.',
        suggestions: [
          'Increase instruction-level parallelism',
          'Reduce data dependencies between instructions',
          'Consider loop unrolling'
        ]
      });
    }
  }

  // Sort findings by severity
  const severityOrder = { high: 0, medium: 1, low: 2, info: 3 };
  findings.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);

  return findings;
}

function formatBottleneckAnalysis(findings) {
  if (findings.length === 0) {
    return {
      ascii: ['No significant bottlenecks detected. Performance appears well-optimized.'],
      html: '<p class="no-issues">No significant bottlenecks detected. Performance appears well-optimized.</p>'
    };
  }

  const ascii = [];
  let html = '';

  const severityColors = {
    high: '#dc3545',
    medium: '#ffc107',
    low: '#17a2b8',
    info: '#6c757d'
  };

  const severityIcons = {
    high: '[!]',
    medium: '[*]',
    low: '[-]',
    info: '[i]'
  };

  for (const finding of findings) {
    // ASCII format
    ascii.push(`${severityIcons[finding.severity]} ${finding.category}: ${finding.issue}`);
    ascii.push(`    ${finding.detail}`);
    ascii.push('    Suggestions:');
    for (const suggestion of finding.suggestions) {
      ascii.push(`      - ${suggestion}`);
    }
    ascii.push('');

    // HTML format
    html += `
    <div class="finding finding-${finding.severity}">
      <div class="finding-header">
        <span class="severity" style="background: ${severityColors[finding.severity]}">${finding.severity.toUpperCase()}</span>
        <span class="category">${finding.category}</span>
        <span class="issue">${finding.issue}</span>
      </div>
      <p class="detail">${finding.detail}</p>
      <ul class="suggestions">
        ${finding.suggestions.map(s => `<li>${s}</li>`).join('\n        ')}
      </ul>
    </div>`;
  }

  return { ascii, html };
}

// =============================================================================
// Section 4: ASCII Reporter Functions
// =============================================================================

function formatASCII(data, selectedRegions) {
  const lines = [];
  const width = 80;
  const sep = '='.repeat(width);
  const sepThin = '-'.repeat(width);

  lines.push(sep);
  lines.push('                        A64FX Performance Report');
  lines.push(sep);
  lines.push(`Profile Time   : ${data.info.measuredTime || 'N/A'}`);
  lines.push(`CPU Frequency  : ${data.cpuFrequency} MHz`);
  lines.push(`Vector Length  : ${data.info.vectorLength || 512} bits`);
  lines.push('');

  const regions = selectedRegions.length > 0
    ? selectedRegions.filter(r => data.regions[r])
    : Object.keys(data.regions);

  for (const regionName of regions) {
    const region = data.regions[regionName];
    if (!region) continue;

    lines.push(sepThin);
    lines.push(`Region: ${regionName}`);
    lines.push(`Elapsed Time: ${region.elapsed.toFixed(6)} s`);
    lines.push(sepThin);
    lines.push('');

    const metrics = calculateMetrics(region.counters, region.elapsed, data.cpuFrequency);

    // Group by category
    const byCategory = {};
    for (const m of metrics) {
      if (!byCategory[m.category]) byCategory[m.category] = [];
      byCategory[m.category].push(m);
    }

    for (const [category, categoryMetrics] of Object.entries(byCategory)) {
      lines.push(`${category}:`);
      for (const m of categoryMetrics) {
        let valueStr;
        if (m.format === 'count') {
          valueStr = formatValue(m.value, 2);
        } else {
          valueStr = m.value.toFixed(2);
        }
        const line = `  ${m.name.padEnd(25)} : ${valueStr.padStart(12)} ${m.unit}`;
        lines.push(line);
      }
      lines.push('');
    }

    // Bottleneck Analysis
    const findings = analyzeBottlenecks(region.counters, region.elapsed, data.cpuFrequency);
    const analysis = formatBottleneckAnalysis(findings);

    lines.push(sepThin);
    lines.push('BOTTLENECK ANALYSIS');
    lines.push(sepThin);
    lines.push('');
    for (const line of analysis.ascii) {
      lines.push(line);
    }
  }

  lines.push(sep);
  return lines.join('\n');
}

// =============================================================================
// Section 5: HTML Reporter Functions
// =============================================================================

function formatHTML(data, selectedRegions) {
  const regions = selectedRegions.length > 0
    ? selectedRegions.filter(r => data.regions[r])
    : Object.keys(data.regions);

  let html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>A64FX Performance Report</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      background: #f5f5f5;
    }
    h1 {
      color: #333;
      border-bottom: 3px solid #007acc;
      padding-bottom: 10px;
    }
    h2 {
      color: #007acc;
      margin-top: 30px;
    }
    h3 {
      color: #555;
      margin-top: 20px;
    }
    .info-box {
      background: #fff;
      padding: 15px 20px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 20px;
    }
    .info-box p {
      margin: 5px 0;
    }
    .region {
      background: #fff;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 20px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
    }
    th, td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid #ddd;
    }
    th {
      background: #f0f0f0;
      font-weight: 600;
    }
    tr:hover {
      background: #f9f9f9;
    }
    .value {
      text-align: right;
      font-family: 'Consolas', 'Monaco', monospace;
    }
    .unit {
      color: #666;
      font-size: 0.9em;
    }
    .good { color: #28a745; }
    .warn { color: #ffc107; }
    .bad { color: #dc3545; }
    .analysis-section {
      margin-top: 30px;
      padding-top: 20px;
      border-top: 2px solid #007acc;
    }
    .analysis-section h3 {
      color: #007acc;
      margin-bottom: 15px;
    }
    .finding {
      background: #fff;
      border-left: 4px solid #ccc;
      padding: 15px;
      margin-bottom: 15px;
      border-radius: 0 8px 8px 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .finding-high { border-left-color: #dc3545; background: #fff5f5; }
    .finding-medium { border-left-color: #ffc107; background: #fffdf5; }
    .finding-low { border-left-color: #17a2b8; background: #f5fbff; }
    .finding-info { border-left-color: #6c757d; background: #f8f9fa; }
    .finding-header {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    .severity {
      color: white;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75em;
      font-weight: bold;
      text-transform: uppercase;
    }
    .category {
      font-weight: 600;
      color: #555;
    }
    .issue {
      color: #333;
    }
    .detail {
      color: #666;
      margin: 10px 0;
      font-size: 0.95em;
    }
    .suggestions {
      margin: 10px 0 0 0;
      padding-left: 20px;
    }
    .suggestions li {
      color: #555;
      margin: 5px 0;
      font-size: 0.9em;
    }
    .no-issues {
      color: #28a745;
      font-style: italic;
      padding: 20px;
      text-align: center;
      background: #f0fff4;
      border-radius: 8px;
    }
  </style>
</head>
<body>
  <h1>A64FX Performance Report</h1>

  <div class="info-box">
    <p><strong>Profile Time:</strong> ${data.info.measuredTime || 'N/A'}</p>
    <p><strong>CPU Frequency:</strong> ${data.cpuFrequency} MHz</p>
    <p><strong>Vector Length:</strong> ${data.info.vectorLength || 512} bits</p>
  </div>
`;

  for (const regionName of regions) {
    const region = data.regions[regionName];
    if (!region) continue;

    const metrics = calculateMetrics(region.counters, region.elapsed, data.cpuFrequency);

    html += `
  <div class="region">
    <h2>Region: ${regionName}</h2>
    <p><strong>Elapsed Time:</strong> ${region.elapsed.toFixed(6)} s</p>
`;

    const byCategory = {};
    for (const m of metrics) {
      if (!byCategory[m.category]) byCategory[m.category] = [];
      byCategory[m.category].push(m);
    }

    for (const [category, categoryMetrics] of Object.entries(byCategory)) {
      html += `
    <h3>${category}</h3>
    <table>
      <tr><th>Metric</th><th class="value">Value</th><th>Unit</th></tr>
`;
      for (const m of categoryMetrics) {
        let valueStr;
        if (m.format === 'count') {
          valueStr = formatValue(m.value, 2);
        } else {
          valueStr = m.value.toFixed(2);
        }
        html += `      <tr><td>${m.name}</td><td class="value">${valueStr}</td><td class="unit">${m.unit}</td></tr>\n`;
      }
      html += `    </table>\n`;
    }

    // Bottleneck Analysis
    const findings = analyzeBottlenecks(region.counters, region.elapsed, data.cpuFrequency);
    const analysis = formatBottleneckAnalysis(findings);

    html += `
    <div class="analysis-section">
      <h3>Bottleneck Analysis</h3>
      ${analysis.html}
    </div>
`;

    html += `  </div>\n`;
  }

  html += `
</body>
</html>`;

  return html;
}

// =============================================================================
// Section 5.5: JSON Reporter Functions
// =============================================================================

function formatJSON(data, selectedRegions) {
  const regions = selectedRegions.length > 0
    ? selectedRegions.filter(r => data.regions[r])
    : Object.keys(data.regions);

  const result = {
    info: {
      profilerVersion: data.info.profilerVersion,
      measuredTime: data.info.measuredTime,
      cpuFrequency: data.cpuFrequency,
      vectorLength: data.info.vectorLength || 512
    },
    regions: {}
  };

  for (const regionName of regions) {
    const region = data.regions[regionName];
    if (!region) continue;

    const metrics = calculateMetrics(region.counters, region.elapsed, data.cpuFrequency);
    const findings = analyzeBottlenecks(region.counters, region.elapsed, data.cpuFrequency);

    // Group metrics by category
    const metricsByCategory = {};
    for (const m of metrics) {
      if (!metricsByCategory[m.category]) {
        metricsByCategory[m.category] = {};
      }
      metricsByCategory[m.category][m.name] = {
        value: m.value,
        unit: m.unit,
        format: m.format || 'number'
      };
    }

    result.regions[regionName] = {
      elapsed: region.elapsed,
      metrics: metricsByCategory,
      counters: region.counters,
      bottleneckAnalysis: {
        summary: findings.length === 0
          ? 'No significant bottlenecks detected'
          : `${findings.length} issue(s) found`,
        findings: findings.map(f => ({
          severity: f.severity,
          category: f.category,
          issue: f.issue,
          detail: f.detail,
          suggestions: f.suggestions
        }))
      }
    };
  }

  return JSON.stringify(result, null, 2);
}

// =============================================================================
// Section 6: CLI Main
// =============================================================================

function parseArgs(args) {
  const options = {
    files: 17,
    regions: [],
    format: 'ascii',  // 'ascii', 'html', 'json'
    output: null,
    help: false
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '-n' || arg === '--files') {
      options.files = parseInt(args[++i]) || 17;
    } else if (arg === '-r' || arg === '--region') {
      options.regions.push(args[++i]);
    } else if (arg === '--html') {
      options.format = 'html';
    } else if (arg === '--json') {
      options.format = 'json';
    } else if (arg === '-o' || arg === '--output') {
      options.output = args[++i];
    } else if (arg === '-h' || arg === '--help') {
      options.help = true;
    }
  }

  return options;
}

function printHelp() {
  console.log(`
A64FX Profile Report Tool

Usage: node preport.js [options]

Options:
  -n, --files <num>     Number of pa*.csv files to read (1-17, default: 17)
  -r, --region <name>   Region to report (can be used multiple times)
  --html                Output HTML format
  --json                Output JSON format
  -o, --output <file>   Output file (default: stdout)
  -h, --help            Show this help message

Output Formats:
  (default)             ASCII text format for terminal
  --html                HTML format with styled tables
  --json                JSON format for programmatic access

Examples:
  node preport.js                    # ASCII output, all regions, pa1-pa17
  node preport.js -n 5               # Read pa1.csv to pa5.csv
  node preport.js -r dgemm_kernel    # Report only dgemm_kernel region
  node preport.js --html -o out.html # HTML output to file
  node preport.js --json -o out.json # JSON output to file
`);
}

function main() {
  const args = process.argv.slice(2);
  const options = parseArgs(args);

  if (options.help) {
    printHelp();
    process.exit(0);
  }

  const baseDir = process.cwd();

  console.error(`Loading pa1.csv to pa${options.files}.csv...`);
  const data = loadFiles(options.files, baseDir);

  if (Object.keys(data.regions).length === 0) {
    console.error('Error: No data found in CSV files');
    process.exit(1);
  }

  console.error(`Found regions: ${Object.keys(data.regions).join(', ')}`);

  let output;
  switch (options.format) {
    case 'html':
      output = formatHTML(data, options.regions);
      break;
    case 'json':
      output = formatJSON(data, options.regions);
      break;
    default:
      output = formatASCII(data, options.regions);
  }

  if (options.output) {
    fs.writeFileSync(options.output, output);
    console.error(`Output written to: ${options.output}`);
  } else {
    console.log(output);
  }
}

main();
