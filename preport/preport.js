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

  return metrics;
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

    html += `  </div>\n`;
  }

  html += `
</body>
</html>`;

  return html;
}

// =============================================================================
// Section 6: CLI Main
// =============================================================================

function parseArgs(args) {
  const options = {
    files: 17,
    regions: [],
    html: false,
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
      options.html = true;
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
  --html                Output HTML format (default: ASCII)
  -o, --output <file>   Output file (default: stdout)
  -h, --help            Show this help message

Examples:
  node preport.js                    # ASCII output, all regions, pa1-pa17
  node preport.js -n 5               # Read pa1.csv to pa5.csv
  node preport.js -r dgemm_kernel    # Report only dgemm_kernel region
  node preport.js --html -o out.html # HTML output to file
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
  if (options.html) {
    output = formatHTML(data, options.regions);
  } else {
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
