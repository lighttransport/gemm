# A64FX Sector Cache: Related PMU Events and Performance Measurement Formulas

## 1. Sector Cache Overview

The sector cache is a feature that partitions L1D/L2 caches into up to 4 sectors, with configurable maximum capacity per sector. By assigning sectors according to data temporal locality, it prevents important data from being evicted from the cache.

- **L1D**: 4 sectors (sector_id 0–3), controlled per PE
- **L2**: 4 sectors (2 sets × 2 sectors), shared per CMG
- sector_id is specified via HPC tag address override (address upper bits [57:56]) or default_sector

---

## 2. Related PMU Events

### 2.1 Direct Observation of Sector Utilization

| ID | Event Name | Description |
|------|------|------|
| 0x0250 | L1_PIPE0_VAL_IU_TAG_ADRS_SCE | Requests in L1D pipeline #0 with SCE bit = 1 |
| 0x0252 | L1_PIPE1_VAL_IU_TAG_ADRS_SCE | Requests in L1D pipeline #1 with SCE bit = 1 |
| 0x02a0 | L1_PIPE0_VAL_IU_NOT_SEC0 | Requests in L1D pipeline #0 with sector_id ≠ 0 |
| 0x02a1 | L1_PIPE1_VAL_IU_NOT_SEC0 | Requests in L1D pipeline #1 with sector_id ≠ 0 |
| 0x0251 | L1_PIPE0_VAL_IU_TAG_ADRS_PFE | Requests in L1D pipeline #0 with PFE bit = 1 |
| 0x0253 | L1_PIPE1_VAL_IU_TAG_ADRS_PFE | Requests in L1D pipeline #1 with PFE bit = 1 |

### 2.2 L1D Cache Performance

| ID | Event Name | Description |
|------|------|------|
| 0x0004 | L1D_CACHE | L1D cache access count |
| 0x0003 | L1D_CACHE_REFILL | L1D cache miss (refill) count |
| 0x0200 | L1D_CACHE_REFILL_DM | L1D refills caused by demand access |
| 0x0202 | L1D_CACHE_REFILL_HWPRF | L1D refills caused by HW prefetch |
| 0x0049 | L1D_CACHE_REFILL_PRF | L1D refills caused by SW/HW prefetch |
| 0x0015 | L1D_CACHE_WB | L1D write-back count |
| 0x0208 | L1_MISS_WAIT | Outstanding L1D cache miss requests per cycle |

### 2.3 L2 Cache Performance

| ID | Event Name | Description |
|------|------|------|
| 0x0016 | L2D_CACHE | L2 cache access count |
| 0x0017 | L2D_CACHE_REFILL | L2 cache miss (refill) count |
| 0x0300 | L2D_CACHE_REFILL_DM | L2 refills caused by demand access |
| 0x0302 | L2D_CACHE_REFILL_HWPRF | L2 refills caused by HW prefetch |
| 0x0059 | L2D_CACHE_REFILL_PRF | L2 refills caused by SW/HW prefetch |
| 0x0018 | L2D_CACHE_WB | L2 write-back count |
| 0x0308 | L2_MISS_WAIT | Outstanding L2 cache miss requests per cycle (per CMG) |
| 0x0309 | L2_MISS_COUNT | L2 cache miss count (per CMG) |
| 0x0325 | L2D_SWAP_DM | Demand access hits refill buffer allocated by prefetch |
| 0x0326 | L2D_CACHE_MIBMCH_PRF | Prefetch hits refill buffer allocated by demand access |

### 2.4 Stalls and Commits

| ID | Event Name | Description |
|------|------|------|
| 0x0184 | LD_COMP_WAIT | Load/store/prefetch wait stall (L1D + L2 + memory) |
| 0x0182 | LD_COMP_WAIT_L1_MISS | L2 access wait stall (caused by L1 miss) |
| 0x0180 | LD_COMP_WAIT_L2_MISS | Memory access wait stall (caused by L2 miss) |
| 0x0011 | CPU_CYCLES | Total cycle count |
| 0x0008 | INST_RETIRED | Retired instruction count |

### 2.5 Energy

| ID | Event Name | Description |
|------|------|------|
| 0x01e0 | EA_CORE | Core energy consumption (8 nJ/count @ 2.2 GHz, 48 cores) |
| 0x03e0 | EA_L2 | L2 energy consumption (32 nJ/count @ 2.2 GHz, 48 cores) |
| 0x03e8 | EA_MEMORY | Memory energy consumption (256 nJ/count) |

---

## 3. Measurement Formulas

### 3.1 Basic Metrics (Compare with Sector Cache Enabled vs. Disabled)

Below, `_on` denotes sector cache enabled, `_off` denotes disabled (default).

#### L1D Cache Miss Rate

```
L1D_miss_rate = L1D_CACHE_REFILL / L1D_CACHE
```

**Effect:**
```
ΔL1D_miss_rate = L1D_miss_rate_off - L1D_miss_rate_on
```

A positive value indicates that sector cache reduced evictions of important data, lowering the miss rate.

#### L2 Cache Miss Rate

```
L2D_miss_rate = L2D_CACHE_REFILL / L2D_CACHE
```

**Effect:**
```
ΔL2D_miss_rate = L2D_miss_rate_off - L2D_miss_rate_on
```

---

### 3.2 Separating Demand Access vs. Prefetch

The primary goal of sector cache is **reducing demand access misses**. Prefetch-caused misses may be intentional.

#### L1D Demand Miss Ratio

```
L1D_demand_miss_rate = L1D_CACHE_REFILL_DM / (L1D_CACHE - prefetch-equivalent accesses)
```

Simplified:
```
L1D_demand_refill_ratio = L1D_CACHE_REFILL_DM / L1D_CACHE_REFILL
```

A decrease in this ratio indicates that sector cache relatively reduced demand misses.

#### L2 Demand Miss Ratio

```
L2D_demand_refill_ratio = L2D_CACHE_REFILL_DM / L2D_CACHE_REFILL
```

---

### 3.3 Reduction of Memory Access Stalls

#### Memory Stall Rate (fraction of total cycles)

```
mem_stall_rate = LD_COMP_WAIT_L2_MISS / CPU_CYCLES
```

#### L2 Access Stall Rate

```
l2_stall_rate = LD_COMP_WAIT_L1_MISS / CPU_CYCLES
```

#### Total Load/Store Stall Rate

```
total_ld_stall_rate = LD_COMP_WAIT / CPU_CYCLES
```

**Effect:**
```
Δmem_stall_rate = mem_stall_rate_off - mem_stall_rate_on
Δl2_stall_rate  = l2_stall_rate_off  - l2_stall_rate_on
```

---

### 3.4 Average L1D Miss Penalty (Outstanding Degree)

```
avg_L1_miss_penalty = L1_MISS_WAIT / L1D_CACHE_REFILL
```

L1_MISS_WAIT is the cumulative count of outstanding miss requests × cycles. This approximates the average latency per miss.

```
avg_L2_miss_penalty = L2_MISS_WAIT / L2_MISS_COUNT
```

If sector cache is effective, the number of misses decreases, which may indirectly improve these values.

---

### 3.5 Sector Utilization Metrics

#### HPC Tag Address Override Usage Rate

```
SCE_usage_ratio = (L1_PIPE0_VAL_IU_TAG_ADRS_SCE + L1_PIPE1_VAL_IU_TAG_ADRS_SCE)
                / (L1_PIPE0_VAL + L1_PIPE1_VAL)
```

→ Fraction of accesses with the SCE bit enabled. Indicates the ratio of instructions under sector cache control.

#### Non-Default Sector Usage Rate

```
non_sec0_ratio = (L1_PIPE0_VAL_IU_NOT_SEC0 + L1_PIPE1_VAL_IU_NOT_SEC0)
              / (L1_PIPE0_COMP + L1_PIPE1_COMP)
```

→ Fraction of accesses with sector_id ≠ 0. Confirms whether multiple sectors are actively used.
  If this value is near 0, sector partitioning is not being utilized.

---

### 3.6 Write-Back Reduction Effect

If sector cache reduces unnecessary evictions, write-backs should also decrease.

```
L1D_WB_per_access = L1D_CACHE_WB / L1D_CACHE
L2D_WB_per_access = L2D_CACHE_WB / L2D_CACHE
```

**Effect:**
```
ΔL1D_WB = L1D_WB_per_access_off - L1D_WB_per_access_on
ΔL2D_WB = L2D_WB_per_access_off - L2D_WB_per_access_on
```

---

### 3.7 Energy Efficiency

```
energy_total = EA_CORE * 8[nJ] + EA_L2 * 32[nJ] + EA_MEMORY * 256[nJ]  (@ 2.2 GHz, 48 cores)
```

#### Energy per Instruction

```
energy_per_inst = energy_total / INST_RETIRED
```

If sector cache reduces memory accesses, EA_MEMORY decreases, improving energy efficiency.

**Effect:**
```
Δenergy_per_inst = energy_per_inst_off - energy_per_inst_on
```

Memory energy fraction:
```
mem_energy_ratio = (EA_MEMORY * 256) / energy_total
```

A decrease in this ratio when sector cache is enabled indicates effective memory traffic reduction.

---

### 3.8 IPC Improvement

```
IPC = INST_RETIRED / CPU_CYCLES
```

**Effect:**
```
ΔIPC = IPC_on - IPC_off
IPC_improvement = ΔIPC / IPC_off * 100 [%]
```

---

## 4. Recommended Measurement Procedure

1. **Baseline Acquisition**: Measure the above events with sector cache disabled (default_sector=0, all sectors at max capacity)
2. **Sector Cache Enabled**: Configure sector_id assignments and max capacities, then measure the same workload
3. **Comparison**: Compute the Δ metrics above

### Key Events for Baseline

```
perf stat -e r0004,r0003,r0200,r0208,r0016,r0017,r0300,r0308,r0309,r0180,r0182,r0184,r0011,r0008 ./app
```

### Additional Events for Sector Utilization Verification

```
perf stat -e r0250,r0252,r02a0,r02a1,r0240,r0241,r0260,r0261 ./app
```

---

## 5. Typical Tuning Scenarios

### Scenario A: Mixed Streaming + Reused Data

- Reused data → sector_id=0 (limited capacity, resistant to eviction)
- Stream data → sector_id=1 (uses remaining capacity)

**Expected Effect**: Decrease in L1D_CACHE_REFILL_DM and LD_COMP_WAIT

### Scenario B: Tiled Matrix Operations

- Matrix A (read-only) → sector_id=0
- Matrix B (read-only) → sector_id=1
- Matrix C (read-write) → sector_id=2

**Expected Effect**: Decrease in L1D_CACHE_WB, improvement in L1D_miss_rate

### L2 Sector Cache Verification

Since L2 is shared per CMG:
```
L2_effectiveness = L2_MISS_COUNT_on / L2_MISS_COUNT_off
```

A value less than 1 indicates improvement. Changes in L2D_SWAP_DM are also useful as an inter-sector interference metric.

---

## 6. Measuring Sector Cache PMU Events with fapp

Use the `event_raw` option of `fapp` to directly measure sector cache-related PMU events.
Since a maximum of 8 events can be specified per measurement run, measurements are split into multiple groups by purpose.

### 6.1 Inserting Measurement Region Markers in Source Code

Insert `fapp_start` / `fapp_stop` around the region where you want to evaluate sector cache effects.

**C/C++:**
```c
#include "fj_tool/fapp.h"

// After applying sector cache configuration
fapp_start("sector_kernel", 1, 0);
// ... kernel under measurement ...
fapp_stop("sector_kernel", 1, 0);
```

**Fortran:**
```fortran
CALL fapp_start("sector_kernel", 1, 0)
! ... kernel under measurement ...
CALL fapp_stop("sector_kernel", 1, 0)
```

### 6.2 Measurement Group Definitions

To comprehensively evaluate sector cache effects, measurements are divided into 5 groups.

#### Group SC1: Basic Performance Metrics (Baseline)

| Event | ID | Purpose |
|---|---|---|
| CPU_CYCLES | 0x0011 | Total cycle count |
| INST_RETIRED | 0x0008 | Retired instruction count |
| L1D_CACHE | 0x0004 | L1D access count |
| L1D_CACHE_REFILL | 0x0003 | L1D miss count |
| L1D_CACHE_REFILL_DM | 0x0200 | L1D demand miss count |
| L1D_CACHE_WB | 0x0015 | L1D write-back count |
| LD_COMP_WAIT | 0x0184 | Load/store wait stall |
| LD_COMP_WAIT_L2_MISS | 0x0180 | Memory access wait stall |

#### Group SC2: L2 Cache Performance

| Event | ID | Purpose |
|---|---|---|
| CPU_CYCLES | 0x0011 | Total cycle count (for normalization) |
| L2D_CACHE | 0x0016 | L2 access count |
| L2D_CACHE_REFILL | 0x0017 | L2 miss count |
| L2D_CACHE_REFILL_DM | 0x0300 | L2 demand miss count |
| L2D_CACHE_WB | 0x0018 | L2 write-back count |
| L2_MISS_WAIT | 0x0308 | L2 miss outstanding/cycle |
| L2_MISS_COUNT | 0x0309 | L2 miss count |
| LD_COMP_WAIT_L1_MISS | 0x0182 | Stall caused by L1 miss |

#### Group SC3: Sector Utilization

| Event | ID | Purpose |
|---|---|---|
| CPU_CYCLES | 0x0011 | Total cycle count (for normalization) |
| L1_PIPE0_VAL | 0x0240 | L1D pipeline #0 valid cycles |
| L1_PIPE1_VAL | 0x0241 | L1D pipeline #1 valid cycles |
| L1_PIPE0_VAL_IU_TAG_ADRS_SCE | 0x0250 | Pipeline #0 SCE bit = 1 |
| L1_PIPE1_VAL_IU_TAG_ADRS_SCE | 0x0252 | Pipeline #1 SCE bit = 1 |
| L1_PIPE0_VAL_IU_NOT_SEC0 | 0x02a0 | Pipeline #0 sector_id ≠ 0 |
| L1_PIPE1_VAL_IU_NOT_SEC0 | 0x02a1 | Pipeline #1 sector_id ≠ 0 |
| L1_PIPE0_COMP | 0x0260 | Pipeline #0 completed requests |

#### Group SC4: Prefetch and Miss Details

| Event | ID | Purpose |
|---|---|---|
| CPU_CYCLES | 0x0011 | Total cycle count (for normalization) |
| L1D_CACHE_REFILL_HWPRF | 0x0202 | L1D refills by HW prefetch |
| L1D_CACHE_REFILL_PRF | 0x0049 | L1D refills by SW/HW prefetch |
| L2D_CACHE_REFILL_HWPRF | 0x0302 | L2 refills by HW prefetch |
| L2D_CACHE_REFILL_PRF | 0x0059 | L2 refills by SW/HW prefetch |
| L2D_SWAP_DM | 0x0325 | Demand → prefetch buffer hit |
| L2D_CACHE_MIBMCH_PRF | 0x0326 | Prefetch → demand buffer hit |
| L1_MISS_WAIT | 0x0208 | L1D miss outstanding/cycle |

#### Group SC5: Energy and Commits

| Event | ID | Purpose |
|---|---|---|
| CPU_CYCLES | 0x0011 | Total cycle count (for normalization) |
| INST_RETIRED | 0x0008 | Retired instruction count |
| EA_CORE | 0x01e0 | Core energy consumption |
| EA_L2 | 0x03e0 | L2 energy consumption |
| EA_MEMORY | 0x03e8 | Memory energy consumption |
| STALL_FRONTEND | 0x0023 | Frontend stall |
| STALL_BACKEND | 0x0024 | Backend stall |
| L1_PIPE1_COMP | 0x0261 | Pipeline #1 completed requests |

### 6.3 Measurement Scripts

#### A) Baseline Measurement (Sector Cache Disabled)

```bash
#!/bin/bash
# measure_baseline.sh
# Measure with sector cache disabled (default state)

APP="./a.out"
BASE_DIR="./prof_baseline"

# SC1: Basic performance metrics
fapp -C -d ${BASE_DIR}/sc1 -Icpupa \
  -Hevent_raw=0x0011,0x0008,0x0004,0x0003,0x0200,0x0015,0x0184,0x0180 \
  ${APP}

# SC2: L2 cache performance
fapp -C -d ${BASE_DIR}/sc2 -Icpupa \
  -Hevent_raw=0x0011,0x0016,0x0017,0x0300,0x0018,0x0308,0x0309,0x0182 \
  ${APP}

# SC3: Sector utilization
fapp -C -d ${BASE_DIR}/sc3 -Icpupa \
  -Hevent_raw=0x0011,0x0240,0x0241,0x0250,0x0252,0x02a0,0x02a1,0x0260 \
  ${APP}

# SC4: Prefetch and miss details
fapp -C -d ${BASE_DIR}/sc4 -Icpupa \
  -Hevent_raw=0x0011,0x0202,0x0049,0x0302,0x0059,0x0325,0x0326,0x0208 \
  ${APP}

# SC5: Energy and commits
fapp -C -d ${BASE_DIR}/sc5 -Icpupa \
  -Hevent_raw=0x0011,0x0008,0x01e0,0x03e0,0x03e8,0x0023,0x0024,0x0261 \
  ${APP}

echo "Baseline measurement complete."
```

#### B) Sector Cache Enabled Measurement

```bash
#!/bin/bash
# measure_sector.sh
# Measure after enabling sector cache (assumes sector config is applied in the application)

APP="./a.out"  # Binary with sector cache configuration code
BASE_DIR="./prof_sector"

# SC1–SC5: Measure with the same event sets as the baseline
fapp -C -d ${BASE_DIR}/sc1 -Icpupa \
  -Hevent_raw=0x0011,0x0008,0x0004,0x0003,0x0200,0x0015,0x0184,0x0180 \
  ${APP}

fapp -C -d ${BASE_DIR}/sc2 -Icpupa \
  -Hevent_raw=0x0011,0x0016,0x0017,0x0300,0x0018,0x0308,0x0309,0x0182 \
  ${APP}

fapp -C -d ${BASE_DIR}/sc3 -Icpupa \
  -Hevent_raw=0x0011,0x0240,0x0241,0x0250,0x0252,0x02a0,0x02a1,0x0260 \
  ${APP}

fapp -C -d ${BASE_DIR}/sc4 -Icpupa \
  -Hevent_raw=0x0011,0x0202,0x0049,0x0302,0x0059,0x0325,0x0326,0x0208 \
  ${APP}

fapp -C -d ${BASE_DIR}/sc5 -Icpupa \
  -Hevent_raw=0x0011,0x0008,0x01e0,0x03e0,0x03e8,0x0023,0x0024,0x0261 \
  ${APP}

echo "Sector cache measurement complete."
```

### 6.4 Exporting Results

```bash
#!/bin/bash
# export_results.sh
# Export results in text, CSV, and XML formats

for CONFIG in baseline sector; do
  BASE_DIR="./prof_${CONFIG}"
  OUT_DIR="./results_${CONFIG}"
  mkdir -p ${OUT_DIR}

  for i in 1 2 3 4 5; do
    # Text format (for visual inspection)
    fapppx -A -Icpupa -ttext -o ${OUT_DIR}/sc${i}.txt -d ${BASE_DIR}/sc${i}

    # CSV format (for script processing)
    fapppx -A -Icpupa -tcsv -o ${OUT_DIR}/sc${i}.csv -d ${BASE_DIR}/sc${i}

    # XML format (for programmatic analysis)
    fapppx -A -Icpupa -txml -o ${OUT_DIR}/sc${i}.xml -d ${BASE_DIR}/sc${i}
  done
done

echo "Export complete."
```

### 6.5 Combining with Standard CPU PA Report

In addition to raw sector cache measurements, collecting the standard CPU PA Report provides a comprehensive overview.

```bash
#!/bin/bash
# measure_standard_pa.sh
# Also collect standard CPU PA Report (Standard = pa1–pa11)

APP="./a.out"

for CONFIG in baseline sector; do
  PA_DIR="./pa_${CONFIG}"

  for i in $(seq 1 11); do
    fapp -C -d ${PA_DIR}/rep${i} -Hevent=pa${i} ${APP}
  done

  # CSV export
  mkdir -p ${PA_DIR}/csv
  for i in $(seq 1 11); do
    fapppx -A -d ${PA_DIR}/rep${i} -Icpupa,nompi -tcsv -o ${PA_DIR}/csv/pa${i}.csv
  done
done
```

The Standard Report automatically aggregates Cycle Accounting, Cache, Power Consumption, and other sections.
Cross-referencing these with raw event measurement results enables multi-faceted evaluation of sector cache effects.

### 6.6 Comparative Analysis Script Example

A simple script to extract values from XML and compare baseline vs. sector cache enabled:

```javascript
#!/usr/bin/env node
// compare_sector_cache.mjs - Comparative analysis of sector cache effects

import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

// ---- XML Parser (lightweight: no external dependencies) ----

/**
 * Extract <event name="...">value</event> from fapp XML output
 * @param {string} xmlPath
 * @returns {Record<string, number>}
 */
function parseFappXml(xmlPath) {
  const xml = readFileSync(xmlPath, "utf-8");
  const events = {};
  const re = /<event\s+name="([^"]+)"\s*>\s*(\d+)\s*<\/event>/g;
  let m;
  while ((m = re.exec(xml)) !== null) {
    events[m[1]] = Number(m[2]);
  }
  return events;
}

// ---- Helpers ----

/** Get event value (returns default if undefined) */
const ev = (events, id, def = 0) => events[id] ?? def;

/** Safe division */
const safeDiv = (a, b) => (b !== 0 ? a / b : 0);

// ---- Analysis Functions ----

/** SC1: Compute basic performance metrics */
function analyzeSC1(events) {
  const cycles  = ev(events, "0x0011", 1);
  const inst    = ev(events, "0x0008");
  const l1dAcc  = ev(events, "0x0004", 1);
  const l1dMiss = ev(events, "0x0003");
  const l1dDm   = ev(events, "0x0200");
  const l1dWb   = ev(events, "0x0015");
  const ldWait  = ev(events, "0x0184");
  const ldL2m   = ev(events, "0x0180");

  return {
    IPC:                  safeDiv(inst, cycles),
    L1D_miss_rate:        safeDiv(l1dMiss, l1dAcc),
    L1D_demand_miss_rate: safeDiv(l1dDm, l1dAcc),
    L1D_WB_per_access:    safeDiv(l1dWb, l1dAcc),
    LD_stall_rate:        safeDiv(ldWait, cycles),
    Mem_stall_rate:       safeDiv(ldL2m, cycles),
  };
}

/** SC2: Compute L2 cache performance metrics */
function analyzeSC2(events) {
  const cycles   = ev(events, "0x0011", 1);
  const l2Acc    = ev(events, "0x0016", 1);
  const l2Miss   = ev(events, "0x0017");
  const l2Dm     = ev(events, "0x0300");
  const l2Wb     = ev(events, "0x0018");
  const l2MissWt = ev(events, "0x0308");
  const l2MissCt = ev(events, "0x0309");
  const l1MissWt = ev(events, "0x0182");

  return {
    L2D_miss_rate:        safeDiv(l2Miss, l2Acc),
    L2D_demand_miss_rate: safeDiv(l2Dm, l2Acc),
    L2D_WB_per_access:    safeDiv(l2Wb, l2Acc),
    avg_L2_miss_penalty:  safeDiv(l2MissWt, l2MissCt),
    L2_stall_rate:        safeDiv(l1MissWt, cycles),
  };
}

/** SC3: Compute sector utilization metrics */
function analyzeSC3(events) {
  const pipe0Val = ev(events, "0x0240", 1);
  const pipe1Val = ev(events, "0x0241", 1);
  const sce0     = ev(events, "0x0250");
  const sce1     = ev(events, "0x0252");
  const nsec0    = ev(events, "0x02a0");
  const nsec1    = ev(events, "0x02a1");

  const totalVal = pipe0Val + pipe1Val;
  return {
    SCE_usage_ratio: safeDiv(sce0 + sce1, totalVal),
    Non_sec0_ratio:  safeDiv(nsec0 + nsec1, totalVal),
  };
}

/** SC4: Compute prefetch and miss detail metrics */
function analyzeSC4(events) {
  const cycles    = ev(events, "0x0011", 1);
  const l1Hwprf   = ev(events, "0x0202");
  const l1Prf     = ev(events, "0x0049");
  const l2Hwprf   = ev(events, "0x0302");
  const l2Prf     = ev(events, "0x0059");
  const l2Swap    = ev(events, "0x0325");
  const l2Mibmch  = ev(events, "0x0326");
  const l1MissWt  = ev(events, "0x0208");

  return {
    L1_hwprf_refill_per_cycle: safeDiv(l1Hwprf, cycles),
    L1_prf_refill_per_cycle:   safeDiv(l1Prf, cycles),
    L2_hwprf_refill_per_cycle: safeDiv(l2Hwprf, cycles),
    L2_prf_refill_per_cycle:   safeDiv(l2Prf, cycles),
    L2_swap_dm_per_cycle:      safeDiv(l2Swap, cycles),
    L2_mibmch_prf_per_cycle:   safeDiv(l2Mibmch, cycles),
    avg_L1_miss_outstanding:   safeDiv(l1MissWt, cycles),
  };
}

/** SC5: Compute energy and commit metrics */
function analyzeSC5(events) {
  const cycles  = ev(events, "0x0011", 1);
  const inst    = ev(events, "0x0008");
  const eaCore  = ev(events, "0x01e0");
  const eaL2    = ev(events, "0x03e0");
  const eaMem   = ev(events, "0x03e8");
  const stallFe = ev(events, "0x0023");
  const stallBe = ev(events, "0x0024");

  // Energy (nJ): assuming 2.2 GHz 48-core configuration
  const totalEnergy = eaCore * 8 + eaL2 * 32 + eaMem * 256;

  return {
    energy_per_inst_nJ:    safeDiv(totalEnergy, inst),
    mem_energy_ratio:      safeDiv(eaMem * 256, totalEnergy),
    frontend_stall_rate:   safeDiv(stallFe, cycles),
    backend_stall_rate:    safeDiv(stallBe, cycles),
  };
}

// ---- Comparison Display ----

const ANALYZERS = [
  { file: "sc1.xml", label: "SC1: Basic Performance",     fn: analyzeSC1 },
  { file: "sc2.xml", label: "SC2: L2 Cache Performance",  fn: analyzeSC2 },
  { file: "sc3.xml", label: "SC3: Sector Utilization",    fn: analyzeSC3 },
  { file: "sc4.xml", label: "SC4: Prefetch / Miss Detail", fn: analyzeSC4 },
  { file: "sc5.xml", label: "SC5: Energy / Commit",       fn: analyzeSC5 },
];

function compare(baseline, sector, label) {
  const pad = (s, n) => s.length >= n ? s : " ".repeat(n - s.length) + s;
  const padL = (s, n) => s.length >= n ? s : s + " ".repeat(n - s.length);
  const fmt = (v) => v.toFixed(6);
  const fmtD = (v) => (v >= 0 ? "+" : "") + v.toFixed(6);

  console.log(`\n${"=".repeat(72)}`);
  console.log(`  ${label}`);
  console.log(`${"=".repeat(72)}`);
  console.log(
    `${padL("Metric", 32)} ${pad("Baseline", 14)} ${pad("Sector", 14)} ${pad("Delta", 14)}`
  );
  console.log("-".repeat(74));

  for (const key of Object.keys(baseline)) {
    const b = baseline[key];
    const s = sector[key];
    const d = s - b;
    console.log(
      `${padL(key, 32)} ${pad(fmt(b), 14)} ${pad(fmt(s), 14)} ${pad(fmtD(d), 14)}`
    );
  }
}

// ---- Main ----

const { positionals } = parseArgs({ allowPositionals: true });
const baseDir = positionals[0] || "./results_baseline";
const sectDir = positionals[1] || "./results_sector";

for (const { file, label, fn } of ANALYZERS) {
  const basePath = resolve(baseDir, file);
  const sectPath = resolve(sectDir, file);

  if (existsSync(basePath) && existsSync(sectPath)) {
    const b = fn(parseFappXml(basePath));
    const s = fn(parseFappXml(sectPath));
    compare(b, s, label);
  } else {
    console.log(`\n[SKIP] ${label} — file not found (${file})`);
  }
}
```

Usage:
```bash
node compare_sector_cache.mjs ./results_baseline ./results_sector
```

### 6.7 Important Notes

1. **CPU Binding is Required**: Sector cache is a per-PE/CMG feature, so thread-to-CPU binding is mandatory.
   Improper binding may mix L2 sector settings from different CMGs, producing incorrect measurements.

2. **Execution Reproducibility**: Use identical input and execution paths for baseline and sector cache enabled runs.
   Since `fapp` measurements are performed per run, be mindful of execution variability.

3. **CMG-Scoped Events**: `L2_MISS_WAIT` (0x0308), `L2_MISS_COUNT` (0x0309), `EA_L2` (0x03e0),
   `EA_MEMORY` (0x03e8), etc. count across the entire CMG. Contributions from a specific PE cannot be isolated.

4. **Recommend method=fast**: Counter precision is important for sector cache evaluation.
   For kernel-focused measurements where sleep-state counting is not a concern, `method=fast` is effective.
   ```bash
   fapp -C -d ${DIR} -Icpupa \
     -Hevent_raw=0x0011,...,method=fast,mode=user \
     ${APP}
   ```

5. **Maximum 8 Events per event_raw**: Since more than 8 events cannot be measured simultaneously,
   they must be split into groups across multiple runs. Including CPU_CYCLES (0x0011) in each group
   enables normalization and comparison across different runs.

6. **Shared L2 Sector Cache Registers**: The L2 sector setting registers (`IMP_SCCR_SET0_L2_EL1`,
   `IMP_SCCR_SET1_L2_EL1`) are shared within a CMG. When multiple processes share a CMG,
   configuration changes from other processes may affect measurements.
