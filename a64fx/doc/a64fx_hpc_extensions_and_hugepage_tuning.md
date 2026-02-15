# A64FX HPC Extensions & Large Page Tuning Guide

Consolidated from Fujitsu official documentation:
- *A64FX Specification HPC Extension v1* (EN, Nov 2020)
- *Job Operation Software End User Guide HPC Extension Edition* (j2ul-2535-01z0, JA, Sep 2022)
- *Processors Programming Guide* (EN/JA, 2021)
- *Tuning Programming Guide* (JA, v2.2, Mar 2023)
- *C Language User's Guide* (j2ul-2560-01z0, JA)

---

## Part I: HPC Extensions Overview

The A64FX processor implements five proprietary Fujitsu HPC extensions, succeeding features from SPARC64 VIIIfx/IXfx/XIfx:

1. **HPC Tag Address Override** — Control sector cache and hardware prefetch from applications
2. **Sector Cache** — Cache partitioning by data access characteristics (temporal locality)
3. **Hardware Prefetch Assistance** — Software hints for complex prefetch patterns
4. **Hardware Barrier** — Inter-thread synchronization support
5. **Large Page Library (libmpg)** — Efficient HugeTLBfs utilization

---

## Part II: HPC Tag Address Override

### 2.1 Mechanism

The HPC tag address override repurposes the upper 8 bits of 64-bit virtual addresses (normally unused in ARMv8-A Tagged Addressing) as hardware performance hints. When enabled, these bits are not used for addressing but instead control:

- **Sector cache** behavior (which sector to place data in)
- **Hardware prefetch** behavior (enable/disable, stream detect hints)

This is compatible with ARMv8-A Tagged Addressing at the ISA level — both features use the same top-8-bit mechanism, but they are **mutually exclusive** in practice.

### 2.2 Enable/Disable Settings

| TBI (ARMv8-A) | TBO (Fujitsu) | Behavior |
|:-:|:-:|---|
| 0 | — | Both disabled. Full 64-bit address used. |
| 1 | 0 | ARMv8-A Tagged Addressing enabled. Top 8 bits ignored for addressing. |
| 1 | 1 | **HPC tag address override enabled.** Top 8 bits = performance hints. |

Additional control bits within TBO:
- **SCE** (Sector Cache Enable) — per-region enable/disable
- **PFE** (hardware PreFetch Enable) — per-region enable/disable

When HPC tag override is disabled (TBI=0 or TBO=0), sector cache uses the Default Sector from register settings, and hardware prefetch operates in Stream Detect mode (configured via `IMP_PF_STREAM_DETECT_CTRL_EL0`).

### 2.3 System Registers

| Register | EL | Shared Domain |
|----------|:--:|:---:|
| `IMP_FJ_TAG_ADDRESS_CTRL_EL1` | 1–3 | PE |
| `IMP_FJ_TAG_ADDRESS_CTRL_EL2` | 2–3 | PE |
| `IMP_FJ_TAG_ADDRESS_CTRL_EL3` | 3 | PE |
| `IMP_FJ_TAG_ADDRESS_CTRL_EL12` | 2–3 (alias) | PE |

ARMv8.1 virtualization host extension (HCR_EL2.E2H=1, SCR_EL3.NS=1) adds PFE1/SCE1/TBO1 bits to EL2 and redirects EL1 accesses to EL2.

### 2.4 Tag Address Bit Allocation

The upper 8 bits of the virtual address (bits [63:56]) are divided among HPC features when the override is active. Specific bit assignments control sector ID selection and prefetch hints per load/store/prefetch instruction.

#### 2.4.1 Sector Cache Relevant Bits (Experimentally Verified)

Only bits [57:56] of the virtual address control sector cache behavior:

```
63                58 57 56 55                                 0
┌──────────────────┬─────┬────────────────────────────────────┐
│ other HPC / ign. │secID│          physical address           │
└──────────────────┴─────┴────────────────────────────────────┘
```

| Bits | Function | Verified |
|------|----------|:--------:|
| [57:56] | `sector_id[1:0]` — selects L1D sector (0–3). L2 uses only bit [56]. | Yes |
| [63:58] | Other HPC features (prefetch hints) or unused. **No effect on sector cache.** | Yes |

**Common misconception:** Documentation and some references describe bits [59:56] as `{SCE, PFE, sec_id[1:0]}`, implying SCE and PFE are per-address bits. This is **incorrect** — SCE and PFE are **system register controls** in `IMP_FJ_TAG_ADDRESS_CTRL_EL1`, not per-address fields. See §2.4.2 below.

#### 2.4.2 SCE Bit Investigation (Fugaku PMU Measurements, Feb 2025)

**Question:** Does bit 59 of the virtual address act as a per-access SCE (Sector Cache Enable)?

**Experiment:** Gather-stream benchmark (`result += values[i] * table[indices[i]]`) with fapp PMU profiling. Compared address tag `0xA` (bit 59=1, sector_id=2) against `0x2` (bit 59=0, sector_id=2) — identical except for bit 59.

**PMU Events (SC3 group, single-thread, FLIB env vars enabled):**

| PMU Counter | tag 0xA (bit59=1) | tag 0x2 (bit59=0) | Difference |
|-------------|------------------:|-----------------:|:----------:|
| CPU_CYCLES (0x0011) | 6,560,972,693 | 6,563,190,723 | -0.03% |
| L1_TAG_SCE (0x0250+0x0252) | 2,263,047,023 | 2,263,224,226 | -0.01% |
| L1_NOT_SEC0 (0x02a0+0x02a1) | 1,231,006,377 | 1,231,109,507 | -0.01% |
| L2_SC_MISS (0x0260) | 706,700,108 | 706,675,684 | +0.003% |

All counters match within noise (<0.03%). Bit 59 has **zero observable effect**.

**Additional evidence — PMU event 0x0250 (L1_PIPEx_VAL_IU_TAG_ADRS_SCE):**

| Region | SCE_active / TAG_SCE |
|--------|:--------------------:|
| nohint (no tags at all) | 44.9% |
| tag 0x2 (bit 59=0) | 44.1% |
| tag 0xA (bit 59=1) | 44.1% |
| manual SCCR | 44.2% |

The "SCE_active" ratio is ~44% for **all variants including nohint** (which uses no tagged pointers). This confirms the PMU event counts the system-level SCE state from `IMP_FJ_TAG_ADDRESS_CTRL_EL1`, not a per-address bit.

**Conclusion:**
- **SCE** is a system register control (`IMP_FJ_TAG_ADDRESS_CTRL_EL1`), always ON on Fugaku (HPC tag override enabled by default via `fhetbo enable`)
- **Bit 59** of the virtual address is **not** SCE — it is either unused or assigned to a different HPC feature (possibly prefetch-related)
- **Only bits [57:56]** (sector_id) control sector cache behavior at the per-access level
- The FCC compiler correctly uses only `orr Xn, Xbase, #(1<<56)` (bit 56) for sector 1 assignment

### 2.5 fhetbo Command

The `fhetbo` command controls HPC tag address override at the job level:

```bash
/opt/FJSVxos/fhehpc/bin/fhetbo {enable|disable}
```

- **Scope:** All cores assigned to the current job on the executing node
- **Default:** Enabled (HPC override active, ARMv8-A tagging disabled)
- **Persistence:** Reverts to enabled at job termination
- **Multi-node:** Use via `mpiexec` for multi-node jobs

**Job script example:**
```bash
#!/bin/bash
#PJM -L "node=256"
#PJM -L "elapse=86400"
export NR_PROCS=256
mpiexec -n $NR_PROCS /opt/FJSVxos/fhehpc/bin/fhetbo disable  # Disable HPC override
mpiexec -n $NR_PROCS ./a.out                                   # Run application
mpiexec -n $NR_PROCS /opt/FJSVxos/fhehpc/bin/fhetbo enable   # Re-enable
```

**Messages:**
- `[INFO] xos FHE 1113` — Tag override enabled with core mask
- `[INFO] xos FHE 1114` — Tag override disabled with core mask
- Core mask example: `0xffffffffffff000` = cores 12–59 (job cores)

---

## Part III: Sector Cache

### 3.1 Overview

Sector cache is a **cache partitioning** function that splits L1D and L2 caches into multiple sectors and controls them separately. A maximum capacity is specified for each sector, and the cache controller ensures that data in a given sector is not evicted as long as that sector's consumed capacity is below its configured maximum. Because multiple sectors can coexist in cache with independently configured capacities, applications have a high degree of freedom to separate data with different temporal locality characteristics.

This prevents streaming data from evicting frequently reused data, and vice versa.

### 3.2 Architecture

The sector cache mechanism is implemented on **both L1D cache and L2 cache**, controlled independently:

- **L1D cache:** 4 sectors (sector ID 0–3), per-PE control
- **L2 cache:** 4 sectors (sector ID 0–3), per-CMG control. The 4 L2 sectors are divided into 2 sets of 2 sectors each. Each PE selects one set via `IMP_SCCR_ASSIGN_EL1.assign`.

**Sector assignment behavior:**

- When data is **not** in cache (cache miss): data is fetched from memory and stored with the sector number attached. The eviction candidate is selected considering sector capacity limits.
- When data **is** in cache (cache hit): data on **all sectors** can be accessed regardless of the access's sector number. The `mode` bit in `IMP_SCCR_ASSIGN_EL1` controls whether the cache line's sector ID is updated to match the access sector ID (`mode=0`) or kept as-is (`mode=1`).

**Note:** Instruction fetch (L1I) does not use sector cache directly. However, L1I-to-L2 requests use `assign` and `default_sector<0>` to determine their L2 sector.

**Recommendations:**
- When all 48 computing cores are used by one process, set all CMGs to the same sector cache configuration.
- When cores within a CMG are shared by multiple processes, use consistent sector cache settings across processes within the same CMG.

### 3.3 Sector ID Determination

Sector IDs are determined by the combination of HPC tag address override state and register settings:

| Cache | HPC Tag Override Disabled | HPC Tag Override Enabled |
|-------|---------------------------|--------------------------|
| **L1D** | `default_sector<1:0>` | `TagAddress.sector_id<1:0>` (bits [57:56] of address) |
| **L2** | `assign::default_sector<0>` | `assign::sector_id<0>` (bit [56] of address, mapped via `assign`) |

Where `default_sector` and `assign` are fields from `IMP_SCCR_ASSIGN_EL1`.

The tag address bits [57:56] (`sector_id`) are only used when both `TBI=1`, `TBO=1`, and `SCE=1`. Otherwise sector cache operates with the default sector.

### 3.4 System Registers

| op0 | op1 | CRn | CRm | op2 | Register | Shared | App Access |
|:---:|:---:|:---:|:---:|:---:|----------|:------:|:----------:|
| 11 | 000 | 1011 | 1000 | 000 | `IMP_SCCR_CTRL_EL1` | PE | Read-only (EL0) |
| 11 | 000 | 1011 | 1000 | 001 | `IMP_SCCR_ASSIGN_EL1` | PE | EL1+ only |
| 11 | 011 | 1011 | 1000 | 010 | **`IMP_SCCR_L1_EL0`** | PE | **EL0 accessible** |
| 11 | 000 | 1111 | 1000 | 010 | `IMP_SCCR_SET0_L2_EL1` | CMG | EL1+ only |
| 11 | 000 | 1111 | 1000 | 011 | `IMP_SCCR_SET1_L2_EL1` | CMG | EL1+ only |
| 11 | 011 | 1111 | 1000 | 010 | **`IMP_SCCR_VSCCR_L2_EL0`** | PE(→CMG) | **EL0 accessible** |

Access control: EL0/EL1 access is gated by `IMP_SCCR_CTRL_EL1.el1ae` and `el0ae` bits. When `el1ae=1` and `el0ae=1`, user-space (EL0) can directly read/write `IMP_SCCR_L1_EL0` and `IMP_SCCR_VSCCR_L2_EL0`. Otherwise, access traps to EL1 (EC=0x18).

#### 3.4.1 IMP_SCCR_CTRL_EL1 — Access Control

64-bit register, per-PE. Controls whether lower ELs can access sector cache registers.

```
63    62    61                                           0
┌──────┬──────┬─────────────────────────────────────────────┐
│el1ae │el0ae │                   RES0                      │
└──────┴──────┴─────────────────────────────────────────────┘
```

| Bit | Name | Description |
|:---:|------|-------------|
| [63] | `el1ae` | 1: NS-EL1 R/W to all sector cache registers. 0: NS-EL1 access traps to EL2. Writable only from Secure EL1 / EL2 / EL3. |
| [62] | `el0ae` | 1: EL0 R/W to `IMP_SCCR_L1_EL0` and `IMP_SCCR_VSCCR_L2_EL0` (when `el1ae=1`). 0: EL0 access traps to EL1. |

Access: `MRS/MSR S3_0_C11_C8_0`

#### 3.4.2 IMP_SCCR_ASSIGN_EL1 — Sector Assignment & Mode Control

64-bit register, per-PE. Controls sector assignment behavior and L2 set selection.

```
31                           4    3      2       1      0
┌────────────────────────────┬──────┬────────┬────────────┐
│            RES0            │ mode │ assign │default_sect│
└────────────────────────────┴──────┴────────┴────────────┘
```

| Bit | Name | Description |
|:---:|------|-------------|
| [3] | `mode` | 0: Cache line's sector ID is updated when accessed with a different sector ID. 1: Cache line keeps its original sector ID regardless of access sector ID. |
| [2] | `assign` | Selects which L2 set register `IMP_SCCR_VSCCR_L2_EL0` aliases to. 0: → `IMP_SCCR_SET0_L2_EL1`. 1: → `IMP_SCCR_SET1_L2_EL1`. |
| [1:0] | `default_sector` | Default sector ID used when sector is not specified by instruction (e.g., HPC tag override disabled, or instruction fetch). L1D uses both bits [1:0]. L2 uses only bit [0]. |

Access: `MRS/MSR S3_0_C11_C8_1`

#### 3.4.3 IMP_SCCR_L1_EL0 — L1D Sector Cache Capacity (App-Accessible)

64-bit register, **per-PE**. Sets the maximum capacity for each of the 4 L1D cache sectors. This register is directly accessible from user space (EL0) when `el1ae=1` and `el0ae=1`.

```
31          15  14  12  11  10   8   7   6   4   3   2   0
┌────────────┬───────┬─────┬───────┬─────┬───────┬─────┬───────┐
│    RES0    │sec3_mx│RES0 │sec2_mx│RES0 │sec1_mx│RES0 │sec0_mx│
└────────────┴───────┴─────┴───────┴─────┴───────┴─────┴───────┘
```

| Bits | Name | Description |
|:----:|------|-------------|
| [14:12] | `l1_sec3_max` | Maximum sector count for L1D Sector 3 (3-bit value) |
| [11] | — | Reserved (RES0) |
| [10:8] | `l1_sec2_max` | Maximum sector count for L1D Sector 2 (3-bit value) |
| [7] | — | Reserved (RES0) |
| [6:4] | `l1_sec1_max` | Maximum sector count for L1D Sector 1 (3-bit value) |
| [3] | — | Reserved (RES0) |
| [2:0] | `l1_sec0_max` | Maximum sector count for L1D Sector 0 (3-bit value) |

Each 3-bit field specifies the maximum number of ways that the corresponding sector ID can occupy in L1D cache. The A64FX L1D cache is 64 KiB, 4-way set-associative (1 way = 16KB).

**Effective value range:** 0–4. Values 0–3 partition the cache; value 4 (or higher) means "use all 4 ways" — equivalent to no partition for that sector. The register accepts 0–7 but values ≥4 have no additional effect since L1D has exactly 4 ways.

**Oversubscription:** The sum of sector max values can exceed 4 (e.g., sec0=3, sec1=3). In this case sectors compete for the physical 4 ways and the partitioning guarantee weakens — the controller treats each value as a *soft maximum* rather than a hard reservation.

**Access encoding:**
```
MRS <Xt>, S3_3_C11_C8_2    // Read L1 sector cache settings
MSR S3_3_C11_C8_2, <Xt>    // Write L1 sector cache settings
```
(op0=11, op1=011, CRn=1011, CRm=1000, op2=010)

**Inline assembly example (C/C++):**
```c
#include <stdint.h>

// Read current L1 sector cache configuration
static inline uint64_t read_sccr_l1(void) {
    uint64_t val;
    asm volatile("mrs %0, S3_3_C11_C8_2" : "=r"(val));
    return val;
}

// Write L1 sector cache configuration
// sec0..sec3: 3-bit max sector values (0–7)
static inline void write_sccr_l1(unsigned sec0, unsigned sec1,
                                  unsigned sec2, unsigned sec3) {
    uint64_t val = ((uint64_t)(sec3 & 0x7) << 12)
                 | ((uint64_t)(sec2 & 0x7) << 8)
                 | ((uint64_t)(sec1 & 0x7) << 4)
                 | ((uint64_t)(sec0 & 0x7));
    asm volatile("msr S3_3_C11_C8_2, %0" :: "r"(val));
}
```

#### 3.4.4 IMP_SCCR_SET0_L2_EL1 / IMP_SCCR_SET1_L2_EL1 — L2 Sector Cache Capacity (Privileged)

64-bit registers, **per-CMG** (shared across all PEs in the CMG). Changing from one PE affects all PEs in the same CMG. Two sets exist (SET0 and SET1), each controlling 2 of the 4 L2 sectors.

```
31          13  12       8   7   5   4       0
┌────────────┬───────────┬─────┬───────────┐
│    RES0    │l2_sec1_max│RES0 │l2_sec0_max│
└────────────┴───────────┴─────┴───────────┘
```

**SET0** (op0=11, op1=000, CRn=1111, CRm=1000, op2=010):

| Bits | Name | Description |
|:----:|------|-------------|
| [12:8] | `l2_sec1_max` | Maximum sector count for L2 Sector ID=1 (5-bit) |
| [4:0] | `l2_sec0_max` | Maximum sector count for L2 Sector ID=0 (5-bit) |

**SET1** (op0=11, op1=000, CRn=1111, CRm=1000, op2=011):

| Bits | Name | Description |
|:----:|------|-------------|
| [12:8] | `l2_sec1_max` | Maximum sector count for L2 Sector ID=3 (5-bit) |
| [4:0] | `l2_sec0_max` | Maximum sector count for L2 Sector ID=2 (5-bit) |

Access:
```
MRS/MSR S3_0_C15_C8_2    // SET0 (sectors 0,1)
MRS/MSR S3_0_C15_C8_3    // SET1 (sectors 2,3)
```

These registers are EL1+ only. User-space accesses them indirectly through the window register below.

#### 3.4.5 IMP_SCCR_VSCCR_L2_EL0 — L2 Sector Cache Capacity Window (App-Accessible)

64-bit **window register**, per-PE but updates the underlying CMG-shared SET register. This register is directly accessible from user space (EL0) when `el1ae=1` and `el0ae=1`.

The register acts as an **alias** to either `IMP_SCCR_SET0_L2_EL1` or `IMP_SCCR_SET1_L2_EL1`, selected by `IMP_SCCR_ASSIGN_EL1.assign`:

- `assign=0` → reads/writes go to `IMP_SCCR_SET0_L2_EL1` (controls L2 sectors 0 and 1)
- `assign=1` → reads/writes go to `IMP_SCCR_SET1_L2_EL1` (controls L2 sectors 2 and 3)

```
31          13  12       8   7   5   4       0
┌────────────┬───────────┬─────┬───────────┐
│    RES0    │l2_sec1_max│RES0 │l2_sec0_max│
└────────────┴───────────┴─────┴───────────┘
```

| Bits | Name | Description |
|:----:|------|-------------|
| [12:8] | `l2_sec1_max` | Maximum sector count for L2 Sector ID=1 or 3 (5-bit, depending on `assign`) |
| [4:0] | `l2_sec0_max` | Maximum sector count for L2 Sector ID=0 or 2 (5-bit, depending on `assign`) |

**⚠ Important:** Because the underlying SET registers are CMG-shared resources, writing to `IMP_SCCR_VSCCR_L2_EL0` from any PE **affects all PEs in the same CMG**.

**Access encoding:**
```
MRS <Xt>, S3_3_C15_C8_2    // Read L2 sector cache settings (via window)
MSR S3_3_C15_C8_2, <Xt>    // Write L2 sector cache settings (via window)
```
(op0=11, op1=011, CRn=1111, CRm=1000, op2=010)

**Inline assembly example (C/C++):**
```c
#include <stdint.h>

// Read L2 sector cache capacity via window register
static inline uint64_t read_vsccr_l2(void) {
    uint64_t val;
    asm volatile("mrs %0, S3_3_C15_C8_2" : "=r"(val));
    return val;
}

// Write L2 sector cache capacity via window register
// sec_lo, sec_hi: 5-bit max sector values (0–31)
// Which L2 sectors these map to depends on IMP_SCCR_ASSIGN_EL1.assign
static inline void write_vsccr_l2(unsigned sec_lo, unsigned sec_hi) {
    uint64_t val = ((uint64_t)(sec_hi & 0x1F) << 8)
                 | ((uint64_t)(sec_lo & 0x1F));
    asm volatile("msr S3_3_C15_C8_2, %0" :: "r"(val));
}
```

### 3.5 Typical Usage Pattern from User Space

To configure sector cache from an application (requires OS/runtime to have set `el1ae=1, el0ae=1`):

1. **L1D sectors** — Write `IMP_SCCR_L1_EL0` directly with desired capacity for each of 4 sectors.
2. **L2 sectors** — The `assign` bit and `default_sector` are set by the OS/runtime via `IMP_SCCR_ASSIGN_EL1`. The application writes `IMP_SCCR_VSCCR_L2_EL0` to configure the selected pair of L2 sectors.
3. **Per-instruction sector selection** — With HPC tag address override enabled (`TBI=1, TBO=1, SCE=1`), embed `sector_id` in bits [57:56] of the address for each load/store/prefetch instruction.

### 3.6 Enabling Sector Cache on Fugaku

SCCR registers (`S3_3_C11_C8_2`, `S3_3_C15_C8_2`) are **not directly accessible** from EL0 by default on Fugaku — the kernel ships with `IMP_SCCR_CTRL_EL1.el0ae=0`, so direct MSR/MRS traps with SIGILL.

Access is enabled at runtime through a kernel driver and OS library. The full software stack:

```
Application (EL0)
  │
  ├── FCC pragma:  #pragma procedure scache_isolate_way L1=2
  │     generates:  bl __jwe_xset_sccr      (libfj90i.so.1)
  │
  ├── __jwe_xset_sccr (Fujitsu runtime, libfj90i.so.1)
  │     ├── first call → __jwe_xsccr_init → __jwe_xsccr_init_com
  │     │     ├── dlopen("libsec.so")       → /lib64/libsec.so
  │     │     └── xos_sclib_init()
  │     ├── subsequent calls → direct MSR S3_3_C11_C8_2
  │     └── on return → MSR S3_3_C11_C8_2, 0  (reset to default)
  │
  ├── libsec.so (OS sector cache library, /lib64/libsec.so)
  │     xos_sclib_init():
  │       ├── open("/dev/xos_sec_normal", O_RDONLY)
  │       ├── read /proc/<pid>/cgroup  (cgroup info for multi-job isolation)
  │       ├── ioctl(fd, 0xee08, ...)   → query driver capabilities
  │       ├── ioctl(fd, 0xee06, ...)   → set cgroup/config (24 bytes)
  │       ├── ioctl(fd, 0xee05, ...)   → *** enable EL0 SCCR access ***
  │       ├── ioctl(fd, 0xee01, ...)   → read current L1 SCCR value
  │       ├── ioctl(fd, 0xee04, ...)   → read current L2 SCCR info
  │       └── close(fd)
  │
  └── Kernel driver (/dev/xos_sec_normal, char 240:0)
        ioctl 0xee05 → sets IMP_SCCR_CTRL_EL1.el0ae=1 for this PE
```

After `xos_sclib_init` completes, the process can directly read/write SCCR registers via MSR/MRS for the remainder of its lifetime.

#### 3.6.1 Using FCC Pragmas (Recommended)

**Pragmas:**
```c
/* Inside a function body: */
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign stream_array
```

- `scache_isolate_way L1=N`: assign N ways (of 4) to sector 1 for L1D. Sector 0 gets 4−N ways.
- `scache_isolate_way L2=N`: assign N ways (of 16) to sector 1 for L2.
- `scache_isolate_assign var`: loads/stores to `var` use tag bit 56 → routed to sector 1.

**Compiler emits:**
1. `bl __jwe_xset_sccr` at function entry with packed config: `{L1_line_size=256, L1_config=0x20002, L2_config=0x50009}`
2. `orr Xn, Xbase, #(1<<56)` on assigned pointer — sets tag bit 56 for sector 1
3. `bl __jwe_xset_sccr` at function exit with reset config `{512}`

**Build:**
```bash
fcc -Nnoclang -O2 -Kocl,hpctag -o program program.c
```
`-Kocl` enables OCL pragma recognition, `-Khpctag` enables HPC tag address override codegen.

**Run (environment variables required):**
```bash
export FLIB_SCCR_CNTL=TRUE           # Enable sector cache runtime init
export FLIB_L1_SCCR_CNTL=TRUE        # Enable L1 sector cache specifically
export FLIB_L2_SCCR_CNTL_EX=TRUE     # Enable L2 sector cache (required for L2=N pragma!)
./program
```

Without these env vars, `__jwe_xset_sccr` short-circuits (no init, no MSR, no effect).
**⚠ Without `FLIB_L2_SCCR_CNTL_EX=TRUE`, L2 SCCR stays at sec0=14,sec1=0 even with `L2=N` pragma.**

**Additional FLIB environment variables:**
| Variable | Description |
|----------|-------------|
| `FLIB_SCCR_CNTL` | Master enable for sector cache runtime (`TRUE`/`FALSE`) |
| `FLIB_L1_SCCR_CNTL` | Enable L1 sector cache (`TRUE`/`FALSE`) |
| `FLIB_L2_SCCR_CNTL_EX` | Extended L2 sector cache control |
| `FLIB_SCCR_CNTL_OSCALL` | Use OS-call path for SCCR access |
| `FLIB_SCCR_USE_MAXWAY` | Use maximum way count for sectors |
| `FLIB_L1_SECTOR_NWAYS_INIT_NP` | Initial L1 sector way count (non-parallel) |
| `FLIB_L2_SECTOR_NWAYS_INIT` | Initial L2 sector way count |

#### 3.6.2 Using libsec.so Directly (Low-Level)

The sector cache library exports these functions:

```c
/* Link with -lsec or dlopen("libsec.so") */
int  xos_sclib_init(void);                           /* Enable EL0 SCCR access */
int  xos_sclib_set_l1_way(int s0, int s1, int s2, int s3); /* Set L1 sectors (0-4 each) */
int  xos_sclib_set_l2_way(int s0, int s1);           /* Set L2 sectors */
int  xos_sclib_get_cache_size(int level, int *size);  /* Query cache size */
void xos_sclib_finalize(void);                        /* Cleanup */
```

`xos_sclib_set_l1_way` validates each sector value ≤ 4, packs as `sec0 | (sec1<<4) | (sec2<<8) | (sec3<<12)`, and does `msr S3_3_C11_C8_2, Xn`.

**Note:** Calling `xos_sclib_init` directly from user code may fail with `EBUSY` if the device is already held by another process. The fjomplib path (`FLIB_SCCR_CNTL=TRUE`) handles this internally.

#### 3.6.3 Verified Register Values (Fugaku Measurements)

With `FLIB_SCCR_CNTL=TRUE FLIB_L1_SCCR_CNTL=TRUE` and `#pragma procedure scache_isolate_way L1=2`:

| Point | L1 SCCR (`S3_3_C11_C8_2`) | L2 SCCR (`S3_3_C15_C8_2`) | Meaning |
|-------|:--:|:--:|---|
| Inside pragma fn | `0x0000000000000022` | `0x000000000000000e` | L1: sec0=2ways(32KB), sec1=2ways(32KB). L2: sec0=14ways |
| After pragma fn | `0x0000000000000000` | `0x000000000000000e` | L1: reset to default (all 4 ways shared) |
| Manual write `0x22` | `0x0000000000000022` | — | Readback matches — full R/W confirmed |
| Manual reset `0x00` | `0x0000000000000000` | — | Reset confirmed |

#### 3.6.4 Performance Verification (Fugaku Measurements)

**Test: L1 way-conflict with pointer-chase latency measurement**

Setup:
- Keep array: 32KB (2 ways worth), accessed via pointer-chase (128 dependent loads per rep)
- Evict array: 96KB (6 ways worth), streamed sequentially to create eviction pressure
- Both arrays 2MB-aligned and aliasing to the same 64 L1 cache sets
- Sector config: sec0=2ways(32KB) for keep, sec1=2ways(32KB) for evict
- 500 iterations, 4 chase reps per step

Pattern: prime keep → stream evict (pressure) → reload keep (measure latency)

| Metric | No sector cache | With sector cache | Speedup |
|--------|:-:|:-:|:-:|
| Step 3 reload latency | **17.2 cyc/load** (L2) | **5.9 cyc/load** (L1) | **2.91x** |
| Total time | 7.55 ms | 6.27 ms | **1.20x** |

**Interpretation:** Without sector cache, the 96KB evict stream overflows the 4-way L1 (64KB) and evicts the keep data — reload hits L2 at ~17 cycles/load. With sector cache, keep data is pinned in sector 0 (2 ways) while evict data cycles within sector 1 (2 ways) — reload hits L1 at ~6 cycles/load.

This confirms L1 sector cache partitioning is **fully functional** on Fugaku when enabled via `FLIB_SCCR_CNTL=TRUE`.

#### 3.6.5 PMU Event Verification (fapp)

fapp profiling with raw PMU events confirms the sector cache effect at the hardware counter level.

**L1/L2 Cache Events** (`-Hevent_raw=0x0011,0x0003,0x0004,0x0016,0x0017,0x0015,0x0008,0x0049`):

| PMU Event | nohint | sector | Delta |
|-----------|-------:|-------:|-------|
| CPU_CYCLES (0x0011) | 29,827,760 | 25,233,463 | **-15.4%** |
| L1D_CACHE_REFILL (0x0003) | 520,135 | 417,821 | **-19.7%** (102K fewer misses) |
| L1D_CACHE (0x0004) | 13,691,372 | 13,604,279 | ~same |
| L1D miss rate | 3.80% | 3.07% | **-0.73 pp** |
| L2D_CACHE (0x0016) | 553,350 | 458,984 | -17.1% |
| L1D_CACHE_WB (0x0015) | 2,292 | 11,413 | +4.98x (sector eviction WBs) |

**Sector Cache Tag Events** (`-Hevent_raw=0x0011,0x0240,0x0241,0x02a0,0x02a1,0x0250,0x0252,0x0260`):

| PMU Event | nohint | sector | Interpretation |
|-----------|-------:|-------:|----------------|
| TAG_ADRS (0x0240+0x0241) | 17,411,207 | 16,576,198 | All tagged load ops |
| **NOT_SEC0 (0x02a0+0x02a1)** | **0** | **12,427,729** | **Evict loads → sector 1** |
| NOT_SEC0 ratio | 0% | **75%** | Tag bit 56 discriminating correctly |
| SCE (0x0250+0x0252) | 13,811,809 | 13,592,195 | Always enabled (background) |

**Key PMU insights:**
- `NOT_SEC0 = 0` in nohint confirms no sector tags applied at baseline
- `NOT_SEC0 = 12.4M` in sector confirms `orr Xn, Xbase, #(1<<56)` correctly routes evict loads to sector 1
- 102K fewer L1D refills × ~40 cycle L2 latency ≈ 4M cycles saved, matching the observed 4.6M cycle reduction
- L1D_CACHE_WB increases with sector cache because sector 1 (2 ways) evicts more frequently as 6 ways of data cycles through it

#### 3.6.6 L1 Way Partition Sizes (Fugaku Measurements)

The SCCR L1 register accepts values 0–7 per sector, but the hardware has 4 physical ways. Testing all combinations reveals the effective behavior:

**16KB keep data (1 way worth), 96KB evict stream:**

| Config | sec0 | sec1 | Reload cyc/load | Speedup vs nohint |
|--------|:----:|:----:|:---:|:---:|
| nohint | — | — | 14.5 (L2) | 1.00x |
| sector | 1 | 1 | 4.1 (L1) | **3.54x** |
| sector | 2 | 2 | 4.0 (L1) | **3.58x** |
| sector | 3 | 1 | 4.1 (L1) | **3.58x** |
| sector | **4** | **4** | **14.5 (L2)** | **1.00x** (no partition!) |
| sector | **4** | **0** | **14.6 (L2)** | **1.00x** (no partition!) |

**32KB keep data (2 ways worth):**

| Config | sec0 | sec1 | Reload cyc/load | Speedup |
|--------|:----:|:----:|:---:|:---:|
| nohint | — | — | 15.6 (L2) | 1.00x |
| sector | 2 | 2 | 5.8 (L1) | **2.68x** |
| sector | **3** | **1** | **5.1 (L1)** | **3.08x** (best) |
| sector | 3 | 3 | 10.5 (mix) | 1.49x (oversubscribed) |
| sector | 4 | 4 | 15.6 (L2) | 1.00x |

**48KB keep data (3 ways worth):**

| Config | sec0 | sec1 | Reload cyc/load | Speedup |
|--------|:----:|:----:|:---:|:---:|
| nohint | — | — | 15.8 (L2) | 1.00x |
| sector | **3** | **1** | **5.7 (L1)** | **2.80x** (best) |
| sector | 3 | 3 | 13.0 (mix) | 1.22x (oversubscribed) |
| sector | 4 | 4 | 15.9 (L2) | 1.00x |

**Conclusions:**
- **sec_max ≥ 4 disables partitioning** for that sector (value means "all ways available")
- **Effective range is 0–3** on 4-way L1D (0=sector disabled, 1–3=partition active)
- **Best strategy:** give the keep sector exactly the ways it needs, minimize the evict sector (sec1=1 is optimal for streaming)
- **Oversubscription (sum > 4)** works but weakens isolation — sectors compete for physical ways

#### 3.6.7 L2 Way Partition Verification (Fugaku Measurements)

**Critical: `FLIB_L2_SCCR_CNTL_EX=TRUE` is required for L2 sector cache.**

Without this variable, the runtime sets L2 SCCR to `0x0e` (sec0=14, sec1=0) — no L2 partition regardless of `L2=N` pragma value. With it, the pragma correctly programs the L2 SCCR (e.g., `L2=5` → `0x509` = sec0=9, sec1=5).

Required environment:
```bash
export FLIB_SCCR_CNTL=TRUE
export FLIB_L1_SCCR_CNTL=TRUE
export FLIB_L2_SCCR_CNTL_EX=TRUE    # ← Required for L2 partition!
```

**L2 specs:** 8MB, 16-way set-associative, 256B line. 1 way = 512KB. 14 ways available to application (2 reserved).

**Test: L2 way-conflict with pointer-chase latency measurement**

Setup: same 3-step pattern as L1 test (prime → evict → reload) but with MB-scale data to stress L2 instead of L1.

**1MB keep (2 ways), 7MB evict (14 ways):**

| Config | sec0 | sec1 | Reload cyc/load | Speedup |
|--------|:----:|:----:|:---:|:---:|
| nohint | — | — | 165 (Memory) | 1.00x |
| sector L2=5 | 9 | 5 | 51 (L2) | **3.25x** |
| manual | 14 | 1 | 51 (L2) | **3.25x** |
| manual | 7 | 7 | 51 (L2) | **3.25x** |
| manual | 4 | 10 | 53 (L2) | **3.11x** |
| manual | 14 | 14 | 108 (Memory) | 1.53x (oversubscribed) |
| manual | 0 | 0 | 117 (Memory) | 1.41x |

**2MB keep (4 ways), 6MB evict (12 ways):**

| Config | sec0 | sec1 | Reload cyc/load | Speedup |
|--------|:----:|:----:|:---:|:---:|
| nohint | — | — | 138 (Memory) | 1.00x |
| sector L2=5 | 9 | 5 | 53 (L2) | **2.61x** |
| sector L2=3 | 11 | 3 | 51 (L2) | **2.73x** |
| manual | 14 | 1 | 51 (L2) | **2.73x** |
| manual | 7 | 7 | 51 (L2) | **2.72x** |
| manual | 14 | 14 | 71 (Memory) | 1.96x (oversubscribed) |

**3.5MB keep (7 ways), 5MB evict (10 ways):**

| Config | sec0 | sec1 | Reload cyc/load | Speedup |
|--------|:----:|:----:|:---:|:---:|
| nohint | — | — | 166 (Memory) | 1.00x |
| sector L2=5 | 9 | 5 | 51 (L2) | **3.23x** |
| manual | 14 | 1 | 51 (L2) | **3.25x** |
| manual | 10 | 4 | 51 (L2) | **3.26x** |
| manual | 7 | 7 | 101 (Memory) | 1.64x (7 ways < 7 ways needed) |
| manual | **4** | **10** | **280 (Memory)** | **0.59x (catastrophic!)** |

**Conclusions:**
- **L2 sector cache is fully functional** on Fugaku when `FLIB_L2_SCCR_CNTL_EX=TRUE` is set
- **Effective range:** sec0 and sec1 values 0–14 (14 ways available). Values ≥14 or ≥16 mean "all ways"
- **sec0 must be ≥ keep_ways** to fully protect keep data in L2
- **sec1=1 is sufficient** for streaming — same finding as L1
- **Oversubscription (sec0+sec1 > 14)** weakens isolation
- **Undersized sec0** is catastrophic: sec0=4 with 3.5MB keep (needs 7 ways) = 0.59x (thrashing, worse than no partition)
- **~51 cyc/load = L2 hit** (37-40 cycle L2 latency + pointer-chase overhead)
- **~165 cyc/load = Memory** (~100-200 cycle memory latency)

---

## Part III-B: Hardware Barrier

### 3B.1 Overview

The A64FX hardware barrier provides low-latency inter-thread synchronization via a dedicated system register, replacing software spin-wait barriers. The Fujitsu OpenMP runtime (`-Nfjomplib`) uses this by default for `#pragma omp barrier`.

### 3B.2 Register Architecture

The hardware barrier has **two layers**: EL1 control-plane registers for configuration, and an EL0 data-plane register for the actual barrier trigger.

**EL1 Control Plane** (kernel-only, configure barrier groups at boot):

| Register | Encoding | Purpose |
|----------|----------|---------|
| `IMP_BARRIER_CTRL_EL1` | `S3_0_C11_C12_0` | Master enable, access control |
| `IMP_BARRIER_BST_BIT_EL1` | `S3_0_C11_C12_4` | BST base/mask configuration |
| `IMP_BARRIER_INIT_SYNC_BB0` | `S3_0_C11_C12_5` | Init sync for barrier group 0 |
| `IMP_BARRIER_INIT_SYNC_BB1` | `S3_0_C11_C12_6` | Init sync for barrier group 1 |
| `IMP_BARRIER_INIT_SYNC_BB2` | `S3_0_C11_C12_7` | Init sync for barrier group 2 |
| `IMP_BARRIER_ASSIGN_EL1` | `S3_0_C11_C15_0` | Barrier group assignment |

All EL1 registers use op1=0 and are inaccessible from EL0 (trap with SIGILL).

**EL0 Data Plane** (user-space, actual barrier synchronization):

| Register | Encoding | Purpose |
|----------|----------|---------|
| **IMP_BARRIER_BST_EL0** | **`S3_3_C15_C15_0`** | Barrier Sync Trigger (toggle-and-wait) |

This register is in the ARMv8 **implementation-defined** space (CRn=15, CRm=15) with op1=3, making it directly accessible from EL0 without kernel mediation. On Fugaku, it is **always accessible** — the kernel configures the barrier hardware at boot via the EL1 registers, and user-space applications simply use the EL0 trigger.

**Note:** The EL0 BST encoding (`S3_3_C15_C15_0`) is completely different from the EL1 BST configuration register (`S3_0_C11_C12_4`). They are separate registers serving different roles — the EL1 register sets up barrier parameters, the EL0 register triggers synchronization.

### 3B.3 Mechanism

The hardware barrier uses a **toggle-and-poll** protocol on bit 0 of `IMP_BARRIER_BST_EL0`:

```asm
; Each participating thread executes:
mrs  x0, S3_3_C15_C15_0     ; 1. Read current BST value
mvn  x0, x0                  ; 2. Toggle
and  x0, x0, #1              ; 3. Keep only bit 0
msr  S3_3_C15_C15_0, x0     ; 4. Write toggled value (trigger)
dsb  sy                      ; 5. Full data synchronization barrier
sevl                         ; 6. Send event local (wake WFE loops)
.poll:
mrs  x1, S3_3_C15_C15_0     ; 7. Read BST again
and  x1, x1, #1
cmp  x0, x1                 ; 8. All threads synced?
b.eq .done                  ;    Yes → barrier complete
wfe                          ; 9. Wait for event (low-power)
b    .poll                   ; 10. Poll again
```

The hardware coalesces toggles from all participating threads. When all threads have toggled, the BST value matches what each thread wrote, and the poll loop exits. The `wfe`/`sevl` mechanism avoids busy-spinning.

### 3B.4 Inline Assembly Example

```c
#include <stdint.h>

static inline uint64_t read_bst(void) {
    uint64_t v;
    asm volatile("mrs %0, S3_3_C15_C15_0" : "=r"(v));
    return v;
}

static inline void write_bst(uint64_t v) {
    asm volatile("msr S3_3_C15_C15_0, %0" :: "r"(v));
}

// Single hardware barrier (all threads must call)
static inline void hw_barrier(void) {
    uint64_t val = (~read_bst()) & 1;
    write_bst(val);
    asm volatile("dsb sy" ::: "memory");
    asm volatile("sevl");
    uint64_t cur;
    do {
        asm volatile("wfe");
        cur = read_bst() & 1;
    } while (cur != val);
}
```

### 3B.5 Fujitsu OpenMP Integration

The Fujitsu OpenMP runtime (`libfj90i.so.1`, linked via `-Nfjomplib`) uses the hardware barrier through `__mpc_obar`:

**Call chain:**
```
#pragma omp barrier
  → __mpc_obar (compiler-generated call)
    → libfj90i.so.1:__mpc_obar (runtime)
      → checks FLIB_BARRIER flag at config struct offset [5]
        → if HARD: inline BST toggle + WFE poll (S3_3_C15_C15_0)
        → if SOFT: __jwe_pbar_tree2 / tree3 / cascade / nrnw (memory-based tree)
```

**Environment variable:**
```bash
export FLIB_BARRIER=HARD   # Use hardware barrier (default with -Nfjomplib)
export FLIB_BARRIER=SOFT   # Use software tree-based barrier
```

**Build:**
```bash
fcc -Nnoclang -O2 -Kopenmp -Nfjomplib -o program program.c
```

**Verification:** The BST register (`S3_3_C15_C15_0`) is accessed 55 times within `libfj90i.so.1`, in functions:
- `__mpc_obar` (inline HW barrier path)
- `__jwe_thrbar_sync` (thread barrier sync)
- `__jwe_thrbar_sync_release` (barrier release)

The library also accesses sector cache registers (`S3_3_C11_C8_2`, `S3_3_C15_C8_2`) but guards those with a runtime accessibility flag to handle SIGILL gracefully on systems where `el0ae=0`.

### 3B.6 Performance (Fugaku Measurements)

**12 threads (1 CMG, intra-CMG):**

| Mode | ns/barrier | Speedup |
|------|-----------|---------|
| HARD (HW barrier) | 120.1 | 1.00x |
| SOFT (SW tree) | 120.5 | 1.00x |

**48 threads (4 CMGs, cross-CMG):**

| Mode | ns/barrier | Speedup |
|------|-----------|---------|
| HARD (HW barrier) | 1,372 | **1.28x** |
| SOFT (SW tree) | 1,761 | 1.00x |

Within a single CMG, hardware and software barriers are equally fast (~120 ns). The hardware barrier advantage appears at cross-CMG synchronization (48 threads), where the dedicated synchronization hardware avoids cache coherence traffic between CMGs.

For comparison, `#pragma omp critical` costs 15,983–272,480 ns (133–210x slower than barrier).

### 3B.7 Fugaku EL0 Access Summary

All register probing results on Fugaku (Fugaku kernel, EL0):

| Register | Encoding | Accessible | Notes |
|----------|----------|:----------:|-------|
| IMP_BARRIER_CTRL_EL1 | `S3_0_C11_C12_0` | SIGILL | EL1 only |
| IMP_BARRIER_BST_BIT_EL1 | `S3_0_C11_C12_4` | SIGILL | EL1 only |
| IMP_BARRIER_ASSIGN_EL1 | `S3_0_C11_C15_0` | SIGILL | EL1 only |
| **IMP_BARRIER_BST_EL0** | **`S3_3_C15_C15_0`** | **R/W** | **Always accessible** |
| IMP_SCCR_CTRL_EL1 | `S3_0_C11_C8_0` | SIGILL | EL1 only |
| IMP_SCCR_L1_EL0 | `S3_3_C11_C8_2` | SIGILL → **R/W** | Requires `xos_sclib_init` (via `FLIB_SCCR_CNTL=TRUE` or `libsec.so`) |
| IMP_SCCR_VSCCR_L2_EL0 | `S3_3_C15_C8_2` | SIGILL → **R/W** | Requires `xos_sclib_init` (same as L1) |
| IMP_PF_CTRL_EL1 | `S3_0_C11_C4_0` | SIGILL | EL1 only |
| **IMP_PF_STREAM_DETECT_CTRL_EL0** | **`S3_3_C11_C4_0`** | **R/W** | **Always accessible** |
| IMP_FJ_TAG_ADDRESS_CTRL_EL1 | `S3_0_C11_C2_0` | SIGILL | EL1 only |

2 registers are always accessible from EL0: the HW barrier BST trigger and the HW prefetch stream detect control. 2 more (SCCR L1 and L2) become accessible after `xos_sclib_init` enables `el0ae=1` via the `/dev/xos_sec_normal` kernel driver.

---

## Part III-C: Hardware Prefetch Stream Detect

### 3C.1 IMP_PF_STREAM_DETECT_CTRL_EL0

The hardware prefetch stream detect control register is one of only two EL0-accessible HPC extension registers on Fugaku. It controls the hardware prefetch engine's stream detection behavior.

**Encoding:** `S3_3_C11_C4_0` (op0=3, op1=3, CRn=11, CRm=4, op2=0)

**Default value:** `0x0000000000000000` (all fields zero on Fugaku)

### 3C.2 Bit Field Layout (Empirically Determined)

Valid bit mask: `0x8CC000000F0F0000` (13 writable bits across 5 fields)

```
63       59 58    55 54                    27  24    19  16
┌──┬──────┬────┬────┬────────────────────┬──────┬────┬──────┬────────────────┐
│EN│ RES0 │CfgA│RES0│CfgB│     RES0     │ThA   │RES0│ThB   │     RES0      │
└──┴──────┴────┴────┴────┴──────────────┘└──────┘────└──────┘────────────────┘
```

| Field | Bits | Width | Values | Description (inferred) |
|-------|------|-------|--------|------------------------|
| EN | [63] | 1 bit | 0–1 | Global enable/disable |
| CfgA | [59:58] | 2 bits | 0–3 | Stream detect configuration A |
| CfgB | [55:54] | 2 bits | 0–3 | Stream detect configuration B |
| ThA | [27:24] | 4 bits | 0–15 | Stream detect threshold/count A |
| ThB | [19:16] | 4 bits | 0–15 | Stream detect threshold/count B |

All fields accept all possible values (readback matches write for every valid value).

### 3C.3 Inline Assembly Example

```c
static inline uint64_t read_pf_stream_detect(void) {
    uint64_t v;
    asm volatile("mrs %0, S3_3_C11_C4_0" : "=r"(v));
    return v;
}

static inline void write_pf_stream_detect(uint64_t v) {
    asm volatile("msr S3_3_C11_C4_0, %0" :: "r"(v));
    asm volatile("isb" ::: "memory");
}
```

### 3C.4 Performance Impact

Streaming benchmark (4 MB SVE ld1w sum × 50 iterations) showed **no measurable bandwidth difference** across any field value. The register likely controls stream detection thresholds that don't affect simple sequential scan patterns (which are already perfectly detected by default). Impact may be visible on irregular or multi-stream access patterns.

---

## Part IV: Large Page (HugeTLB) Feature Summary

### 4.1 Background — Memory Address Translation and TLB

When applications access memory, virtual addresses must be translated to physical addresses via the page table. The CPU caches recent translations in the **TLB (Translation Look-aside Buffer)**:

1. Load/store unit receives a virtual address
2. TLB lookup → **hit**: physical address returned immediately
3. TLB lookup → **miss**: expensive page table walk in main memory

For HPC applications with large data, TLB misses become a major bottleneck. Large pages reduce this by covering more memory per TLB entry.

### 4.2 Page Sizes

| Page Type | Size | Use Case |
|-----------|------|----------|
| Normal page | 64 KiB | Default OS page size |
| Large page | 2 MiB | Default for HPC (via libmpg) |
| McKernel large page | 32 MiB | McKernel mode only |
| McKernel extended | 1 GiB, 16 GiB | McKernel-specific |

**Trade-offs:**

| Metric | 64 KiB (Normal) | 2 MiB (Large) |
|--------|:---------------:|:--------------:|
| TLB miss rate | High | **Low** |
| Memory init cost | **Small** | Large |
| Memory usage efficiency | **High** | Low |

### 4.3 Memory Regions and Large Page Coverage

`libmpg.so` selectively applies 2 MiB pages:

| Memory Region | Large Page | Page Size | Notes |
|---------------|:----------:|-----------|-------|
| .text (code) | ✗ | 64 KiB | Always normal pages |
| .data (static, initialized) | ✓ | 2 MiB | Always prepaged |
| .bss (static, uninitialized) | ✓ | 2 MiB | Paging policy configurable |
| Heap (brk/sbrk) | ✗ | 64 KiB | Kernel limitation |
| mmap region (malloc) | ✓ | 2 MiB | Primary path with libmpg |
| Thread heap | ✓ | 2 MiB | Sub-thread heaps |
| Process/main thread stack | ✓ | 2 MiB | Requires `LPG_MODE=base+stack` |
| Thread stack | ✓ | 2 MiB | Requires `LPG_MODE=base+stack` |
| Shared memory | ✗ | 64 KiB | Inter-process shared |

**Key:** When libmpg is linked, `malloc(3)` goes through `mmap(2)` (large-paged), not `brk/sbrk` (normal-paged). Do **not** mix `brk/sbrk` with libmpg.

### 4.4 Variable Placement by Language

**Fortran:**
```fortran
real*8 a(N)                ! → .bss  (or stack with -Kauto/-Kthreadsafe)
real*8 :: b(N) = 1.0       ! → .data
allocatable :: c(:)
allocate(c(N))             ! → dynamic memory (mmap)
```

**C/C++:**
```c
double a[N];               // Global uninitialized → .bss
double b[N] = {1.0};       // Global initialized → .data
double *c = malloc(...);   // → dynamic memory (mmap)
double e[N];               // Local in main → process stack
// In thread function:
double f[N];               // → thread stack
```

**C++ (std::vector, new):** All dynamic → mmap region.

---

## Part V: Compilation and Linking

### Fujitsu Compiler
```bash
fcc -Klargepage program.c        # Enable large pages (default)
fcc -Knolargepage program.c      # Disable large pages
```

### GCC / General Compilers
```bash
gcc -Wl,-T/opt/FJSVxos/mmm/util/bss-2mb.lds \
    -L/opt/FJSVxos/mmm/lib64 \
    -lmpg -lc -lpthread \
    -no-pie \
    test_program.c
```

**Critical rules:**
1. **Link order:** `-lmpg` must precede `-lc` and `-lpthread`.
2. **PIE incompatible:** `.data/.bss` not large-paged if PIE. Use `-no-pie`. Verify: `readelf -h a.out` → `e_type` must be `ET_EXEC`.
3. **Linker script required:** `bss-2mb.lds` aligns `.data/.bss` to 2 MiB boundaries.

---

## Part VI: Environment Variables — Complete Reference

### 6.1 Basic Settings

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `XOS_MMM_L_HPAGE_TYPE` | `hugetlbfs` \| `none` | `hugetlbfs` | Master enable/disable. `none` disables all XOS_MMM_L_ variables. In McKernel+`none`, uses extended THP mode. |
| `XOS_MMM_L_LPG_MODE` | `base+stack` \| `base` | `base+stack` | `base` = only .data/.bss and mmap. `base+stack` adds stack/thread-stack. |
| `XOS_MMM_L_HUGETLB_FALLBACK` | `0` \| `1` | `0` | `1` = fall back to normal pages on failure (vs OOM kill). Only for mmap-region malloc. Requires: HPAGE_TYPE=hugetlbfs, PAGING_POLICY=\*:\*:prepage, ARENA_LOCK_TYPE=1, MAX_ARENA_NUM=1. |
| `XOS_MMM_L_PRINT_ENV` | `on`/`1` \| `off`/`0` | `0` | Print all tuning variables to stderr before main(). |
| `XOS_MMM_L_HUGETLB_SZ` | `2M` \| `32M` | `2M` | **McKernel only.** 2 MiB or 32 MiB page size. |

### 6.2 Paging Policy

| Variable | Format | Default |
|----------|--------|---------|
| `XOS_MMM_L_PAGING_POLICY` | `<bss>:<stack>:<mmap>` | `prepage:demand:prepage` |

Each field is `demand` or `prepage`. The `.data` region is **always prepaged** regardless.

**Paging modes:**

| Mode | Behavior | Pros | Cons |
|------|----------|------|------|
| **Prepage** | Physical pages allocated upfront at region creation | Fewer subsequent page faults, stable performance | Data lands on CMG0, cross-NUMA traffic |
| **Demand** | Physical pages allocated on first access | NUMA-local allocation, pages on accessing CMG | Initial page faults, slight variability |

### 6.3 Tuning Settings (libmpg-specific)

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `XOS_MMM_L_ARENA_FREE` | `1` \| `2` | `1` | **1**: freed memory returned to OS immediately. **2**: all memory pooled/reused (never freed). `2` implies LOCK_TYPE=1, MAX_ARENA=1, MMAP_THRESHOLD=ULONG_MAX, TRIM_THRESHOLD=ULONG_MAX. |
| `XOS_MMM_L_ARENA_LOCK_TYPE` | `0` \| `1` | `1` | **0**: allocation performance priority — thread heaps created on contention (parallel malloc). **1**: memory efficiency priority — serialized through shared arena. |
| `XOS_MMM_L_MAX_ARENA_NUM` | 1–INT_MAX | `1` | Max arenas (process heap + thread heaps). Only with LOCK_TYPE=1. Default `1` = no thread heaps. ≥2 allows thread heap creation. |
| `XOS_MMM_L_HEAP_SIZE_MB` | ≥2×MMAP_THRESHOLD | 2×MMAP_THRESHOLD | Thread heap allocation/expansion size (MiB). Reduce to save memory. |
| `XOS_MMM_L_COLORING` | `0` \| `1` | `1` | Cache coloring for mmap allocations ≥ MMAP_THRESHOLD. Reduces L1 conflicts. Disable if app implements own coloring. |
| `XOS_MMM_L_FORCE_MMAP_THRESHOLD` | `0` \| `1` | `0` | **0**: search heap free-list first for large allocs. **1**: always use mmap (guarantees cache coloring). |

### 6.4 glibc Settings (used by libmpg)

| Variable | Default | Description |
|----------|---------|-------------|
| `MALLOC_MMAP_THRESHOLD_` | 134217728 (128 MiB) | Allocations ≥ threshold use mmap; smaller use heap. |
| `MALLOC_TRIM_THRESHOLD_` | 134217728 (128 MiB) | Contiguous free space ≥ threshold at heap top triggers OS release. |
| `MALLOC_TOP_PAD_` | 131072 (128 KiB) | Heap growth increment. Rounded up to page size. |
| `MALLOC_MMAP_MAX_` | 2097152 | Max outstanding mmap allocations. 0 = force heap only. |
| `MALLOC_CHECK_` | 3 | Debug: 0=ignore, 1=print, 2=abort, 3=print+trace+abort, 5=brief, 7=brief+trace+abort. |
| `MALLOC_PERTURB_` | 0 | Debug: fills alloc'd memory (complement of low byte) and freed memory (low byte). |

---

## Part VII: Tuning Scenarios and Benchmarks

### 7.1 TLB Miss Reduction — Stream Triad

**Config:** 1 thread, n=83,880,960

| Metric | Normal (64 KiB) | Large (2 MiB) | Improvement |
|--------|:---:|:---:|:---:|
| Execution time (s) | 0.435 | 0.347 | 1.25× |
| Throughput (GB/s) | 61.76 | 77.29 | **+25%** |
| L1D TLB miss rate | 0.098% | 0.003% | **30× lower** |
| L2D TLB miss rate | 0.098% | 0.0003% | **296× lower** |

### 7.2 PAGING_POLICY — Multi-CMG NUMA Locality

**Config:** Stream kernel, 48 threads across 4 CMGs, ~1 GB data.

| Paging Policy | Memory Throughput | Ratio |
|:---:|:---:|:---:|
| `prepage` (default for mmap) | 93 GB/s | 1× |
| `demand` | 804 GB/s | **8.6×** |

**Root cause:** Prepaging allocates all data on CMG0 at startup. Other CMGs suffer remote NUMA latency. Demand paging places pages on the CMG that first touches them.

**Compiler options used:** `-Kfast,openmp -Kprefetch_sequential=soft -Kprefetch_line=9 -Kprefetch_line_L2=70 -Kzfill=18`

### 7.3 ARENA_FREE — malloc/free Cycle Optimization

**Config:** 1024 × 8 MiB malloc, then free all, loop twice.

| Operation | FREE=1 (default) | FREE=2 (pool) | Speedup |
|-----------|:---:|:---:|:---:|
| 1st malloc | 0.501 s | 0.511 s | ~1× |
| 1st free | 0.260 s | 0.0005 s | **524×** |
| 2nd malloc | 0.501 s | 0.0003 s | **1,033×** |
| 2nd free | 0.260 s | 0.0003 s | **1,625×** |

**Trade-off:** Memory never returned to OS — peak consumption persists.

### 7.4 ARENA_LOCK_TYPE — Parallel malloc

**Config:** 16 threads × malloc(64 KiB) × 20,480 times, 10 iterations.

| Setting | Time | Speedup |
|:---:|:---:|:---:|
| LOCK_TYPE=1 (default, serialized) | 0.56 s | 1× |
| LOCK_TYPE=0 (parallel) | 0.35 s | **1.60×** |

### 7.5 MALLOC_MMAP_THRESHOLD_ — Cache Thrashing Fix

**Problem:** Multiple identically-sized dynamic arrays align to the same L1 cache sets (e.g., 256×256×8B = 512 KiB = 32×16 KiB boundary). All streams conflict → **L1D cache thrashing**.

**Fortran example:** 8 arrays of 256×256 doubles, 12 threads.

| Metric | Before | After (THRESHOLD=204800) | Change |
|--------|:---:|:---:|:---:|
| L1D miss rate | 0.26 | 0.13 | **2× lower** |
| L1D demand miss % | 51.52% | 9.47% | **5.4× lower** |

**Mechanism:** Changing the mmap threshold shifts allocation base addresses, breaking power-of-2 alignment and eliminating cache set conflicts — **implicit padding without code changes**.

---

## Part VIII: NUMA Architecture

### FX Server NUMA Layout

A64FX has 4 CMGs (Core Memory Groups). In FX server mode, the OS creates a split configuration:

| NUMA Nodes | Cores | Purpose |
|:---:|:---:|---|
| #0–3 | 0–11 | System (OS) — unavailable to jobs |
| #4–7 | 12–59 | Job (application) — 12 cores per CMG |

**User-accessible CPU numbers start at 12.** For manual affinity:
```c
cpuid[i] = i + 12;  // Logical core 0 = OS core 12
```

Use `sched_getaffinity(2)` for portable CPU discovery.

---

## Part IX: Startup Behavior and Caveats

### 9.1 Startup Memory Overhead

At startup, libmpg temporarily uses **2× combined .data + .bss size** for remapping to large pages. Applications with very large static data may be **SIGKILL'd (OOM)** before `main()`.

**Workaround:**
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage
# Or via job submission:
pjsub -x XOS_MMM_L_PAGING_POLICY=demand:demand:prepage jobscript.sh
```

Check static data size: `readelf -S a.out | grep -E '\.data|\.bss'`

### 9.2 brk/sbrk Incompatibility

brk(2)/sbrk(2) memory is never large-paged. Mixing brk/sbrk with libmpg malloc can cause ENOMEM due to normal/large page coexistence. Use `malloc(3)` exclusively with libmpg.

### 9.3 Signal Handler + Prepaging Conflict

When using `timer_create(2)` with frequent SIGALRM/SIGVTALRM, `fork(2)`/`clone(2)` may loop indefinitely due to ERESTARTNOINTR during prepaging. Fix:
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
```

### 9.4 Stack Alignment Caveats

Large-paged stack sizes are aligned up to HugeTLBfs page boundaries. If the aligned stack overlaps adjacent VMAs, large page mapping is skipped (warning emitted, execution continues with normal pages).

---

## Part X: Debugging and Diagnostics

### 10.1 Print All Settings
```bash
export XOS_MMM_L_PRINT_ENV=on
# → all XOS_MMM_L_ and MALLOC_ values printed to stderr before main()
```

### 10.2 /proc Inspection

```
# Normal heap:
00430000-00890000 rw-p 00000000 00:00 0  [heap]
# Large-paged:
aaaae2400000-aaaae2600000 rw-p 00000000 00:0e 452989  /anon_hugepage (deleted)

# Normal stack:
fffffffd0000-1000000000000 rw-p 00000000 00:00 0  [stack]
# Large-paged stack:
fffaeb800000-1000000000000 rw-p 00000000 00:0e 274404  /memfd: [stack] by libmpg (deleted)
```

### 10.3 Valgrind Compatibility

| Tool | With libmpg | Notes |
|------|:-----------:|-------|
| memcheck | ✓ | |
| cachegrind | ✓ | |
| callgrind | ✓ | |
| helgrind | **✗** | Conflicts with malloc hooks. May OOM due to unbounded RLIMIT_DATA/RLIMIT_STACK. Set `pjsub --proc-data=<limit> --proc-stack=<limit>`. |

### 10.4 malloc_stats(3)

glibc's `malloc_stats()` prints per-arena statistics. Requires adding the call to source code.

### 10.5 Job Statistics (HUGETLB_FALLBACK)

```ini
# papjmstats.conf:
Item {
    ItemName=hugetlb_fallback
    ItemNameDisp=HUGETLB_FALLBACK_LPG
    RecordNameList=JN
    DataType=PJMX_DATATYPE_UINT8
    DispFormat=dec
}
```
Output: `HUGETLB_FALLBACK_LPG : 0` (not triggered) or `1` (fallback occurred).

---

## Part XI: Warning Messages Reference

| Code | Meaning | Action |
|------|---------|--------|
| **FHE 1113** | Tag override enabled (core mask shown) | Informational |
| **FHE 1114** | Tag override disabled (core mask shown) | Informational |
| **LPG 2002** | PIE binary — .data/.bss not large-paged | Recompile with `-no-pie` |
| **LPG 2003** | Segment overlap — .data/.bss not large-paged | Use proper linker script |
| **LPG 2004** | Custom stack address in pthread_attr — thread stack not large-paged | Don't set stack address in attr |
| **LPG 2005** | Multi-thread detected during process stack remap | Ensure single-thread at init |
| **LPG 2006** | Process stack bottom overlaps next VMA | Avoid mmap near stack bottom |
| **LPG 2007** | Process stack top overlaps previous VMA | Avoid mmap near stack top |
| **LPG 9999** | Fallback to normal pages occurred | Informational (FALLBACK=1) |

---

## Part XII: Performance Tuning Decision Tree

### Step 1: Is TLB a bottleneck?
Check PA reports for high **mDTLB miss rate** (thrashing) or **uDTLB miss rate** (capacity). Ensure large pages are active (no PIE warnings, proper linking).

### Step 2: Multi-CMG performance far below expectations?
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
```

### Step 3: malloc/free overhead in parallel regions?
```bash
export XOS_MMM_L_ARENA_LOCK_TYPE=0
export XOS_MMM_L_ARENA_FREE=2
```

### Step 4: L1D cache thrashing with identically-sized arrays?
```bash
export MALLOC_MMAP_THRESHOLD_=204800
```

### Step 5: Startup OOM with large static data?
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage
```

### Step 6: Signal handler loops with prepaging?
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
```

---

## Part XIII: Recommended Configurations

### Default (single CMG, simple workload)
```bash
# All defaults — no changes needed
# Large pages enabled, prepage:demand:prepage
```

### Multi-CMG HPC (most common for Fugaku)
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage
```

### Memory-intensive with repeated alloc/free
```bash
export XOS_MMM_L_ARENA_FREE=2
export XOS_MMM_L_ARENA_LOCK_TYPE=0
```

### Maximum memory safety
```bash
export XOS_MMM_L_HUGETLB_FALLBACK=1
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage
```

### Debugging
```bash
export XOS_MMM_L_PRINT_ENV=on
export MALLOC_CHECK_=3
export MALLOC_PERTURB_=0xAA
```

### Sector cache tuning (Fujitsu compiler)
```bash
# In job script:
export FLIB_SCCR_CNTL=TRUE
export FLIB_L1_SCCR_CNTL=TRUE
```
```c
// In source (FCC pragmas, requires -Nnoclang -Kocl,hpctag):
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign stream_ptr
```

### Sector cache tuning (manual, after libsec init)
```c
// After xos_sclib_init() or FLIB_SCCR_CNTL=TRUE enables el0ae:
// Configure L1D: sector0=2ways(32KB), sector1=2ways(32KB)
uint64_t l1_val = 2ULL | (2ULL << 4);
asm volatile("msr S3_3_C11_C8_2, %0" :: "r"(l1_val));
asm volatile("isb" ::: "memory");

// Configure L2 via window (depends on assign bit set by runtime):
uint64_t l2_val = (5ULL << 8) | 11ULL;  // sec0=11ways, sec1=5ways
asm volatile("msr S3_3_C15_C8_2, %0" :: "r"(l2_val));
asm volatile("isb" ::: "memory");

// Tag pointer for sector 1 loads:
float *tagged_ptr = (float*)((uintptr_t)ptr | (1ULL << 56));
```
