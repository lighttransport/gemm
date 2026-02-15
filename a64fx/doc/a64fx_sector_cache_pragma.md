# A64FX Sector Cache Pragma Settings

*Extracted from: Fujitsu Technical Computing Suite V4.0L20 — Development Studio C Language User's Guide (J2UL-2560-01Z0(14), March 2025)*

---

## 1. Overview of Sector Cache (§3.5.1)

The A64FX sector cache is a hardware mechanism that logically partitions the cache into **two sectors (Sector 0 and Sector 1)**, preventing frequently reused data from being evicted by streaming (non-reusable) data. This is especially effective during thread-parallel execution when multiple cores share the L2 cache.

### A64FX Cache Specifications

| Cache Level | Total Ways | User-Available Ways | Notes |
|-------------|-----------|---------------------|-------|
| L1 | 4 | 4 | Per-core, private |
| L2 | 16 | 14 | Shared; assistant core permanently occupies 2 ways |

### Prerequisites

- The **`-Khpctag`** compiler option must be enabled (enables HPC tag address override functionality).
- The `-KA64FX` option must also be active for `-Khpctag` to take effect.
- Both options must be specified at **compile time and link time**.

### Important Caveats

- Sector cache does **not** automatically improve performance. If the way count is set without considering the target array size, cache utilization may actually degrade.
- LRU (Least Recently Used) replacement still operates within each sector, so software control of sector cache does not guarantee speedup.
- The recommended approach is to first collect **hardware monitor (HWPC) data**, confirm that L2 cache misses are occurring, then determine the array size to protect and assign an appropriate number of ways to Sector 1.

---

## 2. Pragma Directives (§3.5.2.1)

### 2.1 `scache_isolate_way` — Set Sector 1 Way Count

Controls the maximum number of ways allocated to Sector 1 for L1 and/or L2 cache.

#### Syntax

```c
/* Function-wide scope */
#pragma procedure scache_isolate_way L2=n1 [L1=n2]

/* Block scope */
#pragma statement scache_isolate_way L2=n1 [L1=n2]
/* ... target code ... */
#pragma statement end_scache_isolate_way
```

#### Parameters

| Parameter | Description | Valid Range |
|-----------|-------------|-------------|
| `L2=n1` | Max ways for Sector 1 in L2 cache | 0 ≤ n1 ≤ 14 (i.e., max L2 ways − 2) |
| `L1=n2` | Max ways for Sector 1 in L1 cache (optional) | 0 ≤ n2 ≤ 4 |

If `L1=n2` is omitted, only the L2 Sector 1 way count is controlled.

#### Scope Rules

| Form | Scope |
|------|-------|
| `#pragma procedure scache_isolate_way ...` | Entire function |
| `#pragma statement scache_isolate_way ... / end_scache_isolate_way` | Enclosed block only |

- **Nesting is not allowed.** However, a function with a `procedure`-level directive may contain `statement`-level block directives.

---

### 2.2 `scache_isolate_assign` — Assign Arrays to Sector 1

Specifies which arrays (or pointers to arrays) should be placed in Sector 1.

#### Syntax

```c
/* Function-wide scope */
#pragma procedure scache_isolate_assign array1[,array2]...

/* Block scope */
#pragma statement scache_isolate_assign array1[,array2]...
/* ... target code ... */
#pragma statement end_scache_isolate_assign
```

#### Constraints

- Only arrays of **arithmetic type** are valid targets.
- Pointers to arrays are also accepted.
- The same nesting rules as `scache_isolate_way` apply.

---

## 3. Usage Examples

### 3.1 Pragma-Only Control (§3.5.2.1)

In this example, array `a` is repeatedly reused across iterations of the outer `j` loop, while `b[j][i]` is accessed in a streaming fashion. By isolating `a` into Sector 1 with 10 ways, it is protected from eviction by `b`.

```c
/* Reuse array a in L2 cache with 10 ways reserved in Sector 1 */
#pragma statement scache_isolate_way L2=10
#pragma statement scache_isolate_assign a
for (int j = 0; j < n; j++) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        a[i] = a[i] + b[j][i];
    }
}
#pragma statement end_scache_isolate_assign
#pragma statement end_scache_isolate_way
```

**Cache partitioning in this example:**

| Sector | Ways | Contents |
|--------|------|----------|
| Sector 1 | 10 | Array `a` (reused data) |
| Sector 0 | 4 | Array `b` (streaming data) + everything else |
| Reserved | 2 | Assistant core |

### 3.2 Environment Variable + Pragma Control (§3.5.2.2)

This approach separates the way allocation (environment variable) from the array assignment (pragma), enabling tuning **without recompilation**.

**Step 1: Set environment variables before execution**

```bash
export FLIB_SCCR_CNTL=TRUE
export FLIB_L2_SECTOR_NWAYS_INIT=2,10   # Sector 0 = 2 ways, Sector 1 = 10 ways
```

**Step 2: Specify array assignment in source code**

```c
/* Assign array a to Sector 1 */
#pragma statement scache_isolate_assign a
for (int j = 0; j < n; j++) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        a[i] = a[i] + b[j][i];
    }
}
#pragma statement end_scache_isolate_assign
```

---

## 4. Related Environment Variables (§3.5.2.2)

### 4.1 `FLIB_SCCR_CNTL`

Controls whether sector cache is enabled or disabled.

| Value | Description |
|-------|-------------|
| `TRUE` | Enable sector cache **(default)** |
| `FALSE` | Disable sector cache entirely |

### 4.2 `FLIB_L1_SCCR_CNTL`

Controls L1 sector cache fallback when L2 sector cache is unavailable (i.e., when a NUMA node is not exclusively owned by a single process). Only effective when `FLIB_SCCR_CNTL=TRUE`.

| Value | Description |
|-------|-------------|
| `TRUE` | Use L1 sector cache when L2 sector cache is unavailable **(default)** |
| `FALSE` | Do not use L1 sector cache; outputs runtime message `jwe1047i-w A sector cache couldn't be used.` and disables sector cache control |

### 4.3 `FLIB_L2_SECTOR_NWAYS_INIT`

Sets the initial way count for each L2 sector at program startup. Only effective when `FLIB_SCCR_CNTL=TRUE`.

**Format:** `n0,n1`

| Parameter | Description | Valid Range |
|-----------|-------------|-------------|
| `n0` | Max ways for Sector 0 | 0 ≤ n0 ≤ 14 |
| `n1` | Max ways for Sector 1 | 0 ≤ n1 ≤ 14 |

**Recommended constraint:** `n0 + n1 = 14` (total L2 ways minus 2 reserved for assistant core) to avoid way contention.

---

## 5. Edge Cases and Error Handling (§3.5.2.3)

| Condition | Behavior |
|-----------|----------|
| Value exceeds upper limit | Clamped to the upper limit |
| Value less than 0 | The pragma directive is ignored |
| Value outside `int16_t` range | **Undefined behavior** (not guaranteed) |

---

## 6. Execution Environment Constraints (§I.2, §J.4.4)

### NUMA Node Occupancy and Sector Cache Availability

| Execution Environment | L1 Sector Cache | L2 Sector Cache |
|-----------------------|-----------------|-----------------|
| NUMA node exclusively owned by 1 process | ✅ Available | ✅ Available |
| NUMA node **not** exclusively owned | ✅ Available (if `FLIB_L1_SCCR_CNTL=TRUE`) | ❌ Not available |

### Conditions That Cause Undefined Sector Cache Behavior

When the runtime cannot determine whether a NUMA node is exclusively owned, sector cache behavior becomes **undefined**. This may cause severe performance degradation, diagnostic message `jwe1048i-u` followed by program termination, or abnormal termination. This occurs when:

- Processes are spawned by mechanisms other than Fujitsu MPI (e.g., `fork()`, `system()`)
- Threads are managed outside OpenMP/auto-parallelization (e.g., `pthread_create()`)

To unconditionally disable sector cache regardless of environment, set `FLIB_SCCR_CNTL=FALSE`.

---

## 7. Tuning Guidelines

1. **Profile first.** Collect HWPC data and confirm L2 cache misses before enabling sector cache.

2. **Size your ways correctly.** On A64FX, L2 is 8 MiB / 16 ways = **512 KiB per way**. For example, 10 ways in Sector 1 protects approximately 5 MiB of data.

3. **Satisfy the constraint `n0 + n1 = 14`.** An imbalanced allocation where the sum is less than 14 wastes available ways; exceeding 14 is clamped.

4. **Use environment variables for iterative tuning.** The `FLIB_L2_SECTOR_NWAYS_INIT` variable allows adjusting the way split without recompiling, making it ideal for performance exploration.

5. **Combine with `scache_isolate_assign` for precision.** Simply setting way counts is insufficient — you must also tell the compiler which arrays belong in Sector 1.

6. **Ensure `-Khpctag` is specified at both compile and link time.**

---

## 8. Quick Reference

```c
/* ---- Pragma-only approach ---- */

// Set Sector 1 to 10 ways in L2, 2 ways in L1
#pragma statement scache_isolate_way L2=10 L1=2

// Assign arrays to Sector 1
#pragma statement scache_isolate_assign myArray, myBuffer

// ... compute kernel ...

#pragma statement end_scache_isolate_assign
#pragma statement end_scache_isolate_way


/* ---- Function-wide approach ---- */

// Apply to entire function
#pragma procedure scache_isolate_way L2=10
#pragma procedure scache_isolate_assign myArray
void compute_kernel(double* myArray, double** B, int m, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            myArray[i] += B[j][i];
        }
    }
}
```

```bash
# ---- Environment variable approach ----
export FLIB_SCCR_CNTL=TRUE
export FLIB_L1_SCCR_CNTL=TRUE
export FLIB_L2_SECTOR_NWAYS_INIT=4,10    # Sector0=4ways, Sector1=10ways
```
