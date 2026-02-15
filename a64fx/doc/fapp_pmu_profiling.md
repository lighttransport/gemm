# PMU Event Profiling with `fapp` on Fujitsu A64FX (Fugaku)

## Overview

The **Detailed Profiler** (`fapp`) on Fujitsu's Fugaku supercomputer provides PMU (Performance Monitoring Unit) event-based profiling for analyzing CPU performance characteristics. It consists of two commands:

- **`fapp`** — Collects profile data (measurement)
- **`fapppx`** — Outputs profile results (analysis, used on login nodes)

The profiler can collect three categories of information:

1. **Time Statistics** — Call count, elapsed time, user/system CPU time
2. **MPI Communication Cost** — MPI function execution counts, message lengths, timings
3. **CPU Performance Analysis (CPUPA)** — PMU hardware counter data for CPU behavior analysis

---

## Step-by-Step Procedure

### Step 1: Add Measurement Region Markers to Source Code

Insert `fapp_start` / `fapp_stop` calls to define the measurement region.

**Fortran:**

```fortran
CALL fapp_start(name, number, level)
! ... code to profile ...
CALL fapp_stop(name, number, level)
```

**C/C++:**

```c
#include "fj_tool/fapp.h"

fapp_start("region_name", 1, 0);
// ... code to profile ...
fapp_stop("region_name", 1, 0);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `name`    | Group name (string). Combined with `number` to form the measurement region identifier. |
| `number`  | Detail number (integer). Combined with `name` for the region identifier. |
| `level`   | Activation level (integer). Used with `-L` option to selectively enable regions. |

**Rules:**
- Always call `fapp_start` before `fapp_stop` for the same region.
- Different region names can be measured concurrently (nested).
- The `level` values must match between `fapp_start` and `fapp_stop`.

### Step 2: Set Environment Variables

Environment variables prefixed with `FAPP_`, `PROF_`, `FJPROF_`, or `FPROF_` are reserved by the profiler. Configure as needed per the system documentation.

### Step 3: Compile the Program

Compile with the Fujitsu compiler, including appropriate debug/profiling options as described in the Fortran/C/C++ user guides.

### Step 4: Collect Profile Data with `fapp`

#### Command Syntax

```
fapp -C -d profile_data \
  [-I{cpupa|nocpupa|cputime|nocputime|mpi|nompi}] \
  [-H{event=EVENT|event_raw=EVENT_RAW}[,method={fast|normal},mode={all|user}]] \
  [-L level] \
  [-W{spawn|nospawn}] \
  exec-file [exec_options ...]
```

#### Key Options

| Option | Description |
|--------|-------------|
| `-C` | **Required.** Instructs data collection. |
| `-d profile_data` | **Required.** Directory to store profile data. Created automatically if it doesn't exist. |
| `-Icpupa` | Enable CPU performance analysis data collection. Automatically enabled when `-H` is specified. |
| `-Inocpupa` | Disable CPU performance analysis data collection. |
| `-Icputime` / `-Inocputime` | Enable/disable user and system CPU time measurement. |
| `-Impi` / `-Inompi` | Enable/disable MPI communication cost measurement. |
| `-L level` | Set measurement activation level (0–2,147,483,647). Only regions with level ≤ this value are measured. |

#### `-H` Sub-options (PMU Event Configuration)

| Sub-option | Description |
|------------|-------------|
| `event=EVENT` | Collect predefined PMU event groups for the CPU Performance Analysis Report. Values: `pa1`–`pa17`, `statistics` (equivalent to `pa1`). |
| `event_raw=EVENT_RAW` | Specify raw PMU event numbers (decimal or hex with `0x` prefix). Up to 8 events, comma-separated. |
| `method=fast` | Directly read hardware counters (high precision, but counts even during sleep states). |
| `method=normal` | Read counters via OS (default). Cannot specify duplicate event numbers with `event_raw`. |
| `mode=all` | Measure both kernel and user mode (default). |
| `mode=user` | Measure user mode only. |

#### Basic Example

```bash
# Collect CPU performance analysis data with the "statistics" event group
fapp -C -d ./prof_data -Icpupa -Hevent=statistics ./a.out
```

#### Raw PMU Event Example

```bash
# Collect specific PMU events by event number (up to 8)
fapp -C -d ./prof_data -Icpupa -Hevent_raw=0x0011,0x0013,0x0018 ./a.out
```

### Step 5: Output Profile Results

Use `fapppx` (login node) or `fapp` (compute node) with the `-A` flag:

```
{fapppx|fapp} -A [-I{cpupa|nocpupa|mpi|nompi}] [-o outfile] [-p p_no] [-t{csv|text|xml}] [-d] profile_data
```

#### Key Options

| Option | Description |
|--------|-------------|
| `-A` | **Required.** Instructs result output. |
| `-Icpupa` | Include CPU performance analysis info in output. |
| `-ttext` | Output in text format (default). |
| `-tcsv` | Output in CSV format (required for CPU PA Report). |
| `-txml` | Output in XML format. |
| `-o outfile` | Output file path (`stdout` for standard output). |
| `-p` | Select processes to output (e.g., `-pall`, `-plimit=16`, `-p3,5`). |

#### Example

```bash
# Output text-format results
fapppx -A -ttext -o result.txt -d ./prof_data

# Output CSV for CPU PA Report
fapppx -A -Icpupa,nompi -tcsv -o pa1.csv -d ./prof_data
```

---

## CPU Performance Analysis Report Workflow

The CPU Performance Analysis Report aggregates multiple PMU event measurement runs into an Excel-based (.xlsm) visual report. Four report levels are available:

| Report Level | Measurements Required | Event Options |
|---|---|---|
| **Single** | 1 run | `pa1` |
| **Brief** | 5 runs | `pa1`–`pa5` |
| **Standard** | 11 runs | `pa1`–`pa11` |
| **Detailed** | 17 runs | `pa1`–`pa17` |

### Measurement Phase

Run `fapp` multiple times, each with a different `-Hevent=paN` value:

```bash
# Example: Standard Report (11 measurements)
for i in $(seq 1 11); do
  fapp -C -d ./rep${i} -Hevent=pa${i} ./a.out
done
```

> **Important:** All runs must use identical program input/execution behavior.

### CSV Export Phase

Export each measurement directory as a CSV file:

```bash
# Example: Standard Report
for i in $(seq 1 11); do
  fapppx -A -d ./rep${i} -Icpupa,nompi -tcsv -o pa${i}.csv
done
```

### Report Generation

1. Copy all `pa*.csv` files and `cpu_pa_report.xlsm` into the same directory.
   - The report template is located at: `/INSTALL_PATH/misc/cpupa/cpu_pa_report.xlsm`
2. Transfer the directory to a machine with Microsoft Excel.
3. Open `cpu_pa_report.xlsm` (enable macros if prompted).
4. Follow the dialogs to select process number, CMG number, and measurement region.
5. The report is generated automatically based on available CSV files.

The report type is determined by which CSV files are present (pa1–pa17 → Detailed, pa1–pa11 → Standard, pa1–pa5 → Brief, pa1 only → Single).

### Report Contents

| Section | Single | Brief | Standard | Detailed | Description |
|---------|:------:|:-----:|:--------:|:--------:|-------------|
| Information | ✓ | ✓ | ✓ | ✓ | Measurement environment info |
| Statistics | △ | ✓ | ✓ | ✓ | Memory throughput, instruction counts, FLOPS |
| Cycle Accounting | - | △ | ✓ | ✓ | Execution time breakdown (9–20 categories) |
| Busy | - | △ | △ | ✓ | Cache/memory/pipeline busy rates |
| Cache | - | △ | ✓ | ✓ | Cache miss details (L1D, L2) |
| Instruction | - | △ | △ | ✓ | Instruction mix (9–28 categories) |
| FLOPS | - | △ | ✓ | ✓ | Floating-point operation details |
| Extra | - | - | - | ✓ | Gather instructions, misc. instruction info |
| HW Prefetch Rate | - | - | - | ✓ | Hardware prefetch mode breakdown |
| Data Transfer CMGs | - | - | - | ✓ | Inter-CMG/memory/Tofu/PCI throughput |
| Power Consumption | - | - | ✓ | ✓ | Core, L2, and memory power consumption (W) |

(✓ = full, △ = partial, - = not shown)

---

## PMU Event XML Output Format

When using `-txml`, CPU performance analysis data is output in XML under a `<cpupa>` element:

```xml
<cpupa>
  <event name="CNTVCT"> 123456789 </event>
  <event name="PMCCNTR"> 987654321 </event>
  <event name="0x0011"> 42000 </event>
  <!-- one <event> per PMU counter measured -->
</cpupa>
```

- **CNTVCT**: Counter-timer Virtual Count (see Arm documentation)
- **PMCCNTR**: Performance Monitors Cycle Counter (see Arm documentation)
- **PMU events**: Refer to [A64FX PMU Events documentation](https://github.com/fujitsu/A64FX/tree/master/doc/)

---

## Important Notes and Caveats

1. **CPU Binding is required.** Threads must be 1:1 bound to CPUs during measurement.
   - OpenMP/thread-parallel programs: Use compiler runtime options.
   - MPI-only programs: Use VCOORD files with `core=1`.
   - Non-MPI, non-threaded programs: Use `taskset` or `numactl`.

2. **`method=fast` vs `method=normal`:**
   - `fast`: Reads hardware counters directly. Higher precision but counts during sleep states, so execution time values may appear larger.
   - `normal` (default): Reads via OS. Cannot specify duplicate event numbers with `event_raw`.

3. **`event_raw` constraints:**
   - Maximum 8 events per measurement run.
   - With `method=normal`, duplicate event numbers cause an error: `RTINF2xxx: Internal error. PAPI return code = xxx.`

4. **Do not use `-Inocpupa` when collecting CPU PA data.** It disables the `-H` option.

5. **MPI threading level:** MPI_THREAD_SERIALIZED and MPI_THREAD_MULTIPLE are not supported; profile data will be incorrect.

6. **Report CSV filenames are fixed:** The output filename must match the pattern `paN.csv` (e.g., `pa1.csv`, `pa2.csv`, ...).

7. **Measurement order is arbitrary.** You can run `pa3` before `pa1`, etc.

8. **Profiler overhead is included** in the CPU performance analysis measurements.
