#!/usr/bin/env python3
"""Generate the benchmark report from completed campaign CSV files."""
import csv
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))

def geo(values):return math.exp(sum(math.log(x) for x in values)/len(values))
def sci(s):return f'{float(s):.3g}'

def main():
    rows=read('results.csv');phases=read('phase_timings.csv');raw=read('raw_wmma.csv')
    main=[r for r in rows if r['family'] not in ('sweep','stress')]
    order=[]
    for r in main:
        key=(r['family'],r['m'],r['n'],r['k'])
        if key not in order:order.append(key)
    out=['# RX 9070 XT Ozaki-II benchmark','',
         'The reported latency covers exponent analysis, input packing, the complete Ozaki-II pipeline, and the FP64 epilogue. Native FP64 is direct rocBLAS DGEMM on the same inputs. Each row uses five warmups and 30 HIP-event samples, with calls repeated to target at least 100 ms per sample. Native baselines are measured separately for INT8 and FP8 runs.', '',
         'Inputs are deterministic uniform FP64, seed 42, with alpha=1 and beta=0. INT8 uses 15 moduli and FP8 11, fast scaling, and the selected variants from [tuning.csv](tuning.csv). The maximum normalized error and ULP statistics compare sampled outputs with a 256-bit MPFR reference; all outputs are checked when there are at most 4096, otherwise 256 positions are sampled.', '',
         '| Family and M×N×K | Native ms / TFLOP/s (INT8 / FP8 run) | INT8 ms (speedup) | FP8 ms (speedup) | INT8 / FP8 sampled normalized error |',
         '|---|---:|---:|---:|---:|']
    family_scores={'int8':{},'fp8':{}}
    for key in order:
        matched={r['backend']:r for r in main if (r['family'],r['m'],r['n'],r['k'])==key}
        if set(matched)!=set(('int8','fp8')):continue
        a,b=matched['int8'],matched['fp8'];family,m,n,k=key
        for r in (a,b):
            if r['quality_pass']=='1':family_scores[r['backend']].setdefault(family,[]).append(float(r['speedup']))
        flag_a='' if a['quality_pass']=='1' else ' †';flag_b='' if b['quality_pass']=='1' else ' †'
        out.append(f'| {family} {m}×{n}×{k} | {float(a["native_ms"]):.3f} / {float(a["native_tflops"]):.2f}; {float(b["native_ms"]):.3f} / {float(b["native_tflops"]):.2f} | {float(a["ms"]):.3f} ({float(a["speedup"]):.2f}×){flag_a} | {float(b["ms"]):.3f} ({float(b["speedup"]):.2f}×){flag_b} | {sci(a["sampled_norm_error"])} / {sci(b["sampled_norm_error"])} |')
    out+=['','† Failed the stated sampled precision criterion.','', 'Equal-family geometric means (the five square sizes first count as one family):']
    for backend in ('int8','fp8'):
        if family_scores[backend]:out.append(f'- {backend.upper()}: **{geo([geo(x) for x in family_scores[backend].values()]):.2f}×** native FP64.')
    out+=['','## Precision / performance sweep','','At 512³, these are the fastest configurations in this campaign satisfying the sampled precision criterion. All moduli counts and failures are in [results.csv](results.csv).','','| Backend | Fastest passing count | Time ms | Speedup | Max sampled normalized error | Max absolute error (emulated / native) | Median / 95th / max ULP |','|---|---:|---:|---:|---:|---:|---:|']
    for backend in ('int8','fp8'):
        passing=[r for r in rows if r['family']=='sweep' and r['backend']==backend and r['quality_pass']=='1']
        if passing:
            r=min(passing,key=lambda x:float(x['ms']));out.append(f'| {backend.upper()} | {r["moduli"]} | {float(r["ms"]):.3f} | {float(r["speedup"]):.2f}× | {sci(r["sampled_norm_error"])} | {sci(r["max_abs_mpfr"])} / {sci(r["native_max_abs_mpfr"])} | {r["ulp_p50"]} / {r["ulp_p95"]} / {r["ulp_max"]} |')
    out+=['','## Raw WMMA product','','These separate device-resident 1024³ modular-product measurements exclude input conversion, CRT, and the FP64 epilogue. They are not FP64 GEMM throughput.','','| Backend | Median ms | Raw TOP/s or TFLOP/s | Max sampled integer error |','|---|---:|---:|---:|']
    for r in raw:out.append(f'| {r["backend"].upper()} | {float(r["median_ms"]):.4f} | {float(r["raw_tops"]):.2f} | {r["max_abs_error"]} |')
    out+=['','![Performance versus native FP64](performance.svg)','','![Precision versus speed](precision_frontier.svg)','','## Numerical stress','','The criterion is `max(4 * native maximum normalized error, 32 * 2^-53)` for each case. Subnormal outputs have an unavoidable quantization floor, so the native FP64 error term is important.','',
         '| Input | Scaling | INT8 count | INT8 error / pass | FP8 count | FP8 error / pass |','|---|---|---:|---:|---:|---:|']
    for distribution in ('wide','cancellation','zero','subnormal-input','subnormal-output','huge'):
        for scaling in ('fast','accurate'):
            for extended in (False,True):
                pair=[]
                for backend in ('int8','fp8'):
                    count=20 if extended else (15 if backend=='int8' else 11)
                    matches=[r for r in rows if r['family']=='stress' and r['distribution']==distribution and r['scaling']==scaling and r['backend']==backend and int(r['moduli'])==count]
                    pair.append(matches[0] if matches else None)
                if all(pair):
                    a,b=pair;out.append(f'| {distribution} | {scaling} | {a["moduli"]} | {sci(a["sampled_norm_error"])} / {"yes" if a["quality_pass"]=="1" else "no"} | {b["moduli"]} | {sci(b["sampled_norm_error"])} / {"yes" if b["quality_pass"]=="1" else "no"} |')
    out+=['','## Internal phase profile','','These medians come from a separate instrumented build with synchronization. They cover GEMMul8 internals on normalized uniform inputs; outer exponent scanning, input staging, and the final public-API epilogue are excluded. They should not be summed or compared directly with the headline event timings.','','| Backend | M=N=K | Scaling / packing ms | WMMA ms | Modular reduction ms | CRT / rescaling ms |','|---|---:|---:|---:|---:|---:|']
    for r in phases:out.append(f'| {r["backend"].upper()} | {r["m"]} | {float(r["scale_pack_ms"]):.3f} | {float(r["wmma_ms"]):.3f} | {float(r["mod_reduce_ms"]):.3f} | {float(r["crt_rescale_ms"]):.3f} |')
    out+=['','## Conditions and limits','','GPU: RX 9070 XT, gfx1201, 64 CUs, 16 GB. [Environment and telemetry](benchmark_environment.txt) records the compiler, ROCm, repository revision, and sampled telemetry. Clocks were not locked. AMD SMI reported the GPU as throttled during the 8192³ run, but did not report the reason. These measurements therefore describe this run and should not be treated as an architectural peak.','',
         'WMMA ISA and resource checks: Both selected variant 27 kernels have 192 VGPRs, 16 KiB LDS, and two reported VGPR spills (12 bytes of private storage per thread). Their code objects contain the expected INT8/FP8 WMMA instructions. The exact rocBLAS DGEMM dispatch was traced to a double-precision code object containing `v_fma_f64`.', '',
         'The accuracy sweep and large-matrix checks are sampled, and fast/accurate scaling at a fixed modulus count does not guarantee FP64-quality componentwise error for all finite matrices. K-block boundaries, padded output panels, scalar cases, and subnormal cases have separate correctness tests; no performance estimate extrapolates beyond the measured shapes.','',
         'Reproduce with `make -C rdna4/gemmmul8 -j8`, `make -C rdna4/gemmmul8 test`, `python3 rdna4/gemmmul8/benchmark_campaign.py`, `python3 rdna4/gemmmul8/plot_results.py`, and `python3 rdna4/gemmmul8/report_results.py`.']
    (ROOT/'REPORT.md').write_text('\n'.join(out)+'\n')
    print(f'Wrote {ROOT / "REPORT.md"}')
if __name__=='__main__':main()
