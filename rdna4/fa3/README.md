# RDNA4 FP8 FA3-Style Attention Harness

Standalone HD128 FP8 flash-attention benchmark for gfx1201/RX 9070 XT.

Build:

```sh
make -C rdna4/fa3
```

Run:

```sh
rdna4/fa3/bench_fa3_fp8 --n-tok 1024 --heads 16 --iters 100 --mode all
rdna4/fa3/bench_fa3_fp8 --n-tok 256 --heads 16 --iters 20 --mode all --check
```

Modes:

- `exact_b16`: BKV=16, exact online softmax using `__expf`.
- `approx_b32`: BKV=32, log2-domain online softmax, fast bit-approx `exp2`, deferred denominator reduction.
- `approx_b32_exp2`: BKV=32, log2-domain online softmax, native `exp2f`, deferred denominator reduction.
- `approx_b64`: BKV=64, fast approximate softmax.
- `approx_b32_8w_fast`: BKV=32, BQ=128, 8-wave CTA, fast bit-approx `exp2`.
- `approx_b32_8w`: BKV=32, BQ=128, 8-wave CTA, native `exp2f`.
- `approx_b32_8w_db`: same BQ/BKV as `approx_b32_8w`, but double-buffers K/V LDS staging. Diagnostic; neutral to slightly slower.
- `approx_b64_8w_fast`: BKV=64, BQ=128, 8-wave CTA. Diagnostic; slower from larger state/occupancy pressure.
- `approx_b32_16w_fast`: BKV=32, BQ=256, 16-wave CTA, fast bit-approx `exp2`. Current best measured path.
- `approx_b32_32w_fast`: BKV=32, BQ=512, 32-wave CTA, fast bit-approx `exp2`. Diagnostic; does not beat 16-wave.
- `approx_b32_16w_qg2_fast`: BKV=32, BQ=512, 16-wave CTA with two query groups per wave. Diagnostic; spills/private segment make it slower.
- `pc_b32_8w`: single-kernel producer/consumer diagnostic; four producer waves compute the next QK tile while four consumer waves process the current tile.
- `two_pass_b32_8w`: two-kernel diagnostic; pass 1 computes row max, pass 2 recomputes QK and runs PV.

Measured on this checkout/GPU after initial implementation:

```text
FP8 WMMA peak reference: 342.9 TF/s (rdna4/fp8 peak_8w)

S=256 check:
  exact_b16:             cos=0.999598, maxd=0.00865
  approx_b32_8w_fast:    cos=0.999319, maxd=0.01210
  approx_b32_8w:         cos=0.999600, maxd=0.00865
  approx_b32_16w_fast:   cos=0.999319, maxd=0.01210
  approx_b32_32w_fast:   cos=0.999319, maxd=0.01210
  approx_b32_16w_qg2_fast: cos=0.999294, maxd=0.00743  (S=512)
  pc_b32_8w:             cos=0.999600, maxd=0.00865
  two_pass_b32_8w:       cos=0.999611, maxd=0.00823

S=1024:
  exact_b16:       18.10 TF/s
  approx_b32_exp2: 18.51 TF/s
  approx_b32_8w:   22.97 TF/s
  approx_b32_8w_fast: 22.47 TF/s
  approx_b32_16w_fast: 26.44 TF/s
  approx_b32_32w_fast: 26.41 TF/s
  pc_b32_8w:        6.85 TF/s
  two_pass_b32_8w: 13.34 TF/s

S=2048:
  approx_b32_8w:   24.21 TF/s
  approx_b32_8w_fast: 24.23 TF/s
  approx_b32_8w_db: 23.73 TF/s
  approx_b32_16w_fast: 28.53 TF/s
  approx_b32_32w_fast: 28.42 TF/s
  pc_b32_8w:        3.03 TF/s
  two_pass_b32_8w: 10.81 TF/s

S=4096:
  approx_b32_8w:   24.69 TF/s
  approx_b32_8w_fast: 24.93 TF/s
  approx_b32_8w_db: 24.55 TF/s
  approx_b32_16w_fast: 29.72 TF/s
  approx_b32_32w_fast: 29.16 TF/s
  approx_b32_16w_qg2_fast: 16.95 TF/s
  pc_b32_8w:        5.39 TF/s
  two_pass_b32_8w: 13.25 TF/s
```

The best current result is `approx_b32_16w_fast` at 29.72 TF/s for S=4096,
about 8.7% of the measured FP8 WMMA peak.
This confirms that the remaining work is not a simple FP8 type swap; the serial
online-softmax and P-transpose path still dominates on synchronous RDNA4 WMMA.

The two-pass diagnostic removes the online max recurrence but pays an extra QK
pass and a second launch. It is slower than the online 8-wave kernel on effective
standard attention TF/s, so the current evidence does not support a full 3-pass
or 4-pass recompute design as the next optimization direction.

The producer/consumer diagnostic is also slower. It overlaps four producer waves
against four consumer waves, but duplicated next-tile K/V staging plus LDS score
handoff and extra barriers dominate any softmax/WMMA overlap.

The double-buffered diagnostic also does not materially help. It reduces source
loop synchronization from two barriers per K/V tile to one barrier after a
prologue, but the larger LDS footprint and scheduling pressure offset the win.

The most useful shape change so far is increasing query rows per CTA. BQ=256
with 16 waves improves K/V tile reuse and amortizes barriers better than BQ=128,
while BQ=512/32 waves stops improving.

The P writeback now uses a positive-only FP32-to-FP8 converter because softmax
probabilities are finite and nonnegative. This trims some scalar work in the hot
P-transpose path without changing measured accuracy.

The shuffle P-transpose (`shflp_8w`/`shflp_16w`) replaces the LDS round-trip with
intra-wave shuffles. It is now numerically correct (cos=0.99963) but ~13% slower
than the LDS path (25.2 vs 29.0 TF/s @ S=4096, 16w): 64 shfls/tile cost more than
the LDS write+barrier+read. So the LDS P-transpose is NOT the bottleneck; the
ceiling is small-tile occupancy + serial online softmax, not the transpose path.
