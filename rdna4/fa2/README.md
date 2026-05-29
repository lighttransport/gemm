# rdna4/fa2 — flash attention, gau-nernst blog techniques

HD=128, BQ=256, BKV=32, 16-wave WMMA flash attention for gfx1201/RX 9070 XT.
BF16/FP16 + FP8 e4m3. Standalone HIPRTC (no SDK). The blog's big lever — **LDS
XOR swizzle** — ports cleanly and is the win.

```
make -C rdna4/fa2
rdna4/fa2/bench_fa2 --n-tok 256 --heads 2 --check --mode all   # f16/fp8 cos>0.999
rdna4/fa2/bench_fa2 --n-tok 4096 --heads 16 --mode all
```

Modes: `f16` `fp8` (keepers); `b16` `b16s` `b16d` (bf16, accuracy open — see
BF16_DEBUG.md). peak: bf16/fp16 190, fp8 350. `--dim` sweeps head_dim (128/256).

## Keepers, S=4096, 16 heads
| mode | TF/s | %peak | note |
|---|---|---|---|
| f16 swizzle+db | **37.1** | 19.5% | 16-bit keeper; cos 0.9996 |
| fp8 d256 | ~24 | 6.8% | fat-matrix keeper; d512 LDS-bound |

bf16: builtin verified good (HIPRTC+AOT minimal pass), but full 16-wave kernel
outputs cos≈0; f16 is the equal-perf proxy. 25% peak not reached — RDNA4 sync
WMMA + thin tiles, as fa3 predicted. d256 helps fp8, 16-bit OOMs LDS past d128.

Swizzle (`d ^ ((row&7)<<4)`) spreads HD-stride K across all 32 banks, killing the
8-way conflict on QK reads. Beats fa3's prior fp16 best. FP8 caps ~30 TF/s as
`rdna4/fa3/note.md` found — synchronous WMMA + small register file, not 80–90%.
