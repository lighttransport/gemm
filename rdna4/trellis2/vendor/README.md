# vendor/ — bundled ROCm runtime bits

`runner/run_dump_rocm.sh` activates the vendored hipBLASLt subset by
prepending it to `LD_LIBRARY_PATH` and pointing
`HIPBLASLT_TENSILE_LIBPATH` here. This lets the runner work without
`/opt/rocm-7.2.2/lib/hipblaslt/library/` mounted at the system path —
useful for distribution to other machines that have the rest of ROCm
(libamdhip64, librocblas) but not hipBLASLt.

The actual binaries are *not* committed (.gitignore excludes them);
re-populate after a fresh checkout with:

```bash
./setup_vendor.sh           # default --prune: BB/HH/SS/DD gfx1201 (~60 MB)
./setup_vendor.sh --full    # every gfx1201 .co/.dat (~225 MB)
./setup_vendor.sh --prune --check
```

## What we keep (default --prune)

- `libhipblaslt.so.1.2.70202` (8.4 MB) — PyTorch links against this.
- `library/Kernels.so-000-gfx1201.hsaco` (20 MB) — compiled kernel blob.
- `library/extop_gfx1201.co` + `hipblasltExtOpLibrary.dat` +
  `hipblasltTransform.hsaco` — epilogues and transform glue.
- `library/TensileLibrary_BB_*gfx1201*` (BF16, 18 MB)
- `library/TensileLibrary_HH_*gfx1201*` (FP16, 20 MB)
- `library/TensileLibrary_SS_*gfx1201*` (FP32, ~225 KB)
- `library/TensileLibrary_DD_*gfx1201*` (FP64, ~165 KB)
- `library/TensileLibrary_lazy_*gfx1201*` (catalogue)

## What we drop

- `B8B8`, `B8F8`, `F8B8`, `F8F8` — FP8 variants (~155 MB total).
  TRELLIS-2 stage-2 doesn't dispatch to them.
- `I8I8` — INT8 (~15 MB).
- All non-`gfx1201` architectures (gfx908/90a/942/950/110x/115x/1200).

## Why we still need hipBLASLt at all

PyTorch links `libhipblaslt.so.1` directly. The F32 `SparseLinear` path
(which would otherwise hit the M>2^19 hipBLASLt bug — see
`rdna4/hipblaslt-issue.md`) is now intercepted by `kernels/linear_f32.py`,
so hipBLASLt only services the BF16 ViT/DiT GEMMs that work correctly.

`manifest.txt` records what the most recent `setup_vendor.sh` produced.
