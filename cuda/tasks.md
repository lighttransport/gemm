# CUDA runner tasks — handoff (continue on another machine)

## Resume prompt (paste to next agent)

> I'm continuing work on the CUDA runners under `cuda/{qimg,flux2,llm}`
> and `cuda/cuda_runner_common.h`. The last session landed three commits
> on `main` (`5fbb7b3`, `f9d7a00`, `0def888`) that finished a planned
> batch of reuse / hardening / perf items (B.1–B.4, C.1–C.3, D.1–D.2 from
> `~/.claude/plans/see-recent-code-changes-tranquil-rabin.md`). Working
> tree is clean; we're 16 commits ahead of `origin/main` but nothing has
> been pushed.
>
> Read `cuda/tasks.md` (this file) for the follow-up checklist. Items are
> grouped by why they exist: **audit gaps** are concrete weaknesses found
> after the work shipped; **deferred plan items** are things the original
> plan named but punted; **opportunistic** is everything else. Pick by
> impact, not order. Each item names the CUDA file/line to touch and how
> to verify.
>
> Environment notes: the prior session was on a host with an RTX 5060 Ti
> 16 GB (sm_120). The F16 sidecar cache (`B.4`) was warm-load measured at
> 27.0s → 3.0s on the Klein2-4B text encoder; if this machine doesn't
> have those weights, smoke with whatever BF16-heavy safetensors model is
> available and re-record timings in `doc/flux2-klein.md`. Don't push to
> `origin/main` without checking with me first (memory entry
> [feedback_git_push.md]).

---

## Audit gaps (found after `f9d7a00` shipped)

These were flagged during the post-commit audit. None are correctness
blockers in the tested path, but each is a sharp edge waiting to cut.

1. **[done] F16 sidecar cache: stream-format fragility** (B.4 follow-up)
   - Status: added per-record tensor signature, byte count, and CRC validation;
     mismatched records now invalidate instead of silently consuming the wrong
     payload.
   - File: `cuda/llm/cuda_llm_runner.c:4918-5069`
   - Problem: reader/writer is sequential with **no per-tensor name/size
     keying**. Cache key is only `(path, st_size, st_mtime)`. If a
     future patch adds, removes, or reorders any BF16 callsite of
     `upload_weight_matrix` between cache write and cache read, existing
     `.f16cache` files corrupt silently — the reader still validates
     magic+version+key, but the byte stream is consumed in the wrong
     order. The current underflow check at `cllm_f16cache_read_consume`
     catches running off the end; **overflow** (extra bytes consumed by
     extra calls misaligning the stream) is undetected.
   - Fix options (pick one):
     1. Bump `CLLM_F16CACHE_VERSION` whenever the load graph changes.
        Cheap, manual, error-prone.
     2. Add a load-profile fingerprint to the cache key: hash of
        `(n_layers, n_embd, head_dim, n_kv_heads, n_ff)` + an explicit
        loader version int. Catches structural changes, not call-order
        reshuffles within a fixed model.
     3. Per-block record format: each cache record is
        `(u32 nbytes, u32 crc32_of_payload, payload[nbytes])`. Reader
        validates record-by-record. Heavier but bulletproof.
   - Verify: write cache, mutate one BF16 callsite (e.g. swap order of
     `upload_weight_matrix(q)` and `upload_weight_matrix(k)`), reload.
     Expect either a clean MISS+rewrite or an explicit invalidation
     message, not silent garbage hidden states.

2. **[done] `cuda_llm_reset_state` returns void on invariant bail** (C.3 follow-up)
   - Status: `cuda_llm_reset_state` now returns `int`; callers propagate
     reset failures instead of continuing after an invariant or sync failure.
   - File: `cuda/llm/cuda_llm_runner.c:9856` (and the new check inserted
     at the top of the gemma4 branch in `f9d7a00`)
   - Problem: if the new shared-KV alias invariant check fires, the
     function logs to stderr and returns early — the cache is partially
     or wholly un-reset. Callers see no signal: the next generation runs
     against undefined state. Loud-but-soft failure mode.
   - Fix: change signature to `int cuda_llm_reset_state(...)`, return
     `-1` on invariant bail, propagate to callers in `cuda/llm/test_*.c`
     and `cuda/flux2/cuda_flux2_runner.h` (which calls reset between
     CFG halves on the resident text-encoder path).
   - Verify: temporarily zap `r->d_key_cache[shared_layer]` to a
     mismatched pointer, run a generate, confirm the caller refuses to
     proceed instead of producing garbage tokens.

3. **[done] F16 cache effectiveness for FP8-scaled encoders is measured**
   - Status: added `F8_E4M3` safetensors support in the CUDA LLM upload
     path by expanding scaled FP8 weights to F16 on upload, with optional
     Q/K norm handling for Qwen2.5 VL-style text encoders. Measured
     `/mnt/disk01/models/qwen-image-st/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
     with `test_cuda_flux2 --test-text-gpu --prompt smoke` and
     `/mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf`.
     Cold sidecar-write load: 102.40 s; warm sidecar-hit load: 97.58 s;
     cache payload: 1039.5 MB; repeat max diff: 0.0. The cache still saves
     ~4.82 s on this host, so no heuristic gate was added.
   - File: same as #1.
   - Problem: warm-load measurement (27.0s → 3.0s) was on Klein2-4B
     text encoder where most weights are BF16. For a Qwen 2.5 VL 7B
     FP8-scaled variant (e.g. `qwen_2.5_vl_7b_fp8_scaled.safetensors`)
     most linear weights are FP8 — they don't hit the BF16 branch of
     `upload_weight_matrix`. Cache savings collapse to whatever subset
     (embeddings, norms) is still BF16. Whether the cache is still net
     useful there is empirical.
   - Verify: `make -C cuda/flux2 test_cuda_flux2`,
     `./test_cuda_flux2 --test-kernels`, and the FP8-scaled
     `--test-text-gpu` run above.

## Deferred plan items (A.1/A.2 migration + plan §B.4 option 2 + §C.1 grid)

Named in the original plan but not landed. Land any of these as
follow-ups.

4. **[done] A.1 - qimg `cfg_cache` migrate to `cu_buf_slot`** (plan §A.1)
   - Status: qimg CFG buffers now use `cu_buf_slot[QIMG_CFG_BUF_COUNT]` with
     shared ensure/free helpers.
   - The generic `cu_buf_slot` / `cu_buf_slot_ensure` helpers landed in
     `cuda/cuda_runner_common.h:38-138` (commit `f9d7a00`). qimg still
     uses its own `qimg_cfg_cache` struct + `qimg_cfg_cache_ensure`
     defined locally in `cuda/qimg/cuda_qimg_runner.h`.
   - Port: replace `qimg_cfg_cache` array-of-slots with
     `cu_buf_slot[QIMG_CFG_BUF_COUNT]`; replace
     `qimg_cfg_cache_ensure(r, id, bytes, label)` with
     `cu_buf_slot_ensure(&r->cfg_slots[id], bytes, label)`. Mechanical.
   - Verify: `make test_cuda_qimg && ./test_cuda_qimg --generate
     --steps 8`; output PPM hash should match pre-migration.

5. **[done] A.2 - flux2 `FLUX2_RB_FREE` / `_FAIL` macros -> `CU_FREE` /
   `CU_CHECKED_ALLOC`** (plan §A.2)
   - Status: remaining local flux2 VAE/RB fail/free macros were removed in
     favor of `CU_FREE` and `CU_CHECKED_ALLOC`.
   - File: `cuda/flux2/cuda_flux2_runner.h` — search for `#define
     FLUX2_RB_FREE` / `FLUX2_VAE_*_FREE` / `FLUX2_*_FAIL` blocks. There
     are per-function `#define` / `#undef` dances at several call
     sites; replace with the common `CU_FREE(ptr)` and
     `CU_CHECKED_ALLOC(dst, bytes, label, fail_label)` from
     `cuda/cuda_runner_common.h`.
   - Verify: `make test_cuda_flux2 && ./test_cuda_flux2 --test-vae`.

6. **[done] B.4 option 2 - deterministic GPU BF16->F16 path** (plan §B.4)
   - Status: added opt-in `CUDA_LLM_GPU_BF16_TO_F16=1` path with a stream
     ordered BF16 upload plus in-place GPU conversion kernel. Verified
     bit-identical repeat output; `CUDA_LLM_GPU_BF16_VERIFY=1` compares GPU
     conversion against the CPU converter.
   - The shipped cache is option 1 (persistent sidecar). Option 2 is a
     GPU-side BF16→F16 kernel that produces bit-identical output across
     runs. Earlier attempt was reverted for nondeterminism; plan
     hypothesizes uninitialized padding in chunked HtoD. Retry with a
     fixed-size staging buffer + zeroed tail; compare hashes across 3
     consecutive cold runs in one process.
   - Wins if it works: eliminates the one-time sidecar build cost (the
     27s in our measurement) for users who never reload — useful when
     model weights change frequently during dev.

7. **[partial] C.1 — true multi-block FA2 grid parallelism** (plan §C.1)
   - Status: qimg and flux2 now have opt-in F32 split-key paths
     (`QIMG_FA2_SPLIT_F32` / `FLUX2_FA2_SPLIT_F32`). They launch split-key
     partial CTAs, store explicit `(m, l, O)` workspace, then merge partials
     with a second kernel. Both `--test-kernels` suites compare the raw split
     kernels and the `op_attn` dispatch path against baseline
     `flash_attn_f32`; both also include a 2048-token microbench that sweeps
     `split_kv={256,512,1024}`. Current measurements on this host: qimg F32
     split is faster at 2048 tokens (best `split_kv=256`: 2.470 ms baseline
     -> 1.789 ms split, 1.38x), while flux2 F32 split is slower on the same
     synthetic shape (best `split_kv=256`: 1.363 ms -> 1.790 ms, 0.76x).
     Numeric env values still force a split. `QIMG_FA2_SPLIT_F32=auto`
     enables the measured qimg large-token heuristic (`n_tok>=2048` ->
     `split_kv=256`); `FLUX2_FA2_SPLIT_F32=auto` currently declines to split
     because the measured path regresses. Keep this opt-in until real model
     shapes are tuned.
     BF16/FP8 split-key tensor-core variants remain future work.
   - Shipped in this session was the BKV bump 16→32 (tile-amortization
     win). The plan's original framing was grid parallelism across keys
     for very large n_tok (>2K). For qimg 1024×1024 image stream
     (~4096+512 tokens) this would matter; for 512×512 it doesn't.
   - Reference: `cuda/hy3d_paint/` BKV=64 kernel (head_dim=64 BF16 TC
     variant — not directly transplantable, but the warp-spec /
     producer-consumer structure is the closest template in-tree).
   - Verify: `./test_cuda_qimg --test-kernels` and
     `./test_cuda_flux2 --test-kernels` include 2048-token attention benches.
     A full CPU-reference bench under `cpu/qwen_image/` or `cpu/flux2/` would
     still be useful before default-enabling any split path.

## Opportunistic

8. **Push `origin/main`**
   - 16 commits ahead. Ask before pushing
     (see `memory/feedback_git_push.md`).

9. **[done] Strip pre-existing warnings during builds**
   - Status: removed the listed `n_top`, qimg `nh`/`hd`, and qimg `.npy`
     `fread` warnings. Also marked optional transformer helpers as
     `TF_MAYBE_UNUSED`, consumed the x86-only unused barrier parameter, and
     made `cuda_llm_vision_encode` validate `image_h`.
   - `common/transformer.h:3364` `unused variable 'n_top'` —
     `#pragma GCC diagnostic` or actually use `n_top`.
   - `cuda/qimg/cuda_qimg_runner.h:3446,3859` `unused variable 'hd' /
     'nh'` — same.
   - `cuda/qimg/test_cuda_qimg.c:311-312` `fread warn_unused_result` —
     branch on the return.
   - These don't block; they just clutter every build log.

10. **[done] `.gitignore` for the test-binary artifacts**
    - Status: added generated CPU/CUDA test binary, verify binary, CUDA image/
      tensor output, logs, server build/log, and top-level `trellis2_repo/`
      ignore patterns with source-file exceptions.
    - `git status` shows ~50 untracked items that are all build/run
      products (binaries, `.npy`, `.ppm`, `cuda_output/`, etc.). A
      sweep of `.gitignore` would clean this up. Examples:
      `cuda/*/test_cuda_*`, `cuda/*/verify_*`, `cuda/*/cuda_*.ppm`,
      `cuda/*/cuda_*.npy`, `cuda/*/cuda_*.bin`, `server/build/`,
      `logs/`, `**/output_*/`, `trellis2_repo/`.

## Verification harness

Quick smokes to run after any of the items above:

```bash
# qimg
cd cuda/qimg && make test_cuda_qimg && \
  ./test_cuda_qimg --test-kernels && \
  ./test_cuda_qimg --generate \
    --dit /mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
    --vae /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors \
    --enc /mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
    --prompt "smoke" --height 512 --width 512 --steps 4 --seed 0

# flux2 (kernel + warm-load only — skip full generate unless touching VAE/DiT)
cd cuda/flux2 && make test_cuda_flux2 && \
  ./test_cuda_flux2 --test-kernels && \
  ./test_cuda_flux2 --test-text-gpu \
    --enc /mnt/disk01/models/klein2-4b/text_encoder \
    --tok /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf
```

Expected baselines (RTX 5060 Ti, post `0def888`):
- qimg 512×512 8-step generate: ~1.17 s/step, 0 NaN, valid PPM.
- flux2 GPU text encoder warm load: ~3 s (HIT), cold: ~27 s (MISS).

## Out of scope here

- RDNA4 / HIP port (separate target tree `rdna4/`; the CUDA changes in
  this session would all be portable but that's a separate effort).
- a64fx port (`fcc` / `FCC` cross-compile; tracked in `a64fx/`).
- New model coverage (Klein-Base full pipeline, qimg img2img/ControlNet) —
  tracked in `doc/flux2-klein.md` and `doc/qwen-image.md`.
- sam3.1 server work (already shipped in `3fa91de`).
