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
     `flash_attn_f32`; both also include 2048-token and 4608-token
     microbenches that sweep
     `split_kv={128,256,384,512,768,1024,1536,2048}` while ignoring
     non-split candidates. Current measurements on this host: qimg F32 split
     is faster at 2048 tokens (best `split_kv=384`: 2.467 ms baseline
     -> 1.774 ms split, 1.39x), while flux2 F32 split is slower on the same
     synthetic shape (best `split_kv=384`: 1.364 ms -> 1.776 ms, 0.77x).
     Numeric env values still force a split. `QIMG_FA2_SPLIT_F32=auto`
     enables the measured qimg large-token heuristic (`n_tok>=2048` ->
     `split_kv=384`, `n_tok>=4096` -> `split_kv=1024`) only when qimg's
     faster BF16/FP8 tensor-core attention path is not active; numeric values
     still force F32 split-key for experiments.
     `FLUX2_FA2_SPLIT_F32=auto` currently declines to split because the
     measured path regresses; it also has the same tensor-core-priority guard
     so future auto heuristics will not preempt active BF16/FP8 attention
     unless a numeric split is explicitly requested. Flux2 kernel tests now
     verify that dispatch behavior directly. Keep this opt-in until real model
     shapes are tuned. A 4608-token qimg microbench was added to approximate the
     1024x1024 image-token regime; recent runs measure about 12.14 ms
     baseline -> 8.97-8.99 ms split (1.35x), with `split_kv=512` and
     `1024` close enough to trade places. qimg `auto` keeps `1024` for the
     lower-workspace path. qimg kernel tests now also verify dispatch priority:
     `auto` keeps the active BF16 tensor-core attention path, while numeric
     split values still force F32 split-key for experiments. qimg tests also
     include a 4608-token/24-head production-shape F32 split bench; on this host
     it measured about 514.7 ms baseline -> 205.5 ms split (2.50x), confirming
     split-key remains valuable for the qimg fallback path. The same 4608-token
     bench was added for flux2
     and still regressed: about 6.74 ms baseline -> 8.94 ms split
     (0.75x). Flux2 also now has a 4608-token/24-head production-shape bench;
     it measured about 200.5 ms baseline -> 205.5 ms split (0.98x), so flux2
     `auto` remains disabled.
     The qimg/flux2 split partial workspaces now share the
     `cu_split_attn_f32_workspace` helper in `cuda_runner_common.h`, including
     common overflow-checked sizing and free logic.
     The qimg/flux2 split correctness tests also compare the baseline F32
     flash-attention kernel against an independent CPU online-softmax reference
     before accepting split partial/merge and `op_attn` dispatch equivalence.
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

11. **[done] FP8 safetensors alias + scaled-weight hardening**
    - Status: common safetensors now accepts `F8_E4M3FN` as a 1-byte dtype.
      Flux2 treats `F8_E4M3FN` the same as `F8_E4M3` in FP8 upload/split
      paths. CUDA LLM safetensors maps both names and now fails loudly for
      FP8 `*.weight` tensors missing the required scalar F32 `*.scale_weight`.
    - Verify: `make -C cuda/{qimg,flux2,llm}` and FP8 text-encoder
      `test_cuda_flux2 --test-text-gpu`.

12. **[done] F16 sidecar cold-write temp path race**
    - Status: cache writers now use a PID-suffixed `*.f16cache.tmp.<pid>`
      with `O_EXCL`, so concurrent cold processes do not share a temp stream
      before atomic rename.
    - Verify: build plus a two-process cold-cache race if changing this again.

13. **[done] Split-key env parsing and launch fallback**
    - Status: qimg/flux2 split-key selectors now accept only `0`, `auto`, or
      clean numeric values; malformed env values warn once and disable split.
      Split partial/merge launch failures now log and fall back to the baseline
      attention path instead of returning with an unwritten output buffer.
    - Verify: `./test_cuda_qimg --test-kernels` and
      `./test_cuda_flux2 --test-kernels`.

14. **[done] Checked hot-path copies for qimg/flux2 DiT steps**
    - Status: qimg single/CFG DiT input uploads, timestep uploads, final syncs,
      and output downloads now check CUDA return codes. Flux2 DiT does the same
      for input/timestep/output transfers and frees the per-step timestep device
      buffer that previously leaked.
    - Verify: qimg/flux2 kernel tests plus a short generate smoke after runner
      changes.

15. **[done] Split-attention bench harness checks CUDA failures**
    - Status: qimg/flux2 split-attention correctness tests and timing helpers
      now fail the test on event, launch, sync, alloc/upload, or output-copy
      errors instead of comparing or timing undefined device buffers.
    - Verify: `./test_cuda_qimg --test-kernels` and
      `./test_cuda_flux2 --test-kernels`.

16. **[done] qimg kernel-test upload helpers avoid null-device copies**
    - Status: local qimg test upload helpers now check `cuMemAlloc` before
      `cuMemcpyHtoD`, report alloc/copy errors, and return null cleanly.
      The qimg FP8 GEMM unit test also checks host allocations, device uploads,
      output allocation, launches, syncs, and downloads before comparing output.
      The qimg FP8 attention unit test now does the same for its MMA/ref paths
      and FP8 workspace setup. The qimg `img_in` projection reference test also
      checks host allocations, safetensors uploads, output allocation, sync, and
      download before scoring correlation. The qimg production-shape FP8 GEMM
      regression test now checks weight/bias availability, host allocation,
      input upload, output allocation, syncs, and downloads while restoring the
      saved MMA mode on failure.
    - Verify: `make -C cuda/qimg test_cuda_qimg` and
      `./test_cuda_qimg --test-kernels`.

17. **[done] flux2 kernel tests check BF16/FP8 setup failures**
    - Status: Flux2 BF16 GEMM and FP8 attention unit tests now check CPU
      reference allocation, device uploads, output allocations, syncs, and
      downloads. The FP8 attention test also restores the saved
      `FLUX2_FP8_ATTN_REF` environment and runner attention mode on failure.
    - Verify: `make -C cuda/flux2 test_cuda_flux2` and
      `./test_cuda_flux2 --test-kernels`.

18. **[done] CUDA LLM stale generic SSM prefill note cleaned up**
    - Status: Qwen3.5/Qwen3.5-MoE SSM is implemented in decode and in the
      `cuda_llm_prefill_qwen35` batched path. The stale generic prefill SSM
      placeholder was replaced with a loud unsupported-route error that frees temporary
      batch buffers and returns `NULL`, so future unsupported SSM routing cannot
      silently skip SSM layers. The surrounding generic prefill path now uses a
      shared cleanup exit for temporary batch buffers and checks its batch
      allocations plus embedding upload/download copies. Non-Gemma4 generic
      prefill now falls back to the single-token path before the Gemma4-only
      batched kernels, avoiding the old silent layer skip for unsupported
      batched routes.
    - Verify: `rg -n "unsupported generic SSM prefill" cuda/llm/cuda_llm_runner.c` and
      `make -C cuda/llm test_cuda_llm`.

19. **[done] Trellis2 CUDA runner destructor frees loaded weights**
    - Status: `cuda_trellis2_free` now releases all owned GPU weight groups
      (Stage 1/2/3 DiT weights, DINOv3 weights, dense decoder weights, shape
      decoder weights, cross-attention KV cache, and all 12 scratch slots) plus
      CPU-side streaming copies and CPU shape/texture decoder instances. This
      removes the previous `free all GPU weight buffers` placeholder and prevents
      long-lived tools from leaking VRAM across runner reloads.
    - Verify: `make -C cuda/trellis2 test_cuda_trellis2`.

20. **[done] Trellis2 Stage 2 test step log reports real latent std**
    - Status: `test_cuda_trellis2` now computes the sample standard deviation
      of the Stage 2 latent after each Euler update instead of printing the
      placeholder `0.0000`, removing a stale test-harness placeholder and making
      long sampling logs useful for spotting divergence.
    - Verify: `make -C cuda/trellis2 test_cuda_trellis2`.

21. **[done] INT8 tcgen05 option no longer dispatches incomplete PTX**
    - Status: the experimental INT8 `--tcgen05` path now fails explicitly on
      SM 10.x with an unsupported-path message instead of selecting a PTX body
      whose `tcgen05.mma` issue instruction is intentionally absent. The help
      text now describes the option as disabled until the operand syntax is
      validated on SM 10.x hardware.
    - Verify: `make -C cuda/int8 int8_gemm`.

22. **[done] Removed stale cuew cuGetProcAddress alias note**
    - Status: dropped the commented `cuGetProcAddress_v2_ptsz` alias note from
      `cuew.h`; the active header already maps `cuGetProcAddress` to the loaded
      v2 symbol, so the disabled ptsz alias was not an actionable CUDA task.
    - Verify: `make -C cuda/int8 int8_gemm`.

23. **[done] Trellis2 predict stub reports unsupported mode without placeholder text**
    - Status: the `cuda_trellis2_predict` stub now reports that DINOv3 GPU
      encoding is not available through that entry point and asks callers to
      pass precomputed features, instead of emitting a placeholder-flavored message.
    - Verify: `make -C cuda/trellis2 test_cuda_trellis2`.

24. **[done] SAM3.1 RoPE block-load note clarified**
    - Status: the stale Phase-C `attn.freqs_cis` note in the SAM3.1 ViT block
      loader was replaced with an implementation note. The runner already
      builds shared F32 RoPE cos/sin tables eagerly and applies them inside
      each ViT block; it does not load per-block checkpoint `freqs_cis` tensors.
    - Verify: `make -C cuda/sam3.1 verify_vit`.

25. **[done] Hunyuan3D paint pipeline pyref dependency note clarified**
    - Status: the top-level paint pipeline script now labels the view-map,
      DINOv2-conditioning, and UniPC scheduler dump requirements as current
      external pyref inputs rather than an extraction placeholder, so CUDA marker scans
      only surface actionable code gaps.
    - Verify: shell syntax remains unchanged (`set -euo pipefail` script,
      comment-only change).

26. **[done] Trellis2 runner-local build warnings cleaned up**
    - Status: removed unused runner-local timing/RNG helpers, an unused upload
      byte count, the unused decoder group parameter, and the remaining unused
      predict-stub parameters. Kernel launch argument arrays now use mutable
      local copies for scalar dimensions, avoiding const-discard warnings.
    - Verify: `make -C cuda/trellis2 test_cuda_trellis2` and
      `git diff --check`.

27. **[done] Trellis2 test-harness warning cleanup**
    - Status: the local `.npy` reader now has structured shape parsing and
      checks the data `fread`, the writer no longer has misleading one-line
      indentation, the unused Stage 1 `dt` local was removed, the downsample
      clamp branches are split onto separate statements, and `sparse_coords`
      stays alive until the optional texture decoder has consumed it.
    - Verify: `make -C cuda/trellis2 test_cuda_trellis2`.

28. **[done] Trellis2 shared-header warning cleanup**
    - Status: fixed the remaining warnings emitted by the Trellis2 CUDA build
      from shared headers: split misleading one-line conditionals in PBR and
      shape-decoder helpers, removed dead FDG split-alignment code, made the
      marching-cubes shrink step use a separate realloc result, and marked
      implementation-only shape-decoder helpers as intentionally optional.
    - Verify: `make -B -C cuda/trellis2 test_cuda_trellis2`.

29. **[done] Ignore INT8 CUDA build binaries**
    - Status: added the generated `cuda/int8` binaries to `.gitignore`, so
      `make -C cuda/int8 int8_gemm` and adjacent INT8 targets no longer leave
      untracked build artifacts in the CUDA status view.
    - Verify: `git check-ignore -v cuda/int8/int8_gemm`.

30. **[done] Trellis2 public DINOv3 image-to-occupancy path**
    - Status: `cuda_trellis2_predict` now runs the RGB image path end to end
      through DINOv3 preprocessing/encoding, Stage 1 DiT Euler sampling with CFG,
      and the dense structure decoder, returning a CPU `[64,64,64]` occupancy
      grid instead of an unsupported-mode stub.
    - Verify: `make -B -C cuda/trellis2 test_cuda_trellis2` and
      `git diff --check`.

31. **[done] Trellis2 CUDA image-to-shape smoke can be bounded**
    - Status: `test_cuda_trellis2` now accepts `--sparse-threshold` and
      `--max-sparse` for Stage 2 sparse coord extraction, making real
      DINOv3-image -> Stage 1 -> Stage 2 -> shape-decoder -> FDG mesh smoke
      runs practical without accidentally expanding tens of thousands of
      occupancy voxels.
    - Verify: `make -B -C cuda/trellis2 test_cuda_trellis2`, `git diff --check`,
      and a bounded smoke with `--max-sparse 2048` that wrote
      `/tmp/t2_cuda_image_to_shape_smoke.obj`.

32. **[done] qimg/flux2 split-attention workspace helper**
    - Status: qimg and flux2 now use a shared
      `cu_split_attn_f32_workspace` in `cuda/cuda_runner_common.h` for F32
      split-key partial O/m/l buffers. The helper centralizes overflow-checked
      byte sizing, lazy growth, and free logic while keeping the existing
      kernel launch ABI unchanged.
    - Verify: `make -C cuda/qimg test_cuda_qimg`,
      `make -C cuda/flux2 test_cuda_flux2`,
      `./test_cuda_qimg --test-kernels`, and
      `./test_cuda_flux2 --test-kernels`.

33. **[done] qimg/flux2 split-attention CPU reference gate**
    - Status: the small F32 split-key correctness tests now compute an
      independent CPU online-softmax reference and require the baseline F32
      attention kernel to match it before comparing split partial/merge and
      forced `op_attn` split dispatch. This keeps the fast production-shape
      benches tied to a CPU-checked correctness seed without making the large
      benches CPU-bound.
    - Verify: `make -C cuda/qimg test_cuda_qimg`,
      `make -C cuda/flux2 test_cuda_flux2`,
      `./test_cuda_qimg --test-kernels`, and
      `./test_cuda_flux2 --test-kernels`.

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
