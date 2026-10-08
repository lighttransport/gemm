# Multiview texture continuation on RTX 5060 Ti

Follow-up: [Blender-assisted surface refinement](BLENDER_TEXTURE.md) promotes a
photo-preserving correction of this baseline. The diffusion comparison below
remains unchanged; none of its newly generated views was promoted.

This workflow continues the Edit-2511/MV-Adapter appearance experiment on the
existing Obama `material12` GNM candidate. Generated appearance is synthetic;
the hybrid retains the SDXL base's evaluation-only provenance. It does not
recover hidden anatomy or estimate new normal/roughness maps.

## Local assets and environment

Inputs are under `/mnt/nvme02/models/vhuman-texture-inputs/`, copied from
`b550:~/work/gemm/pixal3d/tmp/`. The Edit-2511 directory is
`/mnt/nvme02/models/qwen-image-edit-2511/`, copied from
`b550:/mnt/disk1/models/qwen-image-edit-2511/` using `rsync -aL --partial`.
Keep the GGUF and base processor, tokenizer, scheduler, text encoder, VAE and
transformer configuration; omit the BF16 transformer shards and AMD packs.
The transferred model files total 30,104,005,701 bytes. All six weight files
were checked against SHA-256 hashes computed on b550.
After inference, NVMe02 fell to 2.3 GB free. The weight directory was moved to
`/mnt/nvme01/models/qwen-image-edit-2511/`; the NVMe02 path is a symlink, so the
commands and recorded model paths below remain valid. Input data stays on NVMe02.
All six weight hashes passed again after relocation (`relocated-weights-check.log`).

The local environment is `tmp/vhuman-texture-venv` (Python 3.12). Its
`cuda_reference.pth` imports the existing `tmp/qimg21-ref-venv` packages:
Torch 2.14.0+cu130, diffusers 0.41.0.dev0 and transformers 5.18.0.dev0.
Additional packages installed locally: scipy 1.18.1, gguf 0.19.0,
onnxruntime 1.30.0, opencv-python-headless 5.0.0.93, timm 1.0.30,
kornia 0.8.3, kornia-rs 0.2.0, einops 0.8.2, flatbuffers 25.12.19 and
websockets 17.2 (browser verification).
Retain the reference environment when reusing this environment.

```sh
export TMPDIR="$PWD/tmp"
export HF_HOME="$PWD/tmp/vhuman-texture-5060ti/hf"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=tmp/vhuman-texture-venv/bin/python
WORK=tmp/vhuman-texture-5060ti/work
MODEL=/mnt/nvme02/models/qwen-image-edit-2511
MATTE=/mnt/nvme01/models/BiRefNet
PARSER=/mnt/nvme02/models/vhuman-texture-inputs/face-parsing/resnet18.onnx
```

Use authorized GPU access outside the filesystem/device sandbox. The 5060 Ti
reports compute capability 12.0 and successfully executes BF16 matmul.
The desktop consumes roughly 1.2 GB of VRAM. A fully resident GGUF edit ran
out of memory at 1024 square pixels; use `--offload-blocks 1` for this machine.
CPU prompt encoding uses FP32 and returns BF16 embeddings to the pipeline.
Each editor caches at most four prompt encodes, keyed by text and exact image
pixels, dimensions and mode; the cache is never shared across model instances.

## Reproduce and compare

Prepare a fresh workspace with the local candidate. The conditions reproduced
all b550 hashes. Link or copy its saved `mvadapter` and
`qwen_edit_seq_old_int4` generation directories into the workspace. The latter
is the source of the original hybrid's face/side views, not the newer unchained
`qwen_edit_seq` run. Keep the imported baseline immutable.

```sh
$PY -m server.vhuman.reconstruction.mv_texture prepare \
  --candidate /mnt/nvme02/models/vhuman-texture-inputs/obama/material12 --work "$WORK"
$PY -m server.vhuman.reconstruction.mv_texture generate \
  --work "$WORK" --backend qwen_edit_seq --name smoke_offload --views right \
  --portrait-mode raw --prompt-recipe original --steps 2 \
  --edit-backend gguf --edit-model-root "$MODEL" --offload-blocks 1
$PY -m server.vhuman.reconstruction.mv_texture compare \
  --work "$WORK" --views right --name recipe --steps 12 --seed 317 \
  --edit-backend gguf --edit-model-root "$MODEL" --offload-blocks 1 --matte-model "$MATTE"
```

`compare` loads the editor once and writes `recipe_comparison.html` and JSON
receipts. It compares raw/matted portraits with the historical/bare-skin
prompts. The right view always uses seed 318 (base seed plus canonical index),
CFG 4, the same negative prompt and fixed 1024-square target. The target is
the last image condition. The strict matted recipe fails if BiRefNet fails;
it never quietly substitutes the face silhouette.

`--views` selects independent edits of the original conditions. There is no
previous-view reference or generated-atlas feedback. Without it, existing
sequential generation remains available. Generation folders must be fresh;
use `--name` to retain all trials. Records include model hash, prompt and
reference hashes, per-view seed, timing and peak allocated/reserved VRAM.

The local comparison selects raw/original for the complete trial. Reuse its
already-generated right view and generate only front/left:

```sh
$PY -m server.vhuman.reconstruction.mv_texture generate --work "$WORK" \
  --backend qwen_edit_seq --name selected --views front left --steps 12 \
  --portrait-mode raw --prompt-recipe original --edit-backend gguf \
  --edit-model-root "$MODEL" --offload-blocks 1
$PY -m server.vhuman.reconstruction.mv_texture compose --work "$WORK" --backend hybrid_selected \
  --spec 'front=selected:raw,right=recipe_raw_original:raw,left=selected:raw,back=mvadapter:view,top=mvadapter:view,bottom=mvadapter:view'
$PY -m server.vhuman.reconstruction.mv_texture bake --work "$WORK" --backend hybrid_selected \
  --parsing-model "$PARSER" --out "$WORK/out_selected"
$PY -m server.vhuman.reconstruction.mv_texture eval --work "$WORK" \
  --out "$WORK/out_baseline" "$WORK/out_selected"
```

Composition validates complete view coverage, source hashes where supplied,
and geometry/material provenance before creating its output. The bake keeps
photographed texels byte-identical and preserves geometry and existing maps.

## Baseline and acceptance

The local `out_baseline` reproduces b550's `out_hybrid8` byte-for-byte:

| Measurement | Baseline |
| --- | ---: |
| Seam score | 0.04674750722519876 |
| Unseen coverage | 0.9181410500779761 |
| Cross-view spread | 0.04674915157753333 |
| Photographed texels changed | 0 |
| Synthetic detail amplitude gain | 1.3178712129592896 |

Basecolor SHA-256: `2a10071223d688959ccec02ac58c116ca940864a93ffca3f6c72af96f3fa2c86`.
Geometry, coverage, confidence, normal and ORM hashes also match exactly.

Promotion requires visible improvement, no identity/camera/anatomical
hallucinations, unchanged geometry and photographed texels, coverage at least
equal to baseline, and seam score no more than 1% above baseline. Metrics alone
do not establish realistic skin. Retain the baseline if an experiment fails.

### Controlled right-view comparison (2026-10-08)

Only the right image is replaced for these bakes; front/left remain the old
Edit-2511 images and the structure views remain MV-Adapter. Every bake changes
zero photographed texels.

| Portrait / prompt | Edit seconds | Seam score | Unseen coverage | Finding |
| --- | ---: | ---: | ---: | --- |
| Raw / original | 429.41 | 0.04731976 | 91.9188% | No collar; gray scalp patch; lowest error among new recipes |
| Raw / bare-skin | 517.47 | 0.04736951 | 91.9181% | Similar appearance; no improvement from the added wording |
| Matted / original | 560.83 | 0.04754613 | 90.2521% | Invented suit collar and fuller hair; 12.48% of right view rejected |
| Matted / bare-skin | 498.42 | 0.04753057 | 90.2565% | Collar persists; 10.16% of right view rejected |

Times include CPU prompt encoding; raw/original reuses the smoke-test encodes.
The matted variants fail the coverage and visual gates. Both raw variants
exceed the seam threshold (baseline × 1.01 = 0.04721498). Raw/original is selected
for a complete face-view trial, not promoted as a final material at this stage.

Seam evaluation now indexes photographed/generated colors once before sampling
the boundary, instead of copying the entire atlas on each of 3,000 samples.
The baseline and raw/original scores are bit-identical to the previous
implementation; scoring takes approximately one second per atlas here.

For browser texture review, use a copy of `offline/fit12` with its candidate
path rebound to the hash-matching local candidate. Disable its older authored
hair undercoat and hair cards in that review copy so they do not obscure the
completed scalp texture. Preserve anatomical geometry and bindings, and record
the review-scene changes separately. Build with the existing mobile exporter
and browser builder and the pinned Three.js 0.163.0 dependency. Set `EM_CACHE`
to a repository-local copy of the Emscripten cache when the SDK is read-only.

```sh
$PY -m unittest server.vhuman.test_mv_texture
$PY -m unittest server.vhuman.test_generated_skin server.vhuman.test_photoreal \
  server.vhuman.test_quality server.vhuman.test_reconstruction \
  server.vhuman.test_mobile_preprocess server.vhuman.test_mobile
$PY -m server.vhuman.mobile.browser_verify --player PLAYER --out CHECK --hardware
```

The browser verifier expects the existing dynamic-detail assets: run
`server.vhuman.mobile.preprocess` for the exported package and pass its output
as `mobile.browser --detail`. This retains the existing authored wrinkle prior;
it is not a new appearance estimate. Keep the verification output path short
(for example `tmp/vht-b2`) to stay within Chromium's Unix socket length limit.

The combined local regression suite ran 108 tests successfully, with one
optional Torch/OpenEXR test skipped because OpenEXR is absent. The baseline hardware browser check passed with
the NVIDIA ANGLE/Vulkan renderer, zero native/WASM vertex error, relighting,
dynamic detail, inspection cameras, audio sample timing and cancellation.
It measured about 30 FPS while the diffusion experiment was also running.

The offloaded two-step smoke test completed in 256.79 seconds, including
161.81 seconds for CPU prompt encoding. Peak PyTorch allocation was
2,073,151,488 bytes and peak reservation 2,629,828,608 bytes. The first full
12-step raw/original edit reused that prompt cache and took 429.41 seconds;
steady denoising was about 35.6 seconds per step.

## Final decision and review artifacts

**Retain `out_baseline`; no new material was promoted.** Raw/original front and
left edits introduced a suit/tie and collar respectively. Clothing rejection
reduced coverage and did not remove all clothing-colored patches from the bake.
Keeping the old front also failed to recover baseline quality:

| Candidate | Seam score | Unseen coverage | Cross-view spread | Photo texels changed |
| --- | ---: | ---: | ---: | ---: |
| Reproduced `out_hybrid8` baseline | 0.04674751 | 91.8141% | 0.04674915 | 0 |
| New front/right/left + MV-Adapter | 0.05118661 | 82.3465% | 0.06756530 | 0 |
| Old front + new right/left + MV-Adapter | 0.04887352 | 90.6688% | 0.07069194 | 0 |

All candidates preserve the source geometry, portrait, coverage, confidence,
normal and ORM files byte-for-byte. The new bakes fail both seam and coverage
gates. The baseline still has ear halos, scalp tone transitions and photographed
projection artifacts; preserving photographed pixels necessarily retains those
artifacts. Realistic texture completion remains unfinished.

Artifacts are under `tmp/vhuman-texture-5060ti/`:

- `index.html`, `final_decision.json`: decision, metrics and review links.
- `work/recipe_comparison.html`: four matched editor outputs and recipes.
- `work/eval.html`: baseline, four right-only bakes and both complete trials.
- `player_final/`: retained baseline viewer, with corrected hybrid provenance.
- `player_sides/`: rejected side-only trial for interactive comparison.
- `source-weights.sha256`: verified remote weight hashes.
- `final-tests.log`: 108 tests, one optional OpenEXR skip.
- `final-texture-tests.log`: 12 texture tests pass after the scoring optimization.

Serve the review directory with:

```sh
python3 -m http.server 8796 --bind 127.0.0.1 --directory tmp/vhuman-texture-5060ti
```

The final browser package was exported from `work/out_baseline` using
`texture_scene`, preprocessed with the existing dynamic-detail defaults and
built with Three.js 0.163.0. The source review scene retains its geometry and
bindings but disables the older hair cards/painted undercoat. Its changes are
recorded in `texture_scene/texture_review.json`.

```sh
export EM_CACHE="$PWD/tmp/vhuman-texture-5060ti/emscripten-cache"
ROOT=tmp/vhuman-texture-5060ti
$PY -m server.vhuman.mobile.export --candidate "$ROOT/work/out_baseline" \
  --scene "$ROOT/texture_scene" --out "$ROOT/mobile_final"
$PY -m server.vhuman.mobile.preprocess --candidate "$ROOT/work/out_baseline" \
  --package "$ROOT/mobile_final" --out "$ROOT/detail_final"
$PY -m server.vhuman.mobile.browser --package "$ROOT/mobile_final" \
  --detail "$ROOT/detail_final" --three "$ROOT/vendor/package" --out "$ROOT/player_final"
$PY -m server.vhuman.mobile.browser_verify --player "$ROOT/player_final" \
  --out tmp/vht-f0 --hardware
```

Use fresh output paths when repeating these commands. The final hardware check
(`tmp/vht-f0/verification.json`) passed on NVIDIA ANGLE/Vulkan after diffusion
finished: 29.84 display FPS, 30.05 animated FPS, zero native/WASM vertex error,
48,000 audio samples with zero underruns, and passing relighting, detail,
inspection-camera and cancellation checks. This validates the desktop review
package, not mobile-device performance or visual realism.
The separate side-only comparison viewer also passed the same hardware checks
(`tmp/vht-s0/verification.json`).

## Reusing a completion after bounded geometry fitting

A fresh portrait bake can reuse a previous completion as an appearance prior
when its skin topology, UVs and source portrait are identical. This does not
rerun generation or establish multiview support on the new geometry:

```sh
TMPDIR="$PWD/tmp" tmp/vhuman-texture-venv/bin/python \
  -m server.vhuman.reconstruction.transfer_skin \
  --candidate tmp/vhuman-quality8h/lip_candidate_ramp_patch \
  --prior tmp/vhuman-blender/out_balanced \
  --out tmp/vhuman-quality8h/completed_anatomy \
  --feather-mm 6 --maximum-displacement-mm 3
```

Use an empty output directory. The new bake's photographed texels remain exact;
old photographed texels and unsupported old texels are excluded from transfer.
The output records both geometry hashes, source completion/material/mask hashes,
and a transfer mask. Old multiview coverage/consistency scores are not inherited.
The original completion's license continues to apply. Review the transferred
material in matched views before promotion; unchanged UV correspondence does
not guarantee tone or shading quality.
