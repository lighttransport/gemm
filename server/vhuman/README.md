# Independent procedural virtual humans

For RX 9070 XT / RDNA4, see [ROCm setup and validation](ROCM.md).

A local eye/head viewer with synthetic procedural textures, analytic refraction,
portable glTF export, portrait-based eye fitting and optional Qwen/Pixal3D jobs.
The runtime requires no Unreal Engine installation, source, content or measured
character data. Engine-specific import/export adapters are not included.

## Portrait expression videos

The head page can queue silent expression clips through the experimental
[repository HunyuanVideo-1.5 CUDA runner](../../cuda/hunyuan_video15_native/).
The default `--video-backend repo` selects repository GEMM and rejects cuBLAS
fallback. Build with `make -C cuda/hunyuan_video15_native`, stage the pinned model
assets, then launch with `--video-model DIR --video-experimental`. The standalone
`video` CLI uses an existing head portrait. This backend currently supports 81
frames. A complete fast12 image-to-video run passed numerical parity and the
16 GB memory budget on RTX 5060 Ti; expression timing remains experimental.
The runner uses public Google SigLIP; FLUX.1-Redux is not used.

The 2026-10-02 validation used 480×848, 81 frames at 24 fps, seed 42 and
12 denoising steps. All encoders, all denoising intermediates, the final latent
and all 81 decoded frames passed independent references (FP32 text/vision
encoders, official FP16 DiT/VAE). Final latent relative L2 was `0.002146`, decoded
RGB `0.003141`, and the worst frame `0.006661`. This prompt contains no quoted
text, so ByT5 used its empty-text branch. The run made 6,068,402 repository-GEMM
calls with zero cuBLAS/fallback calls. Sampled peak process VRAM was 3,108 MiB
and host RSS 16,237 MiB. Wall time was 3,690 seconds; this was a validation run
with two short concurrent encoder probes, not an isolated performance benchmark.

The full independent reference was first accepted for a matching cuBLAS baseline.
`ref/vhuman/verify_hunyuan_backend.py` then checked both generation receipts,
model hashes, recipe, byte-identical noise and prepared images before comparing
the strict-GEMM run directly with those saved reference tensors. This comparison
requires no Torch. Reports, manifests, a contact sheet and the completed video
are under `tmp/hv15-integration-review/`. Reproduce the final comparison with:

```sh
tmp/mhr-native/runtime/bin/python ref/vhuman/verify_hunyuan_backend.py \
  --candidate-run tmp/hv15-integration-review/full-repo-run \
  --candidate-actual tmp/hv15-integration-review/full-repo-captures \
  --baseline-run tmp/hv15-integration-review/full-fast12-v5/run \
  --baseline-actual tmp/hv15-integration-review/full-fast12-v5/actual \
  --baseline-reference tmp/hv15-integration-review/full-fast12-v5/reference \
  --output tmp/hv15-integration-review/full-repo-parity.json
python -m unittest ref.vhuman.test_hunyuan_backend
```

The generated face and framing remain stable, but “smiles gently, then relaxes”
produces a sustained broad smile. Native landmarks detected a face in all 81
frames; mean left/right smile scores were 0.009 initially, 0.975 at peak and
0.963 at the end. These are diagnostics, not perceptual acceptance. Quality50
T2V/I2V profiles, quoted-text rendering and the wider expression matrix remain
unvalidated; this fast12 result does not promote those profiles.

The explicit `--video-backend legacy` retains the
[ggml-based runner](../../cuda/hunyuan_video15/README.md) and its 81/121-frame
interface. The following measurements apply **only to that legacy backend**:
a fast12/81-frame
smoke test on RTX 5060 Ti 16GB measured 12,624 MiB peak VRAM. All 12 denoising
steps passed the official FP16 reference with saved native conditioning/noise
(final latent relative L2 0.002447). The corrected spatial VAE tiling and IEEE
FP32 SigLIP path also passed a full fast12/81 portrait pipeline check against
independently computed official component outputs: decoded relative L2
0.001776, all 81 frames passing. That run peaked at 13,300 MiB VRAM and took
32.1 minutes. Its brief blink prompt gives shorter closures, but requesting
one blink still produces two; count and natural timing remain experimental.
This validates the assembled reference at native encoder precision. Other
profiles, upstream all-FP16 equivalence and the expression matrix remain under
validation. Generated clips
are previews and are not automatically used for rig fitting or training.

The reference tools now accept fast12 121-frame captures and an explicit
`--encoder-dtype float16`. A coverage audit requires three portraits, six
expressions, two seeds, two presets and both lengths (144 cases). Missing runs
remain missing; numerical parity does not certify visual expression quality.
See [validation commands](../../ref/hunyuan_video15/README.md).

## Image-guided reconstruction candidates

Refine a fitted head without replacing its accepted rig:

```sh
python3 -m server.vhuman.cli --work tmp/vhuman-independent \
  rig-refine-portrait --head HEAD_ID --profile full --gaussians 2000
```

`portrait-reconstruct --portrait FILE --face-model gnm_v3` constructs a direct
portrait candidate using native face anatomy and analytic eyes. Both commands
support manual `--observations`, artist `--roughness`/`--f0` priors, and optional
`--depth-installation`. Use `rig --head HEAD_ID --reconstruction-run RUN_ID` to
rebuild a candidate. Outputs live in `heads/HEAD_ID/reconstruction/RUN_ID/`.
The rig viewer offers candidate selection, optional diffuse skin scattering,
and a static RGB Gaussian overlay. New candidates rebuild contacts and source
LODs; prior compact soft-deformer weights are not reused.

Known anchors use topology-pinned ICT anatomy and an eye-aligned, lip-group
constrained GNM transfer. Manual `vertex` or `vertices` overrides remain available.
Observation views can include hash-checked `silhouette_mask` (white foreground)
and `exclusion_mask` (white occlusion) at the original image dimensions. Facial
LODs retain feature detail, average UVs within charts and preserve source normals;
hidden skin uses bounded surface-space completion rather than UV-nearest copying.

Evaluate independent annotated views without silently reusing fitting images:

```sh
tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.evaluate \
  --candidate tmp/vhuman-independent/heads/HEAD_ID/reconstruction/RUN_ID \
  --observations path/to/heldout.json --out tmp/reconstruction-evaluation/RUN_ID
```

Reports include landmark RMS/IPD error, optional mask IoU/boundary F1 and
side-by-side diagnostics. `--surfaces` accepts externally predicted animation
frames; `--reference` accepts independent same-topology geometry and timestamps
for geometric/velocity/acceleration errors. `--allow-training-diagnostic` explicitly
labels fitting-image results. Public data sources are URLs in
`reconstruction/data/evaluation_sources.json`, require explicit downloads, and are
never bundled with the code.

### Download evaluation starters

The Linux/POSIX resumable downloader defaults to `/mnt/nvme02/data/vhuman` and rejects a
destination inside this repository. It selects the official two-expression
Multiface mini set, one SpeakingFaces RGB/audio utterance, and Digital Emily 2
geometry/maps/calibration/reference archives. It checks publisher MD5s where
available, records SHA256 receipts, preserves partial HTTP transfers and safely
extracts ZIP/TAR files. The default selection contains only these three public
datasets. Explicitly selected gated datasets are reported without creating accounts or
submitting access forms. Review the catalog's media terms before invocation:

```sh
python3 -m server.vhuman.reconstruction.download_datasets \
  --root /mnt/nvme02/data/vhuman --accept-research-terms

# Optional RAR extraction; unrar or 7z must also be installed:
UV_CACHE_DIR=tmp/uv-download-cache uv venv tmp/vhuman-download-venv
UV_CACHE_DIR=tmp/uv-download-cache uv pip install \
  --python tmp/vhuman-download-venv/bin/python rarfile==4.2
tmp/vhuman-download-venv/bin/python -m server.vhuman.reconstruction.download_datasets \
  --root /mnt/nvme02/data/vhuman --datasets emily --accept-research-terms
```

`--datasets` selects individual datasets; `--no-extract` retains archives only.
`--workers 1..4` controls concurrent Multiface transfers (default 2). Large
Multiface uses installed `aria2c` for parallel ranges (`--engine auto`); use
`--engine urllib` for the standard-library backend with bounded 8 MiB ranges
and validator checks. Both verify the publisher's checksum before extraction.
A destination lock
prevents concurrent invocations from writing the same partial files.
Rerunning verifies completed archives and resumes partial files. The external
root's `download_manifest.json` records completion/access failures; each dataset
also has its own manifest. Archives and extracted data are retained, so the
Multiface starter needs substantially more than its approximately 16 GB download.
A 4 GiB free-space reserve is enforced, with a 32 GiB limit per archive. Exit
codes are 0 for completed selections, 1 for transfer/validation failures and 2
when selected datasets still require approved access.

Public starter download validated on 2026-09-30 at `/mnt/nvme02/data/vhuman`:

| Dataset | Downloaded selection | Local usage including retained archives |
| --- | --- | --- |
| Multiface | Identity 6795937, E057/E061; all 8 official mini archives, publisher MD5 verified and extracted | 33 GiB |
| SpeakingFaces | Utterance `1_1_2_7_611`, 72 RGB PNG frames at 28 fps and original WAV | 21 MiB |
| Digital Emily 2 | OBJ, texture maps, calibration and four lighting reference RARs; all 7 archives extracted | 5.1 GiB |

SHA256 receipts and extraction completion records are stored beside these
external assets. This is a starter selection, not the full datasets. No
NeRSemble, NoW, FaceScape or FaceOLAT assets were downloaded. Public availability
does not change the catalog's dataset-specific media licenses.

For approved NeRSemble/NoW/FaceScape/FaceOLAT archive URLs, create a **private file
outside Git**, restrict it with `chmod 600`, and supply `--access-file FILE`:

```json
{
  "nersemble": [
    {"name": "approved_subset.zip", "url": "AUTHORIZED_HTTPS_ARCHIVE_URL",
     "sha256": "OPTIONAL_PUBLISHER_SHA256"}
  ]
}
```

Keys are `nersemble`, `now`, `facescape`, and `faceolat`. Omit `sha256` if unavailable,
or use `md5` for a publisher MD5. Provide actual authorized archive links, not login
pages or Google Drive HTML pages. Signed URLs are never written into manifests or
logs. This generic archive path does not convert dataset coordinates/topology or
replace the official access procedures.

### Prepare and evaluate downloaded starters

Use the existing rig interpreter and cached GNM/MediaPipe weights. Preparation
performs no download and writes generated media only outside the repository or
under ignored `tmp/`. Choose a fresh output directory for each run:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 tmp/vhuman-rig-venv/bin/python \
  -m server.vhuman.reconstruction.prepare_datasets \
  --root /mnt/nvme02/data/vhuman --out tmp/public-dataset-evaluation \
  --annotate-multiface --fit-speakingfaces

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 tmp/vhuman-rig-venv/bin/python \
  -m unittest server.vhuman.test_dataset_preparation server.vhuman.test_reconstruction
```

- **Multiface:** selects three tracked frames and three frontal cameras, with
  two cameras for fitting and one held out. Exports observation JSON and separate
  same-topology reference NPZ files, plus photo/reference-render previews. The
  original adapter validates OBJ/bin/head-transform agreement and converts
  calibrated cameras to metres and the renderer's camera convention. Horizontal
  and vertical focal lengths and skew are preserved through resizing, material
  baking, silhouettes and evaluation. Nonzero distortion is rejected.
- **SpeakingFaces:** fits frames 0/24/48/71 and evaluates disjoint frames 12/36/60;
  exports raw holdout, training diagnostic and pose-aligned mouth reports. These
  use estimated MediaPipe annotations and assumed intrinsics/IPD. Preparation
  needs the cached GNM model; `--fit-speakingfaces` enables the bounded 50-iteration
  fitting/evaluation pass.
- **Emily:** indexes the mesh, eight material/displacement EXRs, 28 reference
  EXRs and seven camera files, validating bounded EXR headers. It does not decode
  linear EXR pixels or estimate calibrated reflectance. Lens distortion,
  camera convention and light radiance/exposure need validation first.

The Multiface projection composition is checked against the
[publisher's loader](https://github.com/facebookresearch/multiface/blob/main/dataset.py).
Its reference geometry remains in the publisher's aligned head-local frame;
explicit alignment and anatomical correspondence are still required for GNM/ICT
scan error comparisons. Reference-render masks are labelled as such and are not
silently used as independent segmentation annotations. The captured texture is
appearance, not intrinsic albedo. Reference NPZ files must never be passed as
predicted surfaces and reported as reconstruction quality.

Earlier three-training-frame run `tmp/public-dataset-eval-02` (2026-09-30): nine calibrated Multiface
views, maximum projection difference **0.0064 pixels**, maximum tracked
OBJ/head-transform difference **0.000077 mm**; 30 affected numerical/import tests
passed. SpeakingFaces training landmark RMS was 4.69/4.32/4.32 pixels. Raw
holdout RMS was **14.63/20.52/15.74 pixels**; pose-aligned mouth RMS was
**3.72/13.46/8.71 pixels**. The middle frame remains a failure case, and these
estimated-annotation diagnostics establish neither scan likeness nor temporal
animation quality. All generated fixtures, fitted candidates and preview media
are ignored local artifacts; dataset media licenses still apply.

### Scan, animated holdout, material and learned-cue experiments

The next quality pass adds original offline modules; none changes the runtime rig
format or enables an experimental predictor automatically. All output paths are
restricted to an external directory or ignored `tmp/`.

```sh
# Once: official linear EXR decoder in the existing utility environment.
UV_CACHE_DIR=tmp/uv-download-cache uv pip install \
  --python tmp/vhuman-rig-venv/bin/python OpenEXR==3.4.4

# Requires the annotated Multiface fixture prepared above.
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 tmp/vhuman-rig-venv/bin/python \
  -m server.vhuman.reconstruction.scan \
  --prepared tmp/public-dataset-evaluation/multiface --out tmp/scan-evaluation

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 tmp/vhuman-rig-venv/bin/python \
  -m server.vhuman.reconstruction.emily --root /mnt/nvme02/data/vhuman \
  --out tmp/emily-material-evaluation

tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.learned_cues \
  synthesize --out tmp/cue-training/synthetic.npz
tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.learned_cues \
  train --dataset tmp/cue-training/synthetic.npz --out tmp/cue-training/model --steps 600
tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.learned_cues \
  validate-real --checkpoint tmp/cue-training/model \
  --prepared tmp/public-dataset-evaluation/multiface \
  --candidate tmp/scan-evaluation/pca --out tmp/cue-training/real-evaluation

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 tmp/vhuman-rig-venv/bin/python \
  -m unittest server.vhuman.test_quality server.vhuman.test_dataset_preparation \
  server.vhuman.test_reconstruction server.vhuman.test_app
```

**Scan correspondence:** fitting-camera annotation rays locate scan-surface
anchors; cross-camera agreement gates metric similarity alignment. Cameras remain
fixed, and simultaneous fitting views share one expression state. Dense
correspondence stores reference triangle IDs, barycentrics and confidence.
Normal/distance checks, a sparse Laplacian, protected landmarks, a tapered facial
ROI and a 2 mm displacement cap control correction. Exact point-to-triangle
distance is evaluated over fixed facial sample IDs; it is not nearest-vertex
distance. Camera holdouts and the distinction between scan fitting and unseen
expression evaluation are explicit. The alignment/correction frame is marked in
each geometry result. This is research evaluation, not a released training asset.

**Animated holdout:** `sequence.predict` interpolates captured training geometry
and quaternion pose, using only target timestamps. It rejects extrapolation,
checks topology safety and writes predicted NPZ surfaces independently of
references. The updated SpeakingFaces preparation brackets the holdout with frame
71 and emits `animated-held-out/report.json`. This is a deterministic interpolation
baseline, not an audio-conditioned generator or held-out landmark optimization.

**Material and texture polish:** Emily's OBJ uses negative relative indices;
the loaders resolve them when each face is read. Skin is selected by material,
avoiding eyes/lashes. Linear EXRs retain negative values in recorded samples;
clipping is limited to fitting and previews. Camera-to-world/-Z projection with
centimetre scale and forward radial/tangential distortion is checked visually.
Coarse visibility bounds the reference processing. Regional specular/diffuse
ratio fits use two cameras and score the third, with independently optimized
baseline gains. The absolute flash power, exposure and polarization gains are
unknown: these are **relative roughness diagnostics with fixed artist F0**, not
calibrated physical F0/SSS recovery. The official
[Emily reference source](https://vgl.ict.usc.edu/Data/DigitalEmily2/) and
[OpenEXR decoder API](https://openexr.com/en/latest/python.html) are the references.

For observations with supplied multi-light radiance/exposure calibration,
`--spatial-materials` on `portrait-reconstruct`/`rig-refine-portrait` enables four
UV-region GGX fits, retaining priors in rejected regions. Robust weighted-median/
Huber linear-color fusion suppresses conflicting camera samples. Optional
`--detail-um 5` bakes metric-space authored normal detail on the actual imported
topology, with a common world-space field and atlas-footprint frequency limit.
It changes neither geometry nor animation cost; it is an artist prior and does
not recover portrait pores. Zero detail remains the default. Geometry changes
still require the existing deformer rebuild. `--auto-exclusions` optionally
filters projected skin samples using robust cheek chromaticity and darkness,
preserving lip colors and estimated brow bands. It runs after geometry fitting,
manual masks take precedence, and its derived masks are explicitly not ground
truth. Held-out quality scoring retains its original annotations/masks.

**Learned cues (v3):** the original 19,876-parameter PyTorch model predicts a bounded
normal residual around a pose/crop-matched rendered geometry prior, plus a mask
probability. The v3 residual is capped at 0.15 per component before normalization.
Its 32 synthetic views contain eight independent GNM identities, our NumPy/GGX
renderer, procedural albedo, random poses/lights, white balance, exposure, noise
and backgrounds. Identities 0–5 train the
model; 6–7 are held out. No real dataset images, restricted checkpoints or their
labels enter optimization. The checkpoint uses our existing safetensors writer;
code/model hashes, seeds, identity splits and gates are recorded. Adoption must
beat both the pose-matched synthetic geometry prior and a frozen real scan comparison
against rendered GNM. Predicted confidence cannot hide normal errors because the
evaluation mask is fixed by the reference. Mask probability is not calibrated
normal uncertainty. Learned cues remain experimental/off by default, including
after a synthetic gate passes.

First-pass quality artifacts (2026-09-30; cue v2 is a historical experiment):

| Experiment | Local run | Result |
| --- | --- | --- |
| Fixed-camera GNM scan fit | `tmp/vhuman-quality-scan-04` | PCA RMS 4.41/4.38/4.36 mm → dense 3.76/3.72/3.95 mm; protected landmark RMS unchanged |
| SpeakingFaces interpolation | `tmp/vhuman-quality-speech-final` | Neutral holdout RMS 14.55/20.45/15.69 px → animated 4.69/8.22/8.03 px; target annotations unused |
| Emily relative material fit | `tmp/vhuman-quality-emily-03` | 9 linear EXRs; 3,240/3,384/2,635 visible samples; two of four regions accepted, two retain priors |
| Synthetic residual normal predictor | `tmp/vhuman-quality-cues-02/trained` | 12.46° mean held-out error vs 16.00° mean-template baseline; 600 CPU steps |
| Frozen real normal comparison | `tmp/vhuman-quality-cues-real-02` | Learned 38.85/38.70/37.77° vs rendered GNM 20.59/20.62/19.60°; real gate failed, predictor remains disabled |
| Optional baking exclusions | SpeakingFaces `quality03` | Four views exclude 2,585–3,660 pixels each; robust fusion downweights 2,203 conflicting observations; masks remain heuristic |

The regression command above passed all 53 tests in 60.1 seconds. Its log is
`tmp/vhuman-quality-regressions-final.log`. Generated media, fitted assets and
checkpoints remain outside tracked source files.

These are small research pilots. Scan evaluation covers one identity and supplied
tracked surfaces, not independent manual anatomy. Speech interpolation misses
rapid unsampled articulation. Relative Emily fits do not establish physical
material recovery. Synthetic improvement alone never enables learned geometry
correction or a production training claim.

### Broader evaluation and calibration pass

The second pass uses frozen, training-only geometry for an unseen expression and
adds uniform-area **bidirectional** point-to-triangle scoring. It does not align
predictions to held-out scans or claim to predict unseen expression controls.
Emily is prepared separately as a second subject from its undistorted linear
cross-polarized references; preview exposure normalization is not material
calibration. Detector annotations remain estimated. For independent landmarks
and skin outlines, the photo-only annotation kit starts blank and excludes all
model/detector overlays; it still requires human annotation.

```sh
tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.prepare_datasets \
  --root /mnt/nvme02/data/vhuman --out tmp/cross-expression-target \
  --datasets multiface --expression E061_Lips_Puffed
tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.benchmark \
  --scan tmp/vhuman-quality-scan-04 --target tmp/cross-expression-target/multiface \
  --out tmp/cross-expression-evaluation
tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.prepare_emily \
  --root /mnt/nvme02/data/vhuman --out tmp/emily-scan-fixture
tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.scan \
  --prepared tmp/emily-scan-fixture --out tmp/emily-scan-evaluation
tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.annotation_review \
  --observations tmp/cross-expression-target/multiface/held-out.json \
  --out tmp/independent-annotations

# Existing local Japanese clips only; no corpus download happens here.
tmp/vhuman-rig-venv/bin/python -m speech.ref.run_motion_benchmark \
  --clips tmp/reazon-motion-pilot --work tmp/vhuman-independent \
  --head 650ac67354cd --out tmp/japanese-motion-evaluation \
  --backend cuda --limit 12 --tts
```

The Japanese runner produces source-WAV and TTS takes with the production
aligner, animation JSON, LightRig and USD tracks, hashes, consistency diagnostics
and a human review CSV. ReazonSpeech has no synchronized facial ground truth:
closure scores reuse alignment and cannot establish independent lip-sync timing
or perceptual quality. No ratings are invented and the perceptual gate stays
false. These previously downloaded local clips are not redistributed or added
to the repository; no new gated dataset download is performed.

**Measured materials:** `reconstruction.calibration --capture capture.json
--out tmp/measured-lighting.json` recovers the linear RGB radiance gauge from
measured Lambertian gray-card reflectance, card normal, light direction,
exposure and independently estimated ambient. Supply three or more EXR views in
`vhuman.gray_card_capture.v1`: top-level `reflectance` (RGB, 0..1) and `views`
with `image`, `sha256`, integer `card_roi` `[x0,y0,x1,y1]`, `card_normal_h`,
`direction_h`, `exposure` and `ambient_rgb`. ROI samples must be uniform and
non-grazing. Copy each resulting `lighting` object to the corresponding portrait
observation to use the existing calibrated GGX fitting gate. Measurements from a
separate card capture must use the same illumination and exposure as the portrait.
The RGB gauge is not absolute lamp power in watts; near-field lights need a
position-dependent model. Ordinary Emily references lack these measurements, so
absolute F0 remains unresolved there.

`reconstruction.calibration --line-scan measurement.npz --out tmp/scattering.json`
fits effective Gaussian RGB scattering widths from measured skin distances
`distance_m` and background/illumination-corrected `linear_rgb`. A held-out sample
split gates each channel; widths remain disabled by default. This approximates a
renderer profile and does not recover tissue layers or volumetric BSSRDF.
The narrow-light footprint must be measured/deconvolved before interpreting
widths as skin scattering. Controlled synthetic recovery tests validate the
implementation; no measured real capture has been supplied.

**Texture completion:** surface-neighborhood harmonic completion keeps measured
colors exact, limits edges to 4 mm and normal agreement above 0.7, and limits
completion to 20 mm from observations. Distant regions retain a median prior.
`skin_completion_confidence.png` labels proximity-based completion separately
from measured `skin_coverage.png`; it is a heuristic, not recovered texture or
uncertainty calibration. This does not improve hair/glasses segmentation itself.

**Cue v3:** synthetic train/test identities remain disjoint; training never uses
real capture media. Real inference uses the same pose-matched geometry prior as
its baseline and a crop derived solely from the fitting ROI. Standalone `infer`
requires `--prior` pointing to an NPZ with a unit `normals[64,64,3]` map matched
to the RGB crop. Earlier v2 checkpoints are intentionally rejected. V3 must
improve the stronger prior by 5%, then pass independent real-subject validation;
passing a small pilot never enables it by default.

Second-pass validation (2026-09-30):

| Experiment | Local artifact | Result |
| --- | --- | --- |
| Unseen Multiface expression | `tmp/vhuman-cross-expression-01/report.json` | Symmetric RMS 5.24/5.56/5.29 mm → 4.65/5.20/5.03 mm; no target fitting |
| Japanese production speech | `tmp/vhuman-japanese-motion-01/report.json` | 12 WAV + 12 TTS takes; track consistency passes; 19 source and 24 TTS aligned bilabial events reach closure ≥0.7; 8 source-WAV takes have alignment warnings; no TTS warnings |
| Second scan subject (Emily) | `tmp/vhuman-quality-emily-scan-01/report.json` | One-way PCA 6.71 mm → dense 6.15 mm; landmarks remain 11.98 px; bidirectional 13.05 → 12.68 mm fails the stronger 5% gate; estimated anchors, neutral scan pilot |
| Surface texture completion | `tmp/vhuman-quality-texture-02/skin_material.json` | 23,616 measured texels retained, 8,456 nearby samples completed; evidence coverage unchanged; measured albedo pixels identical; held-out appearance diagnostic 0.09350/0.09414/0.08471 → 0.09343/0.09409/0.08463 |
| Cue v3 synthetic | `tmp/vhuman-quality-cues-03/trained/report.json` | 9.35° vs pose-matched prior 9.57°; fails the 5% gate |
| Cue v3 real, fixed fitting ROI crop | `tmp/vhuman-quality-cues-real-04/report.json` | Learned 21.42/21.40/19.05° vs prior 21.82/21.79/19.07°; mask IoU 0.69–0.71; real gate fails, stays disabled |

The API/speech regression run passed all 67 affected tests in
110.8 seconds (`tmp/vhuman-quality-round2-regressions.log`). After the final
normal-compatibility safeguard, all 47 numerical/dataset/reconstruction tests
passed in 4.1 seconds (`tmp/vhuman-quality-round2-final-numerical.log`).
Compile and diff checks also passed. Primary sources for
capture conventions and missing measurement context are the
[Multiface release](https://github.com/facebookresearch/multiface) and
[Digital Emily 2 release](https://vgl.ict.usc.edu/Data/DigitalEmily2/).

See the [implementation record and setup](../../doc/vhuman-face-reconstruction-adoption-plan.md)
for API uploads/jobs, observation semantics, model pins and validation limits.


## Model and measurement inputs

The eye is a smooth union of a sclera sphere and an offset corneal sphere, with
an opaque iris plane behind it. The limbus determines the spheres' intersection.
Defaults are **synthetic demonstration choices**, not a clinical specification
or measurements of a licensed character:

| Input | Default | Unit |
| --- | --- | --- |
| Sclera radius | 0.012 | metres |
| Limbus radius | 0.006 | metres |
| Cornea curvature radius | 0.008 | metres |
| Junction blend width | 0.0005 | metres |
| Chamber depth | 0.0035 | metres |
| Refractive index | 1.336 | dimensionless |

Users or an LLM may choose different inputs through the same validated JSON
schema. Non-finite inputs and impossible radius combinations are rejected.
For example, save this as `tmp/my-eye.json`:

```json
{
  "optics": {
    "sclera_radius": 0.014,
    "limbus_radius": 0.0065,
    "cornea_radius": 0.009,
    "limbus_blend": 0.0003,
    "chamber_depth": 0.0038,
    "ior": 1.34
  }
}
```

```sh
python3 -m server.vhuman.cli eye --params @tmp/my-eye.json --res 1024 --out tmp/my-eye
```

Physical inputs control standalone eye meshes, CPU rendering and browser
uniforms. Automatic portrait head fitting currently uses the synthetic default
profile for scale, carving and lids. Custom standalone eye measurements are not
silently applied to head fitting. Skin controls use the separate `--skin` JSON.
Keep the provenance of any measurements you supply; the application cannot
establish redistribution rights for arbitrary user-provided values or images.

## Independent implementation

- Texture coordinates use two linear angular intervals: the limbus maps to UV
  radius 0.2 and the back pole to 0.7. These are atlas packing choices. No
  measured UV table is imported, embedded or reconstructed.
- Geometry uses sphere intersections and a smooth union. Refraction uses
  Snell's law; reflection uses Schlick's approximation with F0 derived from
  the selected refractive index. Iris diffuse lighting is Lambertian.
- Iris appearance uses original spectral-noise fibres, crypts, spots and
  furrows, two-colour blending, neutral occlusion, a soft pupil and an outer
  ring. There is no fitted caustic term or external material-graph replica.
- Sclera vessels are original branching random walks. Synthetic optical
  density produces their colour. Lid occlusion is a Gaussian falloff from
  the portrait's eye opening; wet margins are original ribbon meshes.
- The CPU and GLSL implement the same mapping, pupil warp and albedo equations.
  The portable glTF eye bakes albedo and uses standard transmission materials.
- Eye-aligned skin masks protect lips and dark brows. Pores/freckles are
  deterministic object-space fields. Optional `delight_strength` reduces
  broad portrait shading; it does not recover ground-truth albedo.

## Running locally

```sh
sh server/vhuman/run.sh
# http://127.0.0.1:8790/ and /head
python3 -m server.vhuman.cli --mock head --subject "test portrait"
python3 -m server.vhuman.cli head-fit --portrait portrait.png --glb pixal3d.glb
python3 -m server.vhuman.cli render --preset blue --out tmp/eye-preview.png
```

The default work directory is `tmp/vhuman-independent/`. Previous development
exports under `tmp/vhuman/` are legacy local artifacts and are not reused or
redistributed. Regenerate meshes and textures rather than relabel old exports.
All generated work remains outside Git. Qwen and Pixal3D jobs use the existing
local GPU backends and lock; CPU eye/head operations do not need those models.
The server binds to loopback by default and is a local development tool.

Exports include `eye.glb`, a neutral texture ZIP, `params.json` and
`measurements.json`. Texture normals use the OpenGL +Y convention. API export
formats are `glb` and `textures`; the result keys are `glb` and `textures_zip`.
There is no engine-specific parameter schema or asset-measurement command.

For generated portrait heads the pipeline detects eye openings, fits procedural
eyes, carves/drapes the lids, adds lining/wet margins/occlusion, and bakes skin.
The head viewer supports analytic and portable eye rendering. Source portraits,
raw generated heads, fit metadata and maps remain beside each exported GLB.

## Facial rig

`/rig` and `python3 -m server.vhuman.cli rig --head <id>` turn a fitted head
into a rigged face (GNM v3 topology by default, optional ICT-FaceKit Light or
procedural topology, a jaw/eye/teeth/tongue skeleton, expression shapes, a
linear rig evaluator, skinned glTF and UsdSkel for LightUSD's vchar and
LightRig). It needs the interpreter in
`requirements-rig.txt`. See [rig/README.md](rig/README.md).

## Tests and limitations

```sh
python3 -m server.vhuman.test_all
```

Optional reconstruction tests need `server/vhuman/requirements-remesh.txt`.
Browser tests use Chrome/SwiftShader and report skips if dependencies are
unavailable. Tests cover finite geometry, UV round trips, user measurements,
cache invalidation, materials, API behavior and both viewers.

Single-view reconstruction can retain folded lids, uneven canthi and socket
layer gaps. These are accepted fitting limitations, not guarantees of anatomical
accuracy. Hair/beard segmentation is incomplete. Relighting cannot uniquely
separate complexion from cast shadows. See [QUALITY_PLAN.md](QUALITY_PLAN.md)
for the current references and validation.

See [Portrait to 3D face reconstruction and skin materials](../../doc/vhuman-face-reconstruction-research.md)
for the research comparison, model and code license distinctions, and proposed
mesh/PBR and Gaussian development paths.
The [adoption plan](../../doc/vhuman-face-reconstruction-adoption-plan.md) maps
these ideas to staged implementation, permissive dependencies, original
algorithms, artifact compatibility, and validation gates.

## Native MHR decoding

Body assembly and uploaded-motion retargeting use the standalone C MHR runner.
These stages require NumPy, SciPy and Pillow, but do not import PyTorch, ONNX or
ONNX Runtime. They consume an existing facial rig; image generation, portrait
preparation and facial-rig training still have their existing dependencies.
This is the first step of the native migration, followed by portrait depth,
RMBG/MoGe preparation, and the remaining rig/realtime inference modules.

Build from the repository root:

```sh
make -C cpu/sam3d_body mhr_decode
make -C cuda/sam3d_body mhr_decode test_mhr_compile
cuda/sam3d_body/test_mhr_compile
```

The CPU executable dispatches pose-corrective projection to repository AVX2
GEMM when available, with a portable C fallback. The CUDA executable uses our
FP32 GEMV and blend kernels through cuew/NVRTC, keeps weights resident across
frames, and does not link or load cuBLAS. Host skeleton transforms retain the
existing FP64 accumulation. CUDA full decoding allocates about 658 MiB of model
and scratch device buffers, excluding the CUDA context/compiler. Actual peak
VRAM still needs measurement on the target GPU.

Assembly follows the configured CUDA/CPU backend; ROCm configurations currently
use native CPU MHR. Motion retargeting uses the native CPU skeleton-only path
and skips all vertex decoding. Reports identify the actual decoder backend.
MHR outputs retain centimetres and xyzw quaternions; assembly converts to metres.

### One-time asset migration

Existing `sam3d_body_mhr_jit.safetensors` and `.json` exports need two small
companions: `sam3d_body_mhr_jit_rig.safetensors` and `_rig.json`. They contain
joint names, original skin weights/indices, parents and provenance hashes.
Generate them with an **offline export interpreter** containing Torch and
safetensors (substitute your model directory):

```sh
MHR_EXPORT_PYTHON=tmp/qimg21-ref-venv/bin/python
"$MHR_EXPORT_PYTHON" ref/sam3d-body/dump_mhr_assets.py \
  --mhr-model /path/to/sam3d-body/dinov3/assets/mhr_model.pt \
  --out-dir /path/to/sam3d-body/safetensors --rig-only
```

For a new export, omit `--rig-only`. Runtime requires only the exported assets;
there is no automatic TorchScript fallback or checkpoint download. The body
job resolves assets from its configured SAM3D Body directory. Standalone
assembly and motion commands accept `--mhr-assets DIR`. Legacy `--model` only
locates the standard sibling safetensors directory; it never opens the `.pt`.

The native executable accepts `--mhr-assets DIR --params POSES.npy --shape
SHAPE.npy [--face FACE.npy] --output-dir DIR`, plus `--backend cpu|cuda`,
`--device N`, `--threads N` and `--skeleton-only`. Input files are little-endian,
C-order float32 NumPy v1 matrices: pose `[B,204]`, identity `[B,45]`, optional
face `[B,72]`; batch sizes must match. Output files are `vertices.npy`
`[B,18439,3]` and `skeleton.npy` `[B,127,8]`. The skeleton-only mode omits vertices.
The Python adapter also broadcasts a single `[45]` identity across frames.

### Validation

Generate the offline oracle and compare a selected native runner:

```sh
"$MHR_EXPORT_PYTHON" ref/sam3d-body/verify_native_mhr.py \
  --model /path/to/sam3d-body/dinov3/assets/mhr_model.pt \
  --assets /path/to/sam3d-body/safetensors \
  --runner cpu/sam3d_body/mhr_decode --backend cpu \
  --pose-sidecar /path/to/body_mhr.glb.json --out tmp/mhr-reference
make -C cpu/sam3d_body verify_mhr_stages
cpu/sam3d_body/verify_mhr_stages \
  --mhr-assets /path/to/sam3d-body/safetensors --refdir tmp/mhr-reference/stages -t 4
MHR_TEST_ASSETS=/path/to/sam3d-body/safetensors MHR_TEST_REFS=tmp/mhr-reference \
  python -m unittest server.vhuman.body.test_mhr server.vhuman.body.test_body server.vhuman.body.test_motion
```

For the CUDA parity run, change `--runner` to `cuda/sam3d_body/mhr_decode` and
`--backend` to `cuda`. Set `MHR_REQUIRE_FRAMEWORK_FREE=1` when running the unit
suite in a NumPy/SciPy/Pillow-only interpreter. No CUDA device is needed for
`test_mhr_compile`; its success establishes compilation, not numerical parity.

Validation on 2026-10-02: CPU full/skeleton decoding passed batches 1/4/5,
including nonzero identity and facial coefficients. Maximum vertex error was
`1.06812e-4 cm` (about 1.1 micrometres); all seven existing stage checks passed.
The framework-free suite passed 16 tests, including malformed inputs, invalid
asset indices, missing metadata and cancellation of native child processes.
Assembly exported GLB/USD with 127 body joints, 12 face joints, 604 morph targets
and both garments accepted. Body skin indices/weights matched the prior avatar
exactly; five-frame motion quaternion error was below `1.2e-7 rad`.

CPU full decoding took roughly 0.3 s/frame; five-frame skeleton decoding took
about 1.8 ms, excluding process startup and asset checksum verification. Both
CUDA sources compiled for sm_120. CUDA numerical parity and peak-memory checks
remain pending because this validation environment had no `/dev/nvidia*` devices.

## Native Depth Anything V2 Small

The optional portrait-depth stage now runs without Torch, ONNX, or ONNX Runtime.
It reuses the repository DINOv2 backbone and DA3 DPT convolution, fusion and
resize operations, with DA2-specific feature taps (blocks 2/5/8/11), tensor-name
mapping and ReLU output. FP32 dense layers and tiled im2col convolutions use
repository GEMM. Relative inverse depth is returned at the original image size;
the existing alignment rejection gate and protected-eye/lip displacement bound
are unchanged. OpenCV remains a preprocessing dependency for exact cubic resize.

Export the existing pinned Small installation once from an **offline** interpreter
with Torch, safetensors, NumPy and OpenCV:

```sh
python ref/da2/export_reference.py \
  --installation tmp/vhuman-rig/models/depth-anything-v2-small \
  --out tmp/vhuman-rig/models/depth-anything-v2-small/native
make -C cpu/da2 da2_depth test_da2_ops
make -C cuda/da2 da2_depth test_da2_ops
cpu/da2/test_da2_ops
```

`setup_depth` also accepts `--export-python /path/to/export/python`. Runtime
needs `installation.json` and `native/{native.json,dinov2.safetensors,
depth_head.safetensors}`; it verifies model identity and exported checksums.
The `.pth` checkpoint and upstream Python checkout are needed only for export
and reference generation. Existing `--depth-installation` requests continue to
work after conversion. Missing native assets produce an explicit export error.

The CPU implementation is the validated path (AVX2/FMA on x86). The initial CUDA
path offloads FP32 GEMM while retaining attention and image operations on the
CPU; reports identify `cuda_gemm_cpu_attention`. It has no cuBLAS dependency.
ROCm configurations currently select native CPU depth. CUDA execution and
performance remain unverified because this environment has no NVIDIA device.

Input preprocessing preserves upstream RGB normalization, lower-bound resize,
rounding to multiples of 14, and float64 OpenCV interpolation before float32
conversion. Native inference is bounded to 4096 patches and four-megapixel
output images; extreme aspect ratios fail explicitly. No resolution reduction
or model substitution occurs silently.

The port also corrects shared DINOv2 bicubic positional interpolation at image
borders: clamp sample indices, not sampling coordinates. This matters for
non-square/upscaled patch grids and is covered by a PyTorch-derived border test.

Generate and validate independent intermediate/output references:

```sh
# Offline export/reference interpreter:
python ref/da2/export_reference.py \
  --installation tmp/vhuman-rig/models/depth-anything-v2-small \
  --out tmp/da2-reference --image /path/to/portrait.png
# Native runtime interpreter: NumPy + OpenCV, no frameworks.
python ref/da2/verify_native.py --fixture tmp/da2-reference \
  --runner cpu/da2/da2_depth
DA2_TEST_INSTALLATION=tmp/vhuman-rig/models/depth-anything-v2-small \
DA2_TEST_FIXTURE=tmp/da2-reference DA2_REQUIRE_FRAMEWORK_FREE=1 \
  python -m unittest server.vhuman.test_depth_native server.vhuman.test_reconstruction
```

For GPU validation use `cuda/da2/da2_depth --backend cuda` in the verifier and
`cuda/da2/test_da2_ops --cuda` for isolated GEMM/convolution checks. Intermediate
features/fusion use max/mean gates of `1e-3`/`1e-4`. Internal relative-depth
maximum error permits `max(2e-4, 1e-4 * reference_peak)`; final original-image
depth retains absolute max/mean gates of `2e-4`/`2e-5`.

Validation on 2026-10-02 passed three independent fixtures: a small rectangular
image, a 518x518 portrait input, and a 518x686 portrait input. Preprocessing was
byte-exact; maximum final relative-depth errors were `3.58e-6`, `4.10e-5`, and
`1.64e-4`. CPU inference took about 5.0 s (square) and 7.2 s (rectangular) using
four threads; rectangular peak host RSS was about 244 MiB. The framework-free
suite ran 33 tests: 32 passed and the Torch-only reference test was skipped.
GEMM, padded/strided convolution and bicubic-border unit checks passed. Existing
DA3 and DINOv2 runners rebuild with the shared changes.

## Native background, camera, cue and speech-motion inference

Items 1–4 of the framework-removal work are implemented in C, using repository
GEMM. Runtime Python handles image IO, NumPy arrays, SciPy camera fitting and
ctypes/subprocess calls; these four inference paths import neither Torch nor
ONNX. Offline export, training and reference comparisons still use Torch.

| Component | Native implementation | Production integration |
|---|---|---|
| RMBG2 | Two-scale Swin-L, complete BiRefNet decoder, deformable convolutions and alpha output | Pixal3D preparation and Qimg21 background removal |
| MoGe-2 camera | DINOv2-L/14, feature neck, point/mask heads | Pixal3D preparation and Qimg21 camera estimation |
| Learned cues | Geometry-prior residual normal/mask CNN | Optional reconstruction cue inference, with existing quality gate |
| Speech motion | Normalization, code embeddings, projection, two-layer streaming GRU and bounded controls | Live motion adapter via native shared library |

MoGe implements the path needed for camera intrinsics. Its metric-scale head
is not evaluated; this is not a complete metric-depth/normal API. The cue
checkpoint currently fails its synthetic quality gate and remains disabled for
production inference. Numerical equivalence does not establish expression or
appearance quality on new identities. Live inference now uses the repository
C++/CUDA Gaussian renderer; Torch/gsplat remain offline training/reference tools.

Further dependency removal (items 6, 9, 10 and 12):

| Path | Current dependency boundary |
|---|---|
| Live rig and Gaussian rendering | Native C/C++ CUDA deformation, covariance bounds, projection, tile sorting, rasterization, streams/events and sRGB/alpha download; no Torch/gsplat inference |
| Hunyuan server inference | Repository C++/CUDA backend; full fast12 I2V pipeline and all 81 frames passed independent references, 3,108 MiB peak VRAM, zero cuBLAS/fallback calls; quality profiles remain unvalidated |
| Video parity campaign | Framework-free dump conversion, bounded-memory comparison and 144-case coverage audit; official inference is an isolated Torch oracle |
| Rig asset helpers | Contact export and shape smoothing use NumPy/SciPy; corrective training loads `mldeformer_training` lazily |
| Motion evaluation | Native GRU by default; Torch full-sequence comparison only with `--reference-parity` |
| Motion training | Native CPU embedding/projection/GRU forward and backpropagation, weighted Huber/temporal loss, gradient clipping and AdamW; direct safetensors/JSON output |
| Modal soft-deformer training | Repository CPU GEMM for bounded randomized PCA; NumPy thin QR/SVD and oscillator/ridge fitting; no Torch or GPU session |
| Corrective deformer training | Native C++ contact/ARAP rotations and analytic gradients, repository-GEMM MLP backpropagation and AdamW; NumPy small skin transforms/solves and bounded regional PCA; no model framework |
| Frozen normal-cue evaluation | Existing native image runner for real-domain evaluation; no Torch model loading |
| Identity reference creation | Native FLUX.2 F16/repository-GEMM neutral and reference-conditioned expressions; full distilled-4B four-step T2I/I2I parity passed with CUDA text encoding and FP32 KV storage; no Torch/ONNX inference |

Photo and video landmark/blendshape inference now uses the native C++ executor
and repository AVX2 GEMM. The pinned MediaPipe task is converted with NumPy and
the standard library; OpenCV supplies image sampling, with no TFLite/MediaPipe
runtime. All13 expression references passed against MediaPipe IMAGE mode:
worst landmark error0.077pixels. Six independent network-tensor checks passed,
as did rotated/wide images and blank/two-face detection. Video observation uses
independent per-frame inference; it does not reproduce MediaPipe's VIDEO-mode
tracking state. The rig optimizer retains temporal regularization.
Build with `make -C cpu/vhuman libvhuman_landmarks.so`; setup_face_video.sh also
builds this library. Run `python -m unittest server.vhuman.test_native_landmarks`
and the optional oracle `ref/vhuman/verify_landmarks.py` (requires ai-edge-litert
and MediaPipe only in the reference environment). Reports are under
`tmp/vhuman-landmarks/{parity,expression-parity,geometry-parity}`.

Photographic-reference conditioning now has a native VAE encoder and DiT path.
Registration/optimization, training
and checkpoint export may still use Torch. The live speech/GRU/rig/render/record
path ran successfully on RTX 5060 Ti without Torch or ONNX installed. The new
Hunyuan backend passed the complete fast12 image-to-video pipeline described
above; its quality profiles and expression timing remain experimental.

Training dependency removal is incremental. Motion and corrective training and
modal PCA use `cpu/vhuman/libvhuman_training.so`, with repository GEMM and AdamW;
NumPy/SciPy remain ordinary array, factorization and geometry dependencies. CPU
GRU checks cover forward/state, all parameter gradients, chunk boundaries and
optimizer updates. Complete motion, corrective and modal training pass with
Torch, ONNX and other model frameworks blocked. Optional CPU oracles are
`ref/vhuman/verify_motion_training.py` and `ref/vhuman/verify_corrective_training.py`;
reports are under `tmp/vhuman-native-training/` and `tmp/vhuman-native-corrective/`.
New training quality and GPU checks are deferred.
Remaining Torch training paths include the normal-cue CNN, Gaussian appearance
fitting, and some registration and expression optimizers. Legacy checkpoint
conversion and independent reference
checks retain optional framework imports.

GPU checks on 2026-10-02 also passed the saved independent RMBG and MoGe
oracles. RMBG maximum alpha error was 1.42e-5 (output mask at most one uint8
level); MoGe maximum output error was 1.17e-5 and FOV error 8.17e-7 radians.
The CUDA linear initializer was corrected to accept the compiler's positive
SM return value. Reports are under `tmp/vhuman-native-gpu/`.

Build from the repository root:

```sh
make -C cpu/vhuman vhuman_models libvhuman_motion.so libvhuman_training.so test
make -C cuda/vhuman
```

The image runner supports `--task rmbg|moge|cues`, bounded finite raw FP32 CHW
input and FP32 CHW output. RMBG returns logits; MoGe returns XYZ plus mask
probability; cues return normals plus mask probability. The adapters apply
RMBG's exact PIL resize/normalization/uint8-alpha semantics and MoGe's camera
fit. Full RMBG runs at 1024 square; MoGe defaults to 3,600 tokens, with image
size up to 2048 per axis to accommodate padded foreground crops.

CPU math uses AVX2/FMA GEMM on x86, with a scalar alternative. The CUDA image
runner offloads linear projections and convolution GEMMs through
`cuda/gemm/cuda_linear_f32.h`; attention, sampling and layout operations remain
on CPU. It uploads weights per projection and is not yet optimized for GPU
residency. Neither image runner links vendor BLAS. The small streaming GRU
uses CPU GEMM with one thread per packet. CUDA builds pass, but GPU numerical
parity, throughput and 16 GB VRAM behavior remain unverified on this host.

RMBG reads the existing FP32 `model.safetensors` directly. Export MoGe and
existing trained motion checkpoints once using a Torch-capable interpreter:

```sh
python ref/vhuman/moge_reference.py \
  --checkpoint /path/to/moge-2-vitl/model.pt --out tmp/vhuman-native/moge-native
python -m server.vhuman.realtime.src.animation.export_native \
  tmp/vhuman-realtime/motion-v3.pt
```

MoGe export writes `dinov2.safetensors`, `heads.safetensors` and a hashed
`native.json`. Pass the exported directory to `--moge-model` in preparation,
or `--moge` in the Pixal3D server. A legacy `model.pt` argument resolves to its
sibling `native/` directory and does not load Torch. Motion export creates
`motion-v3.native/`; `--adapter` accepts that directory or the original `.pt`
path, checking the source hash when the latter is used. Training now exports
native motion assets automatically. Missing or stale assets fail explicitly.
Cue inference reads its existing FP32 safetensors checkpoint.

```sh
# This interpreter needs NumPy, Pillow and SciPy, but no Torch/ONNX.
python ref/pixal3d/prepare_input.py --device cpu \
  --input portrait.png --output tmp/prepared.png --metadata tmp/prepared.json \
  --rembg-model /path/to/RMBG-2.0 --moge-model tmp/vhuman-native/moge-native
```

Both server preparation paths use the current Python interpreter. ROCm callers
use the native CPU image models; CUDA callers select the hybrid CUDA runner.
Job cancellation and timeout terminate the preparation process group, including
native children.

Independent FP32 Torch comparisons on 2026-10-02 produced:

| Check | Measured maximum error |
|---|---|
| Full RMBG2, 1024 square portrait | Alpha `1.05e-5`; resized uint8 mask differs by at most 1 |
| MoGe camera, 512 square / 3,600 tokens | XYZ/mask `1.17e-5`; FOV `1.4e-6` radians |
| Cue network, 64 square and 57×83 | `8.65e-7` |
| Motion GRU, 40 consecutive packets | `2.69e-7`; reset output bit-exact |

RMBG logits have maximum error `0.00542` in saturated regions; alpha is the
relevant output gate. CPU runs with four threads took about 72 seconds for
RMBG and 87 seconds for MoGe. These are single-fixture measurements, not GPU
performance or visual-quality claims. Both production Python adapters also ran
successfully in an environment without Torch, ONNX or ONNX Runtime.

Reproduce the offline oracles with the appropriate reference environments,
then compare with an ordinary NumPy/Pillow/SciPy interpreter:

```sh
# Requires Torch/torchvision/timm/transformers/safetensors and pinned local RMBG source.
python ref/vhuman/rmbg_reference.py --model /path/to/RMBG-2.0 \
  --image portrait.png --out tmp/vhuman-native/rmbg1024
# Requires the local moge-upstream dependencies; also exports native assets.
python ref/vhuman/moge_reference.py --checkpoint /path/to/moge-2-vitl/model.pt \
  --image portrait.png --side 512 --tokens 3600 --out tmp/vhuman-native/moge-native

python ref/vhuman/verify_images.py --fixture tmp/vhuman-native/rmbg1024
python ref/vhuman/verify_images.py --fixture tmp/vhuman-native/moge-native
# For GPU comparisons add --backend cuda --runner cuda/vhuman/vhuman_models.
# Torch reference + native comparison for the two small models:
python ref/vhuman/verify_small_models.py --cues /path/to/trained-cues \
  --motion tmp/vhuman-realtime/motion-v3.pt --out tmp/vhuman-native/small-models

python -m unittest server.vhuman.test_native_models server.vhuman.test_reconstruction
python -m unittest server.pixal3d.test_app server.pixal3d.test_i23d_studio
```

Image verification records complete-buffer errors and hashes. Fixed gates are
alpha max/mean `<1e-3`/`<1e-5`, resized mask difference `<=1`; MoGe output
max/mean `<1e-4`/`<1e-5` and FOV difference `<1e-4` radians. Small models use
maximum error `<2e-5`. The C unit tests independently check convolution,
deformable sampling and antialiased resize, and pass ASan/UBSan checks.
Framework-free replay tests optionally use `VHUMAN_NATIVE_FIXTURES` pointing to
the fixture root (including `cues/` and `motion.native/` asset directories);
`VHUMAN_REQUIRE_FRAMEWORK_FREE=1` asserts that ML frameworks are absent.

The standalone backbone runners remain available in `cpu/rmbg` and
`cuda/rmbg`. Their independent feature-map oracle and verifier are in
`ref/rmbg`; odd/rectangular and both production encoder scales passed before
the full decoder port.

## Licensing and provenance

The [neural avatar runtime](realtime/README.md) adds timestamped native TTS
features/PCM, causal facial motion, and a native CUDA Gaussian renderer on top of the
existing rig. Its diagnostic benchmarks are separate from appearance quality;
commercial appearance training requires new, cleared identity assets.

Repository-authored code is under the root MIT license. NumPy, Pillow and
Three.js are external dependencies with their own permissive licenses; the
optional mapbox-earcut bindings declare ISC and earcut has its own notices.
No GPL code, Unreal source, Unreal assets or measurements extracted from those
assets are included in this implementation. Standard mathematical operations
and original procedural algorithms are used instead of engine material graphs.
Khronos PBR Neutral tone mapping is used for the preview; third-party libraries
retain their own licenses. The tone-mapping adaptation notice is preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Qwen-generated imagery and model weights have separate terms (Qwen Research
License for the configured Qwen-Image 2.1 model). Their outputs are tagged
`qwen-research` and remain in the local work directory. The MIT code license
does not relicense third-party models or generated assets. No user/LLM-supplied
measurement is treated as independently verified anatomy.
