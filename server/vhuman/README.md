# Independent procedural virtual humans

For RX 9070 XT / RDNA4, see [ROCm setup and validation](ROCM.md).
For Wan/H3/HV1.5 expression capture, downloaded face models and GNM mapping,
see [ROCm expression candidates and GNM](VIDEO_EXPRESSIONS.md).
For identity-frozen head/bust creation, complete GNM anatomy and offline HIP
rendering, see [the implementation and commands](../../doc/vhuman-photoreal-head-implementation.md).

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

**Learned cues (v3):** the original 19,876-parameter CNN predicts a bounded
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

Cue training now uses native C++ convolution forward/reverse passes (all five
layers), SiLU, bilinear resize, tanh residual normalization, cosine/prior loss
and stable mask BCE. Repository GEMM and native AdamW perform the model updates.
It preserves the safetensors names, training-only fallback prior, independent
identity split and synthetic/real adoption gates. Initialization and minibatch
sampling use NumPy's random stream; archived Torch-trained quality results below
do not validate newly trained weights. The native trainer currently supports CPU
with batches up to 8 and square crops up to 128 pixels.

```sh
make -C cpu/vhuman libvhuman_training.so
python -m server.vhuman.reconstruction.learned_cues train \
  --dataset tmp/vhuman-cues/dataset.npz --out tmp/vhuman-cues/native-trained \
  --steps 400 --device cpu --threads 4
python -m unittest server.vhuman.test_native_image_training -v
```

Independent CPU math comparisons passed with normal error 1.20e-7, maximum
gradient error 3.73e-9 and one-step AdamW error 6.64e-7. Gaussian appearance
training is also native CPU; see `realtime/README.md`. The combined optional
oracle is `ref/vhuman/verify_image_training.py`, with reports under
`tmp/vhuman-native-image-training/`. GPU, gsplat kernel parity and newly trained
visual quality remain deferred.

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
ONNX. Legacy export and optional reference comparisons retain Torch.

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
C++/CUDA Gaussian renderer; Torch/gsplat remain optional reference tools.

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
| Cue CNN training | Native convolution/SiLU/resize/normalization reverse passes and normal/mask losses; repository GEMM/AdamW; direct compatible safetensors output |
| Gaussian appearance training | Native CPU trace-v1 deformation/projection and analytic alpha reverse pass for all six parameter groups, repository GEMM and Adam; no Torch/gsplat/CUDA requirement |
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

Training dependency removal is incremental. Motion, corrective, cue CNN and
Gaussian appearance training and modal PCA use `cpu/vhuman/libvhuman_training.so`,
with repository GEMM and AdamW;
NumPy/SciPy remain ordinary array, factorization and geometry dependencies. CPU
GRU checks cover forward/state, all parameter gradients, chunk boundaries and
optimizer updates. Complete motion, corrective, cue CNN, Gaussian and modal training pass with
Torch, ONNX and other model frameworks blocked. Optional CPU oracles are
`ref/vhuman/verify_motion_training.py`, `ref/vhuman/verify_corrective_training.py`
and `ref/vhuman/verify_image_training.py`; reports are under
`tmp/vhuman-native-training/`, `tmp/vhuman-native-corrective/` and
`tmp/vhuman-native-image-training/`.
New training quality and GPU checks are deferred.
Remaining Torch training paths include registration and expression/video
optimizers. Legacy checkpoint
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

## Native mobile avatar reference

`mobile/` contains a C++17 GNM evaluator, a Filament C++ player, and a Swift/UIKit
shell for iOS 16.4+. The renderer uses Metal on Apple platforms and Vulkan for
Linux validation. The iPhone 12 profile targets 30 FPS; no iPhone performance
claim is implied by the desktop checks. The phone receives audio and native
motion from the server; it does not load a diffusion model or TTS/LLM weights.

Export an accepted reconstruction and its matching prepared offline scene:

```sh
python -m server.vhuman.cli mobile-export \
  --candidate tmp/vhuman-public-portraits/obama/head/reconstruction/nativefit11 \
  --scene tmp/vhuman-public-portraits/obama/offline/fit12 \
  --profile iphone12 --out tmp/vhuman-mobile/avatar
python -m server.vhuman.mobile.verify \
  --candidate tmp/vhuman-public-portraits/obama/head/reconstruction/nativefit11 \
  --package tmp/vhuman-mobile/avatar --work tmp/vhuman-mobile/verify
python -m unittest server.vhuman.test_mobile server.vhuman.test_gnm_anatomy
```

The export includes a checksum manifest, static GLB reference, native deformation
weights, mesh streams, native/surface/joint bindings, control ordering, prepared
skin detail, optical anatomy and 1,024 deterministic hair ribbons. The Obama
reference totals 79,716 triangles. Short-hair undercoat coverage is baked into
the scalp material with UV gutters. Geometric normals are shared across UV seams
during animation. Portrait residuals now deform in bind space through GNM LBS;
they no longer remain fixed in world space when the head turns. PCA coefficients
retain their native names and are never relabelled as semantic visemes.

Validation on 2026-10-06: 90 affected tests passed, plus a cancellation rerun
after tightening async-generator shutdown. The 20-pose Obama native check
measured 0.000161 mm p95 vertex error (0.25 mm gate), 0.000620 mm maximum and
6.35 ms median host CPU evaluation. Filament static/posed Vulkan renders passed
on RX 9070 XT; CPU deformation plus vertex-upload submission measured about
8 ms. These are host measurements, not iPhone FPS or end-to-end speech results.

Build the host player against the **official Filament 1.77.2 SDK**:

```sh
cmake -S server/vhuman/mobile -B tmp/vhuman-mobile/build \
  -DCMAKE_BUILD_TYPE=Release -DFILAMENT_ROOT=/path/to/filament \
  -DFILAMENT_LIB_DIR=/path/to/filament/lib/x86_64
cmake --build tmp/vhuman-mobile/build -j4
tmp/vhuman-mobile/build/vhuman_player_probe \
  tmp/vhuman-mobile/avatar tmp/vhuman-mobile/front.ppm
```

The Linux SDK uses libc++; use Clang with matching libc++ headers/libraries
(`-stdlib=libc++`) when building against the precompiled archive. The optional
third probe argument is a little-endian F32 pose file containing 383 expression
coefficients, 12 joint axis-angle values, then 3 native translation values.
Its timing measures CPU deformation and upload submission, not GPU frame time.
The Linux SDK archive used for validation has SHA256
`b01d7aeb3d6877fbd6c9736ce1fa1eb2aeb67fa3a3603017c9aa04a15f8e455a`.

On a Mac, extract the corresponding iOS SDK and point `FILAMENT_LIB_DIR` at
its **device arm64** static libraries, then generate the Xcode project:

```sh
cmake -S server/vhuman/mobile -B tmp/vhuman-mobile/ios -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.4 \
  -DFILAMENT_ROOT=/path/to/filament-ios \
  -DFILAMENT_LIB_DIR=/path/to/filament-ios/lib/arm64
open tmp/vhuman-mobile/ios/vhuman_mobile.xcodeproj
```

Select a signing team, build `VHuman` for a physical device, and copy the exported
folder as `Documents/avatar` using Finder file sharing. **Load avatar** verifies
every package hash before opening it. The app pauses rendering and disconnects
audio on backgrounding, audio interruption or route change. Xcode compilation,
signing and physical-device validation remain unverified in the Linux workspace.

The server transport uses 24kHz mono audio, utterance-relative sample positions,
increasing epochs and an exact avatar-manifest handshake. Cancellation joins the
producer before starting the next epoch. Audio gaps fail closed. The app samples
motion against the AVAudioPlayerNode timeline, subtracting downstream output
presentation latency; it never advances animation using wall time.

```sh
uv pip install --python /path/to/venv/bin/python -r server/vhuman/mobile/requirements.txt
python -m server.vhuman.mobile.speech --package tmp/vhuman-mobile/avatar \
  --mapping /path/to/geometry-matched-gnm-map.json --adapter /path/to/motion.native \
  --runner speech/build/qwen3_tts_rocm --model /path/to/qwen3-tts \
  --revision EXACT_MODEL_REVISION --backend rocm
```

Use `--host` explicitly for LAN binding, or terminate WSS at the existing server
proxy. The bridge accepts English/Japanese text and uses the existing native
Qwen TTS feature stream and causal motion student. Mapping metadata must include
`source_geometry_sha256`, matching native coefficient names, semantic control
ordering and a 383-by-controls matrix. The default rejects diagnostic students;
`--diagnostic` is an explicit local experiment option. This bridge is implemented
but has not yet passed the planned bilingual end-to-end quality evaluation.

For deterministic transport validation, `python -m server.vhuman.mobile.stream`
accepts `--package`, `--wav` (24kHz mono PCM16) and `--motion` (NPZ fields
`sample_positions`, `expression`, `rotations`, `translation`,
`source_geometry_sha256`). Replay does not synthesize the entered text.
The identity generator also accepts `--identity-backend native-rocm` for neutral
FLUX.2 Klein T2I; its HIP runner has no reference-image input, so expression
generation must use the existing Wan I2V path rather than independent T2I faces.

Current reference limitations: LOD1/2, ASTC/KTX packaging, native Filament skin
SSS/wrinkle shaders, refined corneal rendering, oral contact quality, and device
thermal/memory/FPS/lip-sync gates remain open. Full 3D identity still requires
multi-view review; a low frontal landmark error does not establish unseen facial
shape. Asset manifests explicitly set `production_ready=false` and
`device_measured=false`. On-device LLM/TTS and the planned six-identity / bilingual
acceptance corpus are not implemented by this mobile reference.

### Browser target and skin-detail preprocessing

The alternative runtime in `web/vhuman_mobile*` uses **WebGL2 + WASM SIMD**.
It compiles the same `mobile/native.cpp` evaluator used by the native player;
there is no approximate JavaScript replacement for GNM. A worker performs
deformation, and the browser applies native, joint and barycentric attachments,
recomputes normals shared across skin UV seams, and renders the same GLB optical
materials. The package runs locally without a CDN. Three.js 0.163.0 is installed
separately under its MIT license, and the build copies its LICENSE into the output.

Preprocessing fits a per-identity expression-to-regional-area-strain regressor
against synthetic native GNM deformation, then bakes metric tangent-space wrinkle
slopes. UV scale and shear are accounted for; chart padding avoids height-to-zero
edges. The driver keeps the captured reference at zero activation. Train/test
seeds, coefficient distribution, package/detail hashes and held-out metrics are
recorded in `detail.json`; rejected fits retain the analytic driver. This is
**geometric supervision**, not recovery of observed wrinkle depth. Groove depth
and pattern still come from the bounded authored skin-detail prior.

```sh
python -m server.vhuman.mobile.preprocess \
  --candidate tmp/vhuman-public-portraits/obama/head/reconstruction/nativefit11 \
  --package tmp/vhuman-mobile/obama04 --out tmp/vhuman-browser/obama-detail

# Emscripten (em++) must be on PATH. Extract the pinned Three.js package locally.
mkdir -p tmp/vhuman-browser/vendor
(cd tmp/vhuman-browser/vendor && npm pack three@0.163.0 && tar -xzf three-0.163.0.tgz)
python -m server.vhuman.mobile.browser \
  --package tmp/vhuman-mobile/obama04 --detail tmp/vhuman-browser/obama-detail \
  --three tmp/vhuman-browser/vendor/package --out tmp/vhuman-browser/player
python -m http.server 8088 --bind 127.0.0.1 --directory tmp/vhuman-browser/player
```

Open `http://127.0.0.1:8088/`. The review controls provide a head turn, pose sweep,
lighting presets and a dynamic-detail toggle. The speech controls connect to the
same `mobile.speech` / `mobile.stream` server used by the iOS shell. Supply the
original exported avatar directory to that server so its manifest hash matches.
Browser audio uses an AudioWorklet with a bounded PCM queue and device-rate
resampling. Animation follows source-sample markers mapped to the output device
through `AudioContext.getOutputTimestamp()`. Underruns freeze the source timeline;
cancel, disconnect and page backgrounding stop the stream. The visual preview
also works over LAN HTTP: serve with `--bind 0.0.0.0` and open the server's LAN
address. Asset checks use a JavaScript SHA-256 fallback when SubtleCrypto is
unavailable. Speech controls require HTTPS or localhost for Web Audio and are
disabled on LAN HTTP; WSS is needed from HTTPS pages. No microphone permission
is needed.

To serve an exported preview over HTTPS on the LAN, use a certificate with the
server's LAN IP (or DNS name) in its Subject Alternative Name, signed by a CA
trusted on the client PC:

```sh
python -m server.vhuman.mobile.serve \
  --directory tmp/vhuman-browser/player-generated01 \
  --cert tmp/vhuman-browser/tls/server.crt \
  --key tmp/vhuman-browser/tls/server.key --bind 0.0.0.0 --port 8443
```

Keep private keys outside the served directory. Install only the public CA
certificate in the other PC's trusted root certificate store (Firefox may use
its own Authorities store). Then open `https://SERVER_LAN_IP:8443/`. Changing
the server IP requires a certificate covering the new address. This is a static
preview server; speech additionally needs a WSS endpoint.

Validation commands:

```sh
python -m unittest server.vhuman.test_mobile_preprocess server.vhuman.test_mobile
node server/vhuman/mobile/test_browser_audio.mjs
python -m server.vhuman.mobile.browser_verify \
  --player tmp/vhuman-browser/player --out tmp/vhuman-browser/check --hardware
```

`--hardware` requests headless ANGLE/Vulkan and reports the actual renderer.
Without it, the verifier requests SwiftShader. It checks WASM/native vertices,
every attachment class, strain activations, rendered lighting/detail changes,
WebSocket delivery, device-rate audio accounting and cancellation. Its silent
PCM fixture tests timing mechanics, not TTS intelligibility or measured lip sync.

Obama preprocessing: 768 training poses and 192 held-out poses; mean activation
error improved from 0.011743 to 0.001765 (6.65×), p95 from 0.047317 to 0.005835.
Twelve 256px two-channel slope maps use 1.5 MiB of biased signed RG8 data
(previously 6 MiB of float data). Zero is represented exactly; maximum slope
quantization error is 0.001378. RG8 supports linear filtering in core WebGL2,
without a float-linear extension. The viewer also accepts earlier float bakes.
Height reduction normalizes atlas coverage, and derivatives use only valid
neighbors within a UV chart, avoiding height-to-background edges.
Chromium/AMD validation
measured exact native/WASM vertices on three poses, about 6 ms WASM evaluation
and 5–10 ms attachment/normal updates. The final animated check sustained 29.9
rendered FPS and 29.9 pose updates/sec with dynamic detail enabled. Attachment
error was 0.0000064 mm p95; browser strain activations matched Python within
6.1e-10. The audio
fixture played 48,000 source samples through 48 kHz output with zero underruns.
These desktop results do not establish iPhone/Safari performance, a thermal soak,
or photometric wrinkle accuracy. Unseen side-face and neck texture quality still
requires additional observations and review.

Mobile texture baking now matches geometric edges across UV seams and applies
bounded, confidence-weighted corrections in a narrow band in linear RGB.
For Obama `nativefit11` with prepared scene `fit12`, the pre-scalp basecolor
edge RMS decreased from 0.08406 to 0.04990 (40.6%, 1,648 sample pairs);
1,359 texels changed, with a maximum linear correction of 0.08. This measures
edge continuity, not recovered albedo accuracy. Each package includes
`bake_quality.json` with the seam and scalp diagnostics.

Scalp baking uses bilinear parsing-mask sampling instead of nearest-neighbor
sampling. Tangent normals are renormalized after blending; four-texel atlas
gutters remain in place. Projected hair coverage and the inferred crown remain
single-view priors; scalp occlusion and unseen hair coverage still need
additional observations.
The updated Obama package is `tmp/vhuman-mobile/obama08`, its browser build is
`tmp/vhuman-browser/player-bake04`, and hardware verification is saved in
`tmp/vhuman-browser/bake-check04`. Reproduce with the export, preprocess, browser
and browser_verify commands above, using these output paths or fresh directories.

For source-color cleanup before export, rebake the existing candidate with:

```sh
python -m server.vhuman.reconstruction.refine_material \
  tmp/vhuman-public-portraits/obama/head/reconstruction/nativefit11 \
  --out tmp/vhuman-public-portraits/obama/head/reconstruction/material12 \
  --exclude-non-skin
```

This opt-in parsing mask excludes background, clothing, accessories, hair,
eyeballs and mouth cavity from the skin material; lips, brows, ears and neck
remain eligible. Existing manual exclusions are unioned with the prediction.
The candidate records the parser checksum, predicted labels and mask counts.
Labels are model predictions, so inspect the saved masks when using a new subject.
All portrait bakes now interpolate in linear light over non-excluded source
pixels only, carrying the remaining interpolation support into bake confidence.
This avoids both nearest-pixel stepping and color leakage across mask edges.

Obama `material12` preserves the exact fitted geometry hash. Compared with
`nativefit11`, observed texels change from 186,196 to 175,326: 11,790 old
observations are removed and 920 gain sufficient support through interpolation.
Visual inspection confirms removal of the shoulder color contamination.
Excluded areas use the existing bounded completion prior, not recovered detail.
The refreshed package is `tmp/vhuman-mobile/obama09`; browser output and AMD
verification are `tmp/vhuman-browser/player-bake05` and
`tmp/vhuman-browser/bake-check05`. The latter passed native/WASM and attachment
parity, relighting/detail checks and audio timing at 29.9 animated FPS.
Run `python -m unittest server.vhuman.test_photoreal server.vhuman.test_quality
server.vhuman.test_reconstruction server.vhuman.test_mobile_preprocess
server.vhuman.test_mobile` for the 74-test preprocessing/material/runtime suite.

### Experimental wrinkle-color baseline

`reconstruction.wrinkle_skin` is a reproducible color-transfer experiment, not
a photorealistic face-completion solution. Visual review found its repeated,
anatomically unconditioned detail inadequate despite passing the contrast gate.
See [multiview face completion research](MULTIVIEW_FACE_COMPLETION_RESEARCH.md)
for model comparisons, licensing, the proposed material pipeline, and quality
criteria for its replacement. That pipeline is not implemented yet.

The earlier
whole-view masked edits below were visually too flat: changing many texels did
not establish successful texture completion. A stronger CFG-4, strength-0.8
close-up still failed the contrast gate. A blue-skin control verified that
native masking and text conditioning work; full-noise whole-view generation
produced wrinkles but also invented another ear, so it was rejected.

The experimental path edits an **anatomy-free skin material** with Qwen, then
attaches its luminance detail to the captured GNM mesh through continuous
world-space triplanar sampling. Side/front coordinates align folds with the
head-up axis, detail tapers off toward the upper scalp, and a 6mm feather
protects the transition to photographed skin. The transfer removes broad
illumination while retaining larger creases (24px rather than 9px low-frequency
subtraction, maximum 0.08 rather than 0.025 linear RGB residual). Photographed
texels, geometry, normal/ORM and evidence maps remain byte-identical. This is
synthetic wrinkle **color** detail; it does not recover wrinkle depth or a
subject's hidden anatomy, and keeps the `qwen-research` provenance.

```sh
python -m server.vhuman.reconstruction.wrinkle_skin all \
  --candidate tmp/vhuman-public-portraits/obama/head/reconstruction/multiview14 \
  --prior tmp/vhuman-generated-skin/obama01/skin_prior.png \
  --work tmp/vhuman-generated-skin/wrinkle-material01 \
  --out tmp/vhuman-public-portraits/obama/head/reconstruction/wrinkles15
```

Stages are `generate`, `bake`, `review`, `all`. The material edit uses 384px,
10 steps, CFG 4, strength 1 and the native ROCm INT8 `low8` backend; the validated
edit took 471 seconds. Matching generation receipts are reusable. `--period`
sets the world-space material scale in metres (default 0.1). The review gallery
shows the actual material edit and before/final-atlas renders, avoiding any
claim that Qwen successfully edited the entire head view. Pass the workspace
as `mobile.browser --skin-review` to include it in the browser preview.

Both material and final ear/jaw renders must pass a mid-scale contrast gate
inside initially flat, eroded skin masks; tiny pixel changes and near-copies are
rejected. The old ear edit decreased contrast (0.00197 to 0.00183 RMS).
The final baked left/right views increased it from 0.00184/0.00195 to
0.01182/0.01190. These are contrast diagnostics, not anatomical fidelity scores.
All 175,326 photographed texels remain unchanged. The 89-test regression suite
also checks world-space continuity, scalp taper, crop calibration and protection
of observations with the larger detail budget.

Artifacts: `tmp/vhuman-mobile/obama12`,
`tmp/vhuman-browser/player-wrinkles01`, and
`tmp/vhuman-generated-skin/wrinkle-material01/baked_quality.json`.
AMD WebGL2 validation (`tmp/vhuman-browser/wrinkles-check01/verification.json`)
passed native/WASM parity with zero vertex error, all inspection cameras,
relighting/detail and speech timing/cancellation at 30.1 animated FPS.

### Mesh-guided multiview skin completion (subtle-detail baseline)

`reconstruction.multiview_skin` adds ear, rear-head and elevated crown views
rendered from the fitted GNM mesh. It orbits calibrated cameras around the
captured mesh, keeping geometry fixed. Each masked Qwen edit uses a four-panel
condition image: target view, original portrait, adjacent mesh view and frontal
mesh view. The native HIP runner still receives **one contact-sheet reference**,
not multiple independent native image conditions. A separate 512px init image
and mask retain full target resolution and exact protected pixels.

```sh
python -m server.vhuman.reconstruction.multiview_skin all \
  --candidate tmp/vhuman-public-portraits/obama/head/reconstruction/material12 \
  --work tmp/vhuman-generated-skin/obama-multiview01 \
  --prior-work tmp/vhuman-generated-skin/obama01 \
  --out tmp/vhuman-public-portraits/obama/head/reconstruction/multiview14 \
  --steps 20 --edit-steps 12
```

`--prior-work` optionally reuses an existing T2I skin patch after checking its
generation request and checksum. Omit it to generate a new patch. Stages
`prepare`, `edit`, `bake` and `review` can run separately. Six masked edits use
the `low8` INT8 ROCm backend, with five actual denoising updates for a 12-step
schedule at strength 0.45. The workflow releases the GPU after each edit.

The bake projects edits through the exact cameras with depth and facing tests.
Overlapping residuals are fused only when at least two supported views agree
within 0.012 linear RGB; conflicting overlaps are rejected, and single-view
support is reduced. The face parser rejects confidently generated eyes, hair,
mouth cavities and accessories, while the mesh mask determines rear-scalp
coverage where frontal face parsing is unreliable. Broad illumination is
removed and the existing 0.025 residual cap and 6mm observation feather remain.
All photographed texels, geometry and observation-confidence maps are preserved.
`review.html` compares the original atlas, raw edit and final baked atlas in
every camera. Browser review also provides ear, back-head and crown view presets.
Pass `--skin-review tmp/vhuman-generated-skin/obama-multiview01` to
`mobile.browser` when building the viewer to include the comparison gallery.
Its checksummed review manifest must match the exported geometry and completed
base-color hash; the viewer exposes a “Compare multiview skin bake” link.

This path uses spatial multiview agreement instead of requiring six Wan clips.
It does not infer hair, ear geometry, hidden markings or measured wrinkle depth;
generated detail retains `qwen-research` provenance. A single-view region is
explicitly uncorroborated and agreement between generated views is not evidence
of likeness.

Obama validation (`multiview14`): six 512px edits took 244–245 seconds each
(1,468 seconds total). The bake modified 550,365 unseen texels and retained all
175,326 photographed texels byte-for-byte. Of 361,705 overlapping texels,
353,631 had agreeing edits and 8,074 were rejected. A total of 478,080 texels
received accepted edited-view support. Unobserved atlas change was 1.02 sRGB
levels RMS, so this is conservative fine-detail completion. Geometry, portrait,
observation/confidence maps, normal and ORM maps retained their original hashes.
The prior Wan-assisted path still reproduces its original base-color hash.

Artifacts: `tmp/vhuman-mobile/obama11`,
`tmp/vhuman-browser/player-multiview01`, and
`tmp/vhuman-browser/multiview-check01/verification.json`. The AMD WebGL2 check
passed native/WASM parity (zero vertex error), all five inspection cameras,
relighting/detail, audio timing and cancellation at 29.9 animated FPS.

```sh
python -m unittest server.vhuman.test_generated_skin server.vhuman.test_photoreal \
  server.vhuman.test_quality server.vhuman.test_reconstruction \
  server.vhuman.test_mobile_preprocess server.vhuman.test_mobile
# 85 tests passed
python -m server.vhuman.mobile.browser_verify \
  --player tmp/vhuman-browser/player-multiview01 \
  --out tmp/vhuman-browser/multiview-check01 --hardware
```

### Generated detail for unobserved skin

`reconstruction.generated_skin` uses the installed native ROCm Qwen-Image 2.1
runner and Wan2.2 HIP runner. The [upstream Qwen model](https://github.com/QwenLM/Qwen-Image-2.1)
supports both text-to-image and image-conditioned editing; this adapter uses
the repository's single-reference masked editing implementation.

```sh
python -m server.vhuman.reconstruction.generated_skin all \
  --candidate tmp/vhuman-public-portraits/obama/head/reconstruction/material12 \
  --work tmp/vhuman-generated-skin/obama01 \
  --out tmp/vhuman-public-portraits/obama/head/reconstruction/generated13 \
  --steps 20 --edit-steps 12 --frames 9 --video-preset fast5
```

Stages can also run individually (`prepare`, `edit`, `video`, `bake`). Generation
receipts bind prompts, seeds, source hashes and outputs; matching completed
image requests can be reused. Each stage releases Qwen before Wan runs, using
the existing shared AMD device lock. Qwen uses the `low8` INT8 preset; Wan uses
Q8_0 weights, ROCm PyTorch orchestration and repository HIP projections.

The T2I skin patch supplies bounded fine detail in world-space triplanar
coordinates. Calibrated left/right/rear renders define the image-edit masks.
Qwen edits are checked for exact preservation outside those masks. Short static
Wan clips gate edited detail using optical flow, forward/backward consistency
and photometric agreement. Accepted flow-aligned frames contribute a temporal
median, blended with the edited still before transfer. Generated eyes, hair and accessories are excluded
from transfer. Broad illumination is removed from the transferred residual;
linear RGB changes are capped at 0.025 and feathered over 6 mm next to observed
skin. Captured atlas texels are copied back byte-for-byte. Geometry, original
observation coverage and confidence stay unchanged. A separate generated-support
map and report accompany the candidate and mobile export.

These are **synthetic appearance priors**, not additional photographic evidence
or recovered hidden anatomy. A static synthetic clip tests self-consistency;
it does not establish likeness or multi-view accuracy. Pores are color detail,
not measured bump/displacement. The configured Qwen model uses the Qwen Research
License; exported generated-skin assets retain `qwen-research` provenance and do
not enter the permissive appearance-training corpus. All generated artifacts
remain in the local work directories.

Obama validation (`generated13`): a 20-step T2I patch, three masked edits using
12-step schedules (five updates at strength 0.45), and three nine-frame / five-step
Wan clips. The T2I pass took 754 s, edits about 240 s each, and Wan runs 163–212 s
including CPU text encoding and loading. Each clip executed 3,000 HIP projection
calls. Whole-image temporal acceptance was 98.0%, 95.2% and 99.2%; all editable
pixels passed the temporal gate before the separate parsing/projection checks.
The bake changed 573,568 unobserved atlas texels; 362,940 received accepted edited
view support. All 175,326 observed texels remained byte-identical. Geometry,
portrait and all original observation/confidence maps retained their hashes.
The change is conservative color detail (about 1.08 sRGB levels RMS over the
unobserved atlas), not a reconstruction of neck anatomy or unseen markings.

Package: `tmp/vhuman-mobile/obama10`; browser: `tmp/vhuman-browser/player-generated01`;
verification: `tmp/vhuman-browser/generated-check01`. AMD WebGL2 validation passed
native/WASM parity, attachments, relighting/detail, audio timing and cancellation
at 30.1 animated FPS. The 80-test material/mobile suite includes
`server.vhuman.test_generated_skin`. iPhone device quality remains unverified.

Upstream API references: [Filament build and platform guidance](https://google.github.io/filament/dup/building.html),
[Apple audio-player timeline](https://developer.apple.com/documentation/avfaudio/avaudioplayernode),
and [output presentation latency](https://developer.apple.com/documentation/avfaudio/avaudionode/outputpresentationlatency).

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
