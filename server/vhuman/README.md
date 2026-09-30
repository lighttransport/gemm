# Independent procedural virtual humans

A local eye/head viewer with synthetic procedural textures, analytic refraction,
portable glTF export, portrait-based eye fitting and optional Qwen/Pixal3D jobs.
The runtime requires no Unreal Engine installation, source, content or measured
character data. Engine-specific import/export adapters are not included.

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
- **SpeakingFaces:** fits frames 0/24/48 and evaluates disjoint frames 12/36/60;
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

Validated run `tmp/public-dataset-eval-02` (2026-09-30): nine calibrated Multiface
views, maximum projection difference **0.0064 pixels**, maximum tracked
OBJ/head-transform difference **0.000077 mm**; 30 affected numerical/import tests
passed. SpeakingFaces training landmark RMS was 4.69/4.32/4.32 pixels. Raw
holdout RMS was **14.63/20.52/15.74 pixels**; pose-aligned mouth RMS was
**3.72/13.46/8.71 pixels**. The middle frame remains a failure case, and these
estimated-annotation diagnostics establish neither scan likeness nor temporal
animation quality. All generated fixtures, fitted candidates and preview media
are ignored local artifacts; dataset media licenses still apply.

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

## Licensing and provenance

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
