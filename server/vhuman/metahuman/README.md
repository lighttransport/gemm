# MetaHuman-derived priors for vhuman / GNM

Current rendering default: **`peri_profile5`**. Full-quality rendering includes
native-preserving skin diffusion, regional normals and a localized optical layer
crossing the eye/skin contact. See [the eye-side contact default](#eye-side-contact-default-2026-11-08)
for validation and opt-outs. Earlier dated defaults below are historical.

The experimental `material_overrides: {"eye_contact_thin_aperture": 1}`
helps retain eye-side contact coverage when a nearly closed aperture becomes
fragmented by antialiasing. It groups nearby samples and fades the fallback
for weak, round, thick or depth-inconsistent components and tiny requested
contact widths. It requires the skin profile and eye-contact blur, and is off
by default: grouping can join unrelated nearby eye
samples, and the treatment can change opaque eye pixels in a thin aperture.
In two authored eight-pose blink sequences it reduced the excess band change
above the change in eye coverage; those poses are geometric proposals with
unvalidated optical contact, not captured natural blinks (`boundary146`,
`boundary147`). At 128 samples, one selected transition reduced maximum excess
from .367 to zero; the other already had only .008 excess and also fell to
zero (`boundary148`). The second identity was held out from tuning the
component confidence.
The public portrait and posed layered-eye HIP renders passed saved-image
recomposition, preserved alpha and pixels outside the eye, and matched an
omitted option to explicit zero. The open-eye portrait was unchanged; the
posed eye changed 110 pixels (`boundary149`, `boundary150`, peak whole-device
usage 10170 and 10727 MiB).

The separate experimental `material_overrides: {"eye_contact_skin_coverage": 1}`
uses nearby fractional eye coverage to make the skin-side contact band vary
more continuously as an edge moves. It requires a positive skin-side contact
strength and is off by default. A depth-aware local confidence rejects isolated
fractional samples, and an 8-10 mm depth fade limits cross-surface gathering. The
gather fades back to the original band between projected widths of 31 and 32 pixels
and when estimated coverage or gather work becomes large, so very wide or expensive
close-up bands retain the original behavior. On two
authored blink sequences the maximum excess band change above the eye-coverage
change fell for every tested transition; selected 128-sample repeats also
improved (`boundary151`). Still-image comparisons were mixed and the authored
blinks are not captured motion (`boundary152`, `boundary153`), so this remains
an experiment. In 20 synthetic moving-edge cases across five angles and four
widths, peak excess fell in every case, while total excess rose in two
(`boundary167`). In a synthetic depth-crossing test the fade reduced the
largest band step from .618 to .012 (`boundary161`). It can be combined with
`eye_contact_thin_aperture`.
For MetaHuman comparisons, `peri_profile5_temporal` selects profile5 with both
options enabled; the normal default remains `peri_profile5`. Its named
layered-eye HIP render passed saved-image, alpha and support checks
(`boundary170`, peak 11174 MiB).
On this CPU, an upscaled portrait-mask benchmark took about .015, .062 and
.35 seconds at 512, 1024 and 2048 pixels, respectively; peak process RSS in
the 2048 run was about 433 MiB (`boundary151/bench_final.json`). Cost depends
on the eye-region size and projected width; a 1024-pixel eye close-up with a
roughly 19-pixel band took about .62 seconds in the prototype.
Combined public-portrait and posed layered-eye HIP renders reproduced their
saved images exactly, preserved alpha and left pixels outside each treatment's
support unchanged (`boundary163`, `boundary164`; peak whole-device usage
10170 and 10723 MiB). Explicit zero matched omission for both options.
An angled HIP view with an actual depth discontinuity also passed those
checks (`boundary171`, peak 10737 MiB). In the depth-confidence ablation,
nine pixels changed by at most .00445 in scene-linear RGB (`boundary165`).
Across the two authored blink sequences, the combined options reduced the
mean, 99th-percentile and maximum frame-to-frame change of the isolated
contact-only RGB correction in all 14 transitions (`boundary168`). Averaged
99th-percentile changes fell from .00498 to .00285. This does not measure the
stability of the underlying posed geometry or lighting. The selected
128-sample transitions also reduced this 99th-percentile change, from
.00381 to .00200 and .00230 to .00138 (`boundary169`).
A third identity, not used to tune these options, showed lower mean,
99th-percentile and maximum correction changes on all seven transitions;
the average 99th percentile fell from .00205 to .00159 (`boundary173`, 32
samples). Its geometric blink had to be limited to a smaller range to avoid
inverted triangles, and is likewise not captured motion.
For full rendered RGB over the same regions, the average mean change was
.00838 versus .00834, while peak changes were mixed (`boundary172`). The
contact-only result therefore does not establish a broadly smoother animation.
On two further identities at 128 samples, the combined options produced small,
mixed still-image changes; both eye-crop metrics worsened slightly for identity
300 in its front and angled views (`boundary160`).

An experimental option, `material_overrides: {"skin_indirect_firefly_guard": 1}`,
captures a guided, denoised indirect-light pass and substitutes it only for
extreme positive indirect samples before skin diffusion. It is **off by default**.
This addresses isolated dark skin pixels caused when a noisy indirect-light
spike is subtracted from a denoised beauty image. On three studio-lit eyes and
one high-contrast light setup, most flagged values were closer to a 128-sample
estimate, but a few were worse; the scene-linear threshold may also suppress
real concentrated indirect light and is not exposure invariant. Crop similarity
results were mixed. Treat this as a diagnostic option, not a validated general
lighting model. Evidence: `tmp/vhuman-eye-quality/iris83` through `iris86`.
The public GPU render path was also exercised at 512 px / 16 samples with the
option enabled: it recorded 420 guarded opaque-skin pixels, reproduced its
stored image exactly when recomposed, and preserved alpha. Peak whole-device
usage was 10169 MiB (`tmp/vhuman-eye-quality/iris87`).
Mode `skin_indirect_firefly_guard: 2` is a separate experiment that scales the
threshold per pixel using direct illumination, so uniform light-intensity scaling
does not change the selection. It avoids a frame-wide statistic that varies with
camera framing. Tests at studio, dim, high-contrast and bounced-light settings
showed fewer missed dim-light spikes than mode 1, but some selected samples still
became less accurate against a 128-sample estimate; neither mode is a default.
The direct-light floor becomes zero under purely indirect illumination, and
isolated genuine caustics may still be classified as outliers. See
`tmp/vhuman-eye-quality/iris88` through `iris91`.
A synthetic narrow indirect highlight with a blurred guide demonstrates this
failure explicitly: either mode can replace its true peak with the dimmer guide
(`tmp/vhuman-eye-quality/iris93`).
In a separate rendered glass-sphere scene, a broad refractive light pool was
visible in the indirect pass. Both modes changed one of the 328 pixels with
the strongest on/off signal and reduced mean error in that region against a
1024-sample estimate (`sss169`). This is one broad light pattern, not evidence
that the guard preserves narrow caustics in general. A second sphere/light
configuration also produced a broad pool and changed none of its 328 strongest
signal pixels (`sss170`); neither experiment resolves the narrow-highlight case.
A smaller glass sphere produced a resolved 10-by-8-pixel signal region against
a 4096-sample on/off reference. Neither guard selected any of its 56 pixels
(`sss171`). A still smaller rendered caustic occupied a 3-by-3-pixel region.
Against an 8192-sample reference, both guards reduced its mean intensity and
increased error in three of four 64-sample seeds; one seed improved (`sss176`,
`sss177`). This is direct evidence that either guard can erase real narrow
indirect light, and reinforces keeping them off by default. Subpixel features
remain untested.

The separate experimental `skin_scatter_correction_sigma_m` override filters
only the additive diffusion-lighting correction using a separable Gaussian
with skin coverage, geometric-normal and depth rejection. It leaves beauty,
albedo and specular inputs unfiltered. For comparison, `.00013` means 0.13 mm
projected at median visible skin depth; projected sigma is capped at 8 pixels.
The default is zero, which bypasses this filter. Tests on three 512-pixel views
reduced low-sample error against a 128-sample estimate; two 1024-pixel
development crops improved LPIPS while DISTS was mixed. It also suppresses
legitimate high-frequency changes in the diffusion correction, so it remains
experimental pending wider full-resolution and lighting validation. See
`tmp/vhuman-eye-quality/sss108` through `sss124`.
On this CPU, isolated filter measurements were about 0.21 s at 512 pixels,
1.49 s at 1024, and 18.1 s at 2048 for projected sigmas of 1, 2 and 4 pixels;
the 2048 synthetic test used about 620 MiB peak process RSS (`sss118`).
The public portrait and layered-eye HIP paths passed with the option enabled;
their stored results recompose exactly, preserve alpha, and match an explicit
zero to the disabled path (`sss124`, `sss125`). On the previously examined
005/194 identities, fresh 1024-pixel / 128-sample captures improved both
LPIPS and DISTS with the separable filter (`sss131`). Those are resolution
checks, not unseen-identity validation.
In a 512-pixel 194 view, the 128-sample correction filter increased skin MAE
against a 512-sample Blender estimate from 0.00481 to 0.00562 (`sss130`).
At 1024 pixels it also reduced the high-frequency softening produced by the
subsurface effect in two views (`sss132`); Unreal and Blender geometry differ,
so those frequency comparisons are descriptive. These results argue against
enabling the correction filter by default.
Two further held-out identities at 512 pixels improved LPIPS with the filter,
but DISTS improved in only one (`sss127`); this does not resolve the
high-sample and frequency-response concerns.

The optional `skin_scatter_denoise` selects guided denoising of the diffuse
lighting used to compute the scattering correction: `1` denoises indirect
light, and `2` denoises both direct and indirect light. The default `0` uses
the original passes. This does not denoise the native beauty or the albedo.
Mode 2 reduced skin error against 128- and 512-sample Blender estimates in
two views (`sss134`, `sss135`), but 32-sample comparisons against Unreal
regressed in some views (`iris81`), and guided denoising can erase legitimate
small lighting features. It is experimental and cannot be combined with
`skin_indirect_firefly_guard`.
The public portrait HIP path with mode 2 and the layered-eye HIP path with
mode 1 passed saved-image recomposition and alpha checks (`sss136`, `sss138`).

The opt-in `skin_scatter_symmetric=1` uses fourfold-symmetric diffusion
samples. It removes 90-degree rotational asymmetry for a point light on a flat surface
and reduced error against a 3072-sample estimate on four skin-lighting crops,
but its point-light error against that estimate was slightly higher and two
Unreal image comparisons were mixed (`sss139`, `sss140`). It therefore remains
experimental and defaults to zero; fourfold symmetry does not guarantee
rotational invariance at other angles.
Two additional held-out identities also had mixed perceptual results
(`sss143`). The public portrait HIP render passed saved-image recomposition,
disabled-path and alpha checks (`sss141`).

For MetaHuman-to-GNM normal-map conversion, the experimental
`normal_map --frame-sigma-px 1 --orbital-mask-npz <skin_detail.npz>` smooths
the transferred mesh-normal field within the supplied orbital weights while
retaining the source texture detail. Sigma is measured in atlas pixels (the
studies used a 2048-pixel atlas); zero is the unchanged default. In one view
it softened a jagged upper-lid highlight, but comparisons across identities
and metrics were mixed (`sss157`, `sss159`, `sss160`). It is a conversion
option for the MetaHuman comparison workflow, not a general reconstruction
default, and it may alter valid shape-dependent shading.
The Gaussian is validity-weighted but does not identify UV islands, so use a
matching mask and inspect atlas boundaries before applying it to other layouts.
At 1024 pixels and 128 samples, the softened highlight remained visible;
LPIPS worsened in both tested views and DISTS improved in one (`sss162`).
Two additional full-resolution native views changed little visually; LPIPS
increased slightly in both and DISTS improved in one (`sss164`).
In a further paired 512-pixel, 128-sample front/yaw comparison on identities
145 and 256, eye-crop LPIPS decreased in all four views, while DISTS decreased
in three and increased in the other (`sss180`). The changes were visually
subtle, and these identities helped select the smoothing amount, so they are
not an independent identity holdout. The staged normal texture hashes and
rendered alpha were verified for every pair.

The eye-contact band also follows fractional eye coverage immediately beside
the opaque aperture. This reduces the cutoff at 50% coverage without applying
the treatment to isolated fractional samples elsewhere. Interior distance is
also interpolated through fractional coverage; in a synthetic subpixel-motion
check the maximum step in contact weight fell from 0.453 to 0.010
(`boundary112`). Ten cached views
showed small, mixed perceptual changes; the full-resolution crop averages
changed LPIPS from 0.19164 to 0.19149 and DISTS from 0.15541 to 0.15539,
while boundary MAE changed from 23.255 to 23.282 (`boundary115`). Alpha and
pixels outside the newly treated partial-coverage band are preserved.
The public portrait and layered-eye HIP renders recomposed exactly and
preserved alpha and all nonfractional eye pixels (`boundary116`, `boundary117`).


Statistics learned from MetaHuman Creator characters generated locally (UE 5.8.4, Linux, headless) feed vhuman's
reconstruction: a GNM-UV skin albedo/cavity prior, a dense-ear shape/colour prior, eye parameter presets and a GNM
identity prior. Plan: `~/.claude/plans/unreal-engine-eula-now-velvety-giraffe.md`.

**Policy.** Only derived statistics leave the dataset root (see `server/vhuman/README.md`, Provenance and licensing; the
EULA revision/clause are placeholders the user fills). No Epic cloud service is used (no auto-rig, no 2K-8K textures).
Raw meshes/textures stay under the dataset root; artifacts carry `report.json` provenance and are off by default.

## Locations
| What | Path |
|---|---|
| UE build | `/mnt/nvme02/local/ue` (5.8.4, all MetaHuman plugins + Creator Core Data) |
| UE project | `/mnt/nvme01/work/vhuman-mh/MHGen/MHGen.uproject` (Zen store `/mnt/nvme01/work/vhuman-mh/ZenData`, local DDC `.../DDC`) |
| Dataset | `/mnt/nvme01/data/vhuman-mh/v1/` (`jobs.jsonl`, `sampling.json`, `manifest.json`, `presets/`, `samples/<id>/`, `logs/`) |
| Artifacts | `/mnt/nvme02/models/vhuman-mh/` (`binding_v1/`, `skin_prior_v1.npz`, `ear_prior_v1.npz`, `eye_prior_v1.json`, `identity_prior_v1.npz`) |

## Commands (repo root, `export PYTHONPATH=$PWD`, `P=tmp/vhuman-cpu-venv/bin/python`)
| Step | Command |
|---|---|
| Run any UE script headless | `$P -m server.vhuman.metahuman.launch <ue_script.py> [args] --result <json>` |
| Dump preset characters | `launch dump_presets.py --result /mnt/nvme01/data/vhuman-mh/v1/presets/face_presets.json` |
| Dump eye presets (T3D) | `launch dump_eye_presets.py --result .../presets/eye_presets.json`; parse with `$P -m server.vhuman.metahuman.eye_prior_fit` |
| Sample design | `$P -m server.vhuman.metahuman.sampling` (499 jobs: 29 presets, 120 blends, 350 per-patch Gaussian; 10% held out) |
| Generate | `$P -m server.vhuman.metahuman.generate --batch 50` (resumable; ~5 s/sample + ~60 s per editor session) |
| Archetype template | `$P -m server.vhuman.metahuman.template` (from smoke 03 outputs) |
| GNM registration + texel map | `$P -m server.vhuman.metahuman.register --res 1024 [2048]` |
| Identity prior | `$P -m server.vhuman.metahuman.binding --identity-prior` |
| Skin prior | `$P -m server.vhuman.metahuman.skin_prior_fit --eval-mask <candidate>/skin_front_protected.png` |
| Ear prior | `$P -m server.vhuman.metahuman.ear_prior_fit` (closest-point projection; `EAR_PRIOR_PROJECTION=ray` optional) |
| Ear fit with the prior | ear workspace env + `EAR_PRIOR=/mnt/nvme02/models/vhuman-mh/ear_prior_v1.npz EAR_PRIOR_MODES=3 EAR_CROSS_PENALTY=0.02`, then `fit_band_joint.py`, `shape_ears.py`, `synth_ear_texture.py` as in `stage3_ears.sh` (outputs via `EAR_FIT_OUT_DIR`, `EAR_FIT`, `EAR_SHAPE_OUT`) |
| Reference-render cameras | `$P -m server.vhuman.metahuman.render_plan` then `launch render_batch.py <batch.json>` |
| UE-vs-Cycles comparison | ROCm venv: `-m server.vhuman.metahuman.compare_render --every 10` (pseudo-candidates + Cycles renders with the reference cameras), then `--report` (CPU: `HIP_VISIBLE_DEVICES=`); outputs `/mnt/nvme01/data/vhuman-mh/compare_v1/` |
| Light calibration | `-m server.vhuman.metahuman.light_calibration all` (ColorChecker probes: Cycles CPU + one UE session, then the fit), `batch` writes `render_batches/vhstudio.json`; `launch render_batch.py <vhstudio.json>` renders `<cam>__vhstudio.exr`; `compare_render --report --ue-env vhstudio`; outputs `/mnt/nvme01/data/vhuman-mh/light_calib_v1/` |
| MH normal conversion | `-m server.vhuman.metahuman.normal_map <id> [--out PNG]`, `--check <id> [--converted PNG]` (convention tests); `compare_render --update-normal` writes it into the pseudo-candidates, `--variant normal` renders/reports with it (`offline_render(skin_normal_map=True)`) |
| Normal prior | `-m server.vhuman.metahuman.normal_prior_fit [--res 1024 --k 40 --workers 8]` -> `normal_prior_v1.npz` (+ report); `normal_prior_fit.write_png(prior, 'detail_xy', png)` encodes a Blender skin_normal.png |
| Eye mapping | `-m server.vhuman.metahuman.eye_map calibrate` (UE eye close-ups -> `eye_colour_calibration_v1.json`), `eye_map params <id> [--version 1\|2]` (default 2 = sclera tint from the measured MH sclera); `compare_render --variant normal_eyes` / `normal_eyes2` renders with `head/fit_mh_eyes.json` (v1) / `fit_mh_eyes_v2.json`; `--eyes` / `--ears` score the eye and ear sets |
| Windows UE worker | `-m server.vhuman.metahuman.winjob sync` / `render <ids> --env Studio\|VHStudio` / `compare <ids>` (`VHUMAN_WIN_HOST`, DX12 SM5 on RX 570; setup `../vhuman-win10.md`; pushes params/cameras, runs render_batch.py detached via WMI, pulls into `DATASET_ROOT/win_v1`); roots overridable via `VHUMAN_UE_ROOT` / `VHUMAN_MH_PROJECT` / `VHUMAN_MH_DATA` / `VHUMAN_MH_MODELS` / `VHUMAN_UE_RHI` |
| Windows Cycles (CPU) | `-m server.vhuman.metahuman.win_cycles render <tag> [--ids] [--variants] [--cameras] [--profile fast]` / `pull <tag>` / `status|finish|resume <tag>` (disconnectable: `render --score <set>` records the scoring, `finish` waits, pulls and scores later), then `compare_render --confirm --report --profile fast` (Linux stages heads with `offline_render.render(stage_only=True)`, the worker renders 512 px / 32 spp in ~15 s/view into `<variant>__fast` dirs; `../vhuman-win10.md` §10) |
| Material sweep | `compare_render --sweep` (renders SWEEP x SWEEP_IDS x SWEEP_CAMERAS on the GPU), then `--sweep --report --ue-env vhstudio` (CPU) -> `compare_v1/sweep_vhstudio.json` ranked by skin LPIPS |
| Use the skin prior | `bake_portrait(..., skin_prior=PATH)`; `tmp/vhuman-ab-cycles/scripts/rebake.py ... --skin-prior PATH` |
| Tests | `$P -m pytest -q server/vhuman/test_metahuman_prior.py` |

## Verified facts (2026-11-03)
- Editor runs headless on RADV Vulkan (`-vulkan -RenderOffScreen`, never `-nullrhi`), ~5 GB VRAM. Each session segfaults at
  exit *after* the script finished; `launch.py` tolerates it and disables core dumps. UE cache paths default to `~/.config/Epic`
  (root FS was full): the Zen store is redirected with `[Zen.AutoLaunch] DataPath` in the project's DefaultEngine.ini.
- Face model: 22 PCA patches, 1,397 coefficients (`[n]`, per patch scale, quat xyzw, translation, nPCA, weights); PCA sizes
  38,54,53,50,41,54,53,45,44,56,56,82,82,69,40,40,75,46,52,64,52,52; patches 11/12 are the ears. 79 landmarks.
- Preview actor 'Face' component: head = vertices 0..24,048 in DNA order (identity-template vertex ids index it directly),
  teeth 4,246, eyes 770 each, plus lash/shell parts. UE cm, +X subject-left, +Y facing, Z up -> vhuman (x, z, y)/100.
  The glTF export of the head is already in the vhuman frame (metres) with 24,414 UV-split vertices, UVs top-left origin.
- Textures: `character.synthesized_face_textures[type]` -> `ImageWriteBlueprintLibrary.export_to_disk(tex, png,
  ImageWriteOptions(format=PNG, async_=False))` gives exact 1024^2 RGBA8 (9 types incl. animated CM1-3/WM1-3).
- Eye presets: 12 entries; struct fields are not script-visible -> T3D export (`EyePresets.T3D`) parsed by eye_prior_fit.
- Python API names that differ from C++: `iris.pattern`, `iris.rotation`, `unreal.MaterialLibrary`,
  `SystemLibrary.get_component_bounds`, `comp.get_world_transform()`; `unreal.Array` must be listed before json.dump.
- Reference renders (`render_batch.py`) must span editor ticks: run synchronously, every capture happens inside one frame
  and shows placeholder materials (checkerboard skin, no eyes). The script drives a generator from
  `unreal.register_slate_post_tick_callback` (non-reentrant: level loads pump ticks), runs the
  `Editor.Async*CompilationFinishAll` console commands and settles ~8 s per head, and calls
  `unreal.EditorPythonScripting.set_keep_python_script_alive(True)`, because `-ExecutePythonScript` otherwise quits the
  editor once the script returns. Files run in a private scope (`__name__` is not `'__main__'`), so `gen_batch.py` guards
  its entry with `__name__ != 'gen_batch'` (render_batch imports it). Set `preview_material_type = EDITABLE` ("Skin")
  BEFORE `try_add_object_to_edit`; set afterwards, the spawned actor keeps the "Topology" region/wire material.
  `SceneCaptureComponent2D.show_only_actors` cannot be assigned; use `show_only_actor_components`. ~15 s per head
  (10 cameras x lit + base colour EXR, ~14 MB); the torso shows grey body-mask patches (head views unaffected).
- Registration GNM skin -> MH archetype: surface distance median 0.10 mm, p95 0.74 mm (ears 0.43 mm median); texel map
  valid for 98.8% of GNM skin texels (holes in the ear folds). Resampled MH albedo lines up with vhuman atlases.

## Results so far
- Skin prior smoke (14 train heads, 3 held out, s1 front-bake coverage observed): unobserved-texel error 2.84/255 sRGB vs
  4.43 for today's observed-median fill and 3.53 for the gain-matched mean. Full 499-head fit and the s1/s2 render gate:
  `tmp/vhuman-ab-cycles/scripts/stage_prior.sh` (B4 = refined-ear fix, B5 = + skin prior; `render_matrix.py --plan prior`).
- Skin prior, full fit (450 train / 49 held-out heads, k_lf 80, k_hf 60): unobserved-texel error 1.65/255 sRGB vs 5.42 for
  the observed-median fill and 5.18 for the gain-matched mean (-70%).
- Ear prior (450/49 heads, per side, 20 modes): shape held-out 0.43-0.46 mm (mean ear 1.36 mm); relief heights held-out
  0.08 mm (mean only 0.19-0.21 mm), mean |relief| 1.2-1.35 mm.
- Ear prior wired into the fitter (2026-11-03). Representation: relief heights along smoothed normals above the ear's own
  Laplacian low-pass, applied by `ear_mesh.apply_prior_relief` (learned counterpart of `apply_relief`; same root feather,
  band unchanged, 16 smoothing passes). Built on a reference refined-ear topology (`binding_v1/ear_topology_ref.npz`,
  subdivision operator from the s1 workspace) and remapped per subject by `ear_mesh.adapt_prior`: the key of each dense
  vertex is `operator @ GNM template` (s2's 4,108 ear vertices match s1 exactly; only band vertices differ).
  Fitter env: `EAR_PRIOR=/mnt/nvme02/models/vhuman-mh/ear_prior_v1.npz EAR_PRIOR_MODES=3 EAR_PRIOR_RIDGE=.05
  EAR_CROSS_PENALTY=0.5`, outputs via `EAR_FIT_OUT_DIR` / `EAR_FIT` / `EAR_SHAPE_OUT`; start `fit_band_joint.py` from the
  BASELINE `fit_global` (default `EAR_FIT_GLOBAL_OUT=fit_global`). `shape_ears.py` reuses the fit's scored placement
  (`new.npy`) in prior mode. Texture: `synth_ear_texture.py` uses the prior's per-vertex colour pattern instead of
  `anatomical_tint` when `EAR_PRIOR` is set (`EAR_PRIOR_TINT=0` disables, cap +-0.3 log).

  | s1 / s2 | authored relief | learned prior |
  |---|---|---|
  | silhouette IoU L / R | 0.61 / 0.53 ; 0.60 / 0.60 | 0.63 / 0.57 ; 0.61 / 0.61 |
  | min area ratio | 0.03 ; 0.22 | 0.25 ; 0.27 |
  | ear crossings | 0 ; 0 | 0 ; 0 |
  | p99 edge stretch | 1.69 ; 1.30 | 3.39 ; 2.78 (deeper folds on the same mesh; not visible in shaded close-ups) |
  | background spill px | 126+77 ; 72+131 | 294+288 ; 116+148 (larger, rounder lobe) |

  Shaded close-ups (`scripts/ear/ear_relief_sheet.py`): rolled helix rim, antihelix with crura, deep concha, tragus and a
  rounded lobe instead of the generic authored ridges. Outputs: `ears/<s>_B/{fit_band_prior4|3,shape_prior4|3}`.
  Pitfalls: absolute-shape prior (`apply_prior_shape`, `EAR_PRIOR_MODE=shape`) misfits the band by 3.5 mm and folds;
  the band topology varies per subject (key vertices by the subdivision operator, not by index or rebound binding);
  starting from the prior's own global fit folds every evaluation; re-running ARAP with more iterations after the fit
  created self-crossings; `ear_mesh.SIDES` 'left' is the -x ear = the prior's subject-'right'.

- Render gate (2026-11-03, `tmp/vhuman-ab-cycles/scripts/prior_report.py`, Cycles studio neutral, LPIPS on parser crops;
  held-out = Edit-2511 +/-90 views never used in fitting). B3 cleaned bake, B4 refined-ear bake fix (ears05), B5 = B4 +
  skin prior (ears06), B6 = B5 + learned-prior ears with the clay-conditioned Edit-2511 ear texture (ears07, via
  `ears02p`; stage scripts `stage_b6a.sh` / `stage_b6b.sh`).

  | LPIPS (lower better) | s1 B5 | s1 B6 | s2 B5 | s2 B6 |
  |---|---|---|---|---|
  | front face | 0.432 | 0.427 | 0.511 | 0.499 |
  | +40 / -40 | 0.497 / 0.491 | 0.506 / 0.493 | 0.554 / 0.523 | 0.537 / 0.509 |
  | held-out ear +90 | 0.572 | 0.551 | 0.612 | 0.591 |
  | held-out ear -90 | 0.547 | 0.571 | 0.554 | 0.546 |

  Verdict: the skin prior alone (B5 vs B4) is metric-neutral (+/-0.006) and visually a mild warm/rosy fill on the back
  of the head without seams; keep it opt-in, recommended only together with the prior ears. B6 wins 7 of 8 view rows;
  the s1 -90 ear regresses (that ear's two 90-degree generated views failed the projection guard, IoU 0.80/0.83 < 0.9,
  so its texture comes from 45-degree views only). The report drops parser specks smaller than 5% of the largest
  component (a 48 px 'ear' speck on s1 B6's lips had stretched the crop; mask IoU 0.12, LPIPS 0.712).
  B6 has new ear geometry, so B3's expression appearance/motion do not apply (neutral renders only until refit).

## Open
- B6 follow-ups: regenerate s1's 90-degree ear views (other seeds) so the projection guard passes; refit the
  expression appearance on the B6 geometry. Dense-ear L/R correspondence is not one-to-one (prior fitted per side).
- Normal-map prior (PCA of converted normals; conversion exists, see normal_map.py), `skin_templates` cavity from the prior.
- Eye mapping: render the normal_eyes variant and compare eye close-ups (colour is calibrated on albedo, not yet on renders).
- UE-vs-Cycles harness (2026-11-03): reference renders done (120 ids); `compare_render.py` builds GNM pseudo-candidates
  (bound skin exact, GNM identity fit for eyes/mouth, MH albedo via texel_map_2048, flat normal, template ORM/eyes),
  translates head + cameras to vhuman's head anchor so the studio rig applies, and renders through
  `offline_render.render(explicit_camera=...)`. First head: ear view silhouette IoU 0.9998 (cameras and geometry agree);
  Cycles skin reads glossier (forehead/lip specular) and eyes use template parameters. Open: ColorChecker light
  calibration in both renderers, MH tangent-normal conversion, MH eye parameters on the Cycles side, material sweep.
  First run (12 heads = render ids[::10], 10 cameras, ~21 s/view; `compare_v1/summary.json`):

  | camera | head-row IoU | Cycles->UE gain | LPIPS | DISTS | dE |
  |---|---|---|---|---|---|
  | front | .80* | .31 | .378 | .288 | 12.8 |
  | yaw +40 / -40 | .83* | .11 / .37 | .444 / .372 | .332 / .286 | 13.2 / 8.7 |
  | yaw +90 / -90 | .86* | .07 / .37 | .588 / .542 | .377 / .326 | 10.5 / 15.1 |
  | ear L / R | 1.00 | .09 / .50 | .250 / .226 | .235 / .203 | 9.3 / 6.2 |
  | eye L / R | 1.00 | .16 / .35 | .313 / .301 | .226 / .209 | 8.5 / 5.4 |
  | mouth | 1.00 | .31 | .261 | .202 | 11.9 |

  (*) wide views include UE's torso/tank top, which the head-only pseudo-candidate lacks. Gain is per view, exposure-
  matched before LPIPS; it is consistent across heads (spread < .05), so the gap is the light rig, not the skin: the
  vhuman studio key sits on the subject's left (+yaw side, ~5x brighter than the -yaw side relative to UE Studio).
  Visual: Cycles skin is glossier and smoother (freckles/pores attenuated: flat normal, no MH cavity), eyes are the
  template (brown iris, wide sclera, upward gaze). The ColorChecker light calibration below removes that.
- ColorChecker light calibration (2026-11-04, `light_calibration.py`, `chart_probe_blender.py`, `ue_scripts/chart_probe.py`):
  a matte 24-patch chart at the reference face, facing each wide reference camera, captured with that camera. The UE
  'Studio' level is four lights (KeyLight/FillLight/BounceLight spots + an AmbientLight point; no sky light). The vhuman
  rig replicated in UE (equal-area RectLights, same relative placement/aim, Studio lights hidden) converts with one
  factor for all three lamps: UE candela = 0.501 x P_cycles / pi (key .501, fill .501, rim .502; spread over probes
  1-6%); the calibrated replica reproduces the Cycles lamps within 0.4% on every probe, superposition holds (<0.4%),
  and the Cycles studio world (no UE counterpart) is 0.6-1.8% of the chart. Studio/Cycles luma per probe: front .36,
  +40 .24, -40 .50, +90 .11, -90 .56 (neutral colour in both): this is the first run's per-side gain. Render-batch env
  `VHStudio` = the Studio level with its lights hidden plus the calibrated rig around each head.
  Comparison rescored against VHStudio references (12 heads, `compare_v1/summary_vhstudio.json`; before -> after):

  | camera | Cycles->UE gain | LPIPS | DISTS | dE |
  |---|---|---|---|---|
  | front | .31 -> 1.12 | .378 -> .142 | .288 -> .186 | 12.8 -> 5.7 |
  | yaw +40 / -40 | .11/.37 -> 1.10/1.11 | .444/.372 -> .252/.112 | .332/.286 -> .206/.174 | 13.2/8.7 -> 9.2/5.2 |
  | yaw +90 / -90 | .07/.37 -> 1.12/1.12 | .588/.542 -> .350/.365 | .377/.326 -> .216/.235 | 10.5/15.1 -> 10.6/12.5 |
  | ear L / R | .09/.50 -> 1.10/1.23 | .250/.226 -> .179/.224 | .235/.203 -> .173/.199 | 9.3/6.2 -> 5.4/5.4 |
  | eye L / R | .16/.35 -> 1.07/1.07 | .313/.301 -> .282/.285 | .226/.209 -> .207/.205 | 8.5/5.4 -> 4.5/4.9 |
  | mouth | .31 -> 1.14 | .261 -> .254 | .202 -> .199 | 11.9 -> 5.6 |

  The gain is now one number for every view (~1.1: same light, UE skin ~10% brighter), so the remaining differences are
  material: Cycles skin has a stronger, broader forehead/cheek specular, less pore/freckle relief (flat normal, no MH
  cavity) and slightly pinker lips; the eyes are the template. Next: MH tangent-normal conversion, MH eye parameters on
  the Cycles side, then the material sweep against the VHStudio references.
- MH tangent-normal conversion (2026-11-04, `normal_map.py`): MetaHuman normals are red = +u, green = +row (UE/DirectX),
  measured, not assumed: red and green divergences correlate with the cavity detail with the same sign (r -.42..-.50 on
  three heads) and the height integrated under that reading is negatively skewed (pores/creases are pits). Per GNM
  texel the MH normal is expanded in the MH smooth tangent frame (MikkTSpace-like: T from UV per vertex+UV corner,
  B = sign N x T) at the texel map's MH triangle/barycentrics on that identity's MH head and re-expressed in the GNM frame
  of its bound skin, written with Blender's convention (green = +V up). The converted map passes the same tests read
  that way (r -.34/-.36, skew -1.4). Cycles: `skin_normal_map=True` feeds the candidate's skin_normal.png into the full
  skin graph, whiteout-blended with the tiled pores (graph unchanged when off).
  Result (12 heads vs VHStudio, aligned scoring, see below; LPIPS flat -> MH normal -> + MH eyes):

  | camera | LPIPS | dE | DISTS |
  |---|---|---|---|
  | front | .136 -> .128 -> .127 | 5.36 -> 5.24 | .184 -> .178 |
  | yaw +40 / -40 | .093/.110 -> .082/.099 -> .082/.098 | 4.95/5.12 -> 4.81/4.94 | .172/.173 -> .157/.153 |
  | yaw +90 / -90 | .084/.112 -> .075/.100 | 5.70/5.92 -> 5.71/5.82 | .167/.182 -> .155/.173 |
  | ear L / R | .184/.229 -> .194/.257 | 5.31/5.38 -> 5.57/5.93 | .176/.203 -> .185/.200 |
  | eye L / R | .282/.285 -> .269/.246 -> .267/.243 | 4.49/4.87 -> 4.41/4.63 | .207/.205 -> .199/.191 |
  | mouth | .254 -> .228 | 5.58 -> 5.38 | .199 -> .191 |

  MH normals win every face view on every head (+40: 12/12, +90: 11/12) and lose both ears on 9/12 heads: pore
  sparkle much stronger than UE plus streaks in the concha (the GNM ear is coarser than the MH ear, so the MH ear
  normals do not line up with its folds) -> ears should use flat/attenuated normals (not confirmed: the 12-head ear
  run below finds the MH ear normals closer to UE than a faded ear). MH eyes (`normal_eyes`): iris
  colour family now follows MH (blue/grey/green instead of the template brown), pupil size varies; mapped irises are
  less saturated than UE (green-hazel reads grey), the Cycles sclera is whiter, and UE's red freckles are missing on the
  Cycles side; metric gain small (eye LPIPS -.002/-.003).
  Scoring fix (2026-11-04): `compare_region` cropped each image around its own parser labels, so a stray 'skin' speck
  in the Cycles render grew its crop to the full frame (LPIPS jumped .07 <-> .36 between near-identical renders).
  `compare_render` now uses `compare_region(..., aligned=True)` (specks < 5% of the largest component dropped, one box
  for both images; same camera). All three summaries were rescored; the VHStudio table above used the old scoring
  (its wide views were inflated: aligned flat front/+40/-40/+90/-90 = .136/.093/.110/.084/.112; pre-fix summaries in
  `tmp/vhuman-ab-cycles/metrics/pre_aligned/`).
- Normal prior (2026-11-04, `normal_prior_fit.py`, `normal_prior_v1.npz`): 450 training heads converted on their own MH
  head / bound GNM skin at 1024; mean, std, PCA K=40 (58% variance) on tangent xy, mean split into a 12 px low-frequency
  part (mostly MH-vs-GNM mesh shading) and the detail part (pores, creases: what transfers to non-MetaHuman heads).
  Held-out rms xy error: flat .117, prior mean .060, detail-only .070, low-frequency-only .104.
- Eye mapping (2026-11-04, `eye_map.py`): vhuman's iris chart is its own palette, so MH U/V are calibrated through
  measured colour: UE base-colour eye close-ups -> pupil (black, < .004), iris edge (largest radial luminance step in the
  horizontal sectors; the GNM limbus estimate is ~30% too small for MH irises), inner/outer ring colours -> nearest
  chart U/V + saturation/tint. 240 eyes (120 heads): reproduction dE mean .52, p95 2.2; measured pupil/iris ratio
  .12-.65 correlates with MH dilation (r .77) and sets vhuman dilation (ratio / P_REF). Other same-name fields are copied
  with range conversion (pattern IRISnnn -> pattern_n, cornea size x .2/.165). Not yet verified in rendered close-ups.
  `eye_prior_fit` now adds `vhuman_distribution` to eye_prior_v1.json: a 4-component full-covariance GMM (numpy EM) over
  chart U/V, saturation, tint and dilation of the 240 mapped eyes, and `sample_vhuman(prior, rng)` draws validated
  vhuman eye parameters from it. MH irises in base colour are darker than vhuman's chart: mapped V averages .07 (primary)
  and tint ~.8 does the rest, so the chart's dark end is where MetaHuman eyes live. Sclera albedo (measured beside the
  iris) is only weakly explained by MH skin u/v (r2 .10, rms .069): sclera tint is not a skin-tone function in MH data.
- B7 (2026-11-04, `tmp/vhuman-ab-cycles/scripts/make_ears08.py`): B6 + normal prior `detail_xy` as skin_normal.png with
  `skin_normal_map=True` for the real subjects s1/s2; render plan `normal` (B7 on the prior views + eye close-ups, B6 eye
  close-ups), report `prior_report.py B5 B6 B7`.
  Result (2026-11-04, LPIPS B6 -> B7): s1 face .427 -> .424, +40/-40 .506/.493 -> .502/.491, held-out ears +90/-90
  .551/.571 -> .553/.537; s2 face .499 -> .498, +40/-40 .537/.509 -> .533/.509, ears .591/.546 -> .558/.543. B7 wins or
  ties 7 of 8 rows (s1 +90 ear +.002) and fixes most of B6's s1 -90 ear regression; held-out ear DISTS improves 3 of 4.
  Visually close to B6 at sheet scale (slightly more pore/crease texture). Unlike the MetaHuman pseudo-candidates, the
  real subjects use the dense refined ears and only the transferable detail part, so the ears do not regress here.
  VRAM peak 8.2 GB on eye close-ups (7.1 GB wide views).
- Material sweep harness (2026-11-04): `offline_render(material_overrides=...)` scales the full skin graph's roughness and
  specular, sets the skin coat weight (.10) and the SSS weight/scale (.6/.5); empty = the authored graph (no extra nodes).
  `compare_render` variants `sw_<param>_<value>` = 'normal' + one setting; SWEEP: roughness x .8/1.25/1.5, specular
  x .5/.75/1.25, skin coat 0/.05/.2, SSS weight .3/.45/.8, SSS scale .25/.75/1, normal strength .5/1.5, micro .0/.5 on
  mh_000000/145/300 x front/yaw_p40 (114 renders).

  Result (2026-11-04, 3 heads x front/+40 vs VHStudio, `compare_v1/sweep_vhstudio.json`, aligned scoring; baseline
  `normal` LPIPS .1155 / DISTS .1789 / dE 5.67). All effects are small (|dLPIPS| <= .011):
  roughness x1.25 is best on average (LPIPS -.0027, DISTS -.0041, dE -.90) and x1.5 has the best DISTS (-.0048), but
  per head it helps mh_000000 / mh_000300 and hurts mh_000145 (LPIPS +.006 on both views), so the gap is
  identity-dependent (MH roughness varies per character), not a global scale; x0.8 is clearly worse (+.011, dE +1.5).
  Specular x0.5-0.75 lowers LPIPS slightly but raises dE (+.5..+1.4, the skin gets darker); coat 0 is a small,
  consistent gain (-.0013, 5 of 6 views); SSS weight/scale are flat (+-.001); normal strength 1.0 is optimal (0.5 and
  1.5 both worse); micro-pores 0 is neutral (DISTS -.003), 0.5 worse. Candidate change: coat 0.10 -> 0 and roughness
  x1.25, to confirm on more heads and on the s1/s2 photo metrics before changing defaults.
- Fast confirm (2026-11-04, Windows Cycles CPU 512 px / 32 spp, `win_cycles`; 12 heads x 6 cameras = 72 views vs
  `normal`, `compare_v1/confirm_vhstudio__fast.json`; fast vs full probe on mh_000000 front: LPIPS .0936 vs .0937):
  roughness x1.25 LPIPS -.0043 (58/72 views, 10/12 heads), DISTS -.0046, dE -.74, mouth gains most (-.011), front least
  (-.0007), mh_000145 worse (+.0093); coat 0 -.0015 (63/72 views, 12/12 heads); both together -.0044 (54/72, 9/12).
  Roughness stays identity-dependent; defaults unchanged.
- Ear normal fade (2026-11-04, `normal_ear_scale` via the `ears` template mask; 12 heads x ear_left/right, +-90, front,
  `compare_v1/ears_vhstudio__fast.json`; deltas vs `normal`, ear-region metrics on 46 views):

  | variant | ear LPIPS | ear DISTS | ear dE | view LPIPS | heads better |
  |---|---|---|---|---|---|
  | ear normal 0 | +.0156 | +.0123 | +1.90 | +.0037 | 0/12 |
  | ear normal .5 | +.0016 | +.0029 | +.56 | +.0000 | 9/12 |
  | no normal map | +.0145 | +.0183 | +1.71 | +.0062 | 4/12 |

  The MH ear normals carry relief UE also shows; fading them is worse on every ear camera. Default stays 1.
- Eye mapping v2 (2026-11-04, `eye_map.VERSION = 2`): sclera tint = MH sclera albedo measured on a band beside the iris
  (median linear .39/.27/.25) / vhuman SCLERA_ALBEDO (.74/.70/.66); v1 used the MH tint setting and rendered the sclera
  ~2.6x brighter than UE. 12 heads x eye_left/right, front, +-40 (`compare_v1/eyes_vhstudio__fast.json`, vs `normal`
  = template eyes; eye-region metrics on 35 views):

  | variant | view LPIPS | eye LPIPS | eye DISTS | eye dE | heads better |
  |---|---|---|---|---|---|
  | v1 `normal_eyes` | -.0017 | -.0027 | -.0080 | -2.9 | 10/12 |
  | v2 `normal_eyes2` | -.0052 | -.0076 | -.0135 | -7.5 | 12/12 |

  Eye dE front / eye_left / eye_right: template 28.7 / 32.6 / 28.2, v1 25.8 / 29.6 / 25.5, v2 19.0 / 25.5 / 22.6
  (eye_right eye LPIPS +.009 with v2). Remaining gap: the iris keeps MH chroma in the mapping but renders grey under the
  Cycles cornea reflection, and the GNM eye sits offset from the MH eye.
- Eye optics sweep (2026-11-04, `compare_render.EYE_SWEEP`, `--eye-sweep`; Cycles eye material overrides on top of
  `normal_eyes2`; 4 heads x eye_left/right, front, fast profile, `compare_v1/eye_sweep_vhstudio__fast.json`; eye-region
  deltas on 11 views):

  | setting (authored) | eye LPIPS | eye DISTS | eye dE | views better |
  |---|---|---|---|---|
  | cornea roughness .08 (.025) | -.0015 | -.0014 | -.22 | 7/11 |
  | cornea roughness 0 / .15 | +.0001 / +.0086 | -.0001 / -.0021 | -.05 / -.65 | 6 / 4 |
  | cornea IOR 1.30 / 1.336 / 1.42 (1.376) | +.0004 / +.0004 / +.0006 | ~0 | -.27 / +.01 / +.66 | 3-5 |
  | iris roughness .3 / .8 (.55) | +.0017 / +.0037 | -.0036 / +.0011 | -.82 / +.92 | 2 / 2 |
  | sclera roughness .15 / .4 / .6 (.25) | +.0087 / +.0044 / +.0050 | ~0 | +.04 / -.16 / -.22 | 2-5 |
  | cornea Specular IOR Level 0 / .25 / 1 (.5) | 0 | 0 | 0 | - |

  All effects are at the 32 spp noise level; defaults unchanged. The Specular IOR Level rows are exactly zero because
  Cycles ignores it on a fully transmissive (glass) Principled BSDF, so the cornea-reflection test did not run; lowering
  the IOR to 1.30 (less reflection) does not help either. More likely causes of the grey iris are the iris texture
  brightness and the GNM-vs-MH eye placement.
  IOR 1.0 cornea (2026-11-05, `compare_render.EYE_IOR1_SET`, `win_cycles --score eye_ior1`; 12 heads x eye_left/right,
  front = 36 views vs `normal_eyes2`, `compare_v1/eye_ior1_vhstudio__fast.json`): an invisible cornea gives eye dE -.79
  (19/35 views), eye LPIPS +.0025, eye DISTS +.0032, view LPIPS +.0036 (0/36 better; the cornea highlight is lost).
  The cornea veil is not the eye gap (dE ~19-25); cornea unchanged.
  Iris brightness (2026-11-05, `cycles_scene` override `iris_gain` = iris texture x gain, `compare_render.EYE_IRIS_SET`,
  `win_cycles --score eye_iris`; same 36 views, `compare_v1/eye_iris_vhstudio__fast.json`; eye region dE / LPIPS / DISTS,
  view LPIPS): x.7 +.62 / +.0012 / +.0027, +.0002 (eye DISTS better 5/35); x1.5 -.27 / -.0009 / -.0021, -.0002 (29/35);
  x2.5 -.03 / +.0020 / -.0021, -.0003 (22/35). The iris is slightly too dark (x1.5 is the only setting better on every
  metric) but the effect is at the noise level and dE barely moves: neither the cornea nor the iris brightness explains
  the eye gap, which points at the GNM-vs-MH eye placement/size. Defaults unchanged.
- UE-matched skin (2026-11-04, `compare_render.UE_SET`, `--ue-match`; values from the UE skin readout in
  `skin_sss_survey.md`; 12 heads x front/+-40 = 36 views vs `normal`, fast profile, `compare_v1/ue_match_vhstudio__fast.json`):

  | variant | dLPIPS | dDISTS | ddE | views better | heads better |
  |---|---|---|---|---|---|
  | ue_sss25 / 50 / 100 (SSS weight 1, UE mfp 1.05/.41/.26 mm x .25/.5/1) | +.0003 / +.0001 / +.0001 | +.0008 / +.0008 / +.0007 | -.12 / -.11 / -.08 | 12 / 14 / 17 | 2 / 4 / 6 |
  | ue_spec (specular x1.8, F0 ~.05) | +.0065 | +.0052 | +.11 | 2 | 1 |
  | ue_match (roughness x1.25, coat 0, ue_sss50) | -.0015 | -.0025 | -.65 | 27 | 9 |
  | ue_match_spec (ue_match + specular x1.8) | -.0012 | -.0025 | -.96 | 24 | 8 |

  SSS radius/amount is neutral at 512 px although Cycles' mfp is 2-7x UE's; UE's higher F0 alone hurts (UE's lobe mix /
  cavity attenuation is not reproduced). ue_match dE ~5.0 -> 4.4 on every camera.

  6-camera confirm (`compare_render.UE_CONFIRM_SET`, `win_cycles --score ue_confirm`; 12 heads x 6 cameras = 72 views vs
  `normal`, `compare_v1/ue_confirm_vhstudio__fast.json`):

  | variant | dLPIPS | dDISTS | ddE | views better | heads better | mouth dLPIPS |
  |---|---|---|---|---|---|---|
  | mat_r125_c0 (roughness x1.25, coat 0) | -.0044 | -.0039 | -.67 | 54 | 9 | -.0123 |
  | ue_match (+ ue_sss50) | -.0018 | -.0006 | -.67 | 50 | 8 | +.0006 |
  | ue_match_spec (+ specular x1.8) | +.0021 | +.0005 | -.94 | 40 | 4 | +.0209 |

  UE's SSS gives nothing on the wide views and removes the mouth gain; UE's F0 clearly hurts the mouth. The best setting
  stays roughness x1.25 + coat 0 with the current SSS/specular (mh_000145 worse with all three, +.011-.015).

  s1/s2 photo check (2026-11-05, B8 = B7 + `material_overrides` roughness x1.25, coat 0; `render_matrix.py --plan mat`,
  final 128 spp, `prior_report.py B7 B8`; LPIPS B7 -> B8): s1 face .424 -> .424, +40/-40 .502/.491 -> .504/.492, held-out
  ears +90/-90 .553/.537 -> .548/.544; s2 face .498 -> .498, +40/-40 .533/.509 -> .535/.510, ears .558/.543 -> .560/.543.
  A tie or slightly worse (DISTS flat to worse); the MetaHuman gain is matching UE's look, not photo realism.
  Defaults unchanged.
- 12 h goal round (2026-11-05, `goal-vhuman-12h.md`; all renders on the Windows worker, 12 heads, fast profile; DRAFT):
  1. Iris saturation (`iris_sat` = HSV saturation of the iris texture, `EYE_SAT_SET`, 36 views vs `normal_eyes2`):
     x1.5 / 2 / 3 eye dE +.41 / +.85 / +1.45 (11 / 11 / 8 of 35 eyes better), eye LPIPS -.0003 / +.0019 / +.0023.
     Worse with every setting: the texture is not what is missing.
  2. Iris colour audit (`eye_audit.py`, 18 measurable eyes): iris texture median saturation .21, UE render .42, Cycles
     render .27, and the Cycles iris hue lands at ~20-30 deg (warm) whatever the texture hue (blue 228 deg -> 10-23 deg).
     The mapping is right (mh_000020 texture hue 37 / sat .98 = UE 38 / .98); the shading loses it. Iris, cornea and
     sclera geometry are fine (sclera never covers the iris). Cause under test: the iris is lit only through the glass
     cornea (Cycles shadow rays stop at glass; the light arrives as noisy caustics that the denoiser removes), leaving
     warm skin bounce. Test `cornea_shadow 0` (cornea + tear film cast no shadows, `EYE_SHADOW_SET`, 36 views): eye dE
     +.63 (10/35 better), eye LPIPS +.0026; with iris gain x1.5 dE +1.80. The iris gets brighter but stays grey, so the
     caustic lighting is not the cause. Next suspect: the iris's own specular (albedo ~.03 linear vs F0 ~.04 reflecting
     the bright studio sky = a grey veil; a real iris sits under the aqueous humour). Test `iris_specular 0`, alone and
     with cornea_shadow 0 (`EYE_IRISSPEC_SET`): iris specular 0 eye LPIPS -.0027, DISTS -.0025 (19-21/35), view LPIPS
     12/12 heads, but eye dE +1.18 (7/35; the iris turns darker, not more coloured); with cornea_shadow 0 dE +.46.
     Structure improves, colour does not: the remaining iris hue loss is not explained by lighting or the iris
     specular (left open; next suspects: the iris texture's very low value (.17-.26) against UE's brighter, saturated
     MH iris, i.e. the chart lookup's V at the dark end, and the cornea's refraction magnifying the grey limbal ring).
  4. Eye placement (`compare_render.write_eye_place_offsets`, Cycles iris centre vs the MetaHuman eye camera target):
     the GNM eye joints sit a median 1.87 mm higher (range -.7..+3.3 mm) and ~.9 mm medial of the MetaHuman eyes;
     image check: pupil 14 px (512 px close-up) higher. Fix test `ew_eye_place` (eyeballs moved by the per-head x/y
     offset, lids unchanged, `EYE_PLACE_SET`, 36 views vs `normal_eyes2`): **eye dE -6.62 (33/35 better), eye LPIPS
     -.0317 (33/35), eye DISTS -.0144 (28/35), view LPIPS -.0036 (31/36 views, 11/12 heads)** -- the largest eye gain so
     far; the iris now sits centred in the lid opening as in UE. Candidate default: place the eyeballs at the MetaHuman
     eye centres (for the pseudo-candidates; the real-subject eye placement comes from the photo fit).
  5. Dual-lobe specular (`dual_lobe`: base roughness .8r at half F0 + coat lobe 1.1r, IOR 1.27, `mat_r125_dual` vs
     `mat_r125_c0`, 72 views): LPIPS +.0004 (22/72, 5/12 heads), DISTS +.0011, dE +.18, eye-region LPIPS/DISTS worse on
     10/12 and 12/12 eye-region views (eye LPIPS +.0074). UE's dual lobes do not carry over; one lobe at x1.25 stays.
  6. mh_000145: the per-head gain of roughness x1.25 follows MetaHuman's sampled skin roughness multiplier
     (params.json skin.roughness .87-1.15; r = -.71 for sw_roughness_scale_1.25, -.72 for mat_r125_c0): heads at
     .87-.92 (mh_000145 .91, mh_000100 .90) are hurt or flat, all heads >= 1.07 gain. Per-identity test `mat_rmh_c0`
     (roughness x 1.25 x MH roughness / 1.03, coat 0, `RMH_SET`, 72 views vs `normal`): LPIPS -.0053 (57/72, 10/12
     heads), DISTS -.0047, dE -.72, eye LPIPS -.0127 vs mat_r125_c0 -.0044 (54/72, 9/12) / -.0039 / -.67; mh_000145
     +.0119 -> +.0050, mh_000358 +.0043 -> +.0028. Per-identity roughness is better than the global scale; on real
     subjects the identity's roughness is unknown (s1/s2 were a tie at x1.25), so it applies to MH-matched rendering.
  7. Ear thin-part transmission (`ear_sss_scale` k: SSS weight 1 and scale x k on the ears mask, `EAR_SSS_SET`,
     ear cameras, 60 views / 46 ear regions vs `normal`): x2 ear LPIPS +.0013, DISTS +.0038, dE +.73 (23/46 better);
     x4 +.0142 / +.0132 / +1.81 (17/46); view LPIPS +.0012 / +.0043 (4/12 heads). More ear transmission makes the ears
     redder than UE's; the current random walk stays.
  8. Windows render_matrix path (`win_matrix.py`: stage on Linux, Blender CPU on Windows, pull into
     renders/<s>/<cand><suffix>); Linux/Windows equivalence (s1 B7 front, 1024 px / 128 spp: Linux RX 9070 XT 28 s vs
     Windows CPU 65 s): LPIPS .00006, mean |diff| .0001, identical mean colour -- Windows renders can replace the Linux
     GPU for s1/s2. s1/s2 B9 (= B7 + the median MH eye offset, down 1.87 mm / ~.8 mm outward; `win_matrix`, 28 views,
     `prior_report.py B7 B9`; LPIPS B7 -> B9): s1 face .424 -> .425, +40/-40 .502/.491 -> .501/.491, ears .553/.537 ->
     .552/.537; s2 face .498 -> .498, +40/-40 .533/.509 -> .532/.510, ears .558/.543 -> .557/.543. A tie at face scale
     (the report has no eye-region row): the offset is specific to GNM-on-MH registration, not a general GNM eye bias.
  3. Sclera (`sclera_sat` / `sclera_gain` on the sclera texture, `EYE_SCLERA_SET`, 36 views vs `normal_eyes2`):
     saturation x.5 / 0 eye dE +.97 / +2.16 (6 / 5 of 35 better); brightness x.85 eye dE -.74 (32/35), eye LPIPS -.0029,
     DISTS -.0015, view LPIPS better on 12/12 heads; x1.15 dE +.69 (3/35). The sclera colour is right, it is too bright.
  Combined confirm (`COMBO_SET`: per-identity roughness + coat 0 + MH eye placement + sclera x.85, eye_map v2;
  `combo_is0` + iris specular 0; 6 confirm cameras + eye close-ups vs `normal`, 96 views each):
  `combo` LPIPS -.0111 (87/96 views, 11/12 heads), DISTS -.0106, dE -.68; eyes LPIPS -.0406, dE -15.1 (35/35);
  ears LPIPS -.0030, dE +.26. `combo_is0` LPIPS -.0113 (88/96), DISTS -.0113, eyes LPIPS -.0449, dE -14.3 (35/35).
  The gains add up (eye placement dominates the eyes, roughness the skin); iris specular 0 trades eye dE for LPIPS.
  Default changes (applied 2026-11-06: `compare_render.DEFAULT_VARIANT = 'combo'`, the `--variant` default; `''` = the
  original render, `normal` stays the experiment baseline): (a) MH pseudo-candidates place the eyeballs on the MH eye
  centres; (b) per-identity roughness x1.25 x MH roughness/1.03 + coat 0 for MH comparisons; (c) sclera x.85.
  Not proposed: iris specular 0 (mixed), the s1/s2 photo defaults (B8/B9 tie).
- Follow-up round (2026-11-06, `goal-vhuman-next.md`; Windows worker Blender 5.2.3 CPU fast profile, baseline = the new
  default `combo` unless stated; `c_<param>_<value>` = combo + one setting; DRAFT, results complete):
  1. Iris hue diagnostics (`IRIS_DIAG_SET`, 12 heads x eye_left/right/front vs `combo`): hiding the cornea and tear film
     from camera rays worsened eye LPIPS by .0281 and eye dE by 3.17 (2/35 and 3/35 better); unshaded iris texture
     emission improved eye LPIPS by .0065 (21/35) but worsened eye dE by 2.59 (5/35). On the 17 eyes valid in all three
     `eye_audit` runs, median absolute iris saturation error vs UE was .265 for combo, .221 with the cornea hidden, and
     .167 with emission (11/17 and 14/17 closer than combo). This is a fixed-ring, automatically detected image-space
     measurement: pupil detection, refraction and very dark/low-saturation irises can move or contaminate the sampled
     ring; hue is unreliable for nearly grey pixels. The two tests implicate the image formation but do not isolate a
     practical correction, and neither is a candidate default because eye dE worsens. A four-head follow-up is testing
     additive iris-coloured fill at strengths .25/.5/1 while retaining the shaded material (`iris_fill4`, four heads,
     36 views): eye LPIPS / DISTS / dE vs `combo` were -.0028 / -.0012 / -.54, -.0064 / -.0022 / -.98, and -.0093 /
     -.0039 / -1.48, respectively; eye dE improved on 11/11 measurable views for each. This is only a small pilot and
     is a nonphysical emission adjustment. For the eight matched close-up eyes, median absolute iris saturation error
     was .394 / .359 / .333 for combo / fill .5 / fill 1; fill was closer on 5/8 eyes at either strength. Hue error
     median was 10 -> 7 degrees on the six eyes where both measurements had saturation >= .1. An old queued UE job
     unexpectedly overlapped this pilot; the pilot's code sync occurred while UE was running. All 36 outputs passed
     asset validation. The .5/1 settings were rerendered for all four heads without a concurrent worker job: all 24 outputs passed
     validation and differ from the pilot by a mean of .000019 RGB8 levels per channel (maximum .000036 over any view).
     Combining the clean four-head repeat with the eight independent heads (35 measurable eye-region views), fill .5
     changed eye LPIPS / DISTS / dE by -.0026 / -.0034 / -.60 (30 / 30 / 27 views and 11 / 9 / 10 heads better);
     fill 1 by -.0054 / -.0037 / -.44 (24 / 23 / 23 views and 10 / 8 / 8 heads better). The aggregation is reproducible
     with `tmp/vhuman-mh/iris_fill_combined.py`.
     On the eight newly rendered heads alone (24 views), .5 changed them by -.0008 / -.0040 / -.42 (19 / 20 / 16
     better), while 1 changed them by -.0036 / -.0037 / +.04 (15 / 14 / 12 better). Fill .5 is more consistent;
     Photo checks (B11/B12 = B7 + fill .5/1, two subjects, final 1024²/128 spp, front/eye close-ups) do not establish a
     broad eye-region improvement: s1 B7 -> B11 -> B12 eye LPIPS .2576 -> .2582 -> .2584 and mean dE 10.00 -> 10.07 ->
     10.18; s2 .2656 -> .2658 -> .2662 and 11.83 -> 11.72 -> 11.84. These parsed eye regions include brows; this report
     uses the ordinary parser mask, while the B10 `prior_report` below removes small components, so their B7 values differ. As a
     supplemental, approximately projected horizontal iris-ring colour check against the front input, CIE dE76 for
     right/left was s1 7.97/5.96 -> 8.78/5.15 -> 10.88/6.65, and s2 4.96/6.82 -> 3.41/.98 -> 4.58/3.58. Fill .5
     improves 3/4 of these small samples, but the photo/render pupil centres are not independently registered and the
     lighting differs; the native fit's pupil radius is a fixed .45x iris estimate, not a measured pupil boundary.
     See `tmp/vhuman-ab-cycles/bench/iris_ring_projection.png` and `tmp/vhuman-ab-cycles/scripts/iris_photo_report.py`.
     Close-up comparison: `tmp/vhuman-ab-cycles/bench/b7_b11_b12_eyes.jpg`. Results are mixed and the adjustment emits
     light, so no photo or general default change is proposed. A non-emissive follow-up tests iris specular 0 combined
     with albedo gain 1.5/2.5 (`iris_physical4`). Within the inner 90% of the iris radius, the 24 comparison heads'
     source textures have a maximum linear channel value of .238, so gain 2.5 exceeds 1 in none of those pixels; this
     is not a bound for other identities or the outer texture. The photo-subject iris textures are brighter: over the
     same inner region, gain 2.5 exceeds 1 in 1.14% / .22% of s1/s2 pixels; gain 1.5 does not exceed 1 for either
     (`tmp/vhuman-mh/iris_texture_range.py`).
     On four heads / 11 eye-region views, LPIPS / DISTS / dE changed -.0165 / -.0092 / -2.04 and -.0171 / -.0089 / -1.96 versus combo;
     both improve LPIPS and dE in 8 and 7 views respectively. The mean is driven by two heads: mh_000038 eye dE worsens
     by 1.06 / .25 and mh_000145 by .22 / -1.02 (1.5 / 2.5). On eight new heads / 24 views, changes were -.0005 /
     -.0018 / +.007 and -.0003 / -.0026 / +.109, with eye dE better on 9/24 views and 3/8 heads for each. Across all
     12 heads / 35 measurable views, changes were -.0055 / -.0041 / -.64 and -.0056 / -.0046 / -.54; eye dE improved on
     16/35 views and 5/12 or 6/12 heads. Per-head mean dE changes ranged from -5.96 to +1.18 at gain 1.5 and -5.28 to
     +2.20 at 2.5. Over all 36 full views, LPIPS / DISTS / dE changed -.0010 / -.0027 / +.001 at 1.5 and -.0013 /
     -.0026 / -.007 at 2.5. The strongest gains are on two highly saturated warm iris textures (mh_000020 and mh_000404), while
     several low-saturation eyes regress; this small sample is not sufficient for a colour-conditioned default. Pilot
     means therefore do not generalize to the new heads' broader eye crops.
     Against the existing specular-0-only render, gain 1.5 / 2.5 changes eye LPIPS by
     -.0036 / -.0042 and dE by -1.46 / -1.38 (10 / 9 of 11 eye-region views improve dE), suggesting the gain adds to
     the specular change. Across the 12 heads, the approximate audit matched 18 close-up eyes: median absolute iris
     saturation error was .270 for combo and .1405 / .1535 for gains 1.5 / 2.5 (12/18 eyes closer for either). On nine
     paired eyes with saturation >= .1, median hue error was 10 -> 3 / 4 degrees (7 / 8 of 9 closer). Restricting to
     the eight new heads, saturation was closer on 6/10 matched eyes; only three pairs met the hue threshold, with
     median hue error 13 -> 30 / 28 degrees (1 / 2 of 3 closer). The pilot's hue gain does not generalize on this
     small, detector-sensitive sample. Photo follow-up (B13/B14 = B7 + specular 0, gain 1.5/2.5; two subjects,
     1024²/128 spp) yielded eye-region LPIPS / mean dE: s1 B7 .2576 / 10.00, B13 .2575 / 10.03, B14 .2582 / 10.27;
     s2 .2656 / 11.83, .2653 / 11.88, .2657 / 12.24. Approximate right/left iris-ring dE76 was s1 7.97/5.96 ->
     4.77/1.95 -> 8.49/4.97 and s2 4.96/6.82 -> 5.64/2.87 -> 12.01/7.76. As above, the rings are not independently
     registered and lighting differs. Gain 1.5 improves 3/4 iris samples but the broad eye metrics are essentially flat
     or worse; gain 2.5 is worse on most samples. Close-ups: `tmp/vhuman-ab-cycles/bench/b7_b13_b14_eyes.jpg`.
     No default change is proposed from these results.
  2. Iris size: the UE base-colour iris radius (`eye_colour_calibration_v1.json`, 240 eyes) is 1.03x the GNM 6 mm limbus
     (p25-p75 .97-1.10). This compares physical sizes, not apparent rendered sizes, which can also differ with optics and
     camera alignment. The existing render-based limbus detector is too noisy (left/right rays disagree 2-3x); a visual
     check of mh_000020 still suggests a larger apparent Cycles iris. As an exploratory check, Hough circles on the 24
     paired close-ups were manually screened (`tmp/vhuman-mh/iris_radius_probe/circles.jpg`; grayscale median filter 5,
     OpenCV Hough gradient dp=1, minDist=30, edge threshold 80, accumulator 25, radii 25-65 px, first centre in
     x=200..310/y=205..295): 10 pairs had plausible
     circles on both images (both eyes of mh_000020/038/100/145, right 404, left 435), with median Cycles/UE radius 1.00
     (range .97-1.06); mh_000020 left was 1.06. The other 14
     were missing or visibly incorrect detections. This small, selected subset does not rule out a gap on other eyes;
     the x.9/.95 scale test remains skipped pending a reliable projected-radius measurement. A follow-up constraining
     Hough circles to within 8 px of the automatic pupil detector (`tmp/vhuman-mh/iris_radius_probe/constrained.jpg`)
     still picked lids, pupils or the wrong centre on multiple eyes. A second attempt with a compact central dark-component
     pupil anchor (`central.jpg`) improved some centres but still picked wrong iris radii; neither replaces screening.
  3. Auto eye offsets: `compare_render.eye_place_offsets(sid)` writes `head/eye_place_offset.json` on first use (CPU
     stage_only base under `tmp/vhuman-mh/eye_place`, ~40 s); recomputing from all 12 staged eye-map-v2 bases reproduced
     the existing offsets exactly (24/24 sides, maximum x/y difference 0). The 12 new heads staged for the wider check
     also reproduce their generated offsets exactly (another 24/24 sides).
  4. Skin re-check on `combo` (specular x.75/1.25, SSS scale x.75, normal strength x1.5; 72 views each, vs `combo`;
     LPIPS / DISTS / dE, views better, heads better): specular x.75 -.0005 / +.0004 / +.67, 29/72, 10/12; x1.25 +.0011 /
     +.0002 / -.39, 30/72, 2/12; SSS scale x.75 +.0008 / +.0011 / -.01, 24/72; normal x1.5 +.0035 / +.0008 / +.16, 21/72
     (ear LPIPS +.0141). No change wins on both LPIPS and dE: `combo` stays (specular trades structure for colour).
  5. Ears (`EAR_ROUGH_SET`, ear_left/right + yaw +-90 + front, 60 views / 46 ear crops, all vs `normal`): `combo` ear dE
     +.077 (23/46 better), ear LPIPS -.0034; ears-mask roughness x1 instead of the per-identity scale: ear dE -.052 (26/46),
     ear LPIPS -.0021; x1.1: ear dE -.059 (26/46), ear LPIPS -.0033, ear DISTS -.0031 (combo -.0027); view LPIPS -.0056
     for all three. Paired directly against `combo`, x1.1 changes ear LPIPS / DISTS / dE by +.00013 / -.00041 / -.136
     (23 / 28 / 33 of 46 crops and 6 / 7 / 10 of 12 heads better). It is a small colour win with essentially tied
     LPIPS (the per-identity scale reaches x1.39 on some heads; MH ear roughness ~.70).
  6. Landmark eye placement (`eye_offset_fit.py`, 499 MH heads / 998 eyes, CPU): offset = MH corneal-cap centre - GNM iris
     centre, median +1.20 mm lateral / -1.53 mm vertical; ridge on the 12 iBUG lid landmarks relative to the GNM iris
     (x mirrored for the right eye), 5-fold CV by head: RMSE .52 mm (x .37 / y .36) vs 1.18 mm (x .56 / y 1.04) for the
     median offset. On the 12 comparison heads (Cycles-measured offsets, per-side bias -.41/+.36 mm x corrected) RMSE
     .39 / .52 mm. At the selected ridge penalty, held-out per-eye radial error median / p90 / p95 / maximum is
     .415 / .794 / .899 / 1.53 mm (median-offset baseline .983 / 1.78 / 1.985 / 3.405 mm); median errors are .410 mm
     left and .421 mm right. CV uses MetaHuman-derived fitted GNM heads, so it does not establish generalization to
     photo-fitted heads. s1/s2 B10 = B7 + predicted offset (s1 L +.98/-1.24, R -.58/-1.34 mm; s2 L +1.05/-1.26, R -.70/-1.35
     mm; B9 used -1.87/-1.91): all 28 final-quality Windows views completed. B7 -> B10 front eye-region LPIPS / DISTS /
     mean dE: s1 .2526 / .2762 / 9.73 -> .2554 / .2771 / 10.10; s2 .2656 / .2486 / 11.83 -> .2651 / .2499 / 12.45.
     Eye-mask IoU fell .854 -> .841 and .858 -> .844 respectively. The reference is each subject's front input image,
     and the parsed eye region includes brows; the close-up renders do not have independently registered references.
     A paired close-up sheet is at `tmp/vhuman-ab-cycles/bench/b7_b10_eye_closeups.png`. This is no consistent
     improvement on these two subjects, so do not propose a photo default change.
  7. Wider confirm: UE VHStudio references exist for 12 heads only and `/mnt/nvme01` had 5.7 GB free, so the confirm uses
     `render_ids[::5]` = 24 heads (12 new UE VHStudio refs on the Windows worker, UE 5.8.4 DX12; light batch extended,
     the 12 existing entries byte-identical). The 12 new sets completed and their 120 EXRs match the recorded SHA-256
     values; all 36 eye/front EXRs decode at 1024² with finite, nonblack pixels, and the 12 left-eye previews show the
     expected heads (`tmp/vhuman-mh/iris_radius_probe/new_reference_eyes.jpg`). The UE log reached shutdown before
     its exit error. The batch overlapped the iris pilot, whose code sync
     replaced the worker server tree while UE was running. The old queue was stopped before it could sync again.
     Eye-only check (24 heads x front/eye_left/eye_right, `eye_wide24`, vs `normal`): `combo` view LPIPS / DISTS / dE
     -.0147 / -.0158 / -.58 (69 / 68 / 64 of 72 views better; view LPIPS 22/24 heads). Eye LPIPS / DISTS / dE
     -.0328 / -.0278 / -13.64 (64 / 68 / 71 of 71 eye-region views better). On the 12 new heads alone, eye LPIPS /
     DISTS / dE were -.0252 / -.0225 / -12.22 (32 / 34 / 36 of 36 views better). The automatic colour audit retained
     41/48 eye close-ups and reports median texture / UE / Cycles saturation .286 / .427 / .260; its centre-box gate
     does not guarantee a correctly located pupil, as the visual diagnostics above show. The original eight-camera wide
     confirmation completed and validated its remaining 120 views. Full eight-camera result (192 views): `combo` vs
     `normal` LPIPS / DISTS / dE -.0105 / -.0096 / -.63 (171 / 165 / 176 views better; LPIPS 22/24 heads), close to the
     earlier 12-head LPIPS -.0111. On the 12 new heads alone (96 views), LPIPS / DISTS / dE changed -.0099 / -.0086 /
     -.58 (84 / 85 / 88 views better). The two head-level LPIPS regressions were small: mh_000145 +.00028 and mh_000425
     +.00037. Every camera's mean view LPIPS and dE improved. Eye-region results are those above; on 44 measurable ear crops, ear LPIPS / DISTS /
     dE changed -.0042 / -.0027 / +.42 (22 / 27 / 7 better), confirming a residual ear colour trade-off.
  - Default proposals, pending user review: ears-mask roughness x1.1 is a small colour improvement over combo with
    essentially tied ear LPIPS on the original 12 heads; it has not been confirmed on the wider 24-head set and no
    change has been applied. Do not propose the landmark photo offset or iris emission
    based on the mixed photo results. Do not propose the non-emissive iris alternative as a general default: its
    held-out eye-region and photo results are mixed despite iris-colour gains in some samples.


### Isolated eyeball shading experiments

`ue_scripts/eye_isolate.py` captures only the MetaHuman eye material sections with the existing VHStudio lighting and
eye cameras zoomed 2.5x, including a separate base-colour pass. Results are written under
`eye_isolate_eyeball/<identity>` on the Windows worker; re-running an identity overwrites its earlier captures. Set
`include_eye_occlusion` in the batch only to inspect the separate translucent eyelid-occlusion shell. The first
`eye_isolate_ue02/03` references included this shell and are not clean eyeball-only references.

The Windows Cycles comparison runner accepts `iso_base`, `iso_vessel1`, `iso_vessel2`, `iso_sss03`, `iso_sss06`,
`iso_vessel1_sss03`, `iso_ue_n05`, `iso_ue_n1`, `iso_ue_n1_gain15`, `iso_unified`, `iso_unified_sss03`,
`iso_unified_rect`, `iso_unified_rect_sss03`, `iso_tissue*`, `iso_iris3*` and `iso_sclera_*`. These hide the face and other accessories, use the same 2.5x camera zoom, and independently test
procedural sclera vessel strength and random-walk sclera subsurface weight. A separate unrotated vessel mask keeps
the vessel distribution laterally anchored even when the authored sclera atlas is rotated. Both controls are opt-in;
existing renders retain their authored material. These isolated comparisons are visual diagnostics: eye geometry,
exposure and lighting are not fully matched, so pixel differences are not a measure of shader quality alone.

The Unreal source `Engine/Shaders/Private/ShadingModels.ush` EyeBxDF uses a separate iris normal and a light-dependent
caustic factor, followed by subsurface-profile scattering. The measured preset has iris normal strength 1, cornea
roughness .075 and IOR 1.41; `r.Substrate=0` in the captured material dump. The translucent eyeshell is instead the
separate unlit eyelid-occlusion material. The `iso_ue_n*` variants connect the generated iris normal, with the same
pupil warp and rotation as its albedo. The `iso_unified*` variants test a single-surface principled approximation,
not a literal implementation of UE's eye BxDF. The `_rect` variants use square area lights of the same area as the
normal disk lamps, because the reference rig uses UE RectLights.

A first fast 512 px / 32 sample sweep used both eyes of `mh_000020` and `mh_000038`. The procedural overlay makes
vessels visible on the former, but adds too much redness on the latter, which already has vessels in its baked sclera.
The tested .3/.6 subsurface weights produce only subtle visual differences. UE retains a softer limbus, more detailed
iris and different reflections; the reference preview and Cycles preview also use different display pipelines. This
does not establish a UE-quality match or justify a shared default. Side-by-side sheets are under
`tmp/vhuman-mh/eye_isolate/`, and individual Cycles results are under `compare_v1/<identity>/cycles_iso_*__fast/`.

The cleaned reference exposed a missing peripheral tissue gradient in the generated sclera. A seeded, low-frequency
procedural factor adds a vascularity-dependent red tint outside the limbus; `iso_tissue_base` and `iso_tissue1` test it
on the base-color and lit eye respectively. `eye_tissue_audit.py` compares median linear RGB in two outer sclera annuli
normalized by the fitted eyeball silhouette; it is not a registered image metric and geometry, coverage and lighting
still differ. On the six left eyes used to develop the factor it improved 3/6, mean .0754 -> .0553. Four additional left
eyes improved 2/4, .0531 -> .0316; ten right eyes improved 5/10, .0665 -> .0464. The earlier fixed-screen-annulus
figures (6/6, 4/4, 8/10) were sensitive to differences in eye size and should not be used. Reports are in
`tmp/vhuman-mh/eye_isolate/`. The mixed result does not justify a universal material default.
The MetaHuman eye source allows vascularity intensity up to 2 and limbus softness up to .15; diagnostic mapping
version 3 retains those ranges and samples iris colours only in the horizontal sectors to reduce eyelid contamination.
Version 4 also anchors the iris measurement radius to camera geometry, rather than assuming the largest color gradient
is the iris edge. The established version 2 mapping and default render selection remain unchanged. The bounded iris
calibration matches the baked texture's inner median to the UE-derived measurement; it is a per-character reference fit,
not an independent prediction. On a separately captured zoomed view, the silhouette-normalized approximate inner-iris
linear-RGB error fell from .0167 to .0074 on 10/10 eyes with version 3; version 4 changed it to .0080 (3/10 improved).
The earlier fixed-screen-radius estimates (.0153 -> .0055 -> .0033) are affected by framing differences and should not
be used to choose a general mapping. `eye_iris_audit.py` documents that metric's remaining registration limitations.
The separate square-light
orientation experiment aligns the emitter's vertical axis with vhuman +Y / UE +Z; the source baseline uses Blender's
default +Z up. A stronger procedural sclera normal makes the wet reflection irregular, but needs further validation.

The measured v4 sclera already incorporates some of UE's transmission tint; applying the full authored tint again can
darken the limbus twice. On six eyes, compensating half of that tint improved 4/6 in two silhouette-normalized
near-limbus annuli, mean linear RGB error .0378 -> .0302; full compensation was less consistent (.0355). A separate
bounded fit of tissue strength to each left-eye UE reference, evaluated with actual right-eye base-color renders,
improved 8/10, mean .0670 -> .0324 (one tie, one slight regression). This uses per-character UE measurements and is not
a reference-free or independent-character result. `eye_tissue_audit.py --fit-out` writes the diagnostic strength table.

`eye_zoom_fit.py` optionally computes per-eye diagnostic focal lengths from UE and Cycles silhouette radii. The
`iso_iris4_framed*` variants use these per-reference fits and fitted tissue strengths from `tissue_strengths.json`;
they are reference-driven comparison aids, not a portrait camera or material default.

The initial UV-space sclera microtexture improved a contrast statistic but produced obvious radial streaks and is
rejected. A seeded spherical-field version at strength .05 improved the difference-of-Gaussians sclera contrast on
4/4 eyes (.00550 -> .00288); the stronger .1 version looked over-patterned. This narrow statistic does not measure
whether the texture looks anatomical. The `iso_iris4_split*` shader approximates UE's separate iris-diffuse and
surface-specular normals using Cycles nodes, without reproducing UE's per-light caustic. In four highlight-excluded
inner-iris samples, split normals reduced the error on three eyes; the palest eye changed .138 -> .088 in linear RGB.
The experimental independent-eye material uses each eye's own UE input and fitted iris measurements and has been
smoke-tested on one asymmetric character; it is not yet a general-quality conclusion.

An outer-iris gain applied after mixing sclera (`iso_iris4_radial*`) improved 4/6 outer-annulus samples against the
same transmission-compensated baseline (.0176 -> .0090), but introduced a bright limbal halo on a light iris outside
that subset. It is rejected. The ten-eye comparison against an uncompensated baseline improved 7/10 but worsened mean
error .0167 -> .0224; it also confounds the effect of transmission compensation. `iso_iris4_layered*`
instead applies bounded iris gains before the iris/sclera mix; it is a separate experiment to avoid scaling sclera.
In six controlled eyes, this layered variant improved outer-annulus error on 4/6 (.0176 -> .0081) and removed the
visible halo on the inspected light iris. A combined reference-fitted comparison rendered both eyes of six characters
and reloaded each scene to validate its separate-eye textures and iris-normal attributes. Comparison sheets are under
`tmp/vhuman-mh/eye_isolate/combined_left.jpg` and `combined_right.jpg`.

The `iso_iris4_curvy*` experiments use a more curved procedural vascular network. The four-character renders show
additional branching, but the vessels remain too thin and radial compared with UE's irregular, mottled sclera; this
is not yet a visual match. `iso_iris4_contrast*` additionally tests a bounded per-eye angular smoothing estimated from
the UE base-color reference. It is reference-driven and must be evaluated on separate renders before drawing a
quality conclusion. Four left-eye base-color/lit comparisons completed; on the inspected blue/gold eye the adaptive
version restores some radial detail but still misses the reference's distinct blue and gold structure. See
`tmp/vhuman-mh/eye_isolate/contrast_100.jpg`.

UE's material dump identifies a dedicated sclera base-color texture and distinct irritation controls. The experimental
`iso_iris4_irregular*` overlay uses less inward-biased, multiscale procedural branches. Against `iso_iris4_layeredbase`,
four isolated eyes improved the medium-frequency contrast error .00551 -> .00408, but the inspected lit image still
shows predominantly radial vessels from the existing baked sclera. This contrast statistic does not assess anatomical
shape. `iso_iris4_clean_irregular*` separately removes those baked vessel layers before adding the new overlay.
The clean-base experiment revealed an overly sparse network, with contrast error .00551 -> .00691 (1/4 improved)
against the original layered base; the inspected high-vascularity eye visibly loses too much structure. It is not a
replacement for the baked texture. See `tmp/vhuman-mh/eye_isolate/clean_irregular_038.jpg`.
The `iso_iris4_dense_irregular*` experiment triples the number of branches, adds a soft peripheral field and increases
overlay strength. Against the sparse clean-base version, medium-frequency contrast error improved on 4/4 eyes
(.00691 -> .00260). The result remains too heavily veined on the zero-vascularity eye, although it looks closer on
the highly vascular eye; see `tmp/vhuman-mh/eye_isolate/dense_020.jpg` and `dense_038.jpg`. The separate
`iso_iris4_vascular*` experiment scales this dense overlay by the MetaHuman left-eye vascularity input; its linear
mapping is a provisional diagnostic, not a validated general mapping.
It reduces excessive veins on the zero-vascularity eye, but also suppresses the shared tissue field, leaving that eye
too smooth; the four-eye contrast error rose .00260 -> .00634 versus fixed dense strength. Separate vascular and
nonvascular tissue layers are needed. See `tmp/vhuman-mh/eye_isolate/vascular_020.jpg`.
Adding the independent .05 spherical field (`iso_iris4_vascular_surface*`) improved that contrast error on 4/4
(.00634 -> .00376), but the low-vascularity eye shows an overly regular cloudy pattern. The separate
`iso_iris4_vascular_fine*` test uses a finer spherical field and removes the built-in mottling from the vessel mask;
neither version is a validated default. See `tmp/vhuman-mh/eye_isolate/vascular_surface_020.jpg`.
The finer field and vessel-only mask improved the same statistic 4/4 (.00376 -> .00351) and reduced the broad ripples,
but the low-vascularity eye remains too uniformly patterned and has too few visible veins. This comparison changes both
the field frequency and the vessel mask, so it does not isolate either effect. See
`tmp/vhuman-mh/eye_isolate/vascular_fine_020.jpg`.
`iso_iris4_vascular_sparse*` instead maps the left-eye vascularity input to the number of generated vessels while
keeping individual lines visible. Compared with the faded dense network, medium-frequency contrast improved on 2/4
eyes, mean error .00351 -> .00289. The quiet eye now has visible isolated veins, although some are too dark and
straight; the highly vascular eye retains dense branching. See `vascular_sparse_020.jpg` and
`vascular_sparse_038.jpg` in the same directory. This is a small reference-driven diagnostic, not a validated mapping.
The combined `iso_iris4_sparse_split*` renders apply the earlier separate iris/surface normal approximation to this
newer sclera. Four left eyes were rendered with and without stronger shell normal and sclera roughness. On the two
inspected eyes the stronger normal breaks up the large wet highlight more like UE, but also distorts a secondary
reflection and adds streaks; the split-normal effect is subtle. UE's per-light caustic is still absent. Compare
`sparse_split_020.jpg` and `sparse_split_038.jpg` in the same directory.
The `iso_iris4_sparse_bump*` variants replace the amplified vessel-derived normal with bump from the independent fine
spherical tissue field. At .0001 m the secondary reflection is smoother but the main highlight stays too regular; at
.0004 m it breaks up but produces an obvious pebbled surface. Neither is accepted as a UE match. See
`sparse_bump_020.jpg` and `sparse_bump_038.jpg` in the same directory.
`iso_iris4_sparse_patch*` modulates the fine bump with a broader spherical field. In the two inspected eyes the
stronger localized version breaks up the main reflection while retaining the secondary reflection more coherently
than uniform .0004 m bump, but some pebbled patches remain. It is a visual diagnostic, not a validated UE normal
texture. See `sparse_patch_020.jpg` and `sparse_patch_038.jpg` in the same directory.
The reference-driven angular-contrast fit was also combined with the clean sclera and localized wet highlight as
`iso_iris4_sparse_adapt*`. On the inspected blue/gold eye it restores some radial detail but does not recover the
reference's distinct colors. Four base-color inner-iris median RGB errors changed .0045 -> .0046 (2/4 improved); that
metric does not measure detail and the same character's UE data are used to fit the texture. See
`sparse_adapt_100.jpg` in the same directory.
The opt-in `iso_iris4_sparse_bands*` experiment adds three reference-derived radial RGB targets, with bounded gains
applied to the iris before sclera compositing. In four eyes the inner-annulus median error changed .0046 -> .0025
(2/4 improved) and outer error .0077 -> .0053 (3/4 improved). These are the same characters used for the fit, and
the statistics do not assess fiber shape. The blue/gold eye still lacks the reference's distinct fibers; the inspected
light eye shows no new bright scleral halo, but still differs substantially. See `sparse_bands_100.jpg` and
`sparse_bands_300.jpg` in the same directory.
The procedural radial mask favors the secondary chart color near the pupil, whereas the current MetaHuman mapping
places the measured inner color in the primary slot. `iso_iris4_sparse_swap*` tests swapping the slots before bounded
band calibration. On four eyes, inner error changed .0025 -> .0026 (2/4 improved) and outer .0053 -> .0052 (1/4
improved); the blue/gold fibers remain missing in the inspected image. This does not justify changing the default.
See `sparse_swap_100.jpg` in the same directory.
`iso_iris4_sparse_fibers*` colors the generated fiber detail using the difference between measured inner and outer
chromaticity, without copying reference pixels. It adds visible warm fibers to the blue/gold eye, but introduces an
artificial blue ring on the light eye. Four-eye inner median RGB error improved on 3/4 (.0025 -> .0020); that metric
does not capture the artifact. It is reference-driven and not a generally acceptable mapping. See
`sparse_fibers_100.jpg` and `sparse_fibers_300.jpg` in the same directory.
`iso_iris4_sparse_fibers_fit*` estimates median angular chroma variation after subtracting narrow radial-band medians,
then maps it through a provisional bounded strength curve. On the four development eyes it retains full fiber strength
for 100 and suppresses it for 300, removing the visible blue ring; inner median RGB error changes .0025 -> .0024
(2/4 improved). The curve was chosen from these same examples, and each eye uses its UE reference; neither the curve nor
the generality is validated on held-out identities. See `fibers_fit_100.jpg` and `fibers_fit_300.jpg`.
In four additional identities (145, 178, 194, 256), the measured signal falls below the provisional threshold and
selects zero added fiber color. These eyes can check for unwanted changes but cannot validate the active portion of
the curve; identity 358 has a nonzero signal and is a more informative follow-up.
The 358 comparison selected strength .123 and looked nearly unchanged; its inner median RGB error rounded to .0025
for both variants (the fitted value was slightly worse). Both show an iris ring missing from UE. One nonzero held-out
eye is insufficient to validate the curve; see `fibers_fit_holdout_358.jpg`.
For 358, the bright ring is already present in the base-color pass and aligns with the mapped procedural limbal ring.
`iso_iris4_ring_soft*` broadens its transition and `iso_iris4_ring_none*` neutralizes the limbal tint, each recalibrating
the same iris bands. On that eye, a local base-color luminance bump at normalized sphere radius .27 relative to .23
and .31 is .00318 in UE, .02005 in the baseline, .01491 with softening and .00964 with neutralization. The neutral
variant visibly reduces the ring; the inspected blue/gold eye changes less. This is a local diagnostic on three eyes,
not grounds for a default change. See `ring_358.jpg` and `ring_100.jpg`.
Six additional eyes were rendered and compared with their existing baselines using `eye_ring_audit.py`, which removes
a broad trend from silhouette-normalized horizontal base-color profiles. Across nine eyes, neutralization improved
the diagnostic on 6/9, mean error .02614 -> .01998, and worsened it on three. The result is reference-driven and the
metric is sensitive to registration and iris size; it is not a perceptual score. See `ring_audit_expanded.json`,
`ring_194.jpg` (improved diagnostic) and `ring_178.jpg` (worsened diagnostic) in the same directory.
`eye_ring_select.py` compares each unzoomed UE base-color radial profile with the two staged iris textures and writes
an opt-in `iso_iris4_ring_select*` choice table. On these nine eyes it agrees with the separate isolated-eye radial
diagnostic on 8/9; selecting from the already-rendered candidates changes mean radial error .02614 -> .01925.
Actual base-color and lit selector renders for 178 (baseline) and 194 (neutral) were pixel-identical to the chosen
variants. This is not independent-character validation: each texture and choice use that eye's UE reference, and the
unzoomed and isolated views are related. It does not establish full visual quality or justify changing the default.
The opt-in `iso_iris4_bilateral*` path now builds the clean iris and vascular textures independently for each side,
and selects the right limbal-ring treatment from the right UE reference instead of reusing the left selection. Its
right split-normal mask uses right-eye cornea parameters; Blender reload validation checks this mask and the packed
materials. Two characters were rendered in both eyes (base color and lit), and all eight views passed. Their left and
right cornea sizes happened to be equal, so this does not test unequal-size inputs. The visual comparison to the
earlier `side_layered` variant also changes framing and material details; it cannot isolate the bilateral change.
The sclera remains too pebbled. See `bilateral_100_left.jpg` and `bilateral_100_right.jpg` in the same directory.
The right sclera roughness transition now shifts with the right iris size as well as the split-normal mask. The
synthetic `iso_iris4_bilateral_asymprobe` increases only the right texture-space cornea size from .20485 to .22485;
its Windows Blender render and scene-reload checks passed for packed materials, the iris mask and the roughness
transition. This is a controlled implementation check, not an Unreal visual comparison of an unequal-size character.
`iso_iris4_bilateral_smooth` and `iso_iris4_bilateral_bump15` hold the bilateral textures and ring choices fixed while
removing the localized bump or reducing its distance from .0004 to .00015 m. In two inspected eyes the gentler surface
reduces pebbled sclera but makes the main wet reflection smoother than UE; at this resolution the smooth and gentle
versions are similar. Eight views across two characters and both sides completed with scene validation. See
`bilateral_surface_100_right.jpg` and `bilateral_surface_020_left.jpg` in the same directory.
`iso_iris4_bilateral_framed` applies the earlier per-eye silhouette-fitted zoom to the gentler bilateral material.
Across both eyes of two characters, mean normalized-frame radius error changed .02759 -> .00115 (4/4 improved).
The zoom is fitted from related UE/Cycles references, so this is a diagnostic registration aid rather than a
reference-free camera result. It does not address the remaining iris, sclera and reflection differences. See
`bilateral_framed_020_left.jpg`, `bilateral_framed_100_right.jpg`, and `bilateral_framing_audit.json`.
Both eyes of 020 were also rendered with the full 1024 px / 128-sample profile and passed the packed-material and
split-normal reload checks. The eye textures remain 512 px. Shared-display comparison sheets are
`bilateral_full_020_left.jpg` and `bilateral_full_020_right.jpg`; UE/Cycles lighting and scene-linear units are not
fully calibrated. The full renders still show sclera color, vessel shape, iris detail and highlight mismatches.
`iso_iris4_bilateral_framed_tissue*` applies the earlier left-eye reference-fitted tissue strength to the current
bilateral material. For 020, outer base-color linear RGB error changed .0775 -> .0306 on the fitted left eye and
.0763 -> .0297 on the right eye; the latter is a cross-eye check within the same character. Both lit eyes were also
rendered at 1024 px / 128 samples and passed scene reload validation. The sclera is less pink but still too uniform,
and vessels and reflections remain mismatched. This is reference-driven and not yet a general result. See
`bilateral_tissue_full_020_left.jpg`, `bilateral_tissue_full_020_right.jpg`, and the corresponding fast base-color audits.
The `iso_iris4_bilateral_organic` diagnostic uses shorter, more curved and faster-tapering vessel walks, with the
current bilateral tissue and camera settings. Four fast views across two characters passed Blender material and
split-normal reload checks. The inspected results have fewer long straight vessels but are too sparse in places and
still contain some abrupt dark branches. See `bilateral_organic_020_left.jpg` and `bilateral_organic_100_right.jpg`.
Only the four EXRs were retrieved to `tmp/vhuman-mh/organic_selective`, not complete render packages, because the
workspace ran out of space while creating the transfer archive.
`iso_iris4_bilateral_organic_dense` increases the number of short branches while lowering their opacity. The focused
mask test shows more coverage and a lower peak on the default seed, but the inspected rendered eyes remain too faint
and sparse compared with UE. Four views passed the material and split-normal reload checks. Only EXRs were retrieved;
see `organic_dense_020_left.jpg` and `organic_dense_100_right.jpg` in the same directory.
`iso_iris4_bilateral_organic_gain15` and `..._gain2` use the identical denser mask at 1.5x and 2x bounded shader
strength. The 1.5x version is more visible without the darkest 2x lines in the inspected eyes, but some loops are
conspicuous and vessel structure still differs from UE. Eight views passed material and split-normal reload checks;
only EXRs were retrieved. See `organic_gain_020_left.jpg` and `organic_gain_100_right.jpg`.
`iso_iris4_bilateral_organic_guided` adds a weak attraction to each branch's initial heading. A fixed-seed path test
reduced the fraction of long branches with end-to-end/path-length ratio below .3, but the rendered network still has
loop-like intersections and some darker, straighter lines. Four views passed scene reload checks; this is not yet a
reliable visual improvement. See `organic_guided_020_left.jpg` and `organic_guided_100_right.jpg`.
`iso_iris4_bilateral_organic_nocross` trims later centerline paths when they meet earlier paths, allowing a short
parent–child join. A synthetic crossing test passes, and four rendered views passed scene reload checks. The inspected
quiet eye still has a conspicuous loop, while the other eye loses some coverage; the check does not detect a path
crossing itself or every near-crossing. This is not yet a visual improvement. Only comparison EXRs were retrieved;
see `organic_nocross_020_left.jpg` and `organic_nocross_100_right.jpg`.
`iso_iris4_bilateral_organic_selfsafe` instead trims crossings within each individual path, while allowing separate
paths to overlap. A synthetic self-crossing test and four rendered scene reload checks passed. The prominent loop in
020 remains in the procedural mask; the clean base texture does not contain it. The new check changes only a small
fraction of mask pixels, and this experiment is not yet a visual improvement. Only comparison EXRs were retrieved;
see `organic_selfsafe_020_left.jpg` and `organic_selfsafe_100_right.jpg`.
Combining inter-path and self-crossing checks is not an obvious fix: at a mask threshold of .02, coverage fell from
.1845 to .0971 for the 020 left eye and from .3418 to .1501 for the 100 left eye. Adding the self check did not
change either rounded result beyond the inter-path check. These are texture-space diagnostics, not perceptual scores.
UE's `DeferredShadingCommon.ush` documents that the eye base color is applied after subsurface scattering, while the
optional Cycles random-walk material scatters with its colored input. The guided-vessel comparison above does not
enable Cycles subsurface scattering, so this ordering difference does not describe that particular render. An SSS
comparison is needed before judging a lighting/recombination approach.
`eye_sss_recombine.recombine` provides the corresponding linear RGB operation from UE's
`PostProcessSubsurface.usf`: interpolate sharp and scattered lighting using the faded profile tint, multiply by
stored base color, and add the separate specular component. It is a tested arithmetic reference only; it does not
generate the filtered lighting or render passes and is not connected to the Cycles renderer. The two pending
`iso_iris4_bilateral_organic_guided_sss03` and `...sss06` variants instead test Cycles random-walk scattering.
For the guided fast captures, mean scene-linear luminance within 90% of each fitted eyeball radius is UE/Cycles
.512/.491 and .487/.478 for 020 left/right, and .589/.615 and .575/.599 for 100 left/right. At luminance above
2.828, the corresponding pixel fractions are .73%/.81%, .21%/.30%, .77%/.81%, and .32%/.33%. The captures are
not registered or radiometrically calibrated between engines, and the region includes diffuse and specular light;
these values cannot isolate either component or establish perceptual similarity. They do show that a uniform exposure
change alone would move the two identities in opposite directions relative to their references.
The opt-in `iso_iris4_bilateral_caustic*` variants approximate UE's EyeBxDF caustic for the fixed studio key using a
per-point light direction and an approximate concave normal. The node scales iris albedo, so it also changes indirect
lighting and does not model per-light shadows or area-light integration; non-studio lighting is rejected. Across two
characters and both eyes, full strength worsened a lit inner-iris median RGB error on 3/4 views, while the visible
effect was subtle. The UE and Cycles lighting is not fully radiometrically calibrated, and this statistic is not a
perceptual score. The experiment is not accepted as a match. See `bilateral_caustic_100_right.jpg`,
`bilateral_caustic_020_left.jpg`, and `bilateral_caustic_inner.json` in the same directory.

For non-MetaHuman reconstructions, `offline_render.render(..., material_overrides={...})` and the CLI's
`--material-overrides` JSON option expose the uncalibrated eye controls, including `eye_only`, `eye_unified`,
`sclera_vessel_all_strength`, `sclera_tissue_strength`, `sclera_surface_microtexture` (.05 or .1),
`unified_sclera_roughness`, `unified_shell_normal_strength`, `sclera_sss_weight` and `eye_ue_split`. The offline
preparation path creates the optional surface texture when requested. The `iris_calibrate`, `eye_per_side`, framing
and fitted tissue variants rely on UE references and are currently comparison-runner features, not portrait defaults.

### Fresh physically layered eye (`server/vhuman/eye2`, `iso_layered2*`, 2026-11-06)

A new eye path replaces the stacked `iso_iris4_*` texture fixes for eye-only comparisons. It is opt-in
(`material_overrides.eye_model_layered2`); existing renders are unchanged. `eye2.bake` writes per-eye meshes
(`eye2_<side>.npz`, eye-local frame) and polar/spherical textures. `cycles_scene.eye2_layered` places them on the
template eyes: it sphere-fits the old sclera, takes the axis through its corneal apex and hides the old eye meshes.

- Geometry. One smooth outer shell (smooth-max of the sclera and cornea spheres, no corneoscleral crease) whose
  material blends from clear cornea to sclera across a ~0.9 mm soft limbus via a per-vertex alpha. A closed aqueous
  chamber sits behind it, with a displaced iris annulus inside and a dark pupil cup. Proportions use adult ratios to
  the fitted eyeball radius (limbus 5.85/12, cornea 7.8/12, chamber depth 3/12). vhuman's `optics` block pairs a
  14.6 mm eyeball with a 6 mm limbus, which made the iris visibly too small against UE. The shell's back faces over
  the sclera are a dark uvea, which removes a pale limbal halo.
- Sclera texture. Base colour lerps from a near-limbus to a peripheral colour, with cloudy pale patches and a narrow
  limbal band. Vessels are a grown forest: deep trunks from the equator branch toward the limbus with Murray-law
  widths, and short tortuous conjunctival capillaries fill the periphery. Tips stop before touching any other vessel,
  so the drawn paths never cross (`textures.crossings` = 0 on all 8 baked eyes). The old visible loop came from
  crossings; it cannot occur here. Surface relief drives a wet clear coat (bump 80 µm at 12 mm scale).
- Iris texture. Polar stroma: radial fibres, crypts, a wandering collarette, broken furrows, a pupillary ruff and a
  limbal ring. Each zone's median is calibrated to the measured inner/outer colour; mesh displacement plus a bump map
  supply the relief.
- Lighting. Shadow rays pass the clear cornea (Light Path "is shadow" → transparent), as in UE's EyeBxDF, where the
  iris is lit without corneal attenuation. Camera and indirect rays refract. `eye2_cornea_shadow=1` restores physical
  shadowing; with MNEE caustics (`.cycles.is_caustics_*`, verified on Blender 5.2) the iris is darker, not brighter.
  UE's caustic gain has no physical counterpart here.
- Identity inputs (labelled). Colours are medians of each eye's UE base-colour capture
  (`eye_<side>_basecolor.exr`): iris inner/outer, sclera near/far. These are material inputs, not lit pixels; when no
  capture exists the staged angular texture is used. Pupil fraction comes from the staged vhuman texture; vascularity
  from MH params. Nothing is fitted to the lit UE renders. The framing (per-eye zoom) and offsets are the guided
  baseline's reference-fitted ones.
- SSS. Cycles random-walk sclera SSS (radius 1/0.45/0.3 mm) reddens the whole sclera, because UE multiplies base
  colour after scattering. The default is therefore weight 0; `iso_layered2_sss1`, `_sss05` and `_sssr025` keep the
  scattering tests.

Validation (fast 512 px / 32 spp, worker Blender 5.2.3 CPU, every view passed the scene-reload check). Dev
identities are 020/100; holdouts are 038/300 (bases from `eye_clean_irregular01`, no right-eye-specific inputs).
The metric is `tmp/vhuman-mh/eye_layered2/metric.py`: silhouette-normalised horizontal-sector median linear RGB,
highlights excluded, mean abs error versus the lit UE capture. It is unregistered, uncalibrated across engines and
not perceptual. Mean over 8 views (guided baseline: 4 dev views only):

| variant | iris | sclera mid (r .6–.72) | sclera edge (r .85–.95) |
|---|---|---|---|
| guided baseline (dev only) | .0094 | .0508 | .0833 |
| layered2, staged-texture colours (dev10) | .0804 | .0781 | .1492 |
| layered2, UE base-colour colours, SSS 1 (dev11) | .0378 | .1110 | .0413 |
| layered2, SSS .5 | .0376 | .0934 | .0420 |
| **layered2, SSS 0 (default)** | .0375 | .0749 | .0436 |

The UE base-colour source mainly fixed the holdouts (038 iris error .092 → .039, 300 .19 → .08). Sheets:
`tmp/vhuman-mh/eye_layered2/final/eye_layered2_dev12_<id>_<side>.jpg` (UE | guided | layered2 variants).

Visually, layered2 matches UE better than the guided baseline in:
- iris size;
- the soft limbus;
- broken wet highlights;
- red peripheral tissue;
- the branching vessel network with no loops.

It remains worse or different in:
- The mid-sclera is pinker and darker than UE's bright white band, and is still the largest band error on dev.
- Holdout irises are darker than UE (no caustic gain). 300's UE iris is pale with a yellow collarette, which the
  zone model does not reproduce.
- UE vessels in high-vascularity eyes (038, 300) are denser and more saturated than ours.
- Sparkles from the bump near the limbus are stronger than in UE.

Not yet done:
- full-quality renders;
- more identities;
- a portrait-camera (non-isolated) check.
`tmp/vhuman-mh/eye_layered2/round.sh` and `stage_from.py` derive jobs from existing staged bases (the local disk lacks
the 4 GiB staging headroom).

#### Iris/sclera boundary round (2026-11-07, full 1024 px / 128 spp)

The user asked for a natural limbus and allowed non-physical, UE-like blending where it helps. Changes, in order:
- **Veil limbus (default).** The outer shell stays clear over the iris centre, then turns into opaque sclera over a
  soft ramp centred on the limbus. The width follows MH `limbus_softness` (1.4 mm + 12 × softness at 12 mm scale,
  capped at 2.8 mm). The painted iris→sclera colour blend (`iso_layered2_paint`) left a brown/dark halo because MH
  iris rims are dark; it is kept only as an option.
- **No dark limbal rings.** The iris texture's limbal darkening is nearly removed and the sclera's limbal shadow band
  is gone. A non-physical neutral white band (~4 mm) replaces it, because UE renders the near-limbus sclera grey-white
  even on red eyes (038, 300). Without it the veil read as a red ring.
- **Light hidden skirt.** The hidden iris skirt beyond the limbus has the white-band colour, so a partly transparent
  veil never reveals a dark gap. Feathering past the limbus over a dark gap produced a pink-grey ring (full04/05).
- **Limbus ratio 5.1/12.** The visible-iris figure (HVID 11.7 mm) includes corneal magnification. With 5.85/12 the
  silhouette-normalised iris edge was 0.60–0.62 against UE's 0.47–0.55.
- **Less sparkle and SSS.** Sclera bump fades with the veil (`blend^3`); iris SSS defaults to 0.
- **Vessels.** Denser, finer capillaries now sprout from existing vessels; the tree still has no crossings.

`tmp/vhuman-mh/eye_layered2/limbus_profile.py` measures the left horizontal sector, normalised by the silhouette
radius, highlights excluded. It gives the 20–80 % iris→sclera rise and the minimum within the rise (a dark ring shows
as a negative dip). Right-eye sectors contain the key highlight, so those figures are unreliable.

| eye (left) | UE edge / width / dip | layered2 edge / width / dip |
|---|---|---|
| 020 (dev) | .52–.58 / .06 / −.03 | .52–.57 / .05 / −.03 |
| 100 (dev) | .51–.56 / .05 / −.05 | .53–.57 / .04 / −.06 |
| 038 (holdout) | .47–.52 / .05 / +.01 | .55–.56 / .01 / −.07 |
| 300 (holdout) | .30–.43 / .13 / +.19 | .56–.58 / .02 / −.24 |

The limbus ratio and feather were tuned on 020/100. On those the edge position and width now match UE. On the
holdouts the edge is still too sharp and too far out, and 300 keeps a dark dip. UE's 300 iris fades over a very wide
band with a yellow collarette. These are holdout failures, not fixed results.

Sheets: `tmp/vhuman-mh/eye_layered2/final2/overview.jpg`, `limbus_zoom.jpg` and `full_<id>_<side>.jpg`.

Remaining visible differences:
- a thin pink-grey band at the iris edge on 038/300;
- small bump glints on the sclera just outside the limbus;
- vessels still sparser and less saturated than UE's dense reticular network;
- the iris is darker than UE (no caustic gain).

#### Scleral near-limbus band round (`iris95`, 2026-11-08, diagnostic only)

The neutral near-limbus whitening band (`T.sclera_maps`, `base += .85*wb*(white - base)` with the fixed
`white = max(base)*[.98,.97,.96]`) was suspected of erasing the per-identity warm transition UE renders for 300/038.
Evidence first: radial medians of the *staged* textures matched UE basecolour within .0078/channel on 020/100/038/300
(this pass re-implemented iris70's pupil-aware mapping densely), so the texture *colour source* was not the residual
gap — the whitening band itself was. A full-size CPU rebuild with the existing `.85` strength reproduced the staged
texture to mean |.0007-.0009| (8-bit level), and the variants changed only the whitening zone.

Paired 512/32 dicing2/offscreen16 HIP renders with per-mode fresh baselines, production pipeline (scatter, hybrid,
profile5 eye-contact), scoring on fixed UE eye crops. Identities: 020, 038, 300 (dev/previous examined) and
recorded *iris95-only held-outs* 005, 194. Columns: crop LPIPS / DISTS / boundary MAE.

| eye (left) | baseline | band0 (whiten strength 0) | band50 | tint (target = near/max(near), strength .85) |
|---|---|---|---|---|
| 020 | .19274/.17986/24.08 | .19257/.18130/23.41 | .19248/.18068/23.78 | .19250/.18152/23.45 |
| 038 | .19182/.14985/16.50 | .19132/.15484/15.41 | .19122/.15206/15.93 | .19132/.15469/15.43 |
| 300 | .26697/.20147/33.83 | .26263/.19763/27.20 | .26396/.19875/31.28 | .26102/.19774/29.04 |
| 005 (held out here) | .23695/.16802/11.56 | .23649/.16663/11.04 | .23683/.16817/11.27 | .23653/.16660/11.05 |
| 194 (held out here) | .17496/.15702/13.20 | .17675/.16128/13.04 | .17561/.15776/13.11 | .17648/.16000/13.08 |

The chroma-clamped target (`tint`, tested second round) beat strength 0 everywhere: 300 crop LPIPS .26697 → .26102
and boundary 33.83 → 29.04, 005 equal, 194's band0 crop penalty softened. Boundary MAE improved in all ten
identity/mode pairs tested. DISTS still consistently worse than baseline outside 300/005 (+.0013-.0048), so this is
*not* a default promotion: the learned crop metric penalises colour detail that the reference's neutral band
appears to carry while LPIPS/boundary reward the removal of the grey-white drift. Visual:
`tmp/vhuman-eye-quality/iris95/comparison_zoom.jpg` — 300's grey donut is replaced by a warmer UE-like blend under
band0 with a middle ground under tint; 020/038 unharmed.

Distinguishing this from earlier failed attempts (iris66/69/…): those moved geometry/veil strength/opacity; the
visible mismatch source was the *fixed 4 mm neutral white band* painted over measured sclera. `iris95` measured that
exactly and verified reproduction (`variants_report.json`: `mh_000300 rebuild(.85) vs staged meanabs .0007`) before
any render.

Remaining work before considering a default:
- extend identities (145/256/358/178 need staged scenes + UE render references; per-identity `near` colour is
  already measured per eye in the staged bake, no new fit);
- the rendered transition still includes UE's own limbal veil/shadow and lid-geometry effects (previous failed
  attempts), so tint cannot close 300 fully;
- no source default change: variant textures live under `tmp/vhuman-eye-quality/iris95/`
  (`<sid>_band{85,50,00,tint}_sclera_left.png`); production default stays `peri_profile5` with untouched textures.

### Periocular study (face-visible eye cameras, 2026-11-07)

`compare_render.REGION_METRICS['periocular']` crops a window around the parser eye and brow labels with margin .45,
covering lids, canthi, lashes and the brow band. LPIPS and DISTS score the whole window; ΔE scores the labels.
- Score: `tmp/vhuman-mh/peri_score.sh <variant>`. It uses the ROCm venv on CPU; set `LD_LIBRARY_PATH` to all
  `/opt/rocm/core-*/lib`, because the system ROCm moved `libroctx64`/`librocm_smi64`.
- Render rounds: `tmp/vhuman-mh/peri_round.sh TAG VARIANTS`.
- Variants: the `PERI` dict in `compare_render.py`, layered on the default `combo` with eye_map v2 bases.
- New staging: `win_cycles.stage` now bakes eye2 assets with UE base-colour colours and builds periocular geometry
  parts into the shared base. `cycles_scene` renders them only when the variant enables them. Pulls exclude the
  regenerable eye textures; a pull filled /mnt/nvme01 once.

Fast 512 px, dev 020/100 and holdouts 038/300, both eyes (8 views). Columns: mean periocular LPIPS / DISTS / ΔE, and
the number of views better than `peri_l2`:

| variant | LPIPS | DISTS | ΔE | vs `peri_l2` |
|---|---|---|---|---|
| `combo` (default MH comparison render) | .3328 | .2390 | 14.01 | — |
| `peri_l2` = combo + layered eye2 | .3080 | .2280 | 13.08 | 8/8 better than combo |
| + eye AO .6 (`peri_l2_ao`) | .3038 | .2262 | 13.19 | 6/8 |
| + AO + UE-replica rect lights (`peri_l2_rect` / `peri_r2`, with canthal redness) | .2950 / .2948 | .2218 | 13.6 | 8/8 |
| `peri_r2` + upper tear meniscus (`peri_r2_men`) | **.2935** | .2192 | 13.72 | 5/8 vs `peri_r2` |
| caruncle v1 (protruding glossy bead) | .3299 | .2387 | 13.02 | rejected |
| caruncle v2 (recessed, duller), + AO | .3059 | .2271 | 13.27 | ~neutral vs AO; not adopted |
| thin-lid SSS ×2 on the orbital masks (`peri_r2_lid`) | .3240 | .2297 | 13.76 | rejected |

- **Eye AO** (`peri_eye_ao`) is a non-physical, labelled emulation of UE's translucent eye-occlusion shell: the AO of
  the lids within 2 mm darkens the eye2 base colour.
- **Rect lights** change the whole face lighting to match the UE rig; the slight ΔE rise comes from the skin.
- **Canthal redness** is in the eye2 sclera texture (asset v20) and is neutral on this metric.

The best current periocular variant is `peri_r2_men`. Visually it fixes the most obvious defect of `combo`: a flat,
opaque, wrong-colour eyeball with round highlights. Sheets: `tmp/vhuman-mh/eye_layered2/dev/peri03_sheet.jpg`
(UE | combo | rect | car2) and `peri02_sheet.jpg`.

Still unlike UE:
- The eyeball reads brighter and whiter, with less shading under the upper lid and at the corners.
- UE's lower lid shows a wet pink margin and a translucent pink-purple infraorbital area; ours is flatter and browner.
- Skin on 038/300 is more orange and grainier.
- Lashes and brows are not yet upgraded (Phase 3 of the plan).
- No full-quality confirm yet.

#### Periocular rounds 5–10 (2026-11-07, fast 512 px, 8 views)

| variant | LPIPS | DISTS | ΔE | note |
|---|---|---|---|---|
| `peri_r3_ao5`: eye AO 1.0 within 5 mm + rect + meniscus | .2934 | .2173 | 13.63 | ties `peri_r2_men` |
| lower-lid wet rim tube v1 / v2 | .3062 / .2939 | | | rejected (v1 sat on the eyeball) / neutral |
| `peri_r3_back`: eyeball 0.8 mm deeper | .2883 | .2145 | 12.90 | better on 5/8 views |
| `peri_r3_back14`: 1.4 mm deeper | .2874 | .2137 | 12.70 | black line under the lower lid |
| `+ peri_socket` (dark mucosa shell behind the eye) | .2874 | | | no pixel change: the line was not void |
| without tear lines (038/300 only) | ≈ equal | | | line gone, so the floating glass tear tube caused it |
| **`peri_r5`: 1.4 mm deeper; tear lines move back with the eye (×0.7)** | **.2840** | **.2116** | **12.55** | best; eye LPIPS .3135; better than `back14` on 6/8 views |

`peri_r5` vs the default `combo` (.3328 / .2390 / 14.01) is better on all 8 views. Sheet:
`tmp/vhuman-mh/eye_layered2/dev/peri10_sheet.jpg` (UE | combo | peri_r5).

The deeper seating (`peri_eye_back`) is a reconstruction-level change: the template sphere fit leaves the eyeball too
far forward relative to the lids. It is a fixed offset tuned on these four heads, not a fitted per-identity value.

Still different from UE:
- 300's sclera is white where UE's is red (eye colour inputs come from older staged textures for that head);
- UE's lower lid shows a pink wet waterline and translucent infraorbital skin;
- the skin on 038/300 is more orange and grainier;
- lashes and brows are not yet upgraded;
- no full-quality confirm yet.
Results now live on /mnt/nvme02 (`/mnt/nvme02/tmp/vhuman-compare`, symlinked from `compare_v1`).

#### Fast local sweep loop and periocular knobs (2026-11-07)

- **Local GPU Cycles.** Official Blender 5.2.3 runs on the RX 9070 XT via HIP; see `resume-vhuman-eye.md` for the
  environment.
- **Persistent bases.** `win_cycles.render_local` / `render_cached` keep staged bases across rounds.
- **In-session sweeps.** `peri_sweep.py` + `peri_sweep_blender.py` render many hypotheses from one built scene, at about
  10.6 s per hypothesis-view.
- **Fast scorer.** `peri_sweep --score` scores the periocular region only, at about 12 s per variant.
- **Determinism.** The local sweep reproduces worker scores exactly.

Knob sweep on `peri_r7k` (`peri_r5` + skin SSS scale 1.0 + eye AO .7, LPIPS .2789). Knob defaults are now reset per
hypothesis; the first run (`knobs1`) leaked settings and is discarded.

| hypothesis | periocular LPIPS | DISTS | ΔE | views better |
|---|---|---|---|---|
| lid (orbital-mask) roughness ×0.8 | **.2568** | .2036 | 12.20 | 8/8, incl. both holdouts |
| lid ×0.8 + infraorbital ×0.8 | .2559 | .2033 | 12.25 | 7/8 |
| lid ×0.65 / ×0.5 | .2756 / .3038 | | | worse |
| colour tint 1.03/1.11/1.14 on the lid and infraorbital masks | .2790 | | | no effect |
| infraorbital roughness ×0.8 alone | .2788 | | | no effect |

New best: `peri_r8` = `peri_r7k` + `peri_k_lid_rough` .8 (periocular LPIPS .2568 vs `combo` .3328). The default
per-identity skin roughness boost (×~1.32) appears too rough on the eyelids; UE's lids read glossier. A ×0.8 lid
correction is a fixed value tuned on 020/100 and improves 038/300 too. Sheet: `tmp/vhuman-mh/eye_layered2/dev/knobs2_sheet.jpg`.

### Contact-shading experiment (2026-11-07, implementation pass)

Added `peri_contact01`, based on the frozen `peri_r8`, with a smooth 0.5 mm
metric distance band along staged lid margins. The band reduces roughness and
blends local skin SSS toward weight .8 / scale 1.0; eye AO uses a smoothstep
ramp. These are experimental authored controls, not fitted tissue measurements.
The field is stored on skin vertices, so narrow-band accuracy remains limited by
mesh resolution. The existing tear geometry has not yet been replaced.

Local Blender 5.2.3 / HIP RX 9070 XT, 512 px / 32 spp, both eyes of
020/100 (development) and 038/300 (previously examined validation):

| variant | periocular LPIPS | DISTS | delta E | render wall time |
|---|---|---|---|---|
| standalone peri_r8 | .2567 | .2047 | 12.432 | 276 s |
| peri_contact01 | .2590 | .2041 | 12.389 | 315 s |

**Not accepted as default:** LPIPS regressed and inspected development crops
show only subtle contact changes. No full-quality or fresh-identity acceptance
has been established. No claim of improved skin translucency is made.

Artifacts: `tmp/vhuman-eye-quality/first-contact.jpg` (UE | r8 | contact01),
`scores.json` (development and previous-validation scores separately).
Fresh UE 5.8 DX12 captures were completed on Windows for 020/100 (job
`eye_contact_ref01`, exit 0, 107 s); retrieved under
`tmp/vhuman-eye-quality/ue/`. They remain separate from the frozen references
used above. Eye-capture mean absolute linear differences from older references
range .0049–.0117, so they must not silently replace scoring inputs.

Sweep reliability: restore AO, SSS radius, denoising and dicing per hypothesis;
reject unknown named controls; hash scene inputs and shader sources before
reusing built sweep scenes. Metric distance tests cover segment endpoints,
duplicate path points, rigid transforms and invalid inputs.

Commands: `TMPDIR=$PWD/tmp bash tmp/vhuman-mh/peri_local.sh peri_r8` and
`... peri_contact01`; `TMPDIR=$PWD/tmp tmp/vhuman-cpu-venv/bin/python -m unittest
server.vhuman.test_periocular` (3 passed); Python compilation and `git diff
--check` passed.

Still required: per-pixel/atlas contact-field accuracy, waterline replacement,
independent UE SSS/occlusion diagnostics, wider regional skin-scattering work,
full 1024/128 confirmation, front/isolated-eye checks and new identities before
promoting any new shading default.

### Diffuse-only skin SSS continuation (2026-11-07)

A new **experimental** UE-inspired skin-lighting diffusion path is promising;
no production/default switch has been made yet. UE is still captured on Windows
(DX12); all new Cycles renders use Linux HIP / RX 9070 XT.

- `ue_scripts/peri_diagnostics.py` captures SSS on/off × eye-occlusion on/off
  without touching frozen references. Corrected run `peri_diag03` completed
  successfully on 020/100: 15 files per head, all SHA-256 verified. An earlier
  `peri_diag02` produced images but failed light cleanup; do not use its overall
  job status as a success. The corrected script restores the SSS cvar.
- UE's ablation shows strong smoothing of skin-lighting detail and warm lid
  transitions from SSS. Eye occlusion mainly changes exposed eyeball shading.
  Sheet: `tmp/vhuman-eye-quality/ue_ablation02.jpg` (earlier run's completed
  images); authoritative successful captures are under `ue_diag03/`.
- `reconstruction/periocular.py::build_contact_atlas` now evaluates metric lid
  distance at 2048² texel centres, rather than interpolating sparse vertex
  distances. UV seam, mirrored-chart, geometry-cache, and analytic-distance
  tests pass. It is staged by both offline_render and win_cycles.
- `peri_passes_blender.py` exports genuine multi-part EXR diffuse lighting,
  diffuse colour, world position, geometric normal, skin mask and beauty.
  Named compositor inputs are required: the trailing empty input is virtual.
  `peri_scatter.load_passes` validates material masks and geometric normals to
  reject unconnected default-value passes.
- `reconstruction/skin_scatter.py` filters **lighting only**, using a normalized
  Burley radial profile with skin/depth/geometric-normal rejection, then
  recombines diffuse colour and leaves other beauty contributions unchanged.
  This is a labelled screen-space approximation, not Cycles volume transport;
  offscreen transport and thin-part transmission are not reproduced. Do not
  promote it globally without front/ear/lighting regression checks.
- The frozen candidate uses mean-free paths [1.05, .4069, .2625] mm × **0.5**,
  weight 1, 48 deterministic gather samples; native skin SSS is disabled in
  its capture to avoid double diffusion. No tuning on 038/300 was performed
  in this continuation. Those heads were already examined in earlier rounds,
  so report them as previous validation rather than unseen holdouts.

Fast 512 px / 32 spp, native dicing 2, both eyes (same scorer and references):

| set | native r8 LPIPS / DISTS | diffuse-only LPIPS / DISTS |
|---|---|---|
| development 020/100, 4 views | .2276 / .1901 | **.2042 / .1692** |
| previous validation 038/300, 4 views | .2858 / .2193 | **.2568 / .1889** |
| all 8 views | .2567 / .2047 | **.2305 / .1790** |

All eight views improve LPIPS. These results are `peri_scatter06`, generated
with the frozen half-scale profile; source provenance is beside each EXR.
Sheets inspected: `tmp/vhuman-eye-quality/scatter06_eye_left.jpg` and
`scatter06_eye_right.jpg` (fixed baseline exposure for both Cycles columns).
Skin lighting is softer, but geometry/waterline differences and 300's iris
remain visible. Metrics do not establish a complete UE match.

A separate **fixed reference crop + fixed baseline exposure** check on the
2 development left-eye views also improved: LPIPS .2265 → .2069, DISTS
.1882 → .1688, skin-only ΔE 4.21 → 3.97. The standard periocular ΔE uses
*eye/brow labels*, not skin; its small rise is not evidence of worse skin
colour. Details: `fixed_score05.json` and `scatter06_summary.json`.

Rejected/not adopted in this continuation:
- Simply changing native SSS weight/method/radius (`material04.json`, 2 dev
  left eyes): no setting improves both LPIPS and DISTS over r8. Native ref
  .2390/.1920; weight 1 .2415/.2034; Burley 1 .2406/.1840; radius scale 1.4
  .2427/.1981; UE-radius random walk .2481/.1937.
- Removing procedural micro-normal looked promising with coarse dicing but
  does not improve native-quality scores (.2397/.1959 vs .2390/.1920).
  Do not adopt from the coarse screen.
- Atlas contact band alone `peri_surface02`: dev 4-view LPIPS .2290 vs r8
  .2276. Tube-free surface coat `peri_waterline04`: .2317. Both remain
  experiments. Waterline coat composes with existing coat and uses tear-film
  IOR 1.336, but the current highlighted rim is still too prominent.

Verification: 10 focused distance/atlas/scattering tests pass, including
energy conservation, red/blue diffusion extent, pigment/specular preservation,
alpha preservation, and eye-boundary exclusion. Four existing comparison/
variant workflow tests pass. Python compilation and git diff --check pass.

In flight / next steps:
1. Paired full-quality 1024 px / 128 spp confirmation (`full07.py`, log
   `tmp/vhuman-eye-quality/full07.log`) on 020/100/038/300, both eyes, frozen
   settings. First attempt hit the 14336 MiB VRAM guard and stopped cleanly.
   Current attempt uses 256 px Cycles tiles with the same resolution/samples/
   dicing. Poll the existing process before any restart. Outputs are
   `peri_capture07_native` and `peri_scatter07` under full-profile paths.
2. Four additional identities 005/145/178/194 are being staged by the live
   CPU job into `tmp/win/peri_holdout06`; log `holdout06_stage.log`. Includes
   front and both eye cameras. Do not retune the frozen SSS profile on these.
3. Validate fresh identities, front views, and thin-part/backlighting effects.
   The current whole-skin native-SSS-off prototype cannot replace physical ear
   transmission; preserve it in any production integration.
4. Integrate a validated profile into the full-quality default rendering path,
   retaining explicit overrides and frozen historical variants. Currently the
   new diffusion path is a diagnostic CLI, **not** the default render path.
5. Continue waterline/inner-corner work, then the remaining eye2 holdout limbus
   gaps. Do not claim the boundary is solved by the SSS improvement alone.

Useful commands (repo root; scratch only under repository tmp/):
```
TMPDIR=$PWD/tmp tmp/vhuman-cpu-venv/bin/python -m unittest server.vhuman.test_periocular server.vhuman.test_skin_scatter
TMPDIR=$PWD/tmp tmp/vhuman-cpu-venv/bin/python -m pytest -q server/vhuman/test_metahuman_prior.py -k 'win_cycles or eye_sweep or default_variant' --basetemp=tmp/vhuman-eye-quality/pytest
# Rendering scripts use VHUMAN_BLENDER and VHUMAN_BLENDER_LD_LIBRARY_PATH as in the handoff.
# peri_scatter needs OpenEXR; use the existing ROCm venv (CPU venv lacks it).
LD_LIBRARY_PATH=$VHUMAN_BLENDER_LD_LIBRARY_PATH tmp/vhuman-rocm-venv/bin/python -m server.vhuman.metahuman.peri_scatter CAPTURE.exr OUT.exr --scale .5
```

Current worktree is still uncommitted. Preserve all earlier eye work. No push.

### Native-preserving face diffusion and optical eye contact (2026-11-07)

The full-quality default `material_overrides.skin_lighting_profile=1` captures unscattered
Cycles diffuse lighting in a second render, diffuses that lighting with a
normalized Burley radial profile, then restores albedo and the other beauty
contributions. Default channel radii are `[1.05, .4069, .2625] * .5` mm. It uses
actual camera depth, surface normals and skin-material rejection rather than
blurring the beauty image. Exact point-attribute anatomical weights retain the
native render on ears, lips and non-skin surfaces; native alpha is retained.

`eye_contact_blur=1` adds a localized optical scene-colour blend at the lid
boundary. This is a nonphysical approximation of MetaHuman's eye occlusion
material blur, separate from the diffuse skin SSS above. The eye-side kernel has
sigma .25mm, band1.2mm, strength.847213 and the UE material's warm tint.
Profiles1–4 retain their prior eye-side strength.357397. The new
skin-side band has the same sigma/width and strength1; actual skin coverage and
a10mm camera-depth bound restrict its footprint. Central iris, alpha and pixels
beyond the contact band are preserved. Kernels are fitted to matched UE blur-on/off
development captures, independently of validation beauty images.
Explicit overrides include `skin_scatter_scale`, `skin_scatter_weight`,
`eye_contact_width_m`, `eye_contact_sigma_m`, `eye_contact_strength`,
`eye_contact_skin_width_m` and `eye_contact_skin_strength`. The latter set to0
limits the contact layer to the eye side. `eye_contact_blur=0` disables both
contact bands; `skin_lighting_profile=0` disables the extra lighting pass.
The profile is enabled by default for `offline_render` full-quality skin and the
MetaHuman comparison default `peri_profile5`; baseline quality and historical
comparison variants retain their previous settings. Explicit
`material_overrides={"skin_lighting_profile": 0}` restores native rendering.
Install offline-render dependencies from `requirements-render.txt`; this includes
OpenEXR for the lighting/normal AOVs. Multipart reads and signed RGBA round trips
were verified with both OpenEXR3.4.4 and3.5.3.

At 1024px/128spp/dicing2, eight paired eye views (020/100 development and previously
examined 038/300 validation) give:

| Group | Native LPIPS / DISTS | Profile + contact LPIPS / DISTS |
|---|---|---|
| Development, four views | .2222 / .1835 | .2072 / .1677 |
| Previously examined validation, four views | .2867 / .2147 | .2596 / .1930 |

All eight LPIPS improve. The fixed-native-exposure sheet was visually inspected;
diffuse skin softens while pigment remains. Specular sheen, eyelid geometry and
the 300 iris remain mismatches. These results do not establish new-identity
performance. The treatment requires a second GPU render and retains
`beauty_native.exr` for comparison. `head.blend` reproduces the native pass; final
beauty requires the profile finalizer. Detailed evidence is under repository
`tmp/vhuman-eye-quality/full14_scores.json` and `full14_sheet.jpg`.

Fresh validation on 005/145/178/194 used the frozen recipe, with no retuning.
Seven parser-valid eye crops improve from LPIPS/DISTS .2518/.1862 to .2281/.1680;
145 right has no valid parser crop in either version. A separate fixed-camera,
fixed-native-exposure comparison includes all eight views: .2178/.1722 to
.2123/.1639. Identity194 is mixed (DISTS worse in both views and LPIPS slightly
worse in one), so the gain is not universal. Six front views give skin/nose
LPIPS .1121→.1115, DISTS .1651→.1671, and skin ΔE4.30→4.10: modest mixed changes.
Fresh-eye and front sheets were inspected (`validation18_sheet.jpg`,
`front19_sheet.jpg`). Physical thin-part and non-skin shading remains native.
The real-head production capture finished at 13.7GiB whole-device peak and
agrees with the static diagnostic to mean absolute scene-linear difference
.0000652; the difference reflects the exact anatomical/eye material AOVs.

### Procedural pore grid and normal-composition audit (2026-11-07)

The legacy pore generator adds crossed sine waves at three frequencies. Their
normal gradients create a visible specular crosshatch. The optional
`micro_normal(grain='isotropic')` retains Gaussian pore locations and replaces
that fine grain with periodic random Fourier bands of similar height RMS.
The strongest normal-map Fourier-bin fraction drops .0470→.00142. The generator
stays legacy-compatible; the accepted full-quality profile selects a separate
isotropic asset.

The measured/procedural whiteout blend also scales each layer's XY but leaves
its Z active. Consequently, disabling procedural pore strength still multiplies
measured normal slopes by `1 / pore_normal.z` (median1.154, p951.861 on the
current texture). The `skin_normal_weighted_blend=1` control normalizes each
slope-weighted layer before whiteout; a zero-strength layer becomes (0,0,1).
The accepted regional default retains legacy calibration inside the anatomical
protection masks and applies this correction outside them.
This is independent of the skin-diffusion and optical-contact treatment.
Development-only paired captures and fixed-native-exposure comparisons are under
`tmp/vhuman-eye-quality/grain23` and `weighted24`. These development results led
to the global and then regional confirmations described below.

Development screen (020/100 left eye,512px/32spp/dicing2, existing diffuse profile
and optical contact held fixed, full eye-camera frame and baseline exposure):

| Normal/pore treatment | LPIPS | DISTS |
|---|---:|---:|
| Previous default | .18055 | .12542 |
| Corrected blend, legacy grain | .17732 | .12368 |
| Corrected blend, isotropic grain | .17634 | .12443 |
| Corrected blend, no procedural pores | .16800 | .12801 |
| Corrected blend, half-strength isotropic pores | **.16047** | **.11946** |

Half-strength isotropic pores were selected on these two development views and
frozen for further validation (`selection27.json`). `peri_profile2` exposes this
opt-in recipe: `skin_normal_weighted_blend=1`, `skin_micro_isotropic=1`,
`skin_pore_scale=.5`. The separate isotropic PNG is cached with seed, generator
source and NumPy-version provenance. Synthetic spectrum/cache tests pass, and
the production graph matches the experimental graph to mean scene-linear error
~1.95e-7. The production-generated pore PNG matches the experimental PNG exactly.

Initial005/194 validation is mixed: fixed-reference eye crops average
LPIPS/DISTS .21027/.15693→.20714/.15766; fixed camera frames average
.21375/.15172→.20278/.14718. Identity194 right regresses. Subsequent full128spp
tight-eye confirmation regressed .21099/.16603→.21603/.17328, so the global
recipe was not promoted. The regional treatment below addresses that tradeoff.


### Regional normal default (2026-11-07)

The accepted default is `peri_profile3`: the existing native-preserving diffuse
skin profile and optical eye contact, plus corrected normal-layer weighting and
half-strength isotropic added pores outside the orbital/infraorbital regions.
The existing GNM anatomical masks retain the previous calibrated normal treatment
on those regions. Unit tangent normals are mixed in the shader; no beauty-image
blend or skin-color blur is added. With procedural-only normals, the same masks
blend legacy/isotropic pore textures at the existing pore strength.

The global version `peri_profile2` remains opt-in: it removed the grid but
regressed both full-quality tight eye crops. The regional version was selected
on development020/100 and confirmed with a frozen recipe. All displayed
comparisons use UE-only fixed crops and fixed previous-default exposure.

| Comparison | Previous LPIPS / DISTS | Regional default LPIPS / DISTS |
|---|---|---|
| Full1024/128, two development eye crops | .21099 / .16603 | .20671 / .16615 |
| Same two whole eye-camera frames | .18078 / .12370 | .15533 / .11725 |
| Six validation eye crops,005/194/038/300 | .22316 / .16664 | .21297 / .16545 |
| Same six whole frames | .21816 / .15043 | .19760 / .14398 |
| Two development front skin crops | .11698 / .16144 | .11793 / .16190 |

Seven of eight eye-crop LPIPS improve;194 right is slightly worse
(.20995→.21233). Full-quality tight-crop DISTS is essentially flat, and front
changes slightly regress. Full/validation/front sheets were visually inspected;
the surrounding crosshatch is removed while calibrated eyelid detail remains.
The full-quality captures retain dicing2 and reached14251MiB whole-device usage,
below the14336MiB guard. Both public API branches (measured normal map and
procedural-only) passed paired GPU capture/finalization and saved-scene reload
validation on Linux HIP. Tests:24unittest +4 targeted prior tests pass.

The four explicit normal controls are:
```
skin_normal_weighted_blend: 1
skin_micro_isotropic: 1
skin_pore_scale: 0.5
skin_normal_preserve_lids: 1
```
They are defaults when full skin and the skin lighting profile are enabled.
Explicit overrides win. To retain the previous SSS/contact treatment with its
previous normals, use comparison variant `peri_profile`, or set the four controls
to0,0,1,0 respectively. Setting `skin_lighting_profile=0` with no explicit normal
controls disables the default bundle. Baseline quality and historical comparison
variants remain unchanged; cached legacy pore assets are not replaced.

An independent signed world-normal audit captured fresh UE references on the
Windows worker (UE5.8.4-58902052,D3D12; actual capture log/source hash verified)
and Cycles fields on Linux HIP (Blender5.2.3), using identical Cycles geometry
between variants. Mean common-skin angular error on the two development left-eye
frames improves7.25°→6.54°; periocular skin improves8.15°→7.91°. These errors also
include residual UE/GNM geometry correspondence differences. Measured-only
normals improve the angle further but do not improve beauty consistently, so
that result was not promoted. The remaining bright lid-contour error bands
suggest geometry/BRDF calibration deserves further study; this is an inference,
not proof of a tangent-convention bug.

Evidence under `tmp/vhuman-eye-quality/`: `regional35/summary.json`, `scores.json`,
`full_sheet.jpg`, `validation_sheet.jpg`, `additional_sheet.jpg`, `front_sheet.jpg`;
`production36.log`; `normal38/audit/report.json` and `world_normals.png`.
UE reference file hashes were verified. `peri_profile`, `peri_profile2` and the
legacy generator remain available for reproducible comparisons.


Full-quality held-out confirmation also completed on038/300 at1024px/128spp,
dicing2, with the recipe unchanged:

| Full-quality group | Previous eye-crop LPIPS / DISTS | New default LPIPS / DISTS |
|---|---|---|
| Development020/100 | .21099 / .16603 | .20671 / .16615 |
| Validation038/300 | .24857 / .18738 | .23745 / .18314 |
| All four full-quality views | .22978 / .17671 | .22208 / .17464 |

All four full-quality eye-crop LPIPS improve. Whole-frame mean LPIPS/DISTS improve
.20279/.13683→.17583/.12776. The final four-row full-quality comparison was
visually inspected: `tmp/vhuman-eye-quality/regional39/full_confirm_sheet.jpg`;
metadata and split summaries are in `regional39/scores.json` and
`regional39/full_summary.json`. The normal-field capture's actual UE build,
D3D12 adapter choice and source hash are recorded in
`normal37/capture_provenance.json`. All owned render jobs completed.

### Eye-aperture diagnostics (2026-11-08)

`ue_scripts/eye_aperture.py` captures original beauty plus an opaque white-eye,
black-head mask in the Windows UE worker, restoring material slots/sections and
removing temporary assets afterwards. Translucent shells, lacrimal edges and
lashes are hidden. This separates mesh intersections from their optical shading;
material replacement also removes UE vertex/depth offsets, so the mask is not the
final photometric eye opening. The matching Cycles capture keeps skin displacement.

`python -m server.vhuman.metahuman.aperture_audit --ue-root <UE-captures>
--cycles-root <Cycles-captures> --out <report-directory>` compares fixed-camera
masks without alignment fitting, reports IoU and symmetric outline distances,
and overlays cyan UE/magenta vhuman on the matching captured UE beauty. Source
image SHA256 hashes and method limitations are included. Three synthetic tests
cover known translation, isolated speckles/holes, and invalid masks.

Development020/100 left/right,512px: vhuman's opaque eye opening is slightly
larger; IoU .887/.950/.947/.966, mean outline distance2.51/1.10/1.27/.85px.
Adding .3 or .6mm of eye depth improves every outline IoU. These diagnostics use
Linux Blender5.2.3 HIP, visible dicing2 and offscreen dicing16; Windows references
use UE5.8.4 D3D12. They do not justify a geometry default without beauty validation.
Artifacts: `tmp/vhuman-eye-quality/aperture41`, `aperture42/audit`, `aperture43`.

A separate paired512/32 development study replaced glass tearline tubes with a
Fresnel surface film, removed all/upper tubes, or undid their backward following.
Film/removal changes were negligible (eye LPIPS changes below .0003); bringing
existing tubes forward worsened both heads (.20538→.21785, .18813→.19620).
No tearline material change was accepted. Skin/SSS/profile3 remained fixed.
Artifacts: `tmp/vhuman-eye-quality/waterline40/{scores.json,comparison.jpg}`.

A follow-up raw-mesh CPU z-buffer reproduces the UE opaque masks at IoU
.9921–.9947 (mean boundary .14–.19px). Removing the cartilage component changes
zero eye-opening pixels in these four views. Substituting the UE eye meshes only
for this diagnostic under the raw GNM skin raises the largest mismatch,020left,
from .887 to .966 IoU; all four hybrid openings reach about .965–.971. No UE eye
geometry was added to the renderer. These controls point primarily at globe
shape/placement, with a smaller remaining skin binding/subdivision discrepancy.

The staged candidate globes have radius about14.6mm in all four checked
identities, while rear-hemisphere sphere fits of UE eyes range about13.0–14.6mm.
The front UE sclera is also recessed relative to that fitted sphere. Centers
and dimensions must be compared after the camera-derived frame translation;
untranslated UE and GNM coordinates have different origins. This is a mesh
measurement, not a general claim about human eye dimensions. Raw audit artifacts:
`cartilage48`, `skin49` and `aperture43/globe_dimensions.json` under the same
scratch root.

A fixed extra.3mm depth (total1.7mm) improves development020/100 fast eye-crop
LPIPS .19675→.19172, but frozen038/300 validation is essentially flat
(.23700→.23693), with038 slightly worse. The accepted default remains unchanged.
Scleral-shape studies keep the calibrated cornea/iris positions and accepted skin
profile fixed; outline agreement alone does not establish a beauty improvement.

### Skin-side contact default (2026-11-08)

`peri_profile4` extends the existing optical contact layer onto a narrow skin-side
band. The matched UE blur-on/off captures show a substantial effect outside the
opaque eye opening: a shared1.2mm band/.25mm sigma/strength1 explains about80%
of that development difference. The earlier eye-only fit omitted this area.
This is a scene-colour approximation of the translucent eye layer; skin SSS
continues to use the separate diffuse-lighting profile. It adds no GPU render pass.

The new controls are `eye_contact_skin_width_m=.0012` and
`eye_contact_skin_strength=1`. New full-quality offline renders enable them;
explicit overrides win. `peri_profile`, `peri_profile2` and `peri_profile3` retain
skin strength0. The low-level contact helper also defaults to0 for compatibility.
Skin coverage restricts the new band; skin farther than10mm in camera depth from
the nearest visible eye pixel is rejected. Alpha and zero-weight pixels remain
exactly unchanged. Iris centres are preserved.

Recipe selection used only the four Windows UE blur-on/off development captures,
with a fixed common exterior region; validation beauty images did not tune it.
Native Cycles captures use Linux HIP, Blender5.2.3, visible dicing2 and diagnostic
offscreen dicing16. Fresh paired controls share those settings.

| Fixed UE eye crops | Previous LPIPS / DISTS | New LPIPS / DISTS |
|---|---|---|
| Full1024/128, development020/100 | .20570 / .16591 | .19878 / .15986 |
| Six512/32 validation views005/194LR+038/300L | .21279 / .16511 | .20918 / .16328 |

All eight distinct eye views improve crop and whole-frame LPIPS. Both full crops
improve DISTS; validation DISTS improves5/6 (194left slightly worse). Pixel boundary
MAE is mixed: full23.545→23.786, validation19.435→19.339 (5/6 improve). The
perceptual gain should not be described as a universal pixel-error improvement.
The final comparisons use the same eye-material AOV as the public renderer;
original skin and position AOVs match the cached controls exactly. Full and
six-view comparison sheets were visually inspected. Remaining iris,
canthus and skin-material differences are visible in the comparisons.

Artifacts under `tmp/vhuman-eye-quality/`: `contact56/report.json` records the
matched-ablation fit and image hashes; `contact57/{summary.json,full_comparison.jpg,
validation_comparison.jpg}` records the frozen experiment. `contact58/verification.json`
checks the production implementation on all ten cached cases (fast/full repeats
included): new output exactly matches the prototype, and strength0 exactly
reproduces the old output. Final actual-AOV confirmation is recorded in
`mask60` and `contact61/{summary.json,verification.json,full_comparison.jpg,
validation_comparison.jpg}`. All ten cases preserve alpha and zero-weight pixels
exactly; the full capture peak was14208MiB below14336MiB. Tests cover the localized footprint, foreground depth
rejection, invalid inputs, alpha, default opt-outs and historical variants.

Production API validation also passes with measured normal maps and procedural-only
normals on Linux HIP, including saved-scene reload. Both actual eye/skin AOV
compositions change the intended contact pixels while preserving alpha and every
zero-weight pixel exactly. Peak device usage was10190MiB, below14336MiB. Runtime
records: `tmp/vhuman-eye-quality/production59_summary.json`.

Workspace validation commands (17 unit checks +1 prior/default check pass):
```sh
TMPDIR=$PWD/tmp tmp/vhuman-cpu-venv/bin/python -m unittest server.vhuman.test_eye_contact server.vhuman.test_skin_profile server.vhuman.test_aperture_audit -q
TMPDIR=$PWD/tmp tmp/vhuman-cpu-venv/bin/python -m pytest -q server/vhuman/test_metahuman_prior.py::test_default_variant_2026_11_08 --basetemp=tmp/vhuman-eye-quality/pytest58
```

### Eye-side contact default (2026-11-08)

`peri_profile5` strengthens the eye-side contact blend to.8472126447375768.
The original fit used a face-parsing mask; fitting the existing kernel to the
independently captured opaque UE eye openings gives a more accurate boundary.
On those same four development blur-on/off captures, normalized residual SSE
falls.61956→.42854 (about31% lower), improving all four cases. Width1.2mm,
sigma.25mm and the skin-side treatment follow the existing profile. Selection
used only the UE ablations, before checking development/validation beauty scores.

New full-quality render requests receive this strength through `default_overrides`.
Explicit values win. `peri_profile4` and older named profiles explicitly keep
.35739718109334984, and low-level helper/old-request fallbacks retain their prior
values. Use `peri_profile4` or that explicit `eye_contact_strength` to reproduce
the previous default. The new setting changes only pixels in the existing
visible eye-side band; alpha and pixels outside that band are exact.

| Fixed UE eye crops, previous-default exposure | Profile4 LPIPS / DISTS | Profile5 LPIPS / DISTS |
|---|---|---|
| Full1024/128 development020/100 | .19875 / .15982 | .19164 / .15536 |
| Six512/32 validation005/194LR+038/300L | .20908 / .16332 | .20764 / .16238 |

All eight unique eye crops improve LPIPS and boundary MAE. Full boundary MAE
23.793→23.251; validation19.339→18.857. DISTS improves6/8 (194LR worsens), and
whole-frame LPIPS improves7/8 (005right slightly worsens .135483→.135585).
Full and six-view comparison sheets were visually inspected. This is still a
scene-colour approximation of the UE contact layer, separate from skin SSS.

Artifacts under `tmp/vhuman-eye-quality/`: `eye62/fit.json` contains the frozen
fit, per-case residuals and source-image hashes. `contact63/{summary.json,
full_comparison.jpg,validation_comparison.jpg}` contains the actual material-AOV
comparison. Native GPU captures are reused because this change affects only
composition. `contact64/verification.json` confirms all ten cached cases: the
new default matches the candidate exactly, old profile4 reproduces its output
exactly, and alpha/outside-band pixels are exact.

Both public Linux HIP rendering branches (measured/procedural normal maps) pass,
including saved-scene reload and actual AOV composition checks against profile4.
Peak device usage10185MiB<14336MiB; results in `production65_summary.json`.
The targeted17 unit checks and prior/default pytest check pass (same commands
above; latest pytest scratch directory `pytest65`). No new GPU render pass or
render dependency is introduced.

### Experimental wider eye-side contact

`peri_profile6` is an opt-in comparison variant. It keeps profile5's skin-side
width and uses a separate 0.25 mm skin blur sigma, while widening the eye-side
band to 1.8 mm with a 0.35 mm sigma and strength 0.5855533446742217. The
default remains `peri_profile5`. A fit to Windows UE blur-on/off ablations on
020/100 reduced pooled residual SSE from .42854 to .31183. On newly captured
005/194 left and right eye ablations and independently captured opaque eye
openings, the frozen candidate reduced pooled SSE from .39634 to .31914 and
improved all four cases. These measures isolate the UE optical layer; they do
not establish overall face quality.

With the actual eye material AOV, six 512-pixel validation crops improved mean
LPIPS .207660 to .207187 and boundary MAE 18.857 to 18.647, while DISTS changed
.162344 to .162528. Two 1024-pixel development crops changed mean LPIPS .191635
to .192071 and DISTS .155406 to .156934. Those image results and a visual review
are mixed, so the variant is not a new default. The prototype and production
composition differ by at most 0.000089 in scene-linear RGB in the ten cached
views because their skin blur uses a slightly different median depth. Evidence:
`tmp/vhuman-eye-quality/boundary94` through `boundary98` and `aperture95`.
The public Blender GPU render path also passed a 512-pixel / 16-sample test with
the separate sigmas: the saved image recomposed exactly and preserved alpha;
peak whole-device usage was 10169 MiB (`boundary98`).
The named `peri_profile6` layered-eye staging and saved-asset reload path also
completed on Blender HIP at 512 pixels / 32 samples, with 11174 MiB peak
whole-device usage (`boundary99`).

`peri_profile7` is a second opt-in comparison that keeps profile6's eye width
and sigma while using an independently specified, nearly neutral eye-side tint
and strength .670908. The skin-side tint and sigma remain the profile6 values.
Fitting tint and strength on the 020/100 UE ablations reduced pooled residual
SSE .31183 to .24408; on the frozen 005/194 holdout it changed .31914 to
.27250, improving three of four eyes. On cached 1024-pixel development beauty
renders, mean LPIPS changed .192009 to .191056, DISTS .156857 to .156483 and
boundary MAE 23.199 to 23.545; six validation crops give mixed DISTS and boundary
errors. Visual differences are subtle and the default stays `peri_profile5`.
See `boundary101` through `boundary104` under `tmp/vhuman-eye-quality/`.
The named layered-eye Blender HIP path, including saved-asset reload, passed at
512 pixels / 32 samples with 11177 MiB peak whole-device usage (`boundary104`).
Directly against profile5, the six 512-pixel validation crops changed mean
LPIPS .207660 to .207012 and boundary MAE 18.857 to 18.643, but the two
1024-pixel crops changed DISTS .155406 to .156502 and boundary MAE 23.255 to
23.544. This supports keeping the default unchanged (`boundary105`).
On two additional 512-pixel left-eye crops, both variants improved LPIPS and
DISTS for identity 256 and worsened them for identity 145 (`boundary120`).
Fresh UE blur-on/off captures and opaque aperture masks for both eyes of those
identities gave pooled relative SSE .35917 / .30841 / .24419 for profiles
5 / 6 / 7, respectively; all four eye cases improved in this isolated optical
test (`boundary118`, `aperture119`). The 256 right-eye ablation has a much
smaller signal and a large relative error for all three profiles. Overall
beauty results remain mixed, so profile5 remains the default.

`peri_profile8` is an opt-in experiment based on profile7. It estimates each
visible eye's long axis and varies the eye-side contact strength between
0.7702 at the upper lid and 0.4295 at the lower lid; skin-side settings are
unchanged. The weights were fitted to UE 020/100 optical ablations. On fresh
038/300 left- and right-eye ablations, actual production contact code reduced
pooled relative SSE from .16319 for profile7 to .12465, improving all four
eyes (`boundary134`). Overall beauty results remain mixed: two 1024-pixel
development crops changed mean LPIPS .19093 to .19149 and DISTS .15629 to
.15849 (`boundary141`). The axis is inferred from screen-space coverage,
not anatomical landmarks; very small, nearly closed, round or near-vertical
components fade toward neutral weights. Fractional coverage makes that fade
smoother as the eye closes, but upside-down eyes are not disambiguated. The
default remains profile5.
Against profile5 directly, six 512-pixel validation crops changed mean LPIPS
.20768 to .20692 and boundary MAE 18.830 to 18.591, while two 1024-pixel
crops changed DISTS .15542 to .15846 (`boundary142`). On the additional
145/256 left-eye views, profile8 worsened both LPIPS and DISTS for 145 and
slightly improved both for 256 (`boundary143`).
The public portrait and named layered-eye HIP paths passed saved-scene reload
and exact recomposition; the split changed only the eye-contact band and
preserved alpha (`boundary132`, `boundary133`; peak whole-device usage 11176
and 10172 MiB, respectively).
Two additional full-face front and 40-degree yaw comparisons also show mixed
results: profile8 slightly improved both eye-crop LPIPS and DISTS in the two
front views, while the 256 yaw view worsened both. The differences are small
except for DISTS on that yaw view; the assets also differ in face shape and
skin texture (`boundary138`). All four saved HIP renders passed exact baseline
recomposition and alpha checks, with a peak of 11039 MiB (`boundary137`).

An independent on/off skin-scattering diagnostic on those front and yaw views
showed a substantial identity dependence. At a two-pixel high-pass scale, the
UE on/off RMS ratios were .970/.960 for 145 and .905/.911 for 256; the Blender
ratios were .946/.942 and .956/.974, respectively (`sss166`). The comparison
uses eroded common skin masks, but differing geometry, materials and 32-sample
Blender renders limit its physical interpretation. It does not justify a
global increase in scattering strength.
Repeating the two yaw views at 128 samples changed the corresponding Blender
ratios from .942/.974 to .931/.916 (UE .960/.911), using common eroded skin
masks (`sss168`). This sample-count sensitivity is another reason to avoid
fitting scattering strength to the 32-sample diagnostic.
Two more 128-sample identities showed the same stronger red than blue
high-frequency attenuation in both renderers. For the front views, Unreal's
RGB on/off ratios were [.796, .942, .976] and [.913, .979, .992]; Blender's
were [.778, .935, .969] and [.903, .980, .993] (`sss175`). The geometry and
materials differ, so these are descriptive comparisons of the scattering
response rather than a physical equivalence test.
