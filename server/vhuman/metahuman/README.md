# MetaHuman-derived priors for vhuman / GNM

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
| Eye mapping | `-m server.vhuman.metahuman.eye_map calibrate` (UE eye close-ups -> `eye_colour_calibration_v1.json`), `eye_map params <id>`; `compare_render --variant normal_eyes` renders with `head/fit_mh_eyes.json` |
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
