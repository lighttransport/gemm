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
  the IOR to 1.30 (less reflection) does not help either. Untested: an IOR ~1.0 (invisible) cornea; more likely causes
  of the grey iris are the iris texture brightness and the GNM-vs-MH eye placement.
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
