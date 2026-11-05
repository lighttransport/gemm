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
