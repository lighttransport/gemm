# Resume prompt: vhuman texture seams, refined-ear bake round

Paste this into a new session. Repo `/mnt/nvme02/work/gemm/pixal3d`, branch `pixal3d`. Workspace `tmp/vhuman-ab-cycles/` (`$W`, git-ignored); `$W/RESUME.md` has the Cycles A/B study decisions, `server/vhuman/QUALITY_ITERATIONS.md` the write-up. Plans and findings of this round: `seamless-uv-texture-plan-2026-11.md` (sections 6-13; read 7-13 first), `ear-seam-delight-mv-texture-2026-11.md` (survey), `keenface-gap-plan-2026-10.md`. Memory notes: `vhuman-cycles-ab-study`, `vhuman-inverse-texture-plan`.

## Rules (CLAUDE.md and the user)
- NEVER `p4 submit`. Never use /tmp (use repo `tmp/` or the scratchpad). Never `git push` without explicit permission. Commit only when asked; report hashes.
- Mainline licences: Apache, MIT, BSD or GPL only. Research/NC material is evaluation-only and goes in provenance.
- Commit trailer: the one in the session's attribution reminder (last used: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`).
- One heavy GPU job at a time (Blender B render peaks at ~13 GB of the 16 GB card). Render: `VHUMAN_BLENDER_LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`. rocm venv (`tmp/vhuman-rocm-venv`) needs `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`. CPU venv `tmp/vhuman-cpu-venv` for numpy-only bakes and EXR reads (`OPENCV_IO_ENABLE_OPENEXR=1`). Run python with `PYTHONPATH=$PWD TMPDIR=$PWD/tmp`. Kill by explicit PID, never a pkill pattern. `render_matrix.py` skips existing outputs: delete a render dir to force a re-render.

## State (2026-11-01 evening)
HEAD `8aaa1855` (mouth_fill). Already committed: bake cleanup opts (`9060db13`: silhouette guard, feather, gain match, front blend, bg-reject), soft lip coat (`b7a360fb`), teeth shade / lip coat scale (`7e1c041e`). Candidates B (A/B study), B2 (clay ear detail), **B3 = `ears03`** (cleaned bake + ears02 ear) are registered in `scripts/write_candidates.py` and rendered.

### Uncommitted (this round)
- `server/vhuman/reconstruction/materials.py`: `bake_portrait(min_facing, magnification_cap)` + `magnification_weight()`; opt-in, defaults 0. Not applied to the front-only bake that `protect_front` restores.
- `server/vhuman/reconstruction/ear_mesh.py`: `rendered_surface(geometry, view)` = base skin faces minus `ear_removed_triangles` plus the surface-bound refined ear (`ear_dense_*`, positions via `evaluate(full_captured[view], ear_bind_*)`); same atlas UVs.
- `server/vhuman/reconstruction/inverse_texture.py`: experimental inverse texture solver (see Findings). Useful as a scorer: `python -m server.vhuman.reconstruction.inverse_texture <candidate> <out> --iters 0 --no-warp` prints per-view and ear-texel re-projection residuals and LPIPS of a candidate's atlas.
- Docs: `seamless-uv-texture-plan-2026-11.md`, `ear-seam-delight-mv-texture-2026-11.md`, `keenface-gap-plan-2026-10.md`, `pixal3d-quality-ab-cycles-prompt.md` (untracked), this file.
- In the git-ignored workspace: `scripts/rebake.py` (`--ear-surface --min-facing --mag-cap --res`), `scripts/make_ears03.py` (env `REBAKE_DIR`, `OUT_NAME`; records `rebake_options` in provenance), `scripts/inverse_preview.py` (unlit albedo previews; `ears` mode = zoomed ear sheets), `inverse_metrics.py`, `smudge_source.py`, `smudge_metric.py`. PyTorch3D 0.7.9 ROCm wheel: `tmp/wheels/` (+ `.source.txt`), installed in the rocm venv.

### Candidates built this round (all `validate_candidate`-clean, none registered, none rendered)
| Candidate | Meaning |
|---|---|
| `candidates/{s1,s2}/B/reconstruction/ears04` | ears03 with the bake observing the **refined ear**, no fix. s1 good; s2 gets a new orange smudge above the left ear |
| `candidates/s2/B/reconstruction/ears05` | s2, refined ear **+ `--min-facing .25 --mag-cap 6`** (from `rebake/s2/bg_ear_fix`). Smudge back to B3 level (R-B 58) |
| `rebake/s1/bg_ear`, `rebake/s2/bg_ear`, `rebake/s2/bg_ear_fix` | bakes behind the above. The 1024 `fast_*` test bakes and older intermediates were deleted in the 2026-11-01 cleanup; regenerate with `--res 1024` (~100 s) |

## Findings that matter
1. **The base bake saw the coarse ear.** `geometry.npz` `captured`/`triangles` have the coarse GNM ear; renders use the refined ear. Front-photo ear residual .0538 -> .0335 with the refined surface. Control: rebaking with the coarse ear reproduces `rebake/s1/bg` bit-exactly, so only the surface differs.
2. **Refined-ear bake effect:** changes only a ~60-texel ring around the ear mask (5-7k texels). s1: removes the pink smudge and two dark dots behind the ear. s2: new orange blob above the left ear (see 3).
3. **s2 smudge cause:** 2693 texels seen only by view_2 (-40 deg generated), 79 pixels magnified ~12 texels/pixel, facing .14, at the helix top where grey hair overlaps the ear (parsed as ear, support 1.0), then lifted by delight (illumination clip .5-2). Fix = `min_facing .25` + `magnification_cap 6` (cap is texels per source pixel at 2048; `rebake.py` rescales it for `--res`). Side effect: ~27.6k texels (2048) elsewhere change, all with confidence < 80/255 (mean 7), i.e. near-unobserved texels now filled by completion. Front-protected texels with front confidence >= .5: 312 of 835k change.
4. **Inverse texture solver (plan stages 0-4) is a negative result.** Shared albedo pyramid + per-view SH9 + footprint-aware mip sampling fits trained views 10-20% better but held-out views and seam ratio do not improve; per-view warp stays at 0.2-0.7 px; LPIPS crops make held-out LPIPS 33-38% worse; mirror ear priors make held-out ear residual worse (.073 -> .081-.083). Atlas mirrors exactly ((u,v) -> (1-u,v)) but the refined ears are fitted per side. Conclusion: remaining side-view/ear error is input quality and the synthetic ear texture, not lighting or alignment. Do not spend more time on solver variants.
5. **Only ~17% of ear-mask texels are seen by any of the three views**; the rest is `ear_texture.py` / ears02 synthesis. That is the main remaining lever for ear texture.
6. Front pixels cover ~1.49 texels at 2048 (head-on), so footprint thresholds must be relative to that.

## Next steps (in order)
1. **s1 full rebake with the fix** and composite `ears05`: `rebake.py candidates/s1/B/reconstruction/ears03 rebake/s1/bg_ear_fix --guard 3 --feather 6 --gain --inner-lip --front-blend --bg-reject --ear-surface --min-facing .25 --mag-cap 6` (CPU, ~14 min), then `REBAKE_DIR=bg_ear_fix OUT_NAME=ears05 tmp/vhuman-cpu-venv/bin/python scripts/make_ears03.py s1`. The 1024 fast check already showed the s1 gain is kept (seam 1.55 coarse / 1.59 refined / 1.50 refined+fix); confirm at 2048 with `inverse_preview.py ears` and `smudge_metric.py`-style checks.
2. **Register B4 = ears05** in `scripts/write_candidates.py` (like B3 at line ~60: `cands['B4'] = dict(cands['B3'], dir=.../ears05)`, plus `B4-ps` for PixelSmile) and render B3 vs B4: front, yaw_p40, yaw_p90, yaw_m90, ear_left/right, mouth, both subjects (`render_matrix.py --plan b3` pattern, studio, 1024²/128). Sheets like `bench/b3_final_s*.png`. Score ears/face with `stage5_report.score`. Confirm ears stay ahead of A and no new smudges.
3. If B4 holds: commit (materials.py min_facing/magnification cap; ear_mesh.rendered_surface; inverse_texture.py as experimental/scoring; docs; this file), add a short section to `server/vhuman/QUALITY_ITERATIONS.md` and `$W/RESUME.md` (refined-ear finding, min_facing/magnification cap, negative solver result), update the `vhuman-cycles-ab-study` memory. Consider making `--ear-surface` the default in the real pipeline's bake (`pipeline.py` line ~254 passes the coarse mesh to `bake_portrait`); that changes every candidate, so decide with the user.
4. s2 leftovers (older than this round, not caused by it): thin orange line behind the right ear (+90 side) and the beige ring under that ear. Trace with `smudge_source.py`-style forward projection first; likely an unobserved-region or ears02 artefact.
5. Ear texture itself: improve the `ear_texture.py` / ears02 synthesis (83% of ear-mask texels), or fit the refined ear outline (plan stage 3: PyTorch3D soft silhouette, ear vertex offsets) so the front photo explains more of the ear. Both are design work; start from `seamless-uv-texture-plan-2026-11.md` section 9.
6. Older open items: hair/scalp (crown and back bald; needs generated back/±135 views or the VDA-style orbit spike in `ear-seam-delight-mv-texture-2026-11.md` sections 4-5), s1's mouth barely opens (expression-fit limit), s1 surprise tongue/palate glossy, s2 teeth olive in shadowed fear frame (mouth_fill is only a stand-in for translucency), render time ~2x and 13 GB VRAM for B, delight unification (OpenDelight licence of weights still unchecked).

## Cleanup done (2026-11-01)
Deleted: per-render scene files and `cache/` folders in `/mnt/nvme01/vhuman-ab-renders` (`head.blend`, `skin_detail.npz`, `scene_assets.npz`, `skin_templates.npz`, `beauty_noisy.exr`, Blender temp cache; 30 GB freed, nvme01 now ~43 GB free); in `$W`: `inverse/`, `smudge_s2/`, `rendertest/`, `candidates_v1/`, `subjects_v1/`, `review.bak/`, `smoke/`, `ears/s1_test`, `calibration/chart_*`, the `fast_*` and intermediate rebake dirs, and 12 benchmark subfolders under `bench/` (4 GB freed on nvme02). Render dirs keep `beauty.png`, `beauty.exr`, `render_result.json`, `display.json`; `redisplay.py` and the reports still work. Re-rendering a camera regenerates the scene files (delete its dir or set `RM_FORCE=1`). Candidates ears03/04/05 still validate. Not deleted (ask first): `tmp/vhuman-independent/heads`, `vhuman-ear-synth`, `vhuman-rig`, the older quality-run dirs, caches and source trees; "probably safe" list of ~4 GB of unreferenced scratch dirs was listed in the chat, not removed.

## Key technical facts
- `bake_portrait` weight per view = valid x depth agreement (2 mm, <=512 px raster) x facing x mask support (x min-facing ramp x magnification weight if enabled). Images are opaque, so the background is never excluded by alpha: hence the silhouette guard and bg-reject.
- Bake speed: ~14 min at 2048, **~100 s at `--res 1024`** (fast preset for iteration; `make_ears03.py` composite needs 2048, so judge fast bakes directly, e.g. `smudge_metric.py`, `inverse_preview.py ears`).
- Seam stat = gradient on the front-protection ring / interior; ragged = coverage perimeter / sqrt(area), smooth ~4-6.
- GNM UV is one island with 76 seam edges; atlas rows = v*res (no flip); the head mesh is mirror symmetric to .25 mm.
- Index spaces: base skin mesh (11460 vertices) = skin component minus mouth sock, `full_neutral[m]` exact; `ear_removed_triangles` index `full_triangles`; dense ear triangles use full indices (< 17821) or 17821+k.
- `reference.Camera` -> PyTorch3D `PerspectiveCameras`: `S=diag(-1,1,-1)`, `R=(S R_cam)^T`, `T=-(S R_cam) o`, focal (f,fy), principal (cx,cy), `in_ndc=False`, `image_size=(H,W)`.
- Studio preset is calibrated (`STUDIO_ENERGY_SCALE`=2^-1.3632, `CALIBRATED_HDRI`); uncalibrated renders clip skin and fake ear-colour wins. `lip_coat_scale` .6 / `lip_coat_roughness` .30 are the defaults; `mouth_fill` .02 W (quality candidates only); `teeth_glow` defaults to 0.
- Held-out evidence must exclude the held-out view; `inverse_texture` residuals use refit lighting, so they compare textures, not lightings.

## Resume checklist
1. `git status`, `git log -3`; confirm the uncommitted list above.
2. `ls tmp/vhuman-ab-cycles/candidates/s{1,2}/B/reconstruction` (ears04 both, ears05 s2 only).
3. Start with step 1 (s1 rebake in the background) while preparing step 2's `write_candidates.py` edit; no GPU use until the rebake finishes if a render is queued.
