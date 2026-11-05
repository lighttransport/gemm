# Resume prompt: vhuman follow-up round on the Windows worker (2026-11-06, `goal-vhuman-next.md`)

> Update: the queue and pending-result sections below are historical. See the new continuation section at the top of
> `resume-vhuman.md` and the current README draft for the live job, completed photo/iris results, and next steps.

Paste this into a new coding-agent session. Repo `/mnt/nvme02/work/gemm/pixal3d`, branch `pixal3d`. Read first:
`resume-vhuman.md` (environment, rules, Windows job commands), `goal-vhuman-next.md` (the goal: items 1-7),
`server/vhuman/metahuman/README.md` section "Follow-up round (2026-11-06 ...)" (DRAFT results), `server/vhuman/vhuman-win10.md`.

## Rules (short; full list in resume-vhuman.md)
- NEVER `p4 submit`; no `git push`; commit only when the user asks (trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
- No /tmp (repo `tmp/` or a scratchpad). No IPs/hostnames in the repo: `export VHUMAN_WIN_HOST=<address from memory / ask the user>`.
- All Blender/UE work runs on the Windows worker (Blender 5.2.3 CPU via `win_cycles`/`win_matrix`, UE 5.8.4 DX12 via
  `winjob`). Linux only stages, scores (CPU LPIPS/DISTS) and edits code. One Windows job at a time; never `sync`
  while a Windows render runs. Kill by explicit PID only.
- Do not change defaults; propose with evidence and ask. `/mnt/nvme01` is nearly full (5.7 GB free on 2026-11-06):
  `df -h /mnt/nvme01` before pulls.

## Environment
```
cd /mnt/nvme02/work/gemm/pixal3d
export VHUMAN_WIN_HOST=<from memory>
E="env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib TMPDIR=$PWD/tmp PYTHONPATH=$PWD"
P="tmp/vhuman-rocm-venv/bin/python -W ignore -u"; C=tmp/vhuman-cpu-venv/bin/python
$E $P -m server.vhuman.metahuman.win_cycles status|finish|resume <tag>
```

## State
- Commits: `ce4b58dd` (12 h goal round), `9cde45d1` (default MH comparison render `compare_render.DEFAULT_VARIANT =
  'combo'`: MH eye placement + per-identity roughness + coat 0 + sclera x.85). Nothing pushed.
- Uncommitted (this round): `cycles_scene.py` overrides `iris_emit`, `cornea_camera` (diagnostics), `ear_roughness_scale`;
  `compare_render.py` `c_<param>_<value>` variants (= combo + one setting, `COMBO_EXTRA_PARAMS`), `eye_place_offsets(sid)`
  (auto-writes `head/eye_place_offset.json`), sets `IRIS_DIAG_SET`, `SKIN_RECHECK_SET`, `EAR_ROUGH_SET`, `WIDE_SET`;
  `win_cycles.py` score sets `iris_diag`, `skin_recheck`, `ear_rough`, `wide`; new `eye_offset_fit.py`; README draft;
  this file; `goal-vhuman-next.md` (untracked).
- Data changed outside git: `v1/render_batches/vhstudio.json` extended to `render_ids[::5]` (24 heads; the 12 old entries
  byte-identical; backup `tmp/vhuman-mh/vhstudio_every10_backup.json`); `MODELS_ROOT/eye_offset_v1.{npz,json}`;
  s1/s2 `candidates.json` gained `B10`; per-side bias `tmp/vhuman-mh/eye_offset_bias.json`.

## Results so far (README draft has the numbers)
- 2 iris size: UE iris radius 1.03x GNM limbus -> no size gap; scale test skipped.
- 3 auto eye offsets: reproduce the existing files exactly.
- 4 skin re-check on combo: nothing wins on both LPIPS and dE; combo stays.
- 5 ears: ears-mask roughness x1.1: ear dE -.059 vs combo +.077 (26 vs 23/46), ear LPIPS equal -> small win (proposal).
- 6 landmark eye offset regression (499 heads): CV RMSE .52 mm vs 1.18 mm for the median offset.

## Windows queue at hand-off (chain scripts in `tmp/vhuman-ab-cycles/scripts/`, logs in `tmp/vhuman-ab-cycles/logs/`)
1. `chain_next2.sh` -> `iris_diag_fast` (item 1; 72 views; started 04:00 JST) -> `iris_diag_fast_finish.log`.
2. `chain_next3.sh` -> `win_matrix render b10_s12` (item 6; s1/s2 B10, 28 views) -> `chain_next3.log`.
3. `chain_next4.sh` -> `winjob render <12 new ids> --env VHStudio --tag ue_wide_vhstudio` (pulls to `win_v1`, then copies
   `*__vhstudio.exr` + `renders_done_vhstudio.json` into `v1/samples/<id>/renders` if absent) -> `win_cycles render
   wide_fast --ids render_ids[::5] --variants normal,combo --score wide` (item 7) -> `wide_fast_finish.log`.
Chains wait on each other by PID (`kill -0`). If the Linux host rebooted, they are gone: check each tag with
`win_cycles status <tag>` and continue by hand (`finish` / `resume`, or re-run the remaining chain lines).

## Remaining tasks
1. Item 1: read `iris_diag` scores; run `eye_audit` colour on `c_cornea_camera_0__fast` / `c_iris_emit_1__fast` vs
   `combo__fast` (texture vs render iris saturation/hue). Interpretation: if `iris_emit` keeps the hue, the loss is the
   iris shading/lighting; if `cornea_camera 0` restores it, it is the cornea/tear film. If a fix appears, test it on 12
   heads x eye_left/right/front. Fill README item 1.
2. Item 6: B10 vs B7 with `tmp/vhuman-ab-cycles/scripts/prior_report.py B7 B10` (as done for B9); fill README item 6.
3. Item 7: fill the wide confirm row (24 heads, vs normal; compare with the 12-head combo result LPIPS -.0111).
4. Default-change proposal list in the README draft (candidates: ears-mask roughness x1.1 in `combo`; landmark eye offsets
   for photo subjects only if B10 beats B7), update `resume-vhuman.md`, then ask the user before committing or changing
   any default.
