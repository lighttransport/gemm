# Resume prompt: vhuman MetaHuman priors, UE-vs-Cycles matching (after reboot, 2026-11-04 evening)

## Eye-work continuation, 2026-11-06

- `iris_fill12` finished and scored all 48 new views. `iris_fill_repeat` finished and scored its 24 views; they match
  the pilot within .000036 RGB8 per-channel mean per view. Combining clean results, fill .5 improves eye LPIPS / DISTS /
  dE by .0026 / .0034 / .60 over 35 eye-region views; see the README for per-head counts and the fill 1 trade-off.
- The pilot tested additive iris emission (strengths .25/.5/1) and improved eye LPIPS and dE on four heads; see the
  README follow-up draft. It overlapped an old UE queue, so `iris_fill_repeat` reruns the four pilot heads without
  overlap. Do not sync while a worker render is active.
- `eye_wide24` completed, validated and scored: 24 heads, eye LPIPS / DISTS / dE vs normal -.0328 / -.0278 / -13.64,
  64 / 68 / 71 of 71 eye-region views better; the 12 new heads also improve. See the README for all counts.
- `iris_physical12` completed and scored 12 heads: specular 0 plus albedo gain 1.5/2.5 improved mean eye LPIPS, but
  eye dE improved in only 5/12 and 6/12 heads. Iris-only saturation improved for 12/18 approximate matches. Results use
  checksummed baseline snapshots in `tmp/vhuman-mh/iris_physical_baseline` and `iris_physical_combined.py`.
- `wide_full24` completed and scored all 192 views across 24 heads: LPIPS / DISTS / dE vs normal -.0105 / -.0096 /
  -.63, LPIPS better on 171/192 views and 22/24 heads. Ear dE worsened .42 on 44 measurable crops; see the README.
- All 28 B10 photo views are complete; the s1/s2 eye metrics do not support a photo default change. Paired iris-colour
  audit and tests were added; existing defaults remain unchanged. `/mnt/nvme01` had 12 GB free at last check.
- Photo candidates B11/B12 (B7 plus iris emission .5/1) completed 12 final-profile views as `iris_photo`; all passed asset
  validation. Broad eye-region and approximate iris-ring scores were mixed; the README has results and review sheets.
- Non-emissive photo candidates B13/B14 (B7 plus iris specular 0 and albedo gain 1.5/2.5) completed and scored 12 views
  as `iris_photo_physical`; all passed validation. Gain 1.5 improved 3/4 approximate photo iris rings but broad eye scores
  were flat or worse; gain 2.5 generally worsened photo colour. The README has exact results. The worker queue is empty.
- The SSH system config in this sandbox fails ownership checks; direct `ssh -F /dev/null` works when network access is
  permitted. Check host-level processes as well as worker PIDs: sandboxed `ps` alone does not show the old host queue.

Paste this into a new session. Repo `/mnt/nvme02/work/gemm/pixal3d`, branch `pixal3d`. Plan:
`~/.claude/plans/unreal-engine-eula-now-velvety-giraffe.md`. Memory: `vhuman-metahuman-priors` (also holds the Windows
worker address, which is kept out of the repo). Docs: `server/vhuman/metahuman/README.md` (commands + all results),
`server/vhuman/metahuman/skin_sss_survey.md` (SSS models, UE skin readout, mapping), `server/vhuman/vhuman-win10.md`
(Windows worker, section 10 = Cycles CPU). The previous resume prompt (texture seams round) is in git:
`git show 31c82aab:resume-vhuman.md`.

## Rules (CLAUDE.md and the user)
- NEVER `p4 submit`. Never use /tmp (repo `tmp/` or the scratchpad). Never `git push` without permission. Commit only
  when asked; report hashes. Trailer: the one in the session's attribution reminder (last used
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
- Licences Apache/MIT/BSD/GPL only. Never commit raw Epic assets or dataset/model files; derived parameters only.
  EULA revision/clause placeholders are for the user to fill.
- No IP addresses or hostnames in the repo (`export VHUMAN_WIN_HOST=<address from memory>`); `C:\Users\syoyo` is fine.
  Do not touch the Windows checkout `D:\work\gemm` (the worker uses `D:/work/gemm-vhuman`).
- One heavy GPU job at a time on Linux, low VRAM (another app shares the RX 9070 XT). Kill by explicit PID only.
- Do not change defaults or README results unless asked.

## Environment
```
cd /mnt/nvme02/work/gemm/pixal3d
export VHUMAN_WIN_HOST=<from memory>
E="env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib TMPDIR=$PWD/tmp PYTHONPATH=$PWD"
P="tmp/vhuman-rocm-venv/bin/python -W ignore -u"      # scoring (LPIPS/DISTS on CPU)
C=tmp/vhuman-cpu-venv/bin/python                       # tests, numpy-only
```
Data: `COMPARE_ROOT=/mnt/nvme01/data/vhuman-mh/compare_v1` (12 comparison heads = `render_ids[::10]`), dataset
`/mnt/nvme01/data/vhuman-mh/v1`, artifacts `/mnt/nvme02/models/vhuman-mh`, UE project `/mnt/nvme01/work/vhuman-mh/MHGen`.
Windows worker: Blender 5.2.3 CPU (i9-9900K), fast profile 512 px / 32 spp + OIDN, ~15 s/view (eye/ear close-ups ~20 s).

## Windows jobs are disconnectable
The worker job is a detached WMI process that only reads its staged copy (`D:/data/vhuman-mh/cycles/<tag>`), so the
Linux host can reboot while it runs. `win_cycles render <tag> ... --score <set>` records ids/profile/score set in
`tmp/win/<tag>.finish.json`. Afterwards (any session):
```
$E $P -m server.vhuman.metahuman.win_cycles status <tag>     # views done/total, progress tail, exit file
$E $P -m server.vhuman.metahuman.win_cycles finish <tag>     # wait for exit, pull (skips pulled views), score
$E $P -m server.vhuman.metahuman.win_cycles resume <tag>     # worker rebooted / job killed: restart, done views skipped
```
Never `sync` (render does it) while a worker render is running: it replaces `D:/work/gemm-vhuman/server`.

## State (2026-11-05, 12 h goal round `goal-vhuman-12h.md`)
Commits since the reboot (none pushed): `66b91228` UE confirm, `70d5c727` s1/s2 check of x1.25, `d1c64429` IOR 1.0 cornea,
`1abb4752` iris brightness. Uncommitted: overrides in `cycles_scene` (iris_/sclera_ gain/sat, iris_specular,
cornea_shadow, eye_offset_<side>_m, dual_lobe, ear_sss_scale), `compare_render` sets (EYE_SAT/SHADOW/IRISSPEC/PLACE/
SCLERA, RMH, DUAL, EAR_SSS; per-identity variants take `sid`), `win_cycles` score sets, new `win_matrix.py`
(render_matrix candidates on Windows), `eye_audit.py`, a test, the README "12 h goal round" draft, this file.

Windows queue: empty (last job `combo_fast`, chain scripts `tmp/vhuman-ab-cycles/scripts/chain_goal{1..7}.sh`).

## Results this round (README "12 h goal round")
- Eye placement is the eye gap: GNM eye joints ~1.9 mm high / ~.9 mm medial of the MH eyes; moving the eyeballs onto
  the MH eye centres (`ew_eye_place`): eye dE -6.6, eye LPIPS -.032, 33/35 eyes, 11/12 heads.
- Not the cause: iris brightness (x1.5 noise-level), iris saturation (worse), cornea shadow-off (worse), cornea IOR 1.0.
  The rendered iris is still grey with the correct texture hue: testing the iris's own specular (`iris_specular 0`).
- Per-identity roughness x1.25 x MH roughness/1.03 + coat 0: LPIPS -.0053 (57/72, 10/12) beats global x1.25 -.0044;
  the x1.25 gain per head correlates with MH skin.roughness (r -.71).

- Sclera x.85 wins (eye dE -.74, 12/12 heads); dual-lobe, ear SSS x2/x4, sclera desaturation: worse; Linux GPU ==
  Windows CPU (LPIPS .00006); s1/s2 B9 (global eye offset) ties B7.
- Combined (`combo`, 96 views): LPIPS -.0111 (87/96, 11/12 heads), eye dE -15.1 (35/35).

## Next steps
All goal items done and committed. Defaults (a)(b)(c) applied as `compare_render.DEFAULT_VARIANT = 'combo'`. New heads need
`write_eye_place_offsets` first. Open: iris hue loss (shading, not texture).

## Resume checklist
1. `git status`, `git log --oneline -6`; confirm the uncommitted list above.
2. `df -h /mnt/nvme01 /mnt/nvme02 /`.
3. `ssh` reachability: `$E $P -m server.vhuman.metahuman.win_cycles status ue_match_fast` (expects `views 216/216`, `exit 0`).
4. Step 1 above (finish if needed), then report.
