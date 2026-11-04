# Resume prompt: vhuman MetaHuman priors, UE-vs-Cycles matching (after reboot, 2026-11-04 evening)

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

## State at reboot
Commits today (none pushed): `50c06fcb` win_cycles fast path, `77e2eab0` ear fade / eye_map v2 / region scores,
`3bd882c0` eye_map v2 default + README + eye sweep variants, `8bd5014b` SSS survey + UE skin dump + ue_* variants,
`e5908773` README eye sweep results.
Uncommitted: `server/vhuman/metahuman/win_cycles.py` (status/resume/finish, `--score`), this file.

Jobs: nothing runs on Windows. `ue_match_fast` (216 views: ue_sss25/50/100, ue_spec, ue_match, ue_match_spec x 12 heads
x front/yaw_p40/yaw_m40) finished rendering 18:57; its first pull failed (nvme01 full), the re-run
`win_cycles finish ue_match_fast` (log `tmp/vhuman-ab-cycles/logs/ue_match_finish.log`) was started ~19:05. **First thing
after the reboot: check `compare_v1/ue_match_vhstudio__fast.json`; if missing or older than the log, re-run
`finish ue_match_fast` (finish record already written).**

Disk: `/mnt/nvme01` hit 100% at ~18:57 (it had ~43 GB free in the morning; no large new files found, the consumer is
unknown, not ours). Freed 18 GB by deleting regenerable per-render files (`skin_detail.npz`, `beauty_noisy.exr`,
`skin_templates.npz`, `scene_assets.npz`) in finished `compare_v1/*/cycles*` dirs. Check `df -h /mnt/nvme01` before any
pull. `tmp/win/*_in.tar` / `*_out.tar` on nvme02 (~10 GB) are pulled transfer copies and can go once results are in.

## Results so far (details in the metahuman README)
- Fast confirm (72 views vs `normal`): roughness x1.25 LPIPS -.0043 (10/12 heads), coat 0 -.0015 (12/12). Defaults unchanged.
- Ear normal fade: worse on every ear camera (ear0 ear LPIPS +.016, ear dE +1.9); default stays 1.
- Eye mapping v2 (sclera tint from the measured MH sclera) is now the default (`eye_map.VERSION = 2`): eye dE -7.5,
  12/12 heads. Remaining eye dE ~19-25.
- Eye optics sweep: nothing beyond noise; cornea Specular IOR Level is a no-op on glass (test did not run).
- UE skin readout (`ue_scripts/dump_skin_material.py` -> `v1/presets/skin_material.json`): SSP_skin_unified mfp
  1.05/.41/.26 mm x material Opacity (Scatter Distance Base .5 x Global .5), dual lobes .8r/1.1r mix .5, no coat,
  roughness ~.55-.65, F0 ~.06. Cycles: mfp 1.84/.69/.34 mm, SSS weight .6, roughness ~1.25x lower, coat .10, F0 .028.
  The roughness x1.25 and coat-0 wins are what UE predicts.

## Next steps
1. Read `ue_match_vhstudio__fast.json`; report ue_sss* (SSS radius/amount), ue_spec (F0), ue_match(_spec) vs `normal`
   (LPIPS/DISTS/dE, views and heads improved, per camera); add to the README and `skin_sss_survey.md` section 5 when
   the user asks.
2. If a UE-matched setting wins clearly: confirm on all 6 confirm cameras (`win_cycles render <tag> --variants ...
   --cameras front,yaw_p40,yaw_m40,yaw_p90,yaw_m90,mouth --score confirm`, add the variants to the scored set) and on the
   s1/s2 photo metrics before proposing a default change (ask the user).
3. Eye gap: test an IOR ~1.0 cornea (true "no veil" check), then the iris texture brightness and the GNM-vs-MH eye placement.
4. Still open from the plan: dual-lobe specular emulation in Cycles (two Principled lobes mixed .5, or roughness scaled
   to the lobe average), UE thin-part transmission vs Cycles random walk on ears.

## Resume checklist
1. `git status`, `git log --oneline -6`; confirm the uncommitted list above.
2. `df -h /mnt/nvme01 /mnt/nvme02 /`.
3. `ssh` reachability: `$E $P -m server.vhuman.metahuman.win_cycles status ue_match_fast` (expects `views 216/216`, `exit 0`).
4. Step 1 above (finish if needed), then report.
