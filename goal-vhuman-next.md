# Goal: iris hue, eye/skin follow-ups on the new MH default, landmark eye placement (~15 h, from 2026-11-06)

Repo `/mnt/nvme02/work/gemm/pixal3d`, branch `pixal3d`. Context: `resume-vhuman.md`, `server/vhuman/metahuman/README.md`
("12 h goal round"), `skin_sss_survey.md`. Last commits `ce4b58dd`, `9cde45d1` (`compare_render.DEFAULT_VARIANT = 'combo'`).

## Rules
- Follow `resume-vhuman.md` rules (no p4 submit, no /tmp, no push, no hostnames/IPs in the repo, kill by PID, never
  sync during a Windows render). `df -h /mnt/nvme01` before pulls.
- Do not change defaults; collect evidence and propose. Ask before committing.
- **All Blender/UE work runs on the Windows worker** (`win_cycles` fast profile 512 px / 32 spp; `win_matrix` for s1/s2).
  Linux only stages, scores (CPU LPIPS/DISTS) and edits code; light GPU only. One Windows job at a time; chain jobs by
  PID (`kill -0`), not `pgrep -f`.
- Experiment pattern: `cycles_scene` override (absent = unchanged graph), variant in `compare_render`, `*_SET` +
  `SCORE_SETS`, `render --score <set> --no-wait`, `finish <tag>` in the background. Baseline = `combo` (new default)
  unless stated. 12 heads = `render_ids[::10]`. Report each result as it lands (LPIPS/DISTS/dE, region rows,
  views/heads better).
- Run CPU items (3, 6) on Linux while Windows renders.

## Items
1. Iris hue loss (2 h): texture hue is right, the render greys it. On 4 heads x eye close-ups: cornea camera-invisible,
   iris without subsurface/sheen, and the `eye_audit` iris colour vs UE at fixed exposure. Find the shading cause; if
   a fix exists, run it on 12 heads x eye_left/right/front.
2. Iris size / lid cover (1.5 h): measure iris radius vs UE (`eye_audit`), then iris scale x.9/.95 on 12 heads.
3. Auto eye offsets (1 h, CPU): write `eye_place_offset.json` during pseudo-candidate build so `combo` works on any head;
   test it reproduces the 12 existing files.
4. Skin re-check on the new baseline (2 h): specular x.75/1.25, SSS scale .75, normal strength x1.5 vs `combo`
   (CONFIRM_CAMERAS).
5. Ear dE (1.5 h): combo ears dE +.26. Try per-identity roughness excluded from / reduced on the `ears` mask.
6. Landmark eye placement for photos (3 h, CPU first): regress the MH-minus-GNM eye offset on lid/landmark geometry
   over the 500 MH heads (held-out error); then s1/s2 candidate B10 = B7 + predicted offset via `win_matrix` (28
   views) vs B7.
7. Wider confirm (4 h Windows, last): `combo` vs `normal` on `render_ids[::2]` (48 heads), 6 cameras + eye close-ups.

Out of scope: ear/eye prior fitting, Epic cloud, s1/s2 default changes.

## Done when
Every item has a measured result (or a stated skip reason) in a README draft section, a proposed default-change list
with evidence, and an updated `resume-vhuman.md`; ask before committing and before changing any default.
