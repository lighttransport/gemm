# Virtual-human facial rig — status and handoff

Branch `pixal3d`, worktree `/mnt/nvme02/work/gemm/pixal3d`. Rig stage committed as
`19011726 Add facial rig for generated virtual-human heads` (not pushed; no push
is authorized — see AGENTS.md). This file is an untracked handoff note, like
`resume-vhuman.md` (the eye/head stage, whose independence rules still apply).

## Task (user request)

Add facial rigging to the Qwen-Image 2.1 + Pixal3D virtual-character tool.
Study MetaHuman/DNA (`/mnt/nvme02/work/ue/`) and its template rig structure for
concepts; use LightRig (`~/work/LightRig`) and LightUSD
(`~/work/tinyusdz/origin-dev`, `examples/vchar`) as our own replacement for the
auto-rigger and the MetaHuman viewer, with a USD-based pipeline; a script to
clone those projects (no submodules); new web viewer/tool as needed. Order:
head/face mesh + blendshapes + template skeleton/rig fitting (eyes, tongue,
teeth, …) using PyTorch / ML training on this GEMM repo, **then the deformer**.

## Done (stage 1: rig)

- `server/vhuman/rig/` (see its README.md for the table of modules):
  procedural template topology (~13k verts, loops around eyes/mouth/neck, lid
  strips, mouth bag, 2-chart UV); feature detection (lid margins, lip seam,
  profile landmarks); PyTorch registration to the Pixal3D head; skin-map
  transfer; skeleton (root, neck, head, jaw, teeth_upper/lower, tongue_01..04,
  eye_R/L); teeth/gums/tongue; LBS weights; 51 lr.face.v1 shapes +
  4 correctives + tongue/head controls; linear rig (`rigdef.py`, rig.json) with
  the same evaluator in `web/vhuman_rig.html`; exports rig.glb (skin + sparse
  morphs), rig.usda (+ textures, zip; vchar customData; ROM SkelAnimation),
  LightRig track -> USD (`cli rig-track`).
- Server: job kind `rig` (subprocess in `tmp/vhuman-rig-venv`), `/rig` page,
  `/v1/heads/<id>/rig/<file>`, head listing includes rig info. CLI: `rig`,
  `rig-track`.
- `rig/external.sh [--local] [--build]`: clones LightRig (main) + LightUSD
  (dev) into `third_party/`; builds `lightrig` and `vchar` (HIP off by default).
  A local build lives in `tmp/vhuman-rig/external/`.
- Tests: `python3 -m server.vhuman.test_all` (see the latest count in git log) (includes a mock
  end-to-end rig build via `rig/selftest.py` in the rig venv).
- Reference rigs built: `tmp/vhuman-independent/heads/{291dfa911553,650ac67354cd}/rig/`
  (fit mean 0.45 / 0.77 mm). Verified in Chrome and headless vchar.

## Environment

- Rig interpreter: `tmp/vhuman-rig-venv` (python 3.12, torch 2.14 cu130, scipy);
  create with `uv venv` + `server/vhuman/requirements-rig.txt`.
- System python3 runs the server (numpy + Pillow only).
- Dev scratch: `tmp/vhuman-rig/dev/` (pose.py, tmpl_zoom.py debug scripts).
- `python -m server.vhuman.rig.build <head> --out DIR --res 1024 --reuse-fit
  --keep-asset --no-preview` for fast iteration.

## Provenance rules (keep)

MetaHuman DNA Calibration (proprietary) and Unreal sources/plugins (EULA):
concepts only — never copy code, DNA files, joint/control lists, masks or numbers.
OpenRigLogic is MIT (concepts used). LightRig/LightUSD are Apache-2.0,
external clones only. Generated assets stay out of Git.

## Remaining items

1. **Deformer stage — first pass done** (see rig/README.md "Deformer"): PyTorch
   ground truth (ARAP + eyeball/teeth/lip contacts, rest-relative), PCA(48) +
   MLP2 → `deformer.lrm`; runtimes in numpy, JS (ML morph targets), USD, and
   native C `ryzen/vhuman_deformer.c` (0.55 ms/frame, parity 1e-7 m).
   Since then: tongue (sphere sets on the tongue chain) and 4-ring lip-pair
   contacts, 3072 samples / 64 components / 256 hidden; CUDA deformer
   `cuda/vhuman/` (2.6 us/frame batched, equals CPU); viewer Inspect card
   (joint labels, weight heat map, deformation / ML influence / contact heat
   maps from `viz.json`).
   Then: rig-level lip fixes (jaw-scaled mouthClose/tongueOut, per-shape lip
   non-crossing: linear crossings 6060 -> 64), tongue learning (tongue-heavy
   sampling, region-split PCA, MLP on rig inputs incl. correctives, weighted
   loss: tongue 1306 -> 601 held out), posed-space lip penalty, Vulkan
   backend `vulkan/vhuman/` (vkew, BAR inputs; added
   `computeToTransferBarrier` + two constants to vulkan/deps).
   Then: exact post-skinning contact projection (`rig/contacts.py`) in
   numpy, C, CUDA, Vulkan and the viewer (contactOffset attribute) -> zero
   modelled contacts; batched host MLP (sgemm); fixed a glTF bug (ML targets
   lacked NORMAL while others had it -> broken shading in three.js).
   Next: normals in native paths, GL interop / on-GPU consumers, LightRig
   `lr_face_evaluate` integration, real-time tracks, finer tooth/tongue
   contact proxies, USD-side contacts (bake into animation).
2. Shape quality — data-driven expressions, wrinkle maps and LODs done
   (rig/README.md "Subject expressions, wrinkle maps, LODs"): job kind
   `expressions` (Qwen-Image 2.1 edits, preset low8; fast12 runs out of GPU
   memory) -> `rig/expressions/`; the rig build fits 12 expressions from DIS flow
   (man 291dfa911553: 2-17 px -> 0.2-0.8 px), bakes wm_{brow_up,brow_down,
   smile,mouth}.png, exports LOD1 (3.7k) / LOD2 (1.4k) glb/usda/deformer/viz.
   Viewer: "wrinkles" toggle, LOD select. Both heads rebuilt with expressions.
   Fit safeguards: fold repair (smooth falloff around flipped/crushed skin
   triangles) and joint-driven intensity matching; jaw open is skipped on both
   (images show ~0.25 jawOpen; full-range extrapolation tore the mouth corners).
   Known: vertical streaks beside the mouth corners come from the transferred
   normal map (Pixal3D surface), not from the expression work. Next: wrinkle nostril-rim artefacts,
   per-side expression edits (only symmetric prompts now), lip roll detail,
   eyelashes, body LODs.
3. Template: ears/nostrils detail, hair region, higher-res face density option.
4. vchar: joint-driven controls only via animation; LightRig `animate-face`
   targets its own overlay (use our `rig-track` instead).
5. Browser test for `/rig` in `test_browser.py` (only manual check so far).

## Resume prompt

> Continue the virtual-human facial rig in /mnt/nvme02/work/gemm/pixal3d
> (branch pixal3d). Read resume-vhuman-rig.md, server/vhuman/rig/README.md and
> AGENTS.md first. Use tmp/vhuman-rig-venv for torch/scipy work, keep
> MetaHuman/UE material concept-only, never push. Next: finish/extend the
> deformer stage listed under "Remaining items", validate with
> `python3 -m server.vhuman.test_all` and a rig rebuild of head 291dfa911553,
> check /rig in the browser, then commit.
