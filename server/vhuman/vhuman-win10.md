# vhuman on a Windows worker: UE + Blender jobs over ssh

Goal: run MetaHuman/UE jobs (generation, reference renders, chart probes,
eye close-ups) and CPU Blender jobs on the Windows machine, driven over ssh from the Linux host
(this repository). This takes UE work off the shared Linux GPU (RX 9070 XT,
used by another app). The Linux host stays the orchestrator and the only place where results live.
The worker's address is not stored in the repository: set `VHUMAN_WIN_HOST` (ssh host or alias) on the Linux side.

Rules that still apply: never `p4 submit`; never `git push` without explicit permission; never commit raw Epic
assets or anything from the dataset/models roots; no Epic cloud services; kill processes by PID only.

## 1. Facts probed on 2026-11-04 (ssh, read-only)

| Item | Value |
|---|---|
| OS | **Windows 11 Pro**, user `syoyo` logged on at the console (session 1, Active) |
| ssh | OpenSSH server, Automatic/Running; key login from Linux works (`ssh $VHUMAN_WIN_HOST`); **default shell is Windows PowerShell 5.1** (no `&&`/`&`; use `;`) |
| CPU / RAM | i9-9900K, 8 cores / 16 threads; 31.9 GB |
| GPU | **Radeon RX 570, 16 GB** (Polaris, GCN4), AMD driver 31.0.21925.1001, Vulkan 1.3.260. The RTX 3070 registry entries are stale: `nvidia-smi` fails, so there is no NVIDIA GPU present |
| Disks free | C: 42 GB, D: 433 GB, E: 83 GB, G: 40 GB |
| UE | `C:\Program Files\Epic Games\UE_5.8`, **5.8.4 CL 58902052, the same build as Linux**; MetaHuman plugins present, including `MetaHumanCharacter\Content\Optional` (Creator Core Data); `UnrealEditor-Cmd.exe` and the bundled Python present |
| Blender | `D:\local\blender-5.2.3-windows-x64\blender.exe` = 5.2.3 LTS (Linux runs 5.2.2) |
| Repo | `D:\work\gemm` is on branch `pixal3d` at `50a55815` (739 commits behind Linux HEAD, an ancestor of it) with **19 uncommitted files: do not touch this checkout**. Remote `git@github.com:lighttransport/gemm`, git + git-lfs + `tar.exe` (bsdtar) available |
| Python | the existing `.venv-pixal3d-cuda` (uv, CPython 3.12.12) has numpy and PIL but **no scipy**: make a dedicated venv instead |

Consequences:
- **Blender Cycles runs on the CPU only.** Cycles HIP needs RDNA or newer; Polaris (RX 570) is not supported, and Cycles
  has no Vulkan backend. Use `offline_render(device='cpu')`. Expect minutes per 1024² `final` view on 16 threads
  (time it in W4 before planning any large job). The Linux GPU remains the place for big Cycles batches.
- **UE runs on DX12 at SM5** (measured, see §8). Vulkan SM6 is rejected (no mesh shaders on Polaris), and Vulkan SM5
  starts but crashes inside VulkanRHI. DX12 output matches Linux Vulkan closely enough for references (§8). 16 GB VRAM is more than the Linux budget (≤ 7 GB), but Polaris is slow, so
  shader compiles and renders will take longer than on the 9070 XT.
- The machine is dedicated, so the Linux "one heavy GPU job at a time" rule becomes a local lock file on Windows (one UE
  or Blender job at a time on that box). Windows jobs may run **concurrently** with Linux GPU jobs.

## 2. Layout on Windows

| What | Path |
|---|---|
| Code (git worktree of `D:\work\gemm`, own branch) | `D:\work\gemm-vhuman` (branch `vhuman-win`) |
| venv (uv, py 3.12) | `D:\work\vhuman-win\venv` |
| Job inbox / outbox, locks, logs | `D:\work\vhuman-win\{jobs,logs}`; lock file `D:\work\vhuman-win\gpu.lock` |
| UE project (copy of Linux MHGen: uproject + Config + Content, 108 KB) | `D:\work\vhuman-mh\MHGen\MHGen.uproject` (DDC builds locally, once) |
| Dataset root (job-scoped copies only) | `D:\data\vhuman-mh` (`VHUMAN_MH_DATA`) |
| Models root (only the files a job needs) | `D:\data\vhuman-mh-models` (`VHUMAN_MH_MODELS`) |

The authoritative dataset and models stay on Linux (`/mnt/nvme01/data/vhuman-mh`, `/mnt/nvme02/models/vhuman-mh`).
Windows holds disposable copies. Nothing under the dataset or models roots is ever put in git.

## 3. Code changes needed first (W1; on Linux, then sync)

The metahuman package is Linux-only today. These changes should be small and opt-in, and must leave Linux behaviour
byte-identical:
1. `server/vhuman/metahuman/__init__.py`: env overrides `VHUMAN_UE_ROOT`, `VHUMAN_MH_PROJECT`, `VHUMAN_MH_DATA`,
   `VHUMAN_MH_MODELS` (the defaults stay the Linux paths). `UE_EDITOR_CMD` should be `Engine/Binaries/Win64/UnrealEditor-Cmd.exe`
   when `os.name == 'nt'`.
2. `server/vhuman/metahuman/launch.py`:
   - Set `DISPLAY` only on POSIX.
   - Make the RHI flag configurable (`VHUMAN_UE_RHI=vulkan|dx12`; default vulkan on Linux, dx12 on Windows).
   - On Windows build `-ExecutePythonScript="<script> <args>"` with `subprocess.list2cmdline`-safe quoting instead of
     `shlex.quote`, and use forward-slash paths in script arguments.
   - Replace the core-dump guard (`_no_core`, `preexec_fn`) with nothing on Windows.
3. `server/vhuman/gpu.py`: it imports `fcntl` at module level, so it fails on Windows. Guard the import and add an
   `msvcrt.locking` file lock, with `backend()` = `cpu`/`windows` and no ROCm memory probe. `launch.run` already passes
   `check_memory=False`.
4. `server/vhuman/reconstruction/offline_render.py`: already supports `device='cpu'` (no lock, no sysfs probe).
   `VHUMAN_BLENDER` points at the Windows exe. `VHUMAN_BLENDER_LD_LIBRARY_PATH` is not set there.
5. Job JSONs (render batches, chart-probe plans) carry absolute Linux paths. Add a host tool
   `server/vhuman/metahuman/winjob.py` (Linux side) that:
   - rewrites the roots `/mnt/nvme01/data/vhuman-mh` → `D:/data/vhuman-mh` and `/mnt/nvme02/models/vhuman-mh` →
     `D:/data/vhuman-mh-models` in a copy of the job file;
   - packs the inputs, copies them over, runs the job, copies results back, rewrites paths back, and verifies hashes.
6. Tests: one unit test for the path rewrite and the Windows command line (pure string tests, run on Linux).

Keep all of this off by default. Commit only when the user asks.

## 4. One-time setup (W2)

All commands run from Linux. `W="ssh $VHUMAN_WIN_HOST"`; remote commands are PowerShell.

**Code sync** (no push to GitHub; a git bundle over scp):
```bash
# Linux: commit first (only committed code travels). The Windows base 50a55815 is an ancestor of HEAD.
git bundle create tmp/win/vhuman.bundle 50a55815..HEAD
scp tmp/win/vhuman.bundle $VHUMAN_WIN_HOST:D:/work/vhuman-win/vhuman.bundle
$W 'git -C D:\work\gemm fetch D:\work\vhuman-win\vhuman.bundle HEAD:refs/heads/vhuman-win'
$W 'git -C D:\work\gemm worktree add D:\work\gemm-vhuman vhuman-win'        # first time only
# later updates: bundle <last-synced>..HEAD, fetch with a forced ref update (+HEAD:refs/heads/vhuman-win), then
$W 'git -C D:\work\gemm-vhuman reset --hard vhuman-win'                       # worktree has no local edits by design
```
This leaves `D:\work\gemm` (branch `pixal3d`, 19 dirty files) untouched: `fetch` into a new ref and `worktree add`
change neither its index nor its working tree. To try uncommitted Linux work, run `git diff > tmp/win/wip.patch`, scp it
over, and `git -C D:\work\gemm-vhuman apply`. Reset the worktree afterwards.

**venv:**
```bash
$W 'uv venv D:\work\vhuman-win\venv --python 3.12'
$W 'uv pip install --python D:\work\vhuman-win\venv\Scripts\python.exe numpy scipy pillow openexr'
```
Add torch (CPU) or lpips later only if a job needs it; scoring stays on Linux.

**UE project:**
```bash
tar -C /mnt/nvme01/work/vhuman-mh -cf tmp/win/mhgen.tar MHGen/MHGen.uproject MHGen/Config MHGen/Content
scp tmp/win/mhgen.tar $VHUMAN_WIN_HOST:D:/work/vhuman-mh/mhgen.tar
$W 'tar -xf D:\work\vhuman-mh\mhgen.tar -C D:\work\vhuman-mh'
```
- Check `Config/DefaultEngine.ini` for Linux-only settings. The Python `AdditionalPaths` must point at
  `D:/work/gemm-vhuman/server/vhuman/metahuman/ue_scripts` and the plugin's `Content/Python`. Keep Lumen, VSM, ray
  tracing and Nanite off as on Linux.
- Transfer binary data as files (scp, then `tar -xf`). Do not pipe binary stdin through ssh: the remote side is
  PowerShell 5.1, which re-encodes pipeline data.

**Persistent env for jobs** (in a wrapper `D:\work\vhuman-win\env.ps1`, dot-sourced by every job):
```powershell
$env:PYTHONPATH = 'D:\work\gemm-vhuman'
$env:VHUMAN_UE_ROOT = 'C:\Program Files\Epic Games\UE_5.8'
$env:VHUMAN_MH_PROJECT = 'D:\work\vhuman-mh\MHGen\MHGen.uproject'
$env:VHUMAN_MH_DATA = 'D:\data\vhuman-mh'
$env:VHUMAN_MH_MODELS = 'D:\data\vhuman-mh-models'
$env:VHUMAN_BLENDER = 'D:\local\blender-5.2.3-windows-x64\blender.exe'
$env:TMP = 'D:\work\vhuman-win\tmp'; $env:TEMP = $env:TMP      # never the system temp
$py = 'D:\work\vhuman-win\venv\Scripts\python.exe'
```

## 5. How jobs run over ssh (W3: smoke)

The sshd session is a non-interactive service session (session 0). GPU access from it is not guaranteed for a windowed
or off-screen UE editor, so the first test decides the launch mode:
1. **Direct over ssh (try first):**
   `$W '. D:\work\vhuman-win\env.ps1; & $py -m server.vhuman.metahuman.launch smoke_00_env.py'`.
   It passes if the `smoke/00.json` it writes shows the subsystem, the Vulkan RHI and the RX 570.
2. **Fallback, run in the console user's session:** use a scheduled task that runs only when the user is logged on
   (interactive, `/IT`; the console session is active). Create it once:
   `schtasks /Create /TN vhuman-job /TR "powershell -NoProfile -File D:\work\vhuman-win\run_job.ps1" /SC ONCE /ST 00:00 /IT /F`.
   - `run_job.ps1` reads `D:\work\vhuman-win\jobs\current.json` (command line plus a `done` path) and writes its log
     and a `done.json`.
   - From Linux, write `current.json`, run `schtasks /Run /TN vhuman-job`, and poll `done.json` every 60 s. Do not hold
     the ssh session open for hours.
   - Requirement: the user stays logged on, and the screen may be locked but not logged off.

Whichever mode wins, every job:
- takes `D:\work\vhuman-win\gpu.lock`;
- writes a log under `D:\work\vhuman-win\logs`;
- ends with a result or `done.json`, which the Linux side checks;
- gets a timeout. On timeout, stop it by PID (`Stop-Process -Id <pid>`, with the PID recorded at start), never by name.

## 6. Procedures

**A. UE reference renders / generation batch (main use)**
1. Linux: build the batch as today (for example `render_batches/vhstudio.json`, or a `generate.py` batch of N sample
   ids). Then `python -m server.vhuman.metahuman.winjob push <batch.json>`. This rewrites the paths, tars the needed
   `samples/<id>/` inputs (`params.json`, camera placements) and copies them to `D:\data\vhuman-mh`.
2. Windows: `launch render_batch.py <batch.json>`, or `gen_batch.py` for generation. The resume logic (`done.json` per
   sample) works unchanged.
3. Linux: `winjob pull <batch.json>`. This copies back only the new outputs (`renders/*.exr`, `tex/*.png`,
   `head_lod0.bin`, `done.json`) into the Linux dataset, verifies sha256, and runs `manifest.py build`. Record
   `generated_on: <worker name>, ue 5.8.4 CL 58902052, rhi vulkan, gpu RX 570` in the per-sample provenance.
4. Before using Windows renders as references, check that they match Linux. Render 3 heads × the 10 cameras on both
   machines and compare them with `compare_render`'s aligned scoring (UE vs UE). The two machines have different GPUs
   and drivers, so expect small shading differences. Accept them if LPIPS < .02 and ΔE < 1 on the skin crops. Never mix
   the two sources in one comparison set without that check.

**B. Light-calibration UE probe**
- `light_calibration.py ue` runs the UE half. Push the plan and run `chart_probe.py <plan.json>` on Windows.
- Pull `ue/*.exr` and `env_actors.json` back, then run `fit` on Linux.
- The Cycles half stays on Linux.

**C. Blender (CPU) jobs**
- These suit the CPU chart probes (`chart_probe_blender.py`), small smoke renders and overflow `offline_render` views
  when the Linux GPU is busy.
- Push the candidate directory and run `offline_render.render(..., device='cpu')`.
- Before mixing outputs, check Blender 5.2.3 vs 5.2.2 on one view rendered on both machines' CPUs; same seed and preset
  should give ΔE < .5.
- Do not move the material sweep or the matrix renders here: they are GPU-sized and the box has no Cycles GPU backend.

**D. Status and cleanup**
- Status: `$W 'Get-Content D:\work\vhuman-win\logs\<job>.log -Tail 5; Get-Process UnrealEditor*,blender -ErrorAction SilentlyContinue | Select Id,CPU,WS'`.
- VRAM: `vulkaninfo` gives no usage figure. Use the UE log's RHI memory stats, or `Get-Counter '\GPU Process Memory(*)\Dedicated Usage'`.
- Disk: after `pull` verifies the hashes, delete the job's inputs and outputs under `D:\data\vhuman-mh`.

## 7. Order and gates

| Step | Content | Gate |
|---|---|---|
| W1 | Code changes in §3 plus tests (Linux) | tests pass; Linux smoke unchanged (`launch smoke_00_env.py` on Linux) |
| W2 | Bundle sync, venv, project copy, env.ps1 | `& $py -c "import server.vhuman.metahuman"` works on Windows |
| W3 | Smoke 00-03 on Windows; pick the direct or scheduled-task mode | `smoke/00.json` shows the RX 570 on Vulkan; one character created; first shader compile time recorded (expect longer than Linux's 30-90 min) |
| W4 | Timing: one `render_batch` head (10 cameras); one Cycles CPU `final` view | numbers written into this file |
| W5 | Linux-vs-Windows equivalence (UE 3 heads; Blender 1 view) | thresholds in §6A/§6C |
| W6 | Production: UE batches through `winjob`, in parallel with Linux Cycles jobs | provenance records the machine; Linux remains the source of truth |

Open questions for W3:
- Does UE's MetaHuman texture synthesis run on Polaris? It uses compute shaders, so it should, but this is untested.
- If Vulkan fails, try DX12.
- Is session 0 usable for rendering? That decides §5's mode.

## 8. Probe results (2026-11-04)

All probes ran from Linux over ssh, with no repo code on Windows yet. Scratch lives in `D:\work\vhuman-win`:
`probe\`, `ue_scripts\` (a copy of the repo's ue_scripts), `run_ue.ps1`, `chain.ps1`, and `logs\`. The project copy is
`D:\work\vhuman-mh\MHGen`, with its ini patched for the Windows paths and, under `WindowsTargetSettings`,
`DefaultGraphicsRHI_DX12`, `SF_VULKAN_SM5` and `PCD3D_SM5`. Smoke outputs are in `D:\data\vhuman-mh\smoke`.

| Probe | Result |
|---|---|
| Blender devices | HIP/CUDA/OptiX/oneAPI: none (`HIPEW initialization failed`). CPU only, 16 threads |
| Blender timing (SSS sphere, 3 area lights, 1024², 256 spp, OIDN) | Windows CPU 23.6 s; Linux CPU (32 threads) 21.9 s on the same script. Real head scenes are heavier: time one `offline_render` view before planning |
| UE Vulkan SM6 | rejected by the profile check: `VK_EXT_mesh_shader`, `VK_KHR_compute_shader_derivatives` missing |
| UE Vulkan SM5 | accepted; first start compiled global shaders (8 workers), then `EXCEPTION_ACCESS_VIOLATION` in `VulkanRHI` called from `MediaAssets` during module load (about 11 min in). Not pursued |
| **UE DX12 SM5** | works. D3D12 FL 12_1 / SM 6.7 hardware, SM5 feature level used. Note: D3D12 reports the adapter as UMA, 0 MB dedicated and 16 GB shared; performance is not yet measured |
| smoke_00 env | ok; 68 s to start (shaders cached); MetaHuman subsystem, Optional content and TextureSynthesis present |
| smoke_01 create | ok, 56 s |
| smoke_02 skin synthesis | ok, 44 s. **All 27 PNG textures (indices 3/50/120 × 9 maps) are byte-identical to Linux**; the `_rt.exr` readback sizes are identical too (the 11 KB `normal_rt.exr` is the same on Linux) |
| smoke_03 geometry | ok, 55 s; `Body/Face/Head .vhmh.bin` and `Head.bin` **byte-identical** to Linux |
| smoke_04 eyes | ok, 28 s |
| smoke_05 capture (SceneCapture HDR to EXR) | ok, 29 s. Versus Linux Vulkan on the 9070 XT: mean relative abs difference 0.12-0.32% (final colour), 0.03% (base colour), p99 abs diff ≤ 0.0024 linear, mask IoU 1.0 |

Launch mode:
- **Session 0 works.** UE gets the RX 570 from a plain ssh session, so no scheduled task is needed.
- Long jobs must not hang off the ssh session, because Windows kills the session's processes when it closes. Detach
  through WMI instead: `Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="powershell -NoProfile -ExecutionPolicy Bypass -File D:\work\vhuman-win\run_ue.ps1 -Script <s> -Tag <t> -Rhi dx12"}`.
  Then poll `logs\<t>.exit` from Linux with an until-loop.
- `run_ue.ps1` records the PID; stop by PID only.
- Its `exit` line shows an empty exit code, a `Start-Process -NoNewWindow` quirk. Judge success by the script's result
  JSON (`ok`) and by the absence of `Critical error` in the log.

Side effect to know about: the Linux ini carried `[Zen.AutoLaunch] DataPath`, which was rewritten to
`D:/work/vhuman-mh/ZenData`. On first start UE moved the machine-wide **default** Zen data path there ("Migrating default
data path from C:/Users/syoyo/AppData/Local/UnrealEngine/Common/Zen/Data ... Old location will be deleted"). Nothing was
lost: the old folder did not exist or was empty, and every file in the new one dates from this probe. Other UE projects
on this box that use the default Zen location will now use `D:\work\vhuman-mh\ZenData` (0.25 GB so far). To undo,
remove the `[Zen.AutoLaunch]` section from the project ini and point the default back from any UE project, or leave
it, since D: has 433 GB free.

Next (W1-W2 of §7): the code changes in §3 (dx12 default on Windows; WMI-detached launch in `launch.py`/`winjob.py`),
then one `render_batch.py` head (10 cameras) on Windows compared with its Linux references using aligned scoring.

## 9. Code + first head (2026-11-04)

The code is in the repo, uncommitted:
- `metahuman/__init__.py`: `VHUMAN_UE_ROOT`, `VHUMAN_MH_PROJECT`, `VHUMAN_MH_DATA` and `VHUMAN_MH_MODELS` overrides;
  `UE_RHI` (`VHUMAN_UE_RHI`, dx12 on Windows); the Win64 editor path.
- `launch.py`: `command(..., windows, rhi)` builds one quoted command line on Windows and rejects spaces and quotes in
  script arguments. The Linux-only env (DISPLAY, XDG, DDC) and the core-dump guard apply only on POSIX.
- `gpu.py`: `msvcrt.locking` when `fcntl` is missing.
- New `metahuman/winjob.py`: `sync`, `render`, `compare`, `status` and `kill`.

The Windows side needs no third-party Python packages: `D:\work\vhuman-win\venv` is an empty uv venv
(3.12.12), and `D:\work\gemm-vhuman` is a plain copy of `server/` made by `winjob sync`. The project ini's
`AdditionalPaths` points at `D:/work/gemm-vhuman/server/vhuman/metahuman/ue_scripts`.

```bash
python -m server.vhuman.metahuman.winjob sync
python -m server.vhuman.metahuman.winjob render mh_000000 --env Studio [--tag T] [--no-sync]   # push, run detached, wait, pull
python -m server.vhuman.metahuman.winjob compare mh_000000 --env Studio                       # run CPU-only on Linux (LPIPS)
```

Notes on the render path:
- Pulled renders land in `/mnt/nvme01/data/vhuman-mh/win_v1/v1/samples/<id>/renders` and never overwrite the Linux
  dataset.
- `launch` reports exit 1, because the editor access-violates at shutdown after the script finished (exit code
  0xC0000005; Linux segfaults there too). Judge success by `renders_done*.json` `ok`, which `pull` prints.

mh_000000, Studio env, 10 cameras × (lit + base colour) = 20 EXRs:
- **The first run on a new shader platform is not usable.** It took 74 s, and the faces matched, but the tank top
  rendered as a blurry, half-transparent proxy: its materials were still compiling at capture time.
- The second run took **60 s per head**, against about 50-60 s per head on Linux.
- Windows vs Linux, aligned skin-crop scoring:

  | camera | rel abs diff (linear) | LPIPS | dE |
  |---|---|---|---|
  | front / +40 / -40 / +90 / -90 | 0.77 / 0.92 / 0.77 / 0.85 / 1.02 % | .0003 each | .36-.43 |
  | ear L / R | 0.83 / 3.28 % | .0005 / .0009 | .24 / .43 |
  | eye L / R | 3.55 / 1.58 % | .0021 / .0009 | .48 / .35 |
  | mouth | 2.50 % | .0009 | .42 |

  All views are well inside the acceptance thresholds (LPIPS < .02, dE < 1); the remaining difference is fine
  per-pixel noise (TAA/driver).
- Rule: after any change of shader platform, engine, or a new material family, render one warm-up head and discard it.
  Alternatively, check the wide views against Linux before trusting a batch.

Next: VHStudio env for the same head (needs `--env VHStudio`; the lights are pushed inside the batch file), then a
multi-head batch timing with the Linux GPU busy.

VHStudio env (calibrated vhuman rig as RectLights), mh_000000, 10 cameras: 59 s on Windows; no warm-up was needed
(materials already compiled by the Studio runs). Windows vs Linux: wide views rel 1.1-1.8 %, LPIPS .0001-.0003,
dE .42-.45; close-ups (ears/eyes/mouth) rel 2.2-3.7 %, LPIPS .0010-.0013, dE .40-.50. The difference image is fine
noise plus a faint line along the shoulder shadow edge. Within the thresholds, so Windows can render VHStudio references.

Multi-head timing (Studio env, 11 heads mh_000010..mh_000435, 20 shots each, one editor session): 201 s in total =
~60 s editor start + **14.1 s per head** (14.0-14.4, steady), so ~1,000 heads/4 h of reference renders once started.
Linux ran the same batches at ~15-20 s per head, so the worker is at least as fast for UE renders, and it runs in parallel
with Linux Cycles jobs. The launch exited 0 this time (no shutdown crash). Windows vs Linux over all 110 views: LPIPS
median .0004, max .0052; dE median .42, max .54; none over the thresholds. The largest linear difference (6%,
mh_000300 front) is the backdrop only: on the front view (the first capture after each head loads) the grey backdrop
differs by +11% / -10% on 2 of 11 heads while the head pixels match (foreground ratio 1.000). It looks like the
backdrop not having settled for the first shot; it does not affect skin-crop scores. If backdrop pixels ever matter,
add a discarded pre-roll capture per head.

## 10. Cycles CPU renders (win_cycles, 2026-11-04)

Blender on the worker renders on the CPU only (Polaris has no HIP), so it is used for small, fast renders that pair with
each other, not for the full 1024 px / 128 spp comparison set.

```
python -m server.vhuman.metahuman.win_cycles render <tag> [--ids ...] [--variants normal,...] [--cameras ...] [--profile fast]
python -m server.vhuman.metahuman.winjob status <tag>      # progress: D:/data/vhuman-mh/cycles/<tag>/progress.log
python -m server.vhuman.metahuman.win_cycles pull <tag>    # finished views -> compare_v1/<id>/<variant>__fast/<camera>
python -m server.vhuman.metahuman.compare_render --confirm --report --profile fast --ue-env vhstudio
```

- **Linux stages, Windows renders.** `stage` runs `offline_render.render(..., stage_only=True)` once per head
  (prepare + skin templates + wrinkle channels: ~40 s CPU, ~28 MB) and writes one request per variant x camera
  (`view_request`: profile size/samples, camera focal scaled to that size, device cpu, variant options) into
  `tmp/win/<tag>/jobs.json`. Finished views (local `render_result.json`) are not staged again.
- On Windows, `win_cycles.py run` (stdlib only, under Blender's own Python
  `D:\local\blender-5.2.3-windows-x64\5.2\python\bin\python.exe`) copies the head's base into each view directory, points
  `scene.json` at the shipped candidate, runs `cycles_scene.py` then `cycles_validate.py`, and deletes the blend and the
  large npz files. Started detached through `winjob.start(command=...)`; rerunning skips done views.
- `pull` fetches only views with a `render_result.json`, never overwrites local finished views, and writes each result
  file last.
- **Profile `fast`** = 512 px, 32 spp, OIDN denoised: **~15 s per view** on the i9-9900K (14-20 s; mouth views are the
  slowest). Outputs use separate `<variant>__fast` directories; scoring box-downsamples the 1024 px UE references, and a
  fast variant is only ever paired with the fast `normal` baseline.
- Check against full quality (mh_000000 front, normal): fast LPIPS .0936 / DISTS .162 / dE 4.55 vs full (1024 px,
  128 spp, Linux HIP) .0937 / .167 / 4.50. Good enough to rank material variants.
- MH-eye variants (`compare_render.MH_EYES`: `normal_eyes` = eye_map v1, `normal_eyes2` = v2) get their own staged base
  per head (`base_eyes<n>`, prepared with that head fit); other variants share `base`.
