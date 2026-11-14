# rdna4/praxis - praxis-1 policy runner on RDNA4 / ROCm (gfx1201)

Port of `cuda/praxis` (encoder/decoder MoE towers, action expert, window_joint sampler, expert streaming ring, sharded loading, pack cache, server).
HIPRTC at run time (rocew, no hipcc). Files: `hip_praxis_runner.h` (host, shared logic with the CUDA runner through `rdna4/cuda_driver_compat.h`),
`hip_praxis_kernels.h` (kernels), `test_hip_praxis.c` (parity harness = test_cuda_praxis.c), `hip_praxis_server.c` (same binary protocol as `cuda/praxis/praxis_server`),
`test_server_replay.py` (numpy-only protocol replay of a reference dump). The Wan2.2 VAE lives in `hip_wan22_vae.h` (other port).

Differences to CUDA: no cuBLAS (own WMMA GEMM: 64x128x32 tiles, bf16 dense / bf16 expert banks / NVFP4 / FP8 e4m3 weights, grouped by expert); no FP4 tensor cores, so every
NVFP4 plan is **W4A16** (exact e2m1 x e4m3 dequantisation to f16, f16 WMMA; `PRAXIS_NVFP4=1`, `nvfp4`, `nvfp4a16` are the same); FP8 banks are W8A16 (f16 WMMA); flash attention
is a transposed-formulation WMMA kernel (S^T = K Q^T, so P needs no LDS transpose). Pack layout / pack cache / `PRAXIS_EXPERT_FMT=mixed` / `PRAXIS_SLOT_MB` / ring: as in cuda/praxis/README.md.

## Run (tr16)
    make -C rdna4/praxis test_hip_praxis hip_praxis_server
    make -C rdna4/praxis test_hip_praxis_all                     # tiny/mid/mid128 vs torch fp32 dumps (exact 0.02, fast 0.06, stream/sampler/ctx-cache selftests)
    export TMPDIR=/mnt/nvme01/praxis_tmp/tmpdir OMP_NUM_THREADS=8   # real pack: keep threads modest
    PRAXIS_EXPERT_FMT=mixed PRAXIS_PACK_CACHE=/mnt/nvme01/praxis_tmp/praxis-1-droid.mixed.packcache rdna4/praxis/test_hip_praxis \
      --model /mnt/nvme01/models/praxis/praxis-1-droid --ref tmp/praxis_ref/real_fp32 --out tmp/hip_out/real --stream 2 --tol 0.5 --repeat 2   # call1: --ref .../real_fp32/call1
    python3 rdna4/praxis/test_server_replay.py --server rdna4/praxis/hip_praxis_server --policy <real dir> --ref tmp/praxis_ref/real_fp32

## Measured (RX 9070 XT, shared with other work; real checkpoint, dec_vel rel-L2 vs the torch fp32 oracle)
| plan | call2 dec_vel | call1 dec_vel | plain 4-pass call | skip-last-video | peak device |
|---|---|---|---|---|---|
| mixed | 3.0e-2 | 3.4e-2 | 7.25 s (8.2 s before the fp4 dequant fix) | 5.3 s | 7.65 GB first run, 7.87 GiB after latency loops (hipMemGetInfo, incl. ctx) |
| W4A16 | 5.5e-2 | 5.2e-2 | 8.6 s (before fix) | 5.1 s | 7.2 GB |
CUDA 5060 Ti reference: mixed 3.2e-2, W4A16 5.3e-2, ~7.5 s. Profile (mixed): ring stall 33 %, gate_up GEMM 16 %, down 12 %, flash 11 %.
Known gaps: kernels not tuned beyond the fp4 dequant (no double-buffered LDS), bf16 experts untested on the real model. (Server VAE/Qwen and the MuJoCo closed loop: see below.)

## Full server (`hip_praxis_server_full`) and closed-loop sim driven from the AMD host
`hip_praxis_server.c` is now a drop-in of `cuda/praxis/praxis_server` (same binary protocol, same ops ENCODE/SAMPLE/TEXT/DECODE/INFO/QUIT, same 4-f64 header per reply):
in-process Qwen3.5-2B (`../llm/hip_qwen35_st.c`, tokenizer = `praxis_tok.c`, the tokenizer half of `cuda/praxis/praxis_qwen.c`) for the prompt and the blank caption
(all misses of a request in ONE load -> prefill -> free cycle, memory + disk cache `<text-cache>/text_<fnv64 of the int32 ids>.npy`, same key/files as the CUDA server), and
`hip_wan22_vae.h` for ENCODE/DECODE with `--vae-mode keep|reload`; `--trim 2` (default) releases the policy workspace + expert ring (`praxis_trim`/`praxis_release_stream`) before
Qwen/ENCODE/DECODE and regrows it for SAMPLE. NOTE: the old `hip_praxis_server` rule in the praxis-runner block does not link Qwen/tokenizer/VAE any more; build the new target:

    make -C rdna4/praxis hip_praxis_server_full
    # on tr16 (run from the repo root; options: --policy --vae --qwen --text-cache --stream --vae-mode keep|reload --trim 0|1|2; --text/--text-u = fixed conditioning files, tried before Qwen)
    TMPDIR=/mnt/nvme01/praxis_tmp/tmpdir OMP_NUM_THREADS=8 PRAXIS_EXPERT_FMT=mixed PRAXIS_PACK_CACHE=/mnt/nvme01/praxis_tmp/praxis-1-droid.mixed.packcache \
      rdna4/praxis/hip_praxis_server_full --policy /mnt/nvme01/models/praxis/praxis-1-droid --vae /mnt/nvme01/models/praxis/wan2.2-vae --qwen /mnt/nvme01/models/praxis/qwen3.5-2b \
      --text-cache /mnt/nvme01/praxis_tmp/praxis_text_cache --stream 2 --vae-mode reload --trim 2

Device memory per op (hipMemGetInfo polled every 4 ms, GB over the 0.10 GB idle context; mixed, K'=2, policy resident between calls, `--vae-mode reload --trim 2`):
TEXT/Qwen cold prompt **3.00** (load 0.6 s, 2 prefills 41 ms, free 34 ms; +0.15 GB stays allocated after the free), ENCODE **2.03** (policy not yet loaded) / **5.8-6.6** (policy resident),
SAMPLE **6.6** (first call incl. policy load 17-19 s) / **7.5** (warm), DECODE **7.5** (24 frames 224x832, policy resident), resident after a call 3.2-3.5. Max instant 7.54 + 0.10 ctx = **7.64 GB < 8 GB**.

### Launching the server over ssh (frames/latents go through the ssh pipe)
`cuda/praxis/praxis_client.py` is unchanged for the CUDA default; setting `PRAXIS_SERVER_CMD` (or `run_sim_cuda.py --server-cmd`) makes the client start THAT shell command instead
(complete command incl. its own options; the local `--stream/--vae-mode/--expert-fmt` flags are then ignored). The client reads `config.json` of the checkpoint locally (`--checkpoint`), the server uses the tr16 path.

    export PRAXIS_SERVER_CMD='ssh -o BatchMode=yes -o Compression=no tr16 "cd ~/work/gemm/praxis && TMPDIR=/mnt/nvme01/praxis_tmp/tmpdir OMP_NUM_THREADS=8 PRAXIS_EXPERT_FMT=mixed PRAXIS_PACK_CACHE=/mnt/nvme01/praxis_tmp/praxis-1-droid.mixed.packcache exec rdna4/praxis/hip_praxis_server_full --policy /mnt/nvme01/models/praxis/praxis-1-droid --vae /mnt/nvme01/models/praxis/wan2.2-vae --qwen /mnt/nvme01/models/praxis/qwen3.5-2b --text-cache /mnt/nvme01/praxis_tmp/praxis_text_cache --stream 2 --vae-mode reload --trim 2"'
    # numerics, same checks as the CUDA server (local torch, remote server):
    ~/.venv/bin/python -I cuda/praxis/check_client_replay.py
    ~/.venv/bin/python -I cuda/praxis/check_sim_replay.py --obs /mnt/disk01/tmp/praxis_sim/m4_obs/obs_call1.npz --ref /mnt/disk01/tmp/praxis_sim/m4_ref --oracle /mnt/disk01/tmp/praxis_sim/m4_ref_fp32
    # same observation through the local CUDA server AND the remote one, diffed (also times DECODE):
    ~/.venv/bin/python -I cuda/praxis/check_server_compare.py --obs /mnt/disk01/tmp/praxis_sim/m4_obs/obs_call1.npz --oracle /mnt/disk01/tmp/praxis_sim/m4_ref_fp32
    # closed loop (MuJoCo + EGL on the local NVIDIA GPU, policy on the RX 9070 XT), variant 'fast' = mixed, cfg video 1, skip-last-video, rebase 8:
    MUJOCO_GL=egl ~/.venv/bin/python -I cuda/praxis/run_sim_cuda.py --config /mnt/disk1/praxis/praxis-1-oss/configs/sim_droid.yaml --vae-mode reload --expert-fmt mixed \
      --cfg-video 1.0 --skip-last-video --server-cmd "$PRAXIS_SERVER_CMD" --set task.name=lift_cube --set task.prompt=null --set "episodes.seeds=[100,101,102,103,104,105]" --set output.dir=/mnt/disk01/tmp/praxis_sim_hip/A_lift_cube
    (batch used: /mnt/disk01/tmp/praxis_sim_hip/run_hip.sh lift_cube cube_on_plate; summaries: cuda/praxis/eval/summarize.py)

### Results (real checkpoint, mixed experts = W4A16/FP8 on RDNA4)
Replay vs the torch fp32 oracle (sampled action tokens, rel-L2; CUDA mixed numbers from cuda/praxis/README in brackets): reference observation call 1 5.7e-2 (action rows 2.9e-2; xyz max 6.0 mm), call 2 1.1e-1 (rows 5.4e-2; xyz max 12.2 mm) [call 2 6.7e-2 / 12.5 mm, call 1 0.26];
recorded sim observation (cube_on_plate seed 1, call 2): 5.8e-2, xyz rms 16.3 mm [CUDA 4.3e-2 / 13.5 mm], first relative rows |row0..2| 32.6/32.9/31.2 mm (fp32 oracle 34.7/35.2/33.3, CUDA 33.0/33.2/31.2: the boundary jump is the checkpoint's own).
anchor/clean latents vs the torch reference 1.9e-3 / 1.7e-3 (HIP VAE; CUDA value not re-checked here), proprio 0, canvas pixels identical.
Direct HIP-vs-CUDA server on the same observation (identical noise, `check_server_compare.py`): call 1 pred_action 4.7e-2, call 2 2.9e-2 (video 1.3e-1), commands xyz max diff 13.8 / 10.5 mm, i.e. the same size as either backend's distance to the oracle (route flips + W4A16 vs CUDA mixed accumulation), not a protocol error.
DECODE returns uint8 [24][224][832][3]; mean |diff| to the CUDA decode of the CUDA-predicted video 2.3/255 (different predicted latents, not a VAE-only comparison; VAE parity is 18/18 on each backend).
Closed loop (seeds 100-105, 'fast' variant; CUDA = the 12-seed batch /mnt/disk01/tmp/praxis_sim2 restricted to the same seeds, run by `cuda/praxis/eval/e1.sh`):
| task | backend | success | ticks (succ) | per-call mean/p50 s | boundary jump mm | step CV |
|---|---|---|---|---|---|---|
| lift_cube | CUDA 5060 Ti | 6/6 | 165 | 4.53/3.96 | 6.1 | 0.45 |
| lift_cube | HIP 9070 XT via ssh | **4/6** (101, 102 fail) | 151 | 3.73/3.53 | 6.2 | 0.47 |
| cube_on_plate | CUDA | 5/6 (102) | 223 | 4.15/3.93 | 6.2 | 0.45 |
| cube_on_plate | HIP via ssh | 5/6 (102) | 177 | 3.79/3.54 | 5.9 | 0.43 |
Per-call (HIP, 53 calls each task): 3.5 s median end to end incl. client prep + 2 VAE encodes (server SAMPLE 2.7-3.0 s, ENCODE 0.51-0.53 s server-side); the ssh pipe adds ~27 ms per SAMPLE (max 36) and ~80 ms per call for the ENCODE uint8 stacks
(0.59-0.61 s round trip vs 0.51-0.53 s server); first call of an episode set = policy load 15 s. No link problem. Boundary jump and step CV are indistinguishable between backends (rebase 8).
Honest reading: 9/12 vs 11/12 episodes on 6 seeds per task is not a significant difference (lift_cube seeds 101/102 failed on HIP and succeeded on CUDA, cube_on_plate 102 failed on both), but it was not shown to be equal either; no
repeat/other seeds were run, and the failing HIP episodes were not analysed (candidates: W4A16 route flips vs CUDA mixed, Qwen conditioning from the HIP port, which matches HF at 3.8e-4). Not run: 'full' variant (decoded video) closed loop (DECODE op only checked in replay), NVFP4/bf16 variants, other tasks, a CUDA server rerun on the same seeds.
