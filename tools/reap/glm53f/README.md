# GLM-5.3-Flash: local REAP and native GSQ/RCO tools

This workspace implements an experimental bounded-memory compression path for
`/mnt/nvme01/models/glm53f/base`. The source checkpoint is read-only. The target
is one language GGUF below **46,000,000,000 bytes**, an independent FP16 vision
projector, and deployment on **two 32 GB Tesla V100s** with one 131,072-token slot.

## Status and measured limits

Implemented: safetensors row reads and FP8 block dequantization, router-aware
REAP scoring, original-ID expert selection, streaming corpus scripts and
answer masks, native K-quant encode/decode, legal-field Gumbel optimization,
RAM-only candidate storage, full-task-loss choice gradients, discrete byte
allocation, RCO checkpoints, packed GGUF export, and separate host/V100 setup
scripts. The custom GSQ/RCO bridge is experimental; it is not an upstream
ISTA-DASLab implementation or a reproduction of their reported results.

Validated here: 20 tests; byte-for-byte native block round trips against
ggml/gguf; a packed GGUF serialization round trip; task-choice gradients against
a dense reference; a small hybrid HF model's checkpointed task-loss backward;
source FP8 decoding across a block boundary; and real GLM
forward pilots. The four-layer pilot covered KDA, sparse attention and MoE:
8 tokens, 35.93 seconds including initialization, 1.35 GiB peak host RSS.
A CUDA 13 pilot on the RTX 5060 Ti passed with 16 tokens and four layers:
74.60 seconds, finite outputs, 2.44 GiB peak host RSS and 0.43 GiB peak CUDA
allocation (`artifacts/pilot/pilot.json`). This partial pilot cannot predict
full-model throughput. Earlier small GSQ pilots retained their starting candidates. After correcting
FP32 logits, code initialization, and integer-field learning rates, a CUDA
smoke check improved Q2_K and Q3_K validation MSE and preserved starting
Q4_K/Q6_K/Q8_0 candidates. This does not establish full-model quantization quality.

The 45-layer CUDA source smoke check passed on one heldout 16-token window:
256.46 seconds, 1.11 GiB peak CUDA allocation, answer NLL 4.9591 on 15 answer
tokens (`artifacts/quality-cuda-source-smoke.json`). Source tiles now decode FP8
and apply FP32 block scales on CUDA; sampled tiles match the CPU decoder exactly.
The latest GLM5Next converter initializes and prepares real-checkpoint metadata.
A four-layer, 2048-token CUDA forward pilot passed in 36.55 seconds with
2.57 GiB peak host RSS and 3.65 GiB peak CUDA allocation
(`artifacts/pilot-2048/pilot.json`). The default KDA chunk size is now 16; the
upstream size 64 exceeded the 10 GiB allocation cap at 2048 tokens. A regression
test compares size-16 and size-64 KDA outputs. Source weights remain unchanged.

A real image processor/vision-forward check passed with 869 tokens and finite
features, using 1.49 GiB peak CUDA allocation. Torchvision 0.26.0+cu130 is now
installed. The vision rotary buffer is explicitly materialized after constructing
the tower on the meta device.

**Not validated:** representative full-model forward/backward; full GGUF export;
full REAP/GSQ/
RCO quality; full-model GPU peak memory; V100 loading and long-context operation. These are
required acceptance checks before treating the output as usable.

Python setup is complete with PyTorch 2.11.0+cu130, Transformers 5.16.1 and
`datasets` 5.0.1. CUDA device access was enabled and verified. The upstream
llama.cpp checkout and native `ggml-base` library are now present; its revision
is recorded in `artifacts/llama.cpp.revision`. No weights have been pruned,
quantized, or overwritten on NVMe01. The requested final output directory was
outside the session's writable roots at the last preflight; pilot and quality
reports are written under this workspace's `artifacts` directory.


## Research and format decisions

- [Cerebras REAP](https://github.com/CerebrasResearch/reap): use calibration-time
  routing-weighted expert output norms. Keep 116 of 288 experts in every sparse
  layer (59.7% removed), maintain top-8 routing and the shared expert, and retain
  original expert order when slicing routing rows and correction bias.
- [ISTA-DASLab GSQ](https://github.com/IST-DASLab/GSQ) and
  [RCO](https://github.com/IST-DASLab/RCO): optimize quantization and bit allocation.
  This bridge applies Gumbel-softmax to native integer code choices, with
  straight-through integer and FP16 scale constraints. It uses Adam, a local
  five-code neighborhood for formats above two bits, and a hard-packed validation
  safeguard. Those are implementation choices, not claimed upstream parity.
- [The reference GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  is a useful output-format reference, not a GLM-compatible pruning recipe.
  Native Q2_K is **2.625** bits/weight and Q3_K is **3.4375**, including block
  fields. They are not literal two- and three-bit storage.
- [llama.cpp GLM5Next support](https://github.com/ggml-org/llama.cpp/pull/27773)
  is required. The installed September 17 checkout predates this support. Setup
  retrieves an upstream checkout and records exact revisions under `artifacts/`.
  MLA KV projections remain FP16 because export transposes/splits them; optimized
  K-quant bytes must not be transposed or requantized.
- [CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html):
  use CUDA 12.x only for optional V100/sm70 deployment binaries. Local
  preprocessing and quality checks use **CUDA 13 + RTX 5060 Ti**. Setup selects
  PyTorch cu130 and `/usr/local/cuda` (CUDA 13.2 on this PC), and builds local
  llama.cpp for sm120. V100 support does not constrain local preprocessing.

The minimum language payload for this checkpoint/configuration is
**45,750,344,728 bytes**. A 32 MiB metadata reserve leaves only **216,100,840
bytes** for upgrades. Most routed weights must therefore remain Q2_K. If coding
quality needs substantially more Q3_K, reduce the retained expert count or relax
the byte cap; do not assume RCO can overcome this storage constraint.

The candidate bank is estimated at 106.21 GiB and stays in RAM. Configured RSS
limit is 130 GiB, CUDA allocation limit 10 GiB. Activation reservoirs are capped
at 128 samples per projection group and 3 GiB overall to preserve disk headroom.
Expert down-projection reservoirs pool activations within a layer. Embedding GSQ
uses an isotropic reconstruction objective; its choices subsequently receive
answer-token task gradients in RCO. Norms, routing, mHC, KDA convolutions and
small structural fields are protected. The MTP head is omitted; vision is retained.

## Host setup and execution

Run these on the host with network, CUDA access and permission to write the
chosen output directory. Setup installs into this workspace, not global Python.

```bash
bash scripts/setup.sh
bash scripts/test.sh
.venv/bin/python -m glm_reap.cli preflight --report artifacts/preflight.json
bash scripts/download-corpus.sh
.venv/bin/python -m glm_reap.cli pilot --device cuda --tokens 16 --layers 4
```

The sequential production runner is:

```bash
bash scripts/run-compression.sh
```

Its lock prevents duplicate runs. It calibrates, checks pruned-model heldout
loss, generates GSQ candidates, performs task RCO, evaluates final selected bytes
on heldout examples, and exports the final GGUF. Unselected candidate arrays
are released before serialization to reduce host RAM during file verification.
State files and logs use `artifacts/compression` (`work_dir`); the final model
uses the configured NVMe01 `output` directory. Monitor with:

```bash
tail -f artifacts/compression/run.log
cat artifacts/compression/run-state.json
cat artifacts/compression/current-layer.json
cat artifacts/compression/current-tile.json
```

The runner was launched on the GPU. Do not launch a second copy or edit the
active config; resume with the same command if its recorded state is failed.
The run will take substantial time; candidate regeneration is required if
compression is interrupted because the candidate bank stays in RAM.

Inspect the pilot report and extrapolate runtime before starting the large run.
The partial pilot reports its layer count explicitly. Repeat with all 45 layers,
then representative 2048-token windows, and measure backward as well as forward
memory. Do not equate the short forward pilot with a full compression acceptance.

```bash
.venv/bin/python -m glm_reap.cli calibrate --device cuda
.venv/bin/python -m glm_reap.cli compress --device cuda
bash scripts/export-projector.sh
```

`calibrate` scores the source routing, selects expert IDs, then collects GSQ
activations from the pruned model. `compress` generates both expert candidates
and the legal dense candidates; 128-column KDA projections can only use Q8_0
from the configured list. It then optimizes whole-model assistant-token cross
entropy and the byte constraint. Selected expert projections share one type per
layer so GGUF fused expert tensors have a legal uniform encoding.

Candidate generation streams source rows and performs tile-level optimization.
The RAM candidate bank is reconstructed after restart. Completed GSQ tiles are
saved transactionally in `artifacts/compression/gsq-tiles.sqlite` as compressed
XOR differences from the native initial quantization; reconstruction verifies
both baseline and candidate hashes and skips GPU training for those tiles.
The checkpoint defaults to an 8 GiB cap to preserve the limited free space.
If it fills, the process stops safely; add storage and raise the cap with
`GSQ_CHECKPOINT_GIB=16 bash scripts/run-compression.sh`. This environment setting
does not change calibration identity. Worst-case compressed deltas may require
more space; the full candidate bank is about 106 GiB.

Source calibration checkpoints every completed batch with preserved score sums
(eight windows by default in RAM mode; one window in mmap mode). An interrupted
RAM batch is repeated; set `REAP_CALIBRATION_BATCH_WINDOWS=1` for finer checkpoints.
Activation reservoirs and RNG state are checkpointed every eight windows. RCO
checkpoints its logits, optimizer and RNG state after every completed step.
An interruption repeats only unfinished work since the applicable checkpoint.
Source/config/corpus identity mismatches refuse resume. Restart with:

```bash
bash scripts/run-compression.sh
```

Export journals fsynced tensors in `artifacts/compression/export-state.json`.
It verifies saved bytes, discards any uncommitted tail and resumes the `.partial`
GGUF at the last completed tensor. It verifies all tensor hashes through
GGUFReader, enforces the final byte cap and renames the file. An existing completed
GGUF is verified against its source/config/size/tensor manifest and reused.
Keep source and final GGUF on separate paths; no second full output is needed.

A local 10-second calibration sample read 5.08 GB from disk, used about 84% of
one CPU core and showed 28-32% GPU utilization. This points to weight streaming
and CPU orchestration as the current limits. A faster GPU may help later GSQ/RCO
compute, but cannot eliminate repeated checkpoint reads. RAM mode now uses layer-wise source calibration batches and caches source layers
to reduce repeated reads. The sample above predates this change; no full-run
speedup is claimed yet.


## Memory and CPU options

The source payload is 305.78 GiB, which cannot all fit into the available work
RAM. RAM mode is the default: it caches packed source weights (including FP8
scales) in ordinary process memory, processes eight source-calibration windows
layer by layer with their hidden states held on CPU, and keeps the roughly
106 GiB native candidate bank in RAM. The source cache defaults to 112 GiB and
shrinks as candidate storage grows, reserving 12 GiB within the configured
130 GiB RSS cap for activations, modules, decoding and optimizer work. 130 GiB
is approximately 140 decimal GB. `/dev/shm` is not required; using ordinary RAM
avoids a separate tmpfs capacity limit. Checkpoints remain on durable disk.

CPU workers, PyTorch and BLAS thread pools default to half the physical cores
(eight threads on this 16-core machine). The source-layer preload uses that
many CPU workers. The expert/GPU scheduling loop remains sequential; setting
threads does not parallelize independent GPU optimization jobs.

```bash
# Default RAM mode, with optional overrides.
REAP_MEMORY_MODE=ram REAP_CPU_THREADS=8 REAP_SOURCE_CACHE_GIB=112 \
  REAP_CALIBRATION_BATCH_WINDOWS=8 bash scripts/run-compression.sh

# Lower-RAM mode. Requires about 106 GiB plus reserve on the candidate disk.
REAP_MEMORY_MODE=mmap REAP_CPU_THREADS=8 \
  REAP_CANDIDATE_DIR=/path/to/large/disk/reap-candidates \
  bash scripts/run-compression.sh
```

Mmap mode maps source shards and native candidates from disk, does not pin a
source RAM cache, and releases used mapping pages so the OS can reclaim them.
It validates candidate-disk capacity before starting. The current workspace has
only about 20 GiB free, so use a larger disk for this mode. The mapped candidate
files are expendable: durable GSQ tile checkpoints reconstruct their contents.
Export still uses the separate configured output disk.

These settings also accept config keys `memory_mode`, `cpu_threads`,
`source_cache_gib`, `calibration_batch_windows` and `candidate_dir`. Environment
overrides take priority and leave existing calibration identity unchanged; use
them when resuming an existing run. Effective settings are recorded in
`artifacts/compression/execution-options.json`. Mmap pages count toward RSS
while resident, so lower-RAM mode is not a fixed RSS guarantee.

A CUDA pilot compared layer-wise and window-wise results on four real source
layers and two short windows. Results matched, with an 8.11 GiB source cache,
10.58 GiB host RSS, 0.44 GiB peak PyTorch CUDA allocation and eight CPU threads.

## Corpus

Download destination: **`/mnt/nvme02/work/reap/artifacts/corpus`**.
Hugging Face download cache: **`/mnt/nvme02/work/reap/.cache/huggingface`**.
The destination is recorded as `corpus` in `configs/glm53f.json`. Run
`bash scripts/download-corpus.sh`; it prints both paths before downloading.
The corpus has reached its configured 2 GiB budget, including about 1.9 GiB
of vision images. All five categories are present; vision is below its token
quota. `manifest.json` records achieved counts and `budget_reached`.
At the byte cap, the downloader exits successfully and preserves existing data.
New records are size-checked before saving; reruns resume by content ID.


The full downloader output is appended to `artifacts/corpus-download.log`.
Trajectory Parquet files use synchronous eight-row batches with background
scanner threads disabled and explicit stream cleanup at quotas or exceptions.
Nullable tool-call fields are normalized before tokenization.

Calibration windows sample the configured category mix; image tensors are
prepared lazily only for selected windows to avoid retaining the entire image
corpus in RAM.

Sources are streamed into category JSONL files; reruns skip stable record IDs.
Records retain messages, tools, assistant-only token masks, split IDs and image
paths. The train/heldout split is deterministic by content hash. Downloaded
The Stack smol-xs revision is pinned in the loader and recorded in the manifest.
Its per-language JSONL files are streamed with the built-in JSON reader, so
`datasets` 4/5 does not execute its obsolete Python loader. Other dataset
revisions are not pinned yet: archive the corpus and manifest once
downloaded to preserve the exact experiment. Streaming failures do not silently
substitute a different corpus.

Configured token mixture: 45% Evol-CodeAlpaca instructions, 25% multilingual
The Stack smol-xs, 15% SWE-rebench OpenHands trajectories, 5% FineWeb-Edu and
10% Cauldron WebSight/DocVQA/ChartQA/TextCaps. The small multilingual code source
can exhaust before its quota; inspect `manifest.json`. The downloader does not
invent repeated samples to reach a quota. Vision images use a 512-token processor
cap; examples exceeding the whole-window limit are skipped with complete image
spans intact. Consequently usable window proportions must be checked after
tokenization. Stored tool trajectories are data; the downloader never executes
their commands.

## Deployment acceptance

```bash
bash scripts/build-v100.sh
bash scripts/validate-v100.sh
bash scripts/serve-v100.sh
```

Startup must show both devices, all intended layers on GPU, full 128K KV
allocation, and headroom for projector/compute buffers. Q8 KV support and flash
attention must be checked on the exact V100 build. The nominal combined VRAM
capacity does not guarantee balanced allocation or a successful long prefill.
The validation script is a memory/startup smoke check, not a coding benchmark.
Compare heldout coding completion, tool-call correctness and vision behavior
against the existing unpruned GGUF before accepting the compression. Full
benchmark integration and 128K quality evaluation remain outstanding.

## Local streamed quality checks

```bash
# Start with KDA + sparse attention + MoE, then extend to all 45 layers.
bash scripts/quality-local.sh --layers 4 --tokens 64 --windows 1
bash scripts/quality-local.sh --layers 45 --tokens 256 --windows 4
# Isolate calibrated pruning from quantization.
bash scripts/quality-local.sh --mode pruned --reap /mnt/nvme01/models/glm53f/reap/reap.json
# Pruning plus a native low-bit baseline, without storing quantized weights.
bash scripts/quality-local.sh --mode native --reap /mnt/nvme01/models/glm53f/reap/reap.json
```

The checker keeps source and candidate hidden states, streams bounded weight
rows from NVMe to the local GPU, and records accumulated layer MSE, relative MSE
and cosine similarity. Full 45-layer runs also report answer-token negative log
likelihood, with output positions chunked to bound logits memory. It writes a
small JSON report to `/mnt/nvme02/work/reap/artifacts/quality-local.json`; it
creates neither a candidate bank nor a GGUF. Native mode uses tile-wise RTN and
is explicitly a baseline, not validation of optimized GSQ/RCO candidates.
The command was smoke-tested on the first real GLM layer using synthetic token
IDs on CPU; this verifies the streaming/comparison path, not coding quality.
The CUDA allocator is capped at 10 GiB (or lower if available headroom requires
it), leaving room for driver/context allocations on the local 12 GB GPU.

V100 tools are optional deployment steps. The CUDA 13 local workflow does not
invoke or require the V100 build. User authorization to use the GPU has been
received, but the present session still lacks NVIDIA device nodes; its policy
also disables escalated shell execution. GPU checks must run in a session that
exposes the RTX 5060 Ti.

Reports from this session: `artifacts/preflight.json` and
`artifacts/cpu-pilot.json`. The current checkout is a standalone workspace, not
a git repository; no commit or PR was created.
