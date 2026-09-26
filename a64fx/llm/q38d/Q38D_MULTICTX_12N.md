# Qwen3.8-27B NVFP4: 12-node multi-context work

Hardware: Fugaku A64FX, interactive 12-node PJM job 51932259. The runtime
uses three independent TP4 groups, each with one context slot per request.
Every slot owns SSM state, convolution history, and K/V; decode currently
visits slots round-robin. The TP4 groups run concurrently. This is not yet a
fused batch GEMM or an HTTP service.

## Capacity and decode measurements

The `--kv-i6` cache stores each 256-element K and V row in 192 bytes with one
FP32 scale per row: 6,272 bytes per token per TP4 rank across 16 attention
layers. `--prune-model` releases 13.891 GiB of source matrix mappings after
TP4 repack. Steady RSS before KV fill is about 6.02 GiB per rank.

The following screens filled earlier K/V positions with finite zero rows and
scale 1, evaluated one actual prompt token per context at the named depth,
then generated tokens. They establish allocation, attention traversal, and
decode timing at depth. They **do not** measure fresh prefill of the synthetic
history or model quality over real long inputs. All screens passed the
12-rank group/hash checker.

| Depth × contexts | TP4 groups | Busiest rank RSS | Busiest rank MemAvailable | Decode, busiest group |
| --- | --- | ---: | ---: | ---: |
| 262,144 × 32 | 11/11/10 | 23.33 GiB | 5.67 GiB | 7.03 aggregate tok/s, 4 outputs/context, vectorized INT6 |
| 524,288 × 16 | 6/5/5 | 24.63 GiB | 4.63 GiB | 3.31 aggregate tok/s, 4 outputs/context, vectorized INT6 |
| 1,048,576 × 8 | 3/3/2 | 24.49 GiB | 4.92 GiB | 1.80 aggregate tok/s, 4 outputs/context, vectorized INT6 |

At 65,536 depth with two contexts per group, the packed-cache attention
changes raised group decode from 2.87 to 5.12 and then 22.73 aggregate
tok/s for 16 outputs/context. The last change unpacks each row once and uses
SVE dot/PV arithmetic. At 262,144 × 32 it raised the busiest-group rate from
1.34 to 7.03 aggregate tok/s; at 512K × 16 from 0.67 to 3.31; and at
1M × 8 from 0.34 to 1.80. The latter two screens generated four tokens per
context, and all ranks passed the grouped checker. These rates include only
decode after synthetic history placement, not prefill or handoff.
Holding each four-token rate constant, 8,192 outputs per context would take
roughly 3.6–4.1 hours for the busiest group in these shapes. That is only a
projection from short synthetic screens; it shows the current long-context
decode path is a capacity prototype, not an interactive coding-agent service.

`--contexts N --context-prompt-list PATH` accepts one prompt-file path per
slot, each with at least `--prompt-tokens N` tokens. Six distinct C-review
prompts ran across the three groups; all 12 ranks agreed on each stream and
the six output hashes were distinct. `run_tp4_groups_12n.sh` accepts
`Q38_GROUP_CONTEXT_PROMPT_PREFIX` for one manifest per group.

## PP12 prefill and state handoff

PP12 can allocate K/V only for the attention mixers owned by each pipeline
stage. The worst stage owns two attention layers. Its FP32 KV remains too
large at one million tokens: the attempted 1M screen was killed at about
27.5 GiB peak RSS before post-fill measurement. Inputs above 300K now require
`--prefill-kv-i6`; the packed path disables the separate packed-key and
INT16 PV caches. Query tiling up to four is supported above 300K depth,
although tile one uses much less HBM for nearly the same rate there.

With packed PP K/V, a 1,048,575-token synthetic history plus one evaluated
token completed on all 12 stages. The busiest reported stage had 19.68 GiB
RSS and at least 8.39 GiB MemAvailable. The one-token PP12 end-to-end
latency was 18.97 s with serial attention. A new packed INT6 single-query
path splits the deep attention scan over the 12 workers in each KV-head CMG;
the same 1M-depth suffix took **1.80 s**, with the same first token (96066).
At 8K depth it took 0.225 s versus 0.356 s, again with the same first token
(6559). The final residual hashes differ because the softmax/PV reduction
order changes, so these checks do not establish long-generation equivalence.
A 1,025-token **evaluated** random-prefix comparison
produced the same first token (271) from FP32 and packed PP; residual hashes
differed. This is a short first-token check, not a long-context quality gate.
At 32K synthetic depth, the one-query packed path sustained **155.96 prefill
tok/s total (13.00 tok/s/node)** over two 1,024-token evaluated suffixes.
Reusing each unpacked row across eight queries raised this to **176.74 tok/s
total (14.73 tok/s/node)**; the first token and residual hash matched the
one-query path exactly. At 1M synthetic depth, 32 evaluated suffix tokens
took 56.72 s with tile one and 55.73 s with tile four. Both returned first
token 19571 and the same residual hash. Tile four reduced `MemAvailable` on
rank zero from 10.05 to 6.68 GiB; keep tile one for deep contexts. These
rates make full real 1M prefill impractical with the current packed attention.
On the fully evaluated 32,768-token C-source prompt, packed PP12 with query
tile eight reached **317.13 tok/s total (26.43 tok/s/node)** and exported its
version-2 state in 2.66 s. The fast FP32-prefill/INT6-export route below
reached 1,069.68 tok/s total on the same prompt; the packed route remains
for depths that cannot hold FP32 KV. TP4 imported the packed producer state
in 3.46 s and generated 256 tokens at 35.63 tok/s. Against the fast
FP32-prefill/INT6-export state on the identical prompt, only the first
generated ID matched; output index 1 was packed-prefill 333 versus
FP32-prefill 271. Both decoders used INT6 KV, so the discrepancy is caused
by compressed KV during prefill. Deep packed PP is a capacity and timing
prototype pending task-quality evaluation. An optional INT8 PP cache with
INT6 state export was screened on the same 1,024-token C-source prompt used
for the FP32 producer check. It reached 330.49 prefill tok/s total and its
TP4 continuation matched the FP32 producer for only the first two output
IDs (index 2: 274 versus 18601). It was slower and did not resolve the
quality issue, so that experiment was removed from the runner.

State format version 2 stores packed K/V rows and FP32 scales with checksums;
version 1 FP32 state remains available. A 128-token evaluated PP12 state
exported and imported on TP4. The imported first token matched the producer
(271), and TP4 generated 16 tokens at 85.7 tok/s. INT6 and FP32 handoffs
matched the first five generated IDs on that repeated short prompt before
their greedy streams diverged. The longer C-source comparison below likewise
shows INT6 cannot be substituted for FP32 when exact continuation matters.

For 32K coding prompts, `--state-out-i6` writes version-2 state after the
existing fast FP32 PP prefill. This keeps the optimized PP kernels available
while reducing the TP4 handoff and decode cache size. It is a separate option
from `--prefill-kv-i6`, which is needed to hold deep PP history in HBM.
On a 1,024-token evaluated C-source review prompt, this route reached 636
prefill tok/s total, exported in 2.32 s, imported into TP4 in 2.34 s, and
generated 64 tokens at 83.38 tok/s. **All 64 output IDs matched** the FP32
handoff reference. On a 32,768-token evaluated C-source review prompt, PP12
reached **1,069.68 tok/s total (89.14 tok/s/node)** in 30.63 s. Export took
5.39 s; TP4 import took 2.95 s, matched first token 262, and generated 256
tokens at **35.66 tok/s**. Reusing the same INT6 state, TP4 generated
**8,192 tokens at 33.32 tok/s**. The FP32 state from the same evaluated 32K
prompt generated 8,192 tokens at **57.37 tok/s**. Token IDs matched for the
first **275** outputs, then diverged at output index 275 (INT6 1510, FP32
2414). Use FP32 KV for 32K coding continuations where it fits; INT6 remains
an opt-in deep-context capacity path without a long-input quality result.
The one-request prefill rates do not meet the 150 tok/s/node target.

The exported 32K state was also imported into **eight independent slots**
across the three TP4 groups (3/3/2), each generating 256 tokens. The three
groups sustained 35.77, 36.17, and 36.18 **aggregate** tok/s respectively;
all 12 ranks passed the batch checker. Every slot's 256-token FNV hash
(`fc3b6a38ff83144f`) matched a hash recalculated from the single-context
token trace. This run reused the same evaluated prompt/state in each slot,
so it tests independent state storage and scheduling at 32K. The separate
six-state short run above tests distinct prompts and states.

The same eight-slot run with **FP32 KV** reached 60.28/61.21/61.20 aggregate
tok/s across the groups, again matching all 256 IDs per slot. Reading the
same snapshot independently for three slots took 93.39 s of state import in
group zero. The importer now validates the first copy and clones its resident
SSM/convolution/KV state into slots with the identical state directory and
prompt tokens. The second and third FP32 clones took 0.12 s each; total
readiness fell to **38.78 s**, with 59.73/61.03/61.41 aggregate tok/s and
the same output hashes. Distinct prompts retain separate checked imports.
The same clone path passed an eight-slot INT6 run (64 outputs/slot); its extra
slots copied in about 0.05 s each after a checked first import.

Eight **distinct**, fully evaluated 32,768-token C-source review prompts
were also processed with FP32 PP12 and handed to the three TP4 groups. The
prompts differ in their review focus and share the underlying C source.
Each PP12 prefill ran at 1,065.05–1,072.53 tok/s total
(88.75–89.38 tok/s/node), taking 30.55–30.77 s of prefill plus
5.54–5.96 s for state export. Separate process startup brought each
prefill launch to 67.20–67.78 s wall time. The three TP4 groups imported
3/3/2 separate snapshots in 129.38/126.89/80.32 s and decoded 256 tokens
per context at 60.54/61.44/61.23 aggregate tok/s. The 12-rank checker
passed; all eight output stream hashes were distinct. This checks distinct
prompt identities and states at 32K; projection work inside a TP4 group
still runs one context at a time.

Reusing those eight distinct FP32 states for **8,192 output tokens per
context** passed the 12-rank checker again, with eight distinct stream
hashes. The 3/3/2 groups sustained **58.05/58.94/59.19 aggregate tok/s**
over 423.40/417.00/276.81 s of decode. Their separate state imports took
131.03/129.87/79.05 s. The busiest group therefore completes its three
8K outputs in about seven minutes after import; the whole run took
592.80 s including startup and import. This validates long round-robin
decode for eight distinct 32K contexts, not task quality or the requested
deep-context shapes.

The capacity planner (`q38d_capacity.py --json`) uses binary token depths,
the measured 6.02 GiB TP4 baseline, PP12 stage cuts, output reserve, and
load/scratch headroom. It now recognizes packed PP12 KV and version-2 handoff
as implemented, so all three shapes pass its **memory and capability** gates.
That status is not proof of real-input prefill, output quality, or throughput
at the requested depths.

## Commands and remaining checks

Build on Fugaku with `make -C a64fx/llm q38d CC=fcc Q38D_TP=1` and
`bash a64fx/llm/q38p/build_pp.sh`; set `TMPDIR=/local/q38/tmp` for compiler
scratch. The group runner accepts `Q38_GROUP_PRUNE=1`, `Q38_GROUP_KV_I6=1`,
`Q38_GROUP_CONTEXTS_BY_GROUP=3,3,2` and `Q38_GROUP_WARM_DEPTH=1048575`.
Use `--warm-kv-depth` only for synthetic capacity/timing screens.

For deep PP screening, set `Q38P_KV_I6=1 Q38P_ATTN_CACHE=0
Q38P_ATTN_QTILE=1 Q38P_ATTN_PV_INT16=0` with
`a64fx/llm/q38p/run_depth.sh`. For a compressed handoff, set
`HANDOFF_KV_I6=1 HANDOFF_TPS=4` with `run_handoff.sh`.
`HANDOFF_PREFILL_KV_I6=0` selects FP32 PP with compressed export for shorter
prompts; the default selects packed PP attention.

`run_multictx_handoff_12n.sh NEW_RUN_DIR PROMPT_LIST PROMPT_TOKENS GEN CHUNK`
evaluates 6–32 distinct prompt files sequentially on PP12, exports one state
per prompt, and runs their continuations concurrently in three TP4 groups.
`PROMPT_LIST` has one path per line. The default retains FP32 K/V for the
32K coding path; `Q38_MULTI_KV_I6=1` selects packed PP K/V and an INT6
handoff for depths that exceed FP32 HBM capacity. The runner divides contexts
as evenly as possible across the groups and checks all 12 rank outputs.
Sequential PP12 prefill is included in its wall time; this runner does not
fuse model projections across context slots during decode.
The six-context packed-KV smoke run (two distinct prompts per TP4 group)
evaluated 1,024 real tokens per prompt at 569.26–580.34 PP12 tok/s total,
exported version-2 states in 2.07–2.31 s each, then generated 64 tokens
per context at 85.31/85.90/86.55 aggregate tok/s by group. All 12 ranks
passed and all six output hashes differed. This validates the runner's
packed prefill, state-list import, and grouped decode wiring at short depth;
it does not resolve packed-prefill quality or throughput at 256K–1M.

Six distinct short C-review prompts were then exported in separate PP12
version-2 states. Three TP4 groups imported two states each, verified every
first token, and generated eight tokens per context. All six output hashes
exactly matched a separate direct TP4 run on the same six prompts; all 12
ranks passed `check_q38d_batch.py`. `--context-state-list` takes one state
directory per slot and must be paired with `--context-prompt-list` so the
state header's prompt identity is checked. The group runner accepts
`Q38_GROUP_CONTEXT_STATE_PREFIX` for one manifest per group.

The requested full evaluated 1M×8 / 512K×16 / 256K×32 coding-agent
workloads have not run. Long real-input accuracy and performance, and
long-generation quality of INT6 KV, are the next gates. Results above are preserved in the
job's `tmp/q38d-groups/` and `tmp/q38p/` directories and wrapper logs in
`/local/q38/tmp/`.

The non-power-of-two TP9 collective passed 100 FP32 reductions of 5,120
elements at 46.60 µs mean; TP12 passed the same check at 53.80 µs mean.
These are communication tests, separate from the TP4 grouped inference rates.
