# Phase-2 Tensor-Parallel (TP) — Gemma-4 12B BF16 on A64FX

Megatron-style TP across `tp_size` nodes. Gated by env, default OFF (PP/single-node
paths untouched). Correctness-first: replicate the small tensors, shard the big ones.

## Real dims
n_embd=3840, n_ff=15360, n_heads=16, head_dim 256(SWA)/512(full),
l_kvh 8(GQA/SWA) / 1(MQA full layers: 5,11,...,47), vocab=262144, 48 layers.
lm_head TIED to token_embd (no output.weight).

## Per-tensor sharding (rank r of tp_size)
| tensor        | scheme              | why |
|---------------|---------------------|-----|
| attn_q        | ROW-slice (Q heads) | big; each rank computes its head range [h0,h1)*hd |
| attn_k/attn_v | REPLICATE (full)    | small; every rank needs full KV for its Q heads (GQA) |
| attn_output   | COL-slice (head cols)| row-parallel; input is the rank's head outputs -> ALLREDUCE-SUM |
| ffn_gate/up   | ROW-slice (n_ff/N)  | bulk of weights; GELU on local slice |
| ffn_down      | COL-slice (n_ff/N)  | row-parallel -> ALLREDUCE-SUM |
| token_embd    | REPLICATE (full)    | needed for embed lookup (any token); lm_head uses a vocab ROW-RANGE of it |
| all norms, rope_freqs, ple_* | REPLICATE | tiny |

Head split: heads [r*16/N, (r+1)*16/N) (sizes differ by <=1 since 16%N!=0 for N=3,5,6).
o-proj / down all-reduce combine partial sums regardless of per-rank head/ff count.
KV replicated => each rank computes FULL k/v (replicated weights) and attends its Q heads.
Memory win = attn_q + attn_output + ffn_gate/up/down sharded (~the layer bulk).

## Forward wiring (transformer.h gemma4 branch ~5515-5700), gated `m->tp_size>1`
1. QKV: attn_q via tf_qmatvec_row_slice over [h0*hd,h1*hd); k/v full (replicated). q-norm/rope on local heads.
2. attention: loop rank's heads [h0,h1) vs full KV -> xb2 local-head cols.
3. o-proj: tf_qmatvec_col_slice_pool(attn_output, xb2_localcols, col=[h0*hd,h1*hd)) -> ALLREDUCE-SUM -> full n_embd.
4. gate/up: row-slice [r*n_ff/N,(r+1)*n_ff/N); GELU local; 
5. down: col_slice_pool over the same n_ff block -> ALLREDUCE-SUM.
6. lm_head: row-slice token_embd vocab [r*V/N,(r+1)*V/N); decode -> tp_allreduce_argmax.

## Correctness model
disjoint-output shards (attn_q/gate/up/lm_head) = bit-exact; contraction shards
(o-proj/down) = coherent via allreduce-SUM (fp reassoc -> argmax must still match).

## Build checklist (commit each)
- [ ] 1. TP stager: gemma4_stage.c "tp" mode = row/col slice writer (match table above)
- [ ] 2. SVE-threaded col-slice matvec (tf_qmatvec_col_slice_pool is single-threaded today)
- [ ] 3. forward: gate/up + down + allreduce  (validate MLP-only first if cheap)
- [ ] 4. forward: QKV + attn + o-proj + allreduce
- [ ] 5. vocab-parallel lm_head + tp_allreduce_argmax
- [ ] 6. gemma4_tp_runner.c: tp_comm_init + transformer_set_tp + drive prefill/decode
- [ ] 7. validate argmax==single-node (disjoint bit-exact, reduced coherent); sweep tok/s 2-6n

## Reuse
tp_allreduce.h: tp_comm_init(c,vcq,peer_vcq,rank,nprocs,max_count,barrier_fn),
tp_allreduce_sum/max/argmax. transformer.h: transformer_set_tp, resize_kv_for_tp,
tf_qmatvec_row_slice, tf_qmatvec_col_slice_pool. Runner bootstrap = gemma4_pp_runner.c.

## Session-safety (this host is a 32GB A64FX node shared with the agent)
Heavy work offloads to OTHER nodes via mpiexec vcoordfile (exclude login 0,0,0).
Stage in <=1GB fadvise chunks. Forward wiring is pure local source (no OOM risk) ->
do + compile + commit those steps before any compute run. pkill -9 mpiexec/plexec
after a crash; re-stage + re-run topo helper after a restart (alloc gets new nodes).
