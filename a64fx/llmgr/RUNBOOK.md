# llmgr runner runbook

llmgr is a model-neutral supervisor. A runner adapter supplies commands and
environment; llmgr supplies process-group ownership, MPI rank log collection,
passive readiness, stop/reap, result capture, and (for serve runners) HTTP
proxying.

## Generic contract

Each adapter implements the standard operations below and returns
`(argv, env, cwd)`:

| Operation | Use |
|---|---|
| `build` | compile the runner and helper binaries |
| `stage` | place model shards on node-local storage |
| `serve` | start a long-lived HTTP runner |
| `generate` | start a bounded one-shot runner |
| `profile` | describe the per-rank binary invocation for fapp |

Serve adapters also implement passive `readiness(config, start_time)`. It must
read files or other out-of-band state. It must not connect to a rank-0 socket
while peers may still be loading, because Laguna's accepted connections are
collective operations.

The `/models` response advertises `runner_contract: llmgr.v1` and the modes
implemented by each adapter. A new model should add an adapter to
`a64fx/llmgr/models.py`; the supervisor does not need model-specific branches.

## Start the six-hour Laguna FP8 job

Submit from a Fugaku frontend:

```sh
pjsub --no-check-directory a64fx/llmgr/pjsub_llmgr_laguna_fp8_12n.sh
```

This requests the best established shape: 12 A64FX nodes, one EP rank per
node, `freq=2000`, `eco_state=0`, `retention_state=0`, and 80 GiB node-local
storage. The job exposes the llmgr port through a supervised SSH reverse
tunnel on `localhost:21374` of the selected login frontend.

Check the supervisor, not the runner, while weights load:

```sh
curl -sS localhost:21374/health | python3 -m json.tool
curl -sS localhost:21374/models | python3 -m json.tool
```

Start the FP8 server:

```sh
curl -sS -X POST localhost:21374/runner/start \
  -H 'Content-Type: application/json' \
  -d '{"model":"laguna","variant":"fp8","mode":"serve",'
      '"np":12,"port":8080,"maxpos":32768,"stage":true}'
```

Wait for `/runner` to report `state=ready`; do not probe port 8080 during the
load. Then use the tokenizer client from the frontend or through a shell in
the allocation:

```sh
LAGUNA_TOKENIZER=~/models/laguna-s21-fp8/tokenizer.json \
  python3 a64fx/laguna-s21/tools/laguna_cli.py \
  --port 8080 chat 'What is the capital of France?' --no-think
```

Stop cleanly before ending the allocation:

```sh
curl -sS -X POST localhost:21374/shutdown
```

## One-shot FP8 benchmark

For a bounded run, use the generic one-shot endpoint instead:

```sh
curl -sS -X POST localhost:21374/runner/start \
  -H 'Content-Type: application/json' \
  -d '{"model":"laguna","variant":"fp8","mode":"generate",'
      '"np":12,"max_new":128,"chat":"Explain A64FX in one paragraph",'
      '"stage":true,"pchunk":128}'
```

Follow the returned child with `GET /runner/<id>/log?follow=1`. The log and
the runner's `gen_*` directory contain prefill/decode timings.

## Gemma4 TP through llmgr

Use a fresh llmgr process after changing adapters, then start the TP one-shot:

```sh
curl -sS -X POST localhost:21374/runner/start \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","variant":"tp","mode":"generate",'
      '"np":4,"max_new":32,"stage":true,"exclude":"none",'
      '"mtp":"/home/u14346/models/gemma4/12b/mtp-gemma-4-12b-it.gguf",'
      '"prompt_ids":"/home/u14346/work/gemm/glm5-1/a64fx/gemma4-mn/prompt_tp.txt",'
      '"spec_k":4}'
```

The result is written as `gemma4_tp_result.txt` and appended to the child log.

## Safety and troubleshooting

- Keep llmgr bound to loopback. The port executes arbitrary shell through
  `/bash`.
- Use `POST /runner/stop` or `/shutdown`; do not kill only `mpiexec`, because
  Fugaku's `plexec` can detach and continue holding nodes.
- `stage/status` and node fanout may fail while another MPI runner owns the
  allocation. That is expected; pass `fanout=0` for head-only status.
- For a new serve runner, add passive rank-log readiness before enabling proxy
  traffic. Never use a health request as a load-completion probe.
