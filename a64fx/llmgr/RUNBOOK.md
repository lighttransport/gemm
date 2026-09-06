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

### Submit from the Mutagen-synced workstation (preferred)

Run this in the local checkout, not in an `ssh fugaku` shell:

```sh
./a64fx/llmgr/submit_llmgr_over_ssh.sh
```

The helper makes one SSH connection through the `fugaku` alias, observes the
actual frontend selected by the load balancer (for example `fn01sv06`), maps it
to its reachable FQDN (`login6.fugaku.r-ccs.riken.jp`), and passes that exact
frontend to `pjsub`. Discovery and submission must share an SSH session:
separate `ssh fugaku` calls can land on different frontends, while the reverse
forward exists on only one of them.

Useful overrides:

```sh
JOB_SCRIPT=a64fx/llmgr/pjsub_llmgr_4n.sh FRONTEND_PORT=21375 \
  ./a64fx/llmgr/submit_llmgr_over_ssh.sh

# preemptible spot allocation; use a distinct port if another job is queued
RSCGRP=spot-small NODES=12 ELAPSE=06:00:00 FRONTEND_PORT=21375 \
  ./a64fx/llmgr/submit_llmgr_over_ssh.sh

REMOTE=my-fugaku-alias \
REMOTE_REPO=/vol0006/mdt0/data/hp250467/work/gemm/glm5-1 \
  ./a64fx/llmgr/submit_llmgr_over_ssh.sh
```

The output records `SYNCED_HEAD`, `LOGIN_HOST`, `FRONTEND`, `FRONTEND_PORT`, and
the PJM job ID. Save the frontend with the job ID; it is connection state, not
an interchangeable login host.

After the allocation starts, forward the pinned frontend's loopback port to
the workstation. Substitute the `FRONTEND` printed by the helper:

```sh
ssh -N -L 21374:127.0.0.1:21374 login6.fugaku.r-ccs.riken.jp
# another local terminal
curl -sS http://127.0.0.1:21374/health | python3 -m json.tool
```

The first `21374` is the workstation port and may be changed if occupied. The
second is the frontend port recorded at submission.

### Submit from an existing Fugaku frontend

Submit from a Fugaku frontend:

```sh
pjsub --no-check-directory a64fx/llmgr/pjsub_llmgr_laguna_fp8_12n.sh
```

This requests the best established shape: 12 A64FX nodes, one EP rank per
node, `freq=2000`, `eco_state=0`, `retention_state=0`, and 80 GiB node-local
storage. The job exposes the llmgr port through a supervised SSH reverse
tunnel on `localhost:21374` of the selected login frontend. When submitting
manually, pass `LOGIN_NODE` or the `FRONTEND_*` variables explicitly; the batch
script otherwise defaults to `login1`, which may not be your current frontend.

## Operate the allocation

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
load. The direct client below must run on the head compute node because the
reverse tunnel exposes llmgr port 21374, not runner port 8080:

```sh
LAGUNA_TOKENIZER=~/models/laguna-s21-fp8/tokenizer.json \
  python3 a64fx/laguna-s21/tools/laguna_cli.py \
  --port 8080 chat 'What is the capital of France?' --no-think
```

From the workstation, ask llmgr's head-node shell to run that client. This
keeps both tokenization and port 8080 inside the allocation while streaming the
decoded answer back over port 21374:

```sh
python3 a64fx/llmgr/llmgr_cli.py sh \
  'cd a64fx/laguna-s21 && LAGUNA_TOKENIZER=$HOME/models/laguna-s21-fp8/tokenizer.json python3 tools/laguna_cli.py --port 8080 chat "What is the capital of France?" --no-think' \
  --command-timeout 3600
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
