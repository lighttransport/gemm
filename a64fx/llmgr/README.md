# llmgr — control HTTP port for the A64FX LLM runners

Turns a Fugaku allocation into an interactive development box. One HTTP port on
the head compute node lets you compile, stage weights, start/stop `mpiexec`
runners, tail their logs, run inference and capture fapp profiles — without
submitting a new job per experiment.

Reachable from a Fugaku frontend through a supervised SSH reverse tunnel.

```
frontend loginN            compute head node            11 other nodes
  curl localhost:21374 ──ssh -R──▶ llmgr :21274 ──▶ mpiexec ──▶ ranks 1..11
                                       │
                                       └─▶ laguna --serve :8080  (proxied)
```

## What it is (and is not)

llmgr is a **generic supervisor**, not an MPI rank. It runs outside MPI and spawns
runners as `setsid` process groups, so one `killpg` reaps a whole `mpiexec`
tree. No C code was changed to build it.

Inference is **proxied** to the runner's own HTTP server
(`a64fx/laguna-s21/laguna_serve.inc`). That is deliberate: that file's rank-0
invariant — *every accepted connection must reach `lsrv_bcast` exactly once, or
the other ranks strand inside the allreduce* — stays inside the C code where it
is already correct. llmgr never opens a rank-0 socket of its own.

Arbitrary shell is delegated wholesale to `tools/bash_http_server.py` under
`/bash/*` (persistent PTY sessions, NDJSON streaming, `cwd` and shell variables
survive between calls).

Python 3 standard library only — Fugaku's system `python3` is 3.6.

## Quick start (inside an existing interactive allocation)

```sh
a64fx/llmgr/run_llmgr.sh --daemon --verbose      # binds 127.0.0.1:21274, no auth

export LLMGR_URL=http://127.0.0.1:21274
cd a64fx/llmgr
./llmgr_cli.py health
./llmgr_cli.py sh 'make -C ../laguna-s21 all CC=fcc OPENMP=1'   # streams
./llmgr_cli.py build  --model laguna --variant int4
./llmgr_cli.py stage  --model laguna              # ~8 min for 12x13 GB
./llmgr_cli.py stage-status --model laguna
./llmgr_cli.py start  --model laguna --port 8080 --maxpos 8192
./llmgr_cli.py ps                                 # wait for state=ready
./llmgr_cli.py gen run-3 --ids 2,818,1841 --max-new 32
./llmgr_cli.py log run-3 --follow
./llmgr_cli.py stop run-3
```

The runner has no tokenizer: `/generate` is ids-in / ids-out. Encode and decode
on the client with `a64fx/laguna-s21/tools/laguna_tok.py`:

```sh
LAGUNA_TOKENIZER=~/models/laguna-s21-int4/tokenizer.json \
  python3 ../laguna-s21/tools/laguna_tok.py encode "The capital of France is" --bos
```

BOS (id 2) is required — without it the model degenerates to copying the last
token.

## Batch job with the reverse tunnel

From the Mutagen-synced workstation, use the helper that discovers and pins the
actual load-balanced Fugaku frontend:

```sh
./a64fx/llmgr/submit_llmgr_over_ssh.sh
```

See [RUNBOOK.md](RUNBOOK.md) for the workstation local-forward command and the
complete Laguna procedure. From an already pinned Fugaku frontend, submit the
job script directly:

```sh
pjsub --no-check-directory a64fx/llmgr/pjsub_llmgr_12n.sh
# then on the frontend it tunnelled to (login1 by default):
curl localhost:21374/health
```

The tunnel is supervised: if it drops it is re-established, re-selecting a
reachable frontend, up to `MAX_RETRY` consecutive failures. Knobs:
`LOGIN_NODE`, `FRONTEND_SSH_TARGETS`, `FRONTEND_PORT`, `SERVER_PORT`,
`MONITOR_INTERVAL`, `KEEPALIVE_SECONDS`.

**Frontend targets must be FQDNs.** From a compute node `login1` and `fn01sv03`
do not resolve; `login1.fugaku.r-ccs.riken.jp` does.

## HTTP API

Bodies and responses are JSON. Long operations return `202` with a child `id`
immediately; follow them with `/runner/<id>/log`. Requests need no auth unless
`LLMGR_TOKEN` is set, in which case every request must carry
`Authorization: Bearer $LLMGR_TOKEN`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | uptime, job id, every child's state |
| GET | `/models` | adapters, variants, whether each supports serve |
| GET | `/nodes` | PJM env, per-node host/mem/`/local` (`?fanout=0` for head only) |
| GET | `/runner` | list children |
| GET | `/runner/<id>/log?tail=N&follow=1` | tail, or stream until the child exits |
| GET | `/stage/status?model=…` | per-node file count + bytes of the staged blobs |
| GET | `/profile/<id>/artifacts` | the fapp CSV/text reports a profile produced |
| POST | `/build` | `{model, variant, clean}` |
| POST | `/stage` | `{model, variant, stage_dir, model_dir, np}` |
| POST | `/runner/start` | `{model, mode:serve\|generate, port, maxpos, layers, np, extra:[…], env:{…}}` |
| POST | `/runner/stop` | `{id, grace}` |
| POST | `/generate` | `{id, ids:[…], max_new, sample, temp, top_k, top_p, seed}` — proxied |
| POST | `/profile` | `{model, ids, max_new, event, np}` — fapp-wrapped run |
| POST | `/kv` | `{action:save\|load\|clear\|stats, id, path}` |
| POST | `/shutdown` | stop all children, then exit |
| POST | `/bash/{session,run,interrupt,close}` | see `tools/README.md` |
| GET | `/bash/{health,sessions}` | ditto |

## Models

| model | variants | serve? | launcher |
|---|---|---|---|
| `laguna` | `int4` (default), `bf16`, `fp8` | yes | `a64fx/laguna-s21/run_laguna_s21_12n.sh` |
| `gemma4` | `tp` (default), `pp` | no — one-shot only | `a64fx/gemma4-mn/run_gemma4_tp.sh` / `run_gemma4_pp.sh` |
| `k3` | `partial` | no — measured one-shot only | `a64fx/k3/run_k3_ep.sh` |

K3's HTTP interface is the llmgr control API, not a semantic completion API:
the current runner has real TP MXFP4 expert slices but still lacks the tokenizer,
embedding, complete dense/shared path, and LM head. On a K3 llmgr allocation,
`POST /stage` performs a true stage-only operation; `POST /runner/start` with
`mode=generate` launches a bounded partial decode and exposes health, logs, stop,
and profiling through llmgr. `POST /bash/session` can edit sources and rebuild,
but apply a fix by stopping the active MPI child and restarting it against the
retained rank-local stage—already-running machine code is never hot-patched.

The dedicated 96-node wrapper is `pjsub_llmgr_k3_96n.sh` and uses frontend port
21375 by default. The K3 launcher honors llmgr's `MPIEXEC_OF_PROC` prefix, so
rank output is folded into the child log and the supervisor can identify and
reap a detached `plexec` tree reliably.

The K3 stop/restart path was tested on 12 nodes: an HTTP stop reached all ranks
after 7,481 steps, produced coordinated `signal-term` health output, left no MPI
survivor, and a fresh distributed run passed on the same allocation.

Adding a model is one dict in `models.py`. Adapters deliberately do **not** set
performance-critical environment: the launchers already encode it
(`OMP_NUM_THREADS=47` — a pinned thread on every one of the 48 cores costs ~40%;
`XOS_MMM_L_PAGING_POLICY=demand:demand:demand`; and for laguna the *absence* of
`FLIB_BARRIER=HARD`, which would force 48 threads and ~4x-slow the matvecs).
Only env you pass explicitly in `env:{…}` is layered on top.

The adapter/supervisor interface is documented in [RUNBOOK.md](RUNBOOK.md).
It is intentionally command-oriented: a runner can be a C binary, shell
launcher, or another MPI program as long as it supplies the standard operation
builders and a passive readiness check for serve mode.

## Never probe a runner to see if it is ready

This one cost a hung 12-node job to learn, and it is the single most important
thing to know before extending llmgr.

Every accepted connection on rank 0 — **including a `GET /health`** — drives a
collective across all ranks (`laguna_serve.inc`: "every accepted connection
reaches `lsrv_bcast` exactly once"). Rank 0 starts accepting the moment *it*
finishes loading its weights. The other ranks are not there yet: measured on 12
nodes, load times spread **184s to 251s** while rank 0 opened its socket at
206s. One-second `/health` probes in that window ran four collectives that the
slower ranks never joined, and the job never recovered:

```
rank  0: tp_ar: rank 0 bcast timeout sid=4 want=2 got=1
rank  1: tp_ar: rank 1 wait  timeout sid=0 want=2 got=1
rank  8: tp_ar: rank 8 wait  timeout sid=3 want=2 got=1
```

Every rank stranded at a *different* sid — the signature of a desynchronised
collective. The runner exited 1 and the request got a bare TCP close.

So llmgr's readiness check is **passive**: `Adapter.readiness()` counts how many
`laguna_stderr_rank<NN>.txt` files in the launcher's `gen_<timestamp>/` run
directory contain "loaded in", requires all `np` of them plus rank 0's "serving
HTTP on port", and then still waits `READY_SETTLE` (10s) before the child is
marked `ready`. No request is ever sent to a runner that is not `ready`.

If you add a model with a serve mode, give its adapter a `readiness()` that
reads files. Do not reach for a health endpoint.

## Killing a runner takes more than killpg

`mpiexec` re-execs into `plexec`, which puts itself in its own session. The
launcher's process group therefore does **not** contain the ranks: `killpg`
kills the direct child, `stop` cheerfully reports `exit_code: -15`, and the
whole `mpiexec` → `plexec` → `fapp` → runner tree keeps running and keeps
holding all 12 nodes. Both `mpiexec` and `plexec` also ignore `SIGTERM` — only
`SIGKILL` moves them — and a surviving `plexec` poisons the next launch:

```
[ERR.] PLE 0008 plexec must be started sequentially.(nid=own node)(CODE=3746,0,0)
```

So `Child._sweep_tree()` finds survivors by **command line**, not by process
tree (the tree is exactly what got detached). Every mpiexec llmgr launches
carries `-of-proc <child log dir>/rank`, unique per child, which is a precise
handle on that child's launchers; leaf ranks are then matched by the runner
binary path, but only after the launchers are gone, so a concurrently running
child's ranks are never touched.

A graceful `POST /shutdown` (the path `stop` tries first for laguna) avoids all
of this — the ranks exit on their own. The sweep is for when that is not
available or does not work.

## Two Fugaku facts this depends on

**Rank stdout is invisible to the launching process.** `mpiexec -np 12 hostname`
piped, redirected to a file, or run under a PTY all come back *empty* — which is
why every runner in this repo writes rank files instead of printing. The working
mechanism is `mpiexec -of-proc PREFIX`, which writes `PREFIX.<step>.<rank>`, and
that prefix must be on the **shared** filesystem (`/tmp` and `/local` are
node-local, so only rank 0's file would be visible).

Both launchers therefore honour an optional `MPIEXEC_OF_PROC` env var; unset,
their behaviour is byte-for-byte unchanged. llmgr sets it per child and folds
the resulting files into that child's `log.txt`, tagged `[0]`…`[11]`.

**`pjsub` is not on a compute node's PATH.** llmgr manages only the allocation
it lives in; submitting jobs stays a frontend action.

Corollaries worth knowing:

- `mpiexec`'s exit status is not a reliable success signal when the real output
  is a file. `/profile` checks that `tofu_topo.txt` actually appeared rather
  than trusting `rc == 0`.
- Child ids restart at 1 with each llmgr process, so a re-used log directory
  from an earlier instance would have its `rank.*` files re-read from offset 0
  and folded into the new child's log. `Child.start()` moves any pre-existing
  directory to `<id>.prev-<mtime>` rather than deleting it.

## Security

The port executes arbitrary shell as you, so what protects it is **where it is
bound**, not a password:

- It binds **loopback only**. A non-loopback bind requires `--allow-public-bind`;
  don't. Compute nodes sit on shared 10.x Tofu/admin subnets (`eno1`, `tofu0`,
  `tofu1`), so `0.0.0.0` would expose the shell to other users' jobs.
- Exposure to a frontend is via `ssh -R` to `127.0.0.1` on that frontend only,
  and Fugaku is not reachable from the internet.

Bearer auth is therefore **off by default**. Set `LLMGR_TOKEN` (server and
client) to turn it on; the server prints `[auth: token]` or `[auth: none]` at
startup so the mode is never ambiguous. Worth turning on if you forward the port
to a **shared** frontend, where any other logged-in user could reach
`127.0.0.1:21374` and get a shell as you.

## Known seams

- **KV cache** — the laguna runner has no KV endpoint on the wire, so
  `/kv save|load` return the start-time flags to restart with rather than
  pretending a live API exists; `clear` means restart. `stats` reports what is
  observable from the runner's `/health`.
- **`/nodes` and `/stage/status` fanout need the nodes free.** While a serve
  runner owns them, `mpiexec` blocks or fails; the response says so. Pass
  `?fanout=0` for head-node-only answers.
- **Profiling bypasses the launcher.** fapp collects one PMU dataset per
  process, so `/profile` wraps the per-rank binary under `mpiexec` directly
  (`fapp -C -d prof_rank<N>_pa1 -Icpupa,nompi -Hevent=…`). Consequence: weights
  must already be staged, and `ids` must be a pre-tokenized file.
- **gemma4 has no serve mode** — `mode=generate` only; read the log for output.

## Files

```
llmgr_server.py     the daemon
models.py           per-model adapters
llmgr_cli.py        client (frontend or head node)
run_llmgr.sh        start inside an existing allocation
pjsub_llmgr_12n.sh  batch job + supervised reverse tunnel
logs/<child-id>/    log.txt, cmd.txt, rank.<step>.<rank>
state/              llmgr.<jobid>.env  (host/port/pid discovery)
prof/               fapp working directories
```
