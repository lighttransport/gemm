# Remote A64FX development with bash-over-HTTP

This procedure runs a persistent, stateful Bash service inside a Fugaku PJM
allocation and reaches it from the local workstation through two loopback-only
SSH forwards.

```text
local client: 127.0.0.1:42386
    | ssh -L
    v
login1: 127.0.0.1:32386
    | ssh -R (opened by the PJM job)
    v
compute node: 127.0.0.1:21264
    |
    v
bash_http_server.py -> persistent Bash sessions
```

All three listeners use loopback addresses and the two network hops are SSH
forwards. HTTP bearer-token authentication is optional and disabled by default;
do not change any listener bind address to `0.0.0.0` or expose these ports
outside the SSH path.

## Prerequisites

- Local SSH configuration provides `ssh fugaku1` and pins it to
  `login1.fugaku.r-ccs.riken.jp`.
- The remote checkout is `$HOME/work/gemm/glm53f`.
- The compute-node account can SSH back to
  `login1.fugaku.r-ccs.riken.jp`. The job uses `BatchMode=yes` and
  `IdentitiesOnly=yes`.
- Local Python has `requests`. The compute-side server itself uses only the
  Python standard library and supports Fugaku's Python 3.6.8.

Run all local commands below from the Gemm repository root.

## Connection configuration

Copy `a64fx/tools/bash-over-http/setup.json.example` to
`${XDG_CONFIG_HOME:-$HOME/.config}/bash-over-http/setup.json` for user-wide
defaults. A project-local `.bash-over-http.json` in the repository root takes
precedence over the user file. Set `BASH_HTTP_CONFIG=/path/to/file.json` to
select an explicit file. Environment variables such as `REMOTE` and
`LOCAL_PORT` still override JSON settings.

The JSON template covers the local checkout and port, SSH hostname, remote
checkout and forwarded port, server bind/port, optional model storage paths,
and PJM job settings. For
example, project-specific connection settings can be kept in:

```json
{
  "local": { "dir": ".", "port": 42386 },
  "remote": {
    "ssh_host": "fugaku1",
    "hostname": "login1.fugaku.r-ccs.riken.jp",
    "port": 32386,
    "dir": "$HOME/work/gemm/glm53f"
  },
  "server": { "host": "127.0.0.1", "port": 21264 },
  "storage": {
    "model_dir": "$HOME/models/q38fn/bf16",
    "local_dir": "/local/q38fn",
    "staging": "cp"
  }
}
```

The deployment command should be run from `local.dir`; it synchronizes the
bundle to `remote.dir`. The project config is synchronized alongside the
bundle when present.

## Deploy the bridge

Synchronize only the bridge and this procedure. The commands do not delete or
overwrite unrelated remote artifacts.

```bash
REMOTE=fugaku1
REMOTE_DIR='~/work/gemm/glm53f'
rsync -av a64fx/tools/bash-over-http/ \
  "$REMOTE:$REMOTE_DIR/a64fx/tools/bash-over-http/"
rsync -av a64fx/remote-dev-procedure.md \
  "$REMOTE:$REMOTE_DIR/a64fx/remote-dev-procedure.md"
# If this checkout has project-local settings, sync them too:
test ! -f .bash-over-http.json || rsync -av .bash-over-http.json \
  "$REMOTE:$REMOTE_DIR/.bash-over-http.json"
```

When the scripts or project config change, repeat these `rsync` commands before
submitting a new job. A running server keeps using the code loaded when its job
started. Replace `REMOTE` and `REMOTE_DIR` with the values from your config
when using another login host or checkout.

## Open the local forward

```bash
a64fx/tools/bash-over-http/open_local_tunnel.sh
```

The defaults are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `REMOTE` | `fugaku1` | Local SSH target for login1 |
| `LOCAL_PORT` | `42386` | Local HTTP port |
| `REMOTE_PORT` | `32386` | Loopback port on login1 |
| `CONTROL_DIR` | `${XDG_RUNTIME_DIR:-/local}/clair-bash-http-${USER}` (or `tmp/` if `/local` is unavailable) | Private local state directory |

The script uses an SSH ControlMaster and records the pinned login target and
ports in `state.env`. `ExitOnForwardFailure=yes` makes an occupied local port a
hard error.

To recreate a dropped local tunnel automatically, run the watcher in a tmux
pane or in the background:

```bash
state_root=${XDG_RUNTIME_DIR:-/local}; [[ -d "$state_root" ]] || state_root=tmp; state_dir="$state_root/clair-bash-http-${USER}"
mkdir -p "$state_dir"
nohup a64fx/tools/bash-over-http/watch_local_tunnel.sh \
  >"$state_dir/watch.log" 2>&1 &
```

## Submit a job

The standard one-node, 12-hour remote-development allocation is:

```bash
REMOTE=login1.fugaku.r-ccs.riken.jp \
FRONTEND_SSH_TARGET=login1.fugaku.r-ccs.riken.jp \
NODES=1 ELAPSE=12:00:00 \
  a64fx/tools/bash-over-http/submit_bash_http_job.sh
```

The wrapper submits `pjsub` through login1 and the job opens the reverse SSH
tunnel back to that same login node. It prints `JOB_ID=<id>` after submission;
use that ID in the readiness and cleanup commands below.

For boost-eco execution (2.2 GHz, 10% higher HBM2 bandwidth, and FLB disabled),
select the named mode:

```bash
A64FX_MODE=boost-eco NODES=1 ELAPSE=08:00:00 \
  a64fx/tools/bash-over-http/submit_bash_http_job.sh
```

This changes the PJM resource selection to `freq=2200,eco_state=2`.

The submission wrapper executes `pjsub` through the configured SSH target with
these resource settings for the 12-hour example:

```text
-g hp250467
-L freq=2000,eco_state=0,rscgrp=small,node=1,elapse=12:00:00
--no-check-directory
-x PJM_LLIO_GFSCACHE=/vol0004
--llio localtmp-size=87Gi
```

The submitted job ID is stored in `jobid` in the private local state directory.

To allocate between 1 and 12 nodes, set `NODES`:

```bash
NODES=4 ELAPSE=08:00:00 \
  a64fx/tools/bash-over-http/submit_bash_http_job.sh
```

The server runs on the job's initial compute node. Commands submitted through
the service execute inside the allocation and may launch work across the other
allocated nodes with the site's normal `mpiexec` workflow.

The main configurable values are:

| Variable | Default | Constraint or purpose |
| --- | --- | --- |
| `PROJECT_ID` | `hp250467` | PJM group |
| `RSCGRP` | `small` | PJM resource group |
| `NODES` | `1` | Integer from 1 through 12 |
| `A64FX_MODE` | `normal` | `normal` or `boost-eco` |
| `ELAPSE` | `08:00:00` | `HH:MM:SS`; use up to `12:00:00` for the standard batch job |
| `GFSCACHE` | `/vol0004` | `PJM_LLIO_GFSCACHE` value |
| `LOCALTMP_SIZE` | `87Gi` | LLIO local temporary capacity |
| `FRONTEND_SSH_TARGET` | `login1.fugaku.r-ccs.riken.jp` | Compute-to-login reverse SSH target |
| `FRONTEND_PORT` | recorded `REMOTE_PORT`, normally `32386` | Reverse-forward listener on login1 |
| `SERVER_HOST` | `127.0.0.1` | Compute-side server bind address |
| `SERVER_PORT` | `21264` | Compute-side server port |
| `REMOTE_REPO` | `$HOME/work/gemm/glm53f` | Remote checkout used for submission |
| `MAX_RETRY` | `10` | Consecutive tunnel failures before job exit |
| `MONITOR_INTERVAL` | `15` | Tunnel check interval in seconds |
| `KEEPALIVE_SECONDS` | `0` | `0` means run until PJM elapse |

If login1 or the ports are changed, use matching values on both sides:

```bash
REMOTE=some-login-alias LOCAL_PORT=42387 REMOTE_PORT=32387 \
  a64fx/tools/bash-over-http/open_local_tunnel.sh

REMOTE=some-login-alias \
FRONTEND_SSH_TARGET=login2.fugaku.r-ccs.riken.jp \
FRONTEND_PORT=32387 NODES=1 ELAPSE=08:00:00 \
  a64fx/tools/bash-over-http/submit_bash_http_job.sh
```

Use a hostname for `FRONTEND_SSH_TARGET`, not an internal IP address. Keep it a
single hostname: commas in `pjsub -x` values are interpreted as separators.

## Run an interactive job (maximum six hours)

Interactive allocations are limited by Fugaku's `int` resource group to at
most **06:00:00**. A 12-hour allocation must use the batch launcher instead;
`run_bash_http_interactive.sh` rejects any larger `ELAPSE` value before it
contacts `pjsub`.

Use the interactive launcher when the batch `small` queue is delayed. It asks
the `int` resource group for an immediately usable allocation and waits at most
600 seconds; it does not leave a queued batch job behind after that wait
expires.

```bash
NODES=1 ELAPSE=06:00:00 WAIT_TIME=600 \
  a64fx/tools/bash-over-http/run_bash_http_interactive.sh
```

Set `A64FX_MODE=boost-eco` on the same command to request
`freq=2200,eco_state=2` for an interactive allocation.

The launcher runs the following PJM form through `ssh fugaku1` and feeds the
same compute-side server/reverse-SSH supervisor to its standard input:

```text
pjsub --interact -g hp250467 \
  -L "freq=2000,eco_state=0,rscgrp=int,node=1,elapse=06:00:00" \
  --sparam "wait-time=600" --no-check-directory \
  -x PJM_LLIO_GFSCACHE=/vol0004 --llio localtmp-size=87Gi \
  < a64fx/tools/bash-over-http/pjsub_bash_http.sh
```

Additional `-x` options carry ports, hostname, working directory, and
tunnel-supervisor settings. The launcher enforces `ELAPSE <= 06:00:00` and
validates `NODES` as 1 through 12.

The foreground launcher stays attached for the life of the interactive
allocation. Run it in a local tmux session when it must survive terminal
disconnects:

```bash
state_root=${XDG_RUNTIME_DIR:-/local}; [[ -d "$state_root" ]] || state_root=tmp; state_dir="$state_root/clair-bash-http-${USER}"
mkdir -p "$state_dir"
tmux new-session -d -s clair-a64fx-interactive \
  "cd '$PWD' && env NODES=1 ELAPSE=06:00:00 WAIT_TIME=600 \
   a64fx/tools/bash-over-http/run_bash_http_interactive.sh \
   >'$state_dir/interactive.log' 2>&1"
```

Watch startup and obtain the interactive PJM job ID from the log:

```bash
tail -f "$state_dir/interactive.log"
```

Only one bridge may own a given login-node port. Do not overlap interactive
and batch bridges on port 32386. Wait for one to end, cancel it, or give the
second bridge a different matching `REMOTE_PORT` and `FRONTEND_PORT`.

### Concurrent bash-over-HTTP jobs

Every bridge needs a unique port on the login node. Use `PORT_OFFSET` to
increment both ends of the configured port pair together:

```bash
# Job 0: local 42386 -> login1:32386 (the configured default)
# Job 1: local 42387 -> login1:32387
PORT_OFFSET=1 LOCAL_PORT=42386 REMOTE_PORT=32386 \
  a64fx/tools/bash-over-http/open_local_tunnel.sh
PORT_OFFSET=1 NODES=4 ELAPSE=06:00:00 WAIT_TIME=600 \
  a64fx/tools/bash-over-http/run_bash_http_interactive.sh
```

The launcher computes `local_port = configured_local_port + PORT_OFFSET` and
`reverse_port = configured_remote_port + PORT_OFFSET`. Use the same offset for
`open_local_tunnel.sh` and the job submitter, and use a separate
`CONTROL_DIR` for each concurrent local tunnel. For example, use offsets 0,
1, and 2 for three jobs. Do not reuse an offset until its job and tunnel have
terminated.

The interactive procedure is:

1. Choose an unused port offset and a per-job control directory.
2. Open the matching local forward with `open_local_tunnel.sh`.
3. Submit the interactive job, for example:

   ```bash
   PORT_OFFSET=1 CONTROL_DIR=tmp/bash-http-4n \
     NODES=4 ELAPSE=06:00:00 WAIT_TIME=600 \
     a64fx/tools/bash-over-http/run_bash_http_interactive.sh
   ```

   Keep this command attached (or run it in `tmux`). It waits up to 600
   seconds for the allocation and then keeps the reverse tunnel and server
   alive for the job lifetime.
4. Verify `http://127.0.0.1:$((42386 + PORT_OFFSET))/health` locally, then use
   that port for the client.

Interactive jobs are limited to six hours; a 12-hour job must use the batch
launcher. Both launchers pass the selected reverse port to the compute-side
supervisor.

## Stage model data inside the A64FX allocation

Fugaku shared storage is visible from the compute node, so model data does not
need to pass through bash-over-HTTP or be copied from the local workstation.
The `storage` configuration records the shared source and `/local` destination;
it is workload metadata, not a bridge transfer mechanism. Stage data from
inside the running allocation with `cp`:

```bash
MODEL_DIR="$HOME/models/q38fn/bf16"
LOCAL_DIR=/local/q38fn
mkdir -p "$LOCAL_DIR"
cp "$MODEL_DIR/model-00005-of-00131.safetensors" "$LOCAL_DIR/"
```

Use one shard or other bounded subset appropriate for the experiment. Check
`df -h /local` before copying; `/local` is allocation-local and is wiped when
the job ends. For the Qwen3.8 Flash n-gram probe, the copied shard contains an
800 MB n-gram tensor even though the complete safetensor shard is larger. The
probe then reads that tensor from `$LOCAL_DIR` into resident A64FX memory for
the HBM lookup benchmark. Do not use the local deployment `rsync` commands to
transfer model weights.

For the Q38FN tensor-parallel runner, stage logical tensor ranges directly
with `q38fn_tp_stage_mpi` instead of copying whole safetensor files. A full
12-rank shared-filesystem pass can exceed bash-over-HTTP's default five-minute
request timeout. The stager records every completed tensor in
`tp12-v*.manifest.partial`; rerunning the same command validates that prefix,
truncates an interrupted tensor tail, and resumes without rereading committed
tensors. Use a long request timeout for the initial run, while retaining the
partial files if a retry is necessary.

The interactive stdin path is protected by `ssh -n` on all nested
compute-to-login SSH calls, so health probes cannot consume the remainder of
the supervisor script. This path was validated end to end on 2026-08-23 with
interactive job 50790829.

## Wait for readiness

Inspect the saved job ID and PJM state:

```bash
state_root=${XDG_RUNTIME_DIR:-/local}; [[ -d "$state_root" ]] || state_root=tmp; state_dir="$state_root/clair-bash-http-${USER}"
job_id=$(<"$state_dir/jobid")
ssh fugaku1 "pjstat | grep '$job_id'"
```

The batch output and bridge logs live in the remote checkout. The job prints
`SENTINEL bash_http_batch_ready=OK` after both the local compute server and the
reverse tunnel pass health checks. Locate PJM's scheduler-named output file with
the job ID:

```bash
ssh fugaku1 \
  "cd ~/work/gemm/glm53f && find . -maxdepth 4 -type f \
   -name '*${job_id}*' -print"
```

Per-job files under `a64fx/tools/bash-over-http/job-logs/` are:

- `server.<jobid>.log`: HTTP server diagnostics.
- `tunnel.<jobid>.log`: reverse SSH failures and reconnects.
- `runtime.<jobid>.env`: compute hostname, ports, and log paths.

PJM output can be LLIO-buffered, so the local health check is the definitive
readiness test.

## Use and validate the endpoint

The client defaults to `http://127.0.0.1:42386`.

Run the complete demonstration:

```bash
python3 a64fx/tools/bash-over-http/bash_http_example.py
```

It verifies persistent working-directory and variable state, streamed output,
and output truncation. A smaller A64FX identity check is:

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, "a64fx/tools/bash-over-http")
import bash_http_client as bh

print(bh.health())
with bh.Shell() as shell:
    result = shell.run("hostname; uname -m; pwd", normalize=True)
    print(result.stdout, end="")
    if result.code != 0:
        raise SystemExit(result.code)
PY
```

Client configuration can also be supplied explicitly:

```bash
export CLAIR_BASH_HTTP_URL=http://127.0.0.1:42386
```

The API consists of:

- `POST /session` to create a persistent Bash session.
- `POST /run` to execute and stream NDJSON output.
- `POST /interrupt` to send Ctrl-C to a running command.
- `POST /close` to terminate a session.
- `GET /health` and `GET /sessions` for inspection.

Each session is serial: overlapping `/run` calls for the same session receive
HTTP 409. Idle sessions are reaped after 30 minutes by default. A request
timeout or output limit interrupts only the active command; the session remains
usable.

## Stop the service

Cancel the PJM allocation when it is no longer needed:

```bash
state_root=${XDG_RUNTIME_DIR:-/local}; [[ -d "$state_root" ]] || state_root=tmp; state_dir="$state_root/clair-bash-http-${USER}"
job_id=$(<"$state_dir/jobid")
ssh fugaku1 "pjdel '$job_id'"
```

Then close the local ControlMaster and its forward:

```bash
a64fx/tools/bash-over-http/close_local_tunnel.sh
```

The job traps normal termination and stops both the reverse SSH process and the
HTTP server. The private local state remains available for inspection; remove
that state manually only when it is no longer wanted.

## Troubleshooting

- **Local port 42386 is busy:** choose another `LOCAL_PORT`; the login-side
  `REMOTE_PORT` does not need to change unless 32386 is also occupied.
- **Reverse forward cannot bind 32386:** ensure no older bridge job owns the
  port, cancel that job, or choose the same new `REMOTE_PORT`/`FRONTEND_PORT`.
- **Reverse SSH reports too many authentication failures:** keep
  `IdentitiesOnly=yes`; it is already set by the job script.
- **Local health fails while the job is running:** confirm both tunnel ends use
  the same login node. The local `REMOTE` alias and compute-side
  `FRONTEND_SSH_TARGET` must identify that node.
- **Tunnel drops after a network interruption:** the job supervises `ssh -R`.
  Run `watch_local_tunnel.sh` to supervise and recreate the local `ssh -L` end.
