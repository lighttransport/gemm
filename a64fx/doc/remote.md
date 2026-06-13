# remote dev note

if os is ubuntu, filesystem is synched with mutagen. so modifying local file will be reflected to the remote.
remote dir is ~/work/gemm

the remote is 'fugaku'(use ssh to login)

## asset dir

only in remote. ~/models  and ~/data

## submitting job

run pjsub in the remote. as filesystem is synched with mutagen, use working dir in pjsub to ~/work/gemm
for model path and other asset paths, use ~/models and ~/data(use readlink -f in the remote to get abs path upon necessary)

For SSH submission from local files synced by Mutagen, submit from the remote
synced directory, for example:

```sh
ssh fugaku 'cd ~/work/gemm/ds4p && pjsub a64fx/utofu-tests/pjsub_mpi_hostid_16n.sh'
```

On Fugaku, batch stdout/stderr is not returned to the SSH command. After `pjsub`
prints a job ID, inspect the PJM output artifacts from the submit directory.
For a script named `job.sh`, the shell script's combined output may be written as
`job.sh.<jobid>.out`; `mpiexec` rank stdout can also be split under an
`output.<jobid>/` directory in the job's working directory.

```sh
ssh fugaku 'cd ~/work/gemm/ds4p && grep -E "SENTINEL|MPI_HOSTID|^# rank|^[0-9]+ " job.sh.<jobid>.out'
ssh fugaku 'cd ~/work/gemm/ds4p/a64fx/utofu-tests && find output.<jobid> -type f -maxdepth 4 -print -exec grep -H -E "MPI_HOSTID|wrote tofu_topo" {} \;'
```

For the uTofu/MPI host discovery flow, regenerate `tofu_topo.txt` for every
allocation with `mpiexec -np <N> ./tofu_topo_helper`; the helper writes rank to
Tofu 6D coordinate rows, and the batch output file records those rows plus the
rank to hostname lines.

## interactive jobs

For `pjsub --interact`, use an SSH TTY and run `script(1)` on the Fugaku
frontend to capture both typed commands and job output:

```sh
ssh -tt fugaku
cd ~/work/gemm/ds4p
script -q -f a64fx/interactive-test/interactive-1.transcript
./interactive-1.sh
```

Exit once from the compute shell to complete the interactive job, then exit the
logged frontend shell to finalize the transcript. The validated one-node FCC
smoke test is documented in `a64fx/interactive-test/note.md`.

For persistent LLM-driven interaction, run `tmux` on a pinned Fugaku frontend
node and drive it with `tmux send-keys` / `tmux capture-pane`. Do not assume
separate `ssh fugaku` commands reach the same frontend; during testing,
connections alternated between `fn01sv03` and `fn01sv05`, so an existing tmux
socket was invisible from the other frontend. Use nested SSH to the chosen
frontend, for example:

```sh
ssh fugaku 'ssh fn01sv03 "tmux capture-pane -pt ds4p-int-test -S -120"'
```

`a64fx/interactive-test/fugaku_tmux_mcp.py` wraps this workflow as a small
stdio MCP server with tools for `start`, `send`, `capture`, `tail`, `status`,
and `stop`. Configure it with `FUGAKU_FRONTEND`, `FUGAKU_TMUX_SESSION`, and
`FUGAKU_TRANSCRIPT` to match the pinned tmux session.

## batch shell bridge

`tools/bash_http_server.py` can be used from a one-node `pjsub` batch job by
starting the server on the compute node and opening an SSH reverse forward back
to a pinned frontend. The job wrapper is:

```sh
tools/pjsub_bash_http_1n.sh
```

Submit from the frontend:

```sh
ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory tools/pjsub_bash_http_1n.sh'
```

The wrapper:

- runs `python3 tools/bash_http_server.py` on the compute node
- chooses a reachable frontend SSH target from
  `10.6.90.13,10.4.128.23,134.160.188.27,fn01sv03`
- opens `ssh -R 127.0.0.1:21364:127.0.0.1:21264 ...`
- verifies `/health` through the reverse forward before printing readiness

Read the job's `.out` file and runtime env:

```sh
ssh fugaku 'cd ~/work/gemm/ds4p && sed -n "1,200p" pjsub_bash_http_1n.sh.<jobid>.out'
ssh fugaku 'cd ~/work/gemm/ds4p && sed -n "1,120p" tools/bash_http_job/runtime.<jobid>.env'
```

Then connect from the pinned frontend with the shipped client. Export
`DS4P_BASH_HTTP_TOKEN` to the same bearer token the job was submitted with (the
`TOKEN` / `DS4P_BASH_HTTP_TOKEN` env var consumed by the submit scripts) so it is
never written into the doc or shell history:

```sh
ssh fugaku 'ssh fn01sv03 "cd ~/work/gemm/ds4p && DS4P_BASH_HTTP_TOKEN=\$DS4P_BASH_HTTP_TOKEN python3 - <<\"PY\"
import os
import tools.bash_http_client as bh
bh.BASE_URL = \"http://127.0.0.1:21364\"
bh.AUTH_TOKEN = os.environ[\"DS4P_BASH_HTTP_TOKEN\"]
sid = bh.new_session()
r = bh.run(sid, \"hostname; pwd\", normalize=True)
print(r.stdout)
bh.close(sid)
PY"'
```

Validated on 2026-06-13 with job `49224567`:

```text
compute host: g30-1100c
frontend: fn01sv03
health: ok
client sentinel: SENTINEL bash_http_client=OK
```

For efficient local session management, do not rely on the load-balanced
`fugaku` alias after the tunnel is open. Pin one concrete login host — by default
`login1.fugaku.r-ccs.riken.jp` — and reuse that same host for both:

- the local `ssh -L` owner
- the compute node's reverse `ssh -R` target

The pinned frontend is configurable with `LOGIN_NODE` (1..8 →
`loginN.fugaku.r-ccs.riken.jp`, default 1), a single knob honored by all four
local-tunnel scripts so the local forward, the reverse tunnel, and teardown all
agree on the same node. `DS4P_BASH_HTTP_TOKEN` (the bridge bearer token) must be
set before launch — the submit script fails fast if it is unset. Procedure:

```sh
export DS4P_BASH_HTTP_TOKEN=...        # required: the bridge's bearer token
export LOGIN_NODE=1                    # optional: pin loginN.fugaku.r-ccs.riken.jp (1..8)

tools/open_bash_http_local_tunnel.sh   # local ssh -L + ControlMaster to loginN; writes state.env
tools/submit_bash_http_1n_over_ssh.sh  # pjsub the 1-node bridge (reverse tunnel to loginN)
python3 tools/test_bash_http_remote.py http://127.0.0.1:21364 "$DS4P_BASH_HTTP_TOKEN"
tools/close_bash_http_local_tunnel.sh  # tear down the local ControlMaster
```

`open_bash_http_local_tunnel.sh` creates a ControlMaster connection to
`login1.fugaku.r-ccs.riken.jp`, records the actual remote host/IP in
`/tmp/ds4p-bash-http/state.env`, and opens:

```text
local 127.0.0.1:21364 -> login1.fugaku.r-ccs.riken.jp:127.0.0.1:21364
```

`submit_bash_http_1n_over_ssh.sh` reads that state file and submits the batch
job with matching `FRONTEND_HOST` / `FRONTEND_SSH_TARGET`, so the compute node's
reverse tunnel lands on the same pinned host that owns the local forward.
`LOGIN_NODE` (or an explicit `REMOTE=`) overrides which frontend that is;
otherwise the value recorded in `state.env` wins. The target is passed to `pjsub`
as a single **hostname** — never an IP (compute nodes can't ssh the internal
frontend IP) and never a comma-joined list (`pjsub -x` treats commas as variable
separators and rejects the job at submit time, leaving it `ST=ERR` with no
output). Re-validated 2026-06-13 with batch job `49226036` (default
`LOGIN_NODE=1`; compute `a27-6108c`).

Logic:

- `bash_http_server.py` listens only on `127.0.0.1:21264` inside the compute
  node batch job.
- the batch job opens
  `ssh -R 127.0.0.1:21364:127.0.0.1:21264 login1.fugaku.r-ccs.riken.jp`
  so the login node gets a loopback-only view of the compute-side server.
- the local helper opens
  `ssh -L 127.0.0.1:21364:127.0.0.1:21364 login1.fugaku.r-ccs.riken.jp`
  so the local machine gets a stable loopback endpoint.
- the final path is:

```text
local client -> 127.0.0.1:21364
             -> ssh -L -> login1 loopback:21364
             -> ssh -R -> compute loopback:21264
             -> bash_http_server.py
```

Why pin one host:

- the forwarded port exists on exactly one login/frontend node
- a different `loginN` or a later `ssh fugaku` landing elsewhere cannot see it
- reusing one ControlMaster connection avoids that mismatch and keeps the local
  endpoint stable across many client calls

### Long sessions & reconnection

`rscgrp=small` allows `elapse` up to `72:00:00`, so the bridge can stay up for
days. Both ends of the tunnel are supervised and reconnect on drops, capped at
`MAX_RETRY` (default 10) **consecutive** failures (a healthy re-establish resets
the counter):

- **Job side** (`pjsub_bash_http_1n.sh`): a supervisor loop monitors the reverse
  `ssh -R` and, on drop, re-selects a reachable frontend and re-opens it. It runs
  until PJM's `elapse` (or `KEEPALIVE_SECONDS`, if set). Knobs (passed through by
  the submit script via `pjsub -x`): `MAX_RETRY`, `MONITOR_INTERVAL` (default 15 s),
  `KEEPALIVE_SECONDS` (0 = run until elapse), `HEALTH_EVERY` (deep through-frontend
  probe cadence, in ticks).
- **Client/frontend side** (`watch_bash_http_local_tunnel.sh`): monitors the local
  `ssh -L` ControlMaster and re-creates it (via `open_…sh`) on drop, with the same
  retry semantics. Run it alongside the session:

  ```sh
  nohup tools/watch_bash_http_local_tunnel.sh > /tmp/ds4p-bash-http/watch.log 2>&1 &
  ```

Set the session length and retry budget at submit time:

```sh
ELAPSE=24:00:00 MAX_RETRY=20 \
  DS4P_BASH_HTTP_TOKEN=... tools/submit_bash_http_1n_over_ssh.sh
```

`ELAPSE` (default `06:00:00`, max `72:00:00` for `rscgrp=small`) is applied via
`pjsub -L`. Validated 2026-06-13 (job `49226425`): killing the compute's `ssh -R`
triggered `reconnect attempt 1/10` → `re-established` within ~5 s, and killing the
local ControlMaster let the watcher rebuild it — the session survived both.

Operational notes:

- local direct HTTP client access from this Codex sandbox may require an
  unsandboxed command because opening local sockets can be restricted
- compute nodes needed `IdentitiesOnly=yes` to avoid "Too many authentication
  failures" when SSHing back to the login host
- `tools/bash_http_server.py` needed a Python 3.6 fallback for
  `ThreadingHTTPServer` on Fugaku nodes

Validated on 2026-06-13 with:

```text
login host: login1.fugaku.r-ccs.riken.jp
resolved login node: fn01sv01
login IPv4: 10.4.128.21
batch job: 49224908
compute host: a27-4210s
local URL: http://127.0.0.1:21364
```

Sample through the local forwarded endpoint:

```text
ls tools                      -> exit 0
fcc ... hello_fcc.c           -> exit 0
/tmp/bash_http_hello_fcc      -> hello from fcc native A64FX test
```
