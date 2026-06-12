# Fugaku Interactive Job Workflow

## One-node interactive session

Use `script(1)` on the Fugaku frontend to capture all terminal input and output
while `pjsub --interact` attaches to the compute node:

```sh
ssh -tt fugaku
cd ~/work/gemm/ds4p
mkdir -p a64fx/interactive-test
script -q -f a64fx/interactive-test/interactive-1.transcript
./interactive-1.sh
```

If `script` on the frontend starts a logged shell, run `./interactive-1.sh`
inside that shell. Exit the compute shell once to end the interactive job, then
exit the logged frontend shell once more to finalize the transcript.

## Validation run

Validated on 2026-06-12 with interactive job `49215908`.

The allocation attached to compute node `i28-6206c`:

```text
[INFO] PJM 0000 pjsub Job 49215908 submitted.
[INFO] PJM 0082 pjsub Interactive job 49215908 started.
hostname: i28-6206c
uname -m: aarch64
fcc: /opt/FJSVxtclanga/tcsds-1.2.43/bin/fcc
fcc version: fcc (FCC) 4.12.2 20251113
```

Inside the allocation, a small C program was compiled and run natively:

```sh
fcc -O2 -Nclang -march=armv8.2-a+sve \
  -o a64fx/interactive-test/hello_fcc \
  a64fx/interactive-test/hello_fcc.c
./a64fx/interactive-test/hello_fcc
```

Result:

```text
hello from fcc native A64FX test
acc=17500378484502161128
SENTINEL interactive_fcc_native=OK
[INFO] PJM 0083 pjsub Interactive job 49215908 completed.
```

Artifacts:

```text
a64fx/interactive-test/interactive-1.transcript
a64fx/interactive-test/hello_fcc.c
a64fx/interactive-test/hello_fcc
```

## Persistent tmux control

`tmux` works as a persistent owner for the interactive TTY, but Fugaku's
`ssh fugaku` entry point can land on different frontend nodes. Since tmux
sockets live in frontend-local `/tmp`, all control commands must target the
same frontend node that owns the tmux session.

Repeated `ssh fugaku hostname` during the test alternated between `fn01sv03`
and `fn01sv05`. The reliable control pattern is therefore a nested SSH through
the normal Fugaku entry point to a pinned frontend:

```sh
ssh fugaku 'ssh fn01sv03 "hostname"'
```

Start a persistent tmux session on that pinned frontend:

```sh
ssh fugaku 'ssh fn01sv03 "cd ~/work/gemm/ds4p &&
  mkdir -p a64fx/interactive-test &&
  tmux new-session -d -s ds4p-int-test \
    \"cd ~/work/gemm/ds4p && script -q -f a64fx/interactive-test/tmux-live.transcript\" &&
  tmux send-keys -t ds4p-int-test \"sh interactive-1.sh\" C-m"'
```

Send commands into the interactive compute shell:

```sh
ssh fugaku 'ssh fn01sv03 "tmux send-keys -t ds4p-int-test \"hostname\" C-m"'
ssh fugaku 'ssh fn01sv03 "tmux send-keys -t ds4p-int-test \"./a64fx/interactive-test/tmux_hello_fcc\" C-m"'
```

Read back recent terminal state:

```sh
ssh fugaku 'ssh fn01sv03 "tmux capture-pane -pt ds4p-int-test -S -120"'
ssh fugaku 'ssh fn01sv03 "tail -120 ~/work/gemm/ds4p/a64fx/interactive-test/tmux-live.transcript"'
```

End the interactive allocation cleanly:

```sh
ssh fugaku 'ssh fn01sv03 "tmux send-keys -t ds4p-int-test \"exit\" C-m"'
ssh fugaku 'ssh fn01sv03 "tmux send-keys -t ds4p-int-test \"exit\" C-m"'
```

The first `exit` leaves the compute shell and completes the PJM interactive
job. The second exits the `script(1)` frontend shell and finalizes the
transcript.

Validated on 2026-06-12 with persistent tmux session `ds4p-int-test` on
frontend `fn01sv03` and interactive job `49216145`. The job attached to compute
node `a27-2207c`; compiling and running the native FCC smoke binary through
`tmux send-keys` produced:

```text
hello from fcc native A64FX test
acc=17500378484502161128
SENTINEL interactive_fcc_native=OK
```

## MCP server

`fugaku_tmux_mcp.py` is a dependency-free Python stdio MCP server for the same
pinned-frontend tmux workflow. By default it runs locally and controls Fugaku
with:

```text
local MCP client -> ssh fugaku -> ssh fn01sv03 -> tmux ds4p-int
```

Default environment:

```text
FUGAKU_SSH=fugaku
FUGAKU_FRONTEND=fn01sv03
FUGAKU_WORKDIR=~/work/gemm/ds4p
FUGAKU_TMUX_SESSION=ds4p-int
FUGAKU_TRANSCRIPT=a64fx/interactive-test/mcp-live.transcript
FUGAKU_INTERACTIVE_LAUNCHER='sh interactive-1.sh'
```

Example client config:

```toml
[mcp_servers.fugaku-tmux]
command = "python3"
args = ["/mnt/nvme02/work/fugaku/work/gemm/ds4p/a64fx/interactive-test/fugaku_tmux_mcp.py"]
env = { FUGAKU_FRONTEND = "fn01sv03", FUGAKU_TMUX_SESSION = "ds4p-int-test", FUGAKU_TRANSCRIPT = "a64fx/interactive-test/tmux-live.transcript" }
```

When running the MCP server directly on the pinned Fugaku frontend, skip SSH by
setting:

```sh
FUGAKU_SSH=local FUGAKU_FRONTEND=local python3 a64fx/interactive-test/fugaku_tmux_mcp.py
```

Available tools:

```text
fugaku_interactive_start
fugaku_interactive_send
fugaku_interactive_capture
fugaku_interactive_tail
fugaku_interactive_status
fugaku_interactive_stop
```

Local protocol smoke test:

```sh
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | python3 a64fx/interactive-test/fugaku_tmux_mcp.py
```

Validated against the live `ds4p-int-test` session with:

```text
fugaku_interactive_status: saw tmux session, transcript, and PJM job 49216145
fugaku_interactive_send: echo SENTINEL mcp_tmux_send=OK
fugaku_interactive_capture: captured SENTINEL mcp_tmux_send=OK from the pane
```
