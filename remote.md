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
