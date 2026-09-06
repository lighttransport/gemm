#!/bin/bash
# Chunked-prefill checkpointing (ds4f.md fix #2 for the uTofu bcast-timeout on long token-by-token
# prefills). A long prompt is prefilled in M-token chunks; after each chunk the KV+compressor cache is
# saved to disk (cache_save). Each chunk request EXTENDS the previous one, so the prefix cache prefills
# only the new M tokens (M*43 reduces => low per-chunk comm-loss probability). If the runner dies
# mid-chunk (tp_ar bcast timeout -> exit1), we relaunch it with DS4F_SERVE_SYSCACHE=<ckpt> (restores
# all checkpointed chunks at startup) and retry ONLY the failed chunk -- so the prefill always makes
# forward progress and a completed chunk is never redone. Reuses the byte-exact KV save/load + prefix
# cache already validated (validate_ctx_features.sh / validate_cp_shards.sh).
#
# Usage:  CHUNK=256 ./prefill_checkpoint.sh <prompt_ids_file> [ckpt_path]
#   the prompt_ids_file holds whitespace-separated token ids; ckpt_path defaults to ~/.ds4f_ckpt
set -u
cd "$(dirname "$0")"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
IDS_FILE=${1:?usage: [CHUNK=M] prefill_checkpoint.sh <prompt_ids_file> [ckpt_path]}
CKPT=${2:-$HOME/.ds4f_ckpt}
M=${CHUNK:-256}
B=$HOME/.ds4f_serve.ckpt
LOG=/tmp/ds4f_ckpt_runner.log
MAXPOS=${DS4F_MAXPOS:-16384}

read -r -a IDS < "$IDS_FILE"; N=${#IDS[@]}
echo "[ckpt] prompt=$N tokens, chunk=$M, ckpt=$CKPT"

runner_alive(){ pgrep -f 'org/mpiexec' >/dev/null; }
kill_runner(){ local p; for p in $(ps -eo pid,args|grep -E 'org/mpiexec|/bin/plexec'|grep -v grep|awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done; sleep 6; }

launch(){ # $1 = SYSCACHE path ("" = fresh)
  runner_alive && kill_runner
  find . -maxdepth 1 -name 'ds4f_ep_*rank*.txt' -delete 2>/dev/null
  rm -f $B.req $B.resp $B.reqseq $B.respseq
  export DS4F_SERVE=1 DS4F_SERVE_REQ=$B.req DS4F_SERVE_RESP=$B.resp DS4F_SERVE_REQSEQ=$B.reqseq DS4F_SERVE_RESPSEQ=$B.respseq
  export DS4F_REAL=1 DS4F_FP8_BF16=1 DS4F_Q8_DENSE=1 DS4F_TIERB2=1 DS4F_MHC=1 DS4F_HC_PAR=1 DS4F_HC_RMSPAR=1 DS4F_NUMA=1
  export DS4F_MAXPOS=$MAXPOS DS4F_INT8_KV=1 DS4F_INT8_CMP=1 DS4F_INT4_CMP=1 DS4F_IDX_INT4=1 DS4F_PREFILL_GEMM=0 DS4F_SERVE_SLOTS=1
  if [ -n "$1" ]; then export DS4F_SERVE_SYSCACHE="$1"; else unset DS4F_SERVE_SYSCACHE; fi
  echo "[ckpt] launching runner${1:+ (reload $1)} ..."
  ( ./run_ds4f_11n.sh > "$LOG" 2>&1 ) &
  local i; for i in $(seq 1 60); do grep -q 'SERVE ready' ds4f_ep_rank00.txt 2>/dev/null && return 0; runner_alive || { sleep 5; continue; }; sleep 5; done
  grep -q 'SERVE ready' ds4f_ep_rank00.txt
}

send_chunk(){ # prompt = ids[0 .. end); save to CKPT. returns 1 if runner died.
  local end=$1
  { echo "0 0 1 0 0 1 0 0 2"; echo "$CKPT"; echo "${IDS[@]:0:$end}"; } > $B.req
  local nx=$(( $(cat $B.reqseq 2>/dev/null||echo 0)+1 )); echo $nx > $B.reqseq
  local i; for i in $(seq 1 3600); do
    [ "$(cat $B.respseq 2>/dev/null || echo 0)" -ge "$nx" ] && return 0
    runner_alive || return 1; sleep 1
  done; return 1
}

# resume from an existing checkpoint (its header stores npos at byte offset 4, LE uint32).
ckpt_len(){ [ -f "$CKPT" ] && od -An -j4 -N4 -tu4 "$CKPT" 2>/dev/null | tr -d ' ' || echo 0; }
done_to=$(ckpt_len); [ -z "$done_to" ] && done_to=0
if [ "$done_to" -gt 0 ]; then echo "[ckpt] found checkpoint with $done_to tokens -> resume"; launch "$CKPT" || exit 1
else launch "" || { echo "[ckpt] initial launch failed"; exit 1; }; fi

# start at the first chunk boundary strictly beyond what the checkpoint already holds
start=$(( (done_to / M + 1) * M )); [ "$done_to" -eq 0 ] && start=$M
for (( end=start; ; end+=M )); do
  [ $end -gt $N ] && end=$N
  [ $end -le $done_to ] && continue
  tries=0
  until send_chunk $end; do
    tries=$((tries+1)); echo "[ckpt] chunk [$done_to,$end): runner DIED (comm timeout), relaunch+resume try #$tries"
    [ $tries -ge 5 ] && { echo "[ckpt] giving up after 5 tries (checkpoint at $done_to is intact)"; exit 1; }
    launch "$CKPT" || { echo "[ckpt] relaunch failed"; exit 1; }   # SYSCACHE reloads [0,done_to)
  done
  done_to=$end
  echo "[ckpt] chunk done: [0,$end) prefilled+checkpointed  |  $(grep 'SERVE req' ds4f_ep_rank00.txt|tail -1)"
  [ $end -ge $N ] && break
done
echo "[ckpt] DONE: full $N-token prompt prefilled + checkpointed to $CKPT (resumable across restarts)"
