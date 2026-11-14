# sourced by the eval batch scripts. Outputs: $PRAXIS_OUT/praxis_sim2/<name>/ (PRAXIS_OUT defaults to <repo>/tmp) ; seeds 100..111 (the earlier sims used 0-5).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
export MUJOCO_GL=egl
PRAXIS_ROOT=${PRAXIS_ROOT:-$HOME/praxis}
CFG=$PRAXIS_ROOT/praxis-1-oss/configs/sim_droid.yaml
OUT=${OUT:-${PRAXIS_OUT:-$REPO/tmp}/praxis_sim2}
SEEDS="episodes.seeds=[100,101,102,103,104,105,106,107,108,109,110,111]"
# run <name> <task> <expert-fmt> [run_sim_cuda.py flags...]; the default 'fast' flags are added by fast()/full()
run() {
  name=$1; task=$2; fmt=$3; shift 3
  [ -f $OUT/$name/summary.json ] && { echo "skip $name (done)"; return; }
  rm -rf $OUT/$name
  if [ $fmt = nvfp4 ]; then export PRAXIS_NVFP4=1; else unset PRAXIS_NVFP4; fi
  ~/.venv/bin/python -I cuda/praxis/run_sim_cuda.py --config $CFG --vae-mode reload --expert-fmt $fmt "$@" \
     --set task.name=$task --set "task.prompt=${PROMPT:-null}" --set "$SEEDS" --set output.dir=$OUT/$name > $OUT/$name.log 2>&1
  rc=$?
  echo "$name rc=$rc $(grep -h '^success' $OUT/$name.log)"
  ~/.venv/bin/python -I cuda/praxis/eval/prune_videos.py $OUT/$name
}
fast() { run "$@" --cfg-video 1.0 --skip-last-video; }   # new default: --rebase 8 (run_sim_cuda.py default)
full() { run "$@"; }                                     # cfg video 4 + decoded predicted video
