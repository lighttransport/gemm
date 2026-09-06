#!/bin/sh
# Drive a PP run for a given N: gen vcoordfile (2x3x2 grid minus login 0,0,0),
# topo helper, stage shards (1GB chunks), run block-path runner, print tok/s.
# Usage: sweep_pp.sh <N> <maxgen>
set -e
N=$1; MAXGEN=${2:-3}
D=/vol0006/mdt0/data/hp250467/work/gemm/gemma4/a64fx/gemma4-mn
G=$HOME/models/gemma4/12b/gemma-4-12b-it-BF16.gguf
STAGE=/local/gemma4_pp
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$D"
VC=vcoord_g4pp_n${N}.txt
: > "$VC"; n=0
for x in 0 1; do for y in 0 1 2; do for z in 0 1; do
  [ "$x,$y,$z" = "0,0,0" ] && continue
  echo "($x,$y,$z)" >> "$VC"; n=$((n+1)); [ "$n" -ge "$N" ] && break 3
done; done; done
echo "[sweep] N=$N vcoord:"; cat "$VC"
rm -f tofu_topo.txt
mpiexec -np "$N" -vcoordfile "$VC" "$D/../utofu-tests/tofu_topo_helper" >/dev/null 2>&1
echo "[sweep] topo $(wc -l < tofu_topo.txt) lines"
echo "[sweep] staging N=$N ..."
rm -f stageout_*.txt
mpiexec -np "$N" -vcoordfile "$VC" sh -c "mkdir -p $STAGE; GEMMA4_STAGE_FLUSH_GB=1 exec $D/gemma4_stage $G $STAGE \$PMIX_RANK $N pp > $D/stageout_\$(hostname).txt 2>&1"
grep -h "DONE" stageout_*.txt | sort
echo "[sweep] running N=$N maxgen=$MAXGEN ..."
rm -f runout_*.txt
mpiexec -np "$N" -vcoordfile "$VC" sh -c "cd $D; LLM_THREADS=48 GEMMA4_PP_PERSIST=0 exec $D/gemma4_pp_runner $G $STAGE $D/prompt1.txt $MAXGEN > $D/runout_\$(hostname).txt 2>&1"
grep -hE "PPGEN|\[pp\] N=" runout_*.txt
