# Ready hook for a multi-node allocation (run on the first node through the
# bridge): stage the q38d images into every node's /local, build the uTofu
# topology file for this allocation and the MPI-side helpers.
cd ~/work/gemm/qwen38-27b
NN=${Q38_NODES:-4}
S=tmp/q38-lowbit-20260924/hw-51893515/fp4-v1.image
F=tmp/q38-fast-images/fp6-e2m3-skipblk64-v1.image
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
mpiexec -n $NN sh -c "mkdir -p /local/q38/bin /local/q38/final; \
  test -s /local/q38/fp4.image || dd if=$S of=/local/q38/fp4.image bs=8M iflag=direct oflag=direct status=none; \
  test -s /local/q38/fp6.image || dd if=$F of=/local/q38/fp6.image bs=8M iflag=direct oflag=direct status=none; \
  echo \$(hostname) \$(ls -la /local/q38/*.image | wc -l) images"
cp tmp/q38-fast-final/fp4-f32-1024-ref.log /local/q38/ref-f32.log
cp tmp/q38-fast-final/fp6-f32-1024-ref.log /local/q38/q38d-fp6-f32.log
make -C a64fx/utofu-tests tofu_topo_helper qwen38_allreduce_bench MPICC=mpifcc 2>&1 | tail -2
rm -f tofu_topo.txt
mpiexec -n $NN a64fx/utofu-tests/tofu_topo_helper
cat tofu_topo.txt
echo STAGE_DONE
