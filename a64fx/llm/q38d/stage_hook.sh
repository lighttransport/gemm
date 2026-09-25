# Stage q38d images, binaries and F32 references into a fresh allocation.
cd ~/work/gemm/qwen38-27b
mkdir -p /local/q38/bin /local/q38/final
S=tmp/q38-lowbit-20260924/hw-51893515/fp4-v1.image
F=tmp/q38-fast-images/fp6-e2m3-skipblk64-v1.image
test -s /local/q38/fp4.image || dd if=$S of=/local/q38/fp4.image bs=8M iflag=direct oflag=direct status=none
test -s /local/q38/fp6.image || dd if=$F of=/local/q38/fp6.image bs=8M iflag=direct oflag=direct status=none
cp tmp/q38-fast/q38d_v41 /local/q38/bin/
M4=$HOME/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf
M6=$HOME/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf
R=tmp/q38-fast-final
test -s $R/fp4-f32-1024-ref.log || /local/q38/bin/q38d_v41 $M4 --fmt fp4 --image /local/q38/fp4.image --act f32 --prompt-tokens 1024 --gen 256 > $R/fp4-f32-1024-ref.log 2>&1
test -s $R/fp6-f32-1024-ref.log || Q38_LOWBIT_SKIP_PREFIX=blk.64. /local/q38/bin/q38d_v41 $M6 --fmt fp6 --image /local/q38/fp6.image --act f32 --prompt-tokens 1024 --gen 256 > $R/fp6-f32-1024-ref.log 2>&1
cp $R/fp4-f32-1024-ref.log /local/q38/ref-f32.log
cp $R/fp6-f32-1024-ref.log /local/q38/q38d-fp6-f32.log
python3 tmp/q38-fast/compare_tokens.py $R/fp4-f32-1024-ref.log $R/v41-fp4-1024-t1.log
python3 tmp/q38-fast/compare_tokens.py $R/fp6-f32-1024-ref.log $R/v41-fp6-1024-t1.log
ls -la /local/q38
echo STAGE_DONE
