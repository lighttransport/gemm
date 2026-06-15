INT4=/mnt/disk1/models/qwen-image/nunchaku/svdq-int4_r128-qwen-image-logical-v3.safetensors

./server/build/diffusion-server \
      --qwen-variants "qwen-image:/mnt/disk1/models/qwen-image:$INT4" \
      --web-root web --port 8085 --host 0.0.0.0
