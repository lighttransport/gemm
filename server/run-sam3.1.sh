./build/diffusion-server --host 0.0.0.0 --port 8082 --web-root ../web \
    --sam3-ckpt-v31   /mnt/disk01/models/sam3.1/sam3.1.model.safetensors \
    --sam3-vocab  /mnt/disk01/models/sam3.1/vocab.json \
    --sam3-merges /mnt/disk01/models/sam3.1/merges.txt 
