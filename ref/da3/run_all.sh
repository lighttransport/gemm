#!/bin/bash
# End-to-end DA3 reference comparison: PyTorch reference vs our C/CUDA implementation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Configuration
MODEL_DIR="${DA3_MODEL_DIR:-/mnt/nvme02/models/gemm/da3-small}"
INPUT_IMAGE="${DA3_INPUT_IMAGE:-$ROOT_DIR/Brooklyn_Bridge_Manhattan.jpg}"
DEVICE="${DA3_DEVICE:-cuda}"

echo "=== DA3 Reference Comparison ==="
echo "  Model:  $MODEL_DIR"
echo "  Image:  $INPUT_IMAGE"
echo "  Device: $DEVICE"
echo ""

# 1. Setup output directory
mkdir -p "$OUTPUT_DIR"

# 2. Convert input image to PPM (for our C programs)
PPM_PATH="$OUTPUT_DIR/input.ppm"
if [ ! -f "$PPM_PATH" ]; then
    echo "=== Converting image to PPM ==="
    # Use Python/Pillow since it's already a dependency
    cd "$SCRIPT_DIR"
    uv run python -c "
from PIL import Image
img = Image.open('$INPUT_IMAGE').convert('RGB')
w, h = img.size
with open('$PPM_PATH', 'wb') as f:
    f.write(f'P6\n{w} {h}\n255\n'.encode())
    f.write(img.tobytes())
print(f'Wrote {w}x{h} PPM: $PPM_PATH')
"
fi

# 3. Install Python dependencies
echo ""
echo "=== Setting up Python environment ==="
cd "$SCRIPT_DIR"
uv sync

# 4. Run PyTorch reference inference
echo ""
echo "=== Running PyTorch reference inference (device=$DEVICE) ==="
uv run python run_reference.py \
    --model-dir "$MODEL_DIR" \
    --image "$INPUT_IMAGE" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE"

# 5. Build our C/CUDA implementations
echo ""
echo "=== Building C/CUDA implementations ==="
make -C "$ROOT_DIR/common" ARCH=native -j$(nproc) 2>&1 | tail -3
if [ -d "$ROOT_DIR/cuda/da3" ]; then
    make -C "$ROOT_DIR/cuda/da3" -j$(nproc) 2>&1 | tail -3
fi

# 6. Run CPU implementation
echo ""
echo "=== Running CPU implementation ==="
"$ROOT_DIR/common/test_da3" \
    "$MODEL_DIR/model.safetensors" \
    -i "$PPM_PATH" \
    -o "$OUTPUT_DIR/depth_cpu.pgm" \
    --npy "$OUTPUT_DIR/depth_cpu.npy"

# 7. Run GPU implementation (if available)
GPU_NPY=""
if [ -x "$ROOT_DIR/cuda/da3/test_cuda_da3" ]; then
    echo ""
    echo "=== Running GPU implementation ==="
    "$ROOT_DIR/cuda/da3/test_cuda_da3" \
        "$MODEL_DIR/model.safetensors" \
        -i "$PPM_PATH" \
        -o "$OUTPUT_DIR/depth_gpu.pgm" \
        --npy "$OUTPUT_DIR/depth_gpu.npy"
    GPU_NPY="$OUTPUT_DIR/depth_gpu.npy"
fi

# 8. Compare results
echo ""
echo "========================================"
echo "=== Comparisons ==="
echo "========================================"

FAIL=0

# CPU vs Reference
echo ""
echo "--- CPU vs Reference ---"
if uv run python compare.py \
    --reference "$OUTPUT_DIR/depth_ref.npy" \
    --ours "$OUTPUT_DIR/depth_cpu.npy" \
    --label "CPU vs Reference" \
    --output-dir "$OUTPUT_DIR"; then
    echo "  => PASSED"
else
    echo "  => FAILED"
    FAIL=1
fi

# GPU vs Reference (if available)
if [ -n "$GPU_NPY" ] && [ -f "$GPU_NPY" ]; then
    echo ""
    echo "--- GPU vs Reference ---"
    if uv run python compare.py \
        --reference "$OUTPUT_DIR/depth_ref.npy" \
        --ours "$GPU_NPY" \
        --label "GPU vs Reference" \
        --tolerance-r 0.999 \
        --tolerance-ssim 0.99 \
        --output-dir "$OUTPUT_DIR"; then
        echo "  => PASSED"
    else
        echo "  => FAILED"
        FAIL=1
    fi

    echo ""
    echo "--- GPU vs CPU ---"
    if uv run python compare.py \
        --reference "$OUTPUT_DIR/depth_cpu.npy" \
        --ours "$GPU_NPY" \
        --label "GPU vs CPU" \
        --tolerance-r 0.999 \
        --tolerance-ssim 0.999 \
        --output-dir "$OUTPUT_DIR"; then
        echo "  => PASSED"
    else
        echo "  => FAILED"
        FAIL=1
    fi
fi

echo ""
echo "========================================"
if [ $FAIL -eq 0 ]; then
    echo "=== All comparisons PASSED ==="
else
    echo "=== Some comparisons FAILED ==="
fi
echo "========================================"
echo ""
echo "Output files in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

exit $FAIL
