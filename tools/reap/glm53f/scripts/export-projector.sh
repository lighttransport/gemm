#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-/mnt/nvme01/models/glm53f/reap/mmproj-glm53f-f16.gguf}"
[[ ! -e "$output" ]] || { echo "Refusing overwrite: $output"; exit 1; }
exec .venv/bin/python vendor/llama.cpp/convert_hf_to_gguf.py /mnt/nvme01/models/glm53f/base --mmproj --outtype f16 --outfile "$output"
