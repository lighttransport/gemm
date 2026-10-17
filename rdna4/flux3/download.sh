#!/bin/sh
# Persistent/restartable FLUX 3 Klein download. Run with nohup or a batch service.
# Requires uv for initial setup. The isolated hf 0.36 client retains resumable
# .incomplete files; newer clients may use disposable process-unique partials.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
state="$root/tmp/flux3-download"
mkdir -p "$state"
export TMPDIR="$root/tmp" UV_CACHE_DIR="$root/tmp/uv-cache"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONDONTWRITEBYTECODE=1
model_dir=${1:-/mnt/nvme02/model/flux3-klein}
revision=${FLUX3_REVISION:-337337621b8526e120f2275d250242f4a6bedd75}
hf="$root/tmp/flux3-download-venv/bin/hf"
if [ ! -x "$hf" ]; then
    uv venv --python 3.12 "$root/tmp/flux3-download-venv"
    uv pip install --python "$root/tmp/flux3-download-venv/bin/python" 'huggingface_hub==0.36.0'
fi
exec 9>"$state/download.lock"
flock -n 9 || { echo 'A FLUX 3 download job is already running.'; exit 1; }
printf '%s\n' "$$" > "$state/download.pid"
child=
cleanup() {
    if [ -n "$child" ]; then
        kill "$child" 2>/dev/null || :
        wait "$child" 2>/dev/null || :
    fi
    rm -f "$state/download.pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_hf() {
    "$hf" "$@" &
    child=$!
    status=0
    wait "$child" || status=$?
    child=
    return "$status"
}
rm -f "$state/complete.revision"
while :; do
    date -Is
    # Make the compact ROCm bring-up checkpoint available first.
    if run_hf download black-forest-labs/flux-3-image-klein --revision "$revision" \
        --local-dir "$model_dir" --include 'variants/fp8r/*' --max-workers 4 && \
       run_hf download black-forest-labs/flux-3-image-klein --revision "$revision" \
        --local-dir "$model_dir" --max-workers 8; then
        printf '%s\n' "$revision" > "$state/complete.revision"
        echo 'Download complete.'
        exit 0
    fi
    echo 'Download interrupted; retrying in 30 seconds with existing partial files.'
    sleep 30
done
