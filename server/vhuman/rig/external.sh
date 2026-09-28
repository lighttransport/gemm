#!/bin/sh
# Clone (or update) the external rig/USD projects used beside the vhuman rig.
# They are plain clones under third_party/ (ignored by Git), never submodules.
#
#   sh server/vhuman/rig/external.sh [--local] [--build] [--update] [--dest DIR]
#
#   --local   clone from the developer checkouts (LIGHTRIG_SRC / LIGHTUSD_SRC,
#             default ~/work/LightRig and ~/work/tinyusdz/origin-dev) instead
#             of GitHub; handy offline, same history
#   --update  fetch and fast-forward existing clones
#   --build   build `lightrig` (CLI + worker) and `vchar` (LightUSD viewer)
#   --dest    where to clone (default third_party/)
#
# Revisions: LIGHTRIG_REF (default main) and LIGHTUSD_REF (default dev).
# LIGHTUSD_HIP=1 keeps lusdview's optional HIP texture plugin (AMD GPU needed).
# Remotes:   LIGHTRIG_URL, LIGHTUSD_URL.
# The vhuman rig writes plain UsdSkel .usda; these tools are optional
# consumers (inspection, animation tracks, the native vchar viewer).
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
dest="$root/third_party"
local=0 build=0 update=0
while [ $# -gt 0 ]; do
    case "$1" in
        --local) local=1 ;;
        --build) build=1 ;;
        --update) update=1 ;;
        --dest) dest=$2; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

LIGHTRIG_URL=${LIGHTRIG_URL:-git@github.com:lighttransport/LightRig.git}
LIGHTUSD_URL=${LIGHTUSD_URL:-git@github.com:lighttransport/LightUSD.git}
LIGHTRIG_REF=${LIGHTRIG_REF:-main}
LIGHTUSD_REF=${LIGHTUSD_REF:-dev}
if [ "$local" = 1 ]; then
    LIGHTRIG_URL=${LIGHTRIG_SRC:-$HOME/work/LightRig}
    LIGHTUSD_URL=${LIGHTUSD_SRC:-$HOME/work/tinyusdz/origin-dev}
fi
jobs=$(nproc 2>/dev/null || echo 4)
mkdir -p "$dest"

fetch() {  # name url ref
    dir="$dest/$1"
    if [ -d "$dir/.git" ]; then
        if [ "$update" = 1 ]; then
            git -C "$dir" fetch --quiet origin "$3"
            git -C "$dir" checkout --quiet "$3"
            git -C "$dir" merge --quiet --ff-only FETCH_HEAD
        fi
    else
        git clone --quiet --branch "$3" "$2" "$dir"
    fi
    echo "$1 $(git -C "$dir" rev-parse --short HEAD) ($3) -> $dir"
}

fetch LightRig "$LIGHTRIG_URL" "$LIGHTRIG_REF"
fetch LightUSD "$LIGHTUSD_URL" "$LIGHTUSD_REF"

# LightRig vendors LightUSD as a submodule; the build points it at our clone
# instead of fetching a second copy (LIGHTRIG_LIGHTUSD_DIR).
if [ "$build" = 1 ]; then
    command -v cmake >/dev/null || { echo "cmake is required for --build" >&2; exit 1; }
    gen=""
    command -v ninja >/dev/null && gen="-G Ninja"
    # vchar is lusdview renamed; it enters virtual-human mode by argv[0].
    # LightRig links <LightUSD>/build_lightrig/liblightusd_static.a by default.
    # shellcheck disable=SC2086
    # HIP (auto-detected by lusdview's texture tools) needs an AMD GPU to pick
    # an architecture; set LIGHTUSD_HIP=1 to keep it.
    hip=""
    [ "${LIGHTUSD_HIP:-0}" = 1 ] || hip="-DCMAKE_HIP_COMPILER=NOTFOUND"
    cmake -S "$dest/LightUSD" -B "$dest/LightUSD/build_lightrig" $gen -DCMAKE_BUILD_TYPE=Release \
        -DLIGHTUSD_BUILD_GUI_VIEWER=ON $hip
    cmake --build "$dest/LightUSD/build_lightrig" --target lightusd_static lusdview -j "$jobs"
    # Optional heavy dependencies (MuJoCo, OpenRigLogic DNA, GPU backends) are
    # off here: the vhuman rig only needs inspection and USD animation export.
    # shellcheck disable=SC2086
    cmake -S "$dest/LightRig" -B "$dest/LightRig/build-vhuman" $gen -DCMAKE_BUILD_TYPE=Release \
        -DLIGHTRIG_LIGHTUSD_DIR="$dest/LightUSD" -DLIGHTRIG_ENABLE_MUJOCO=OFF \
        -DLIGHTRIG_ENABLE_OPENRIGLOGIC=OFF -DLIGHTRIG_ENABLE_GEMM_MLP2=OFF -DLIGHTRIG_ENABLE_VULKAN=OFF \
        -DLIGHTRIG_ENABLE_CUDA=OFF -DLIGHTRIG_ENABLE_HIP=OFF -DLIGHTRIG_BUILD_TESTS=OFF
    cmake --build "$dest/LightRig/build-vhuman" --target lightrig -j "$jobs"
    echo "vchar:    $(find "$dest/LightUSD/build_lightrig" -name vchar -type f | head -1)"
    echo "lightrig: $dest/LightRig/build-vhuman/lightrig"
fi
