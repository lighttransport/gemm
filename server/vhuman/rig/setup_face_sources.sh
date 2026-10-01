#!/bin/sh
# Reproduce the optional GNM/ICT face models and LightGeom facial solver.
#
#   sh server/vhuman/rig/setup_face_sources.sh [--no-build] [--teacher-only]
#
# Requires git, curl, sha256sum, CMake >= 3.24, and a C/C++ compiler. The
# LightGeom checkout uses GitHub SSH; the caller needs access to its private
# repository. Clones and build products stay in ignored directories here.
# Set VHUMAN_BUILD_JOBS to change the default four parallel build jobs.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
models="$root/tmp/vhuman-rig/models"
lightgeom="$root/third_party/LightGeom"
build=1
teacher_only=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-build) build=0 ;;
        --teacher-only) teacher_only=1 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

die() { printf 'setup_face_sources: %s\n' "$*" >&2; exit 1; }
for tool in git curl sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is required"
done
if [ "$build" -eq 1 ]; then
    command -v cmake >/dev/null 2>&1 || die 'cmake is required (>= 3.24)'
fi

gnm_url=https://huggingface.co/google/gnm-v3
gnm_rev=c01e90d298d82301f9fd18f54806be751775cb7c
gnm_hash=61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb
ict_url=https://github.com/USC-ICT/ICT-FaceKit.git
ict_rev=da5f95a607f5e6b37755b38d3385d7f2853732e5
lightgeom_url=${VHUMAN_LIGHTGEOM_URL:-git@github.com:lighttransport/LightGeom.git}
lightgeom_rev=db64640cbbbb44d73c5ee3ffa3c3b405dee2cec1

staging=
download=
cleanup() {
    [ -z "$staging" ] || rm -rf -- "$staging"
    [ -z "$download" ] || rm -f -- "$download"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

checkout() { # label URL revision destination
    label=$1 url=$2 revision=$3 destination=$4
    if [ -e "$destination" ]; then
        git -C "$destination" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
            die "$destination exists but is not a Git checkout; move it aside first"
        origin=$(git -C "$destination" remote get-url origin 2>/dev/null) ||
            die "$label checkout has no origin remote"
        [ "$origin" = "$url" ] ||
            die "$label origin is $origin, expected $url; use a matching checkout"
        current=$(git -C "$destination" rev-parse HEAD)
        [ "$current" = "$revision" ] ||
            die "$label is at $current, expected $revision; move the checkout aside first"
        [ -z "$(git -C "$destination" status --porcelain --untracked-files=no)" ] ||
            die "$label has tracked changes; move or commit them first"
    else
        mkdir -p "$(dirname "$destination")"
        staging="$destination.clone.$$"
        [ ! -e "$staging" ] || die "staging path already exists: $staging"
        git clone --quiet --filter=blob:none --no-checkout "$url" "$staging" ||
            die "cannot clone $label from $url"
        if ! git -C "$staging" cat-file -e "$revision^{commit}" 2>/dev/null; then
            git -C "$staging" fetch --quiet origin "$revision" ||
                die "$label commit $revision is unavailable from $url"
        fi
        GIT_LFS_SKIP_SMUDGE=1 git -C "$staging" checkout --quiet --detach "$revision" ||
            die "cannot check out $label commit $revision"
        mv "$staging" "$destination"
        staging=
    fi
    printf '%s %s -> %s\n' "$label" "$revision" "$destination"
}

checksum() {
    [ -f "$1" ] || return 1
    actual=$(sha256sum "$1")
    [ "${actual%% *}" = "$gnm_hash" ]
}

if [ "$teacher_only" -eq 0 ]; then
checkout 'GNM v3' "$gnm_url" "$gnm_rev" "$models/gnm-v3/source"
gnm_weight="$models/gnm-v3/gnm_head.npz"
if ! checksum "$gnm_weight"; then
    download="$gnm_weight.download.$$"
    curl --fail --location --retry 3 --output "$download" \
        "$gnm_url/resolve/$gnm_rev/v3_0/gnm_head.npz" || die 'GNM weight download failed'
    checksum "$download" || die 'GNM weight SHA-256 mismatch'
    mv "$download" "$gnm_weight"
    download=
fi
printf 'GNM weight SHA-256 %s -> %s\n' "$gnm_hash" "$gnm_weight"

checkout 'ICT-FaceKit' "$ict_url" "$ict_rev" "$models/ict-facekit/source"
[ -f "$models/ict-facekit/source/FaceXModel/generic_neutral_mesh.obj" ] ||
    die 'ICT-FaceKit neutral mesh is missing'

fi

checkout 'LightGeom' "$lightgeom_url" "$lightgeom_rev" "$lightgeom"
[ -f "$lightgeom/examples/lightphysics_vhuman_face/main.c" ] ||
    die 'LightGeom facial solver is missing from the pinned commit'

if [ "$build" -eq 1 ]; then
    # These are the submodules needed by LightGeom's CMake configuration for
    # this target. Avoid its large, unrelated GUI/MuJoCo/Genesis dependencies.
    git -C "$lightgeom" submodule update --init -- \
        third_party/lightusd third_party/CoACD third_party/tinyvdb third_party/lightrt
    git -C "$lightgeom/third_party/CoACD" submodule update --init -- 3rd/cdt
    set -- -S "$lightgeom" -B "$lightgeom/build-vhuman" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIGHTGEOM_BUILD_EXAMPLES=ON \
        -DLIGHTGEOM_BUILD_SIM_GUI=OFF \
        -DLIGHTGEOM_GUI_ENABLE_VULKAN=OFF \
        -DLIGHTGEOM_ENABLE_LIGHTPHYSICS=ON \
        -DLIGHTGEOM_ENABLE_MUJOCO=OFF \
        -DLIGHTGEOM_ENABLE_RIGLOGIC=OFF \
        -DLIGHTPHYSICS_ENABLE_CUDA=OFF \
        -DLGPHYS_USE_LIGHTRT_SDF=OFF \
        -DLGPHYS_VDB_CACHE=OFF \
        -DLIGHTGEOM_BUILD_PAMO_TESTS=OFF \
        -DLIGHTGEOM_BUILD_MESH_TOOL_TESTS=OFF \
        -DLIGHTGEOM_BUILD_LGPHYS_TESTS=OFF \
        -DLIGHTPHYSICS_BUILD_SCHEMA_TESTS=OFF \
        -DCMAKE_DISABLE_FIND_PACKAGE_OpenGL=TRUE \
        -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE
    if command -v ninja >/dev/null 2>&1; then
        set -- "$@" -G Ninja
    fi
    cmake "$@"
    jobs=${VHUMAN_BUILD_JOBS:-4}
    case "$jobs" in
        *[!0-9]*|0|'') die 'VHUMAN_BUILD_JOBS must be a positive integer' ;;
    esac
    cmake --build "$lightgeom/build-vhuman" --target lightphysics_vhuman_face --parallel "$jobs"
    runner="$lightgeom/build-vhuman/examples/lightphysics_vhuman_face/lightphysics_vhuman_face"
    [ -x "$runner" ] || die "build finished without $runner"
    printf 'LightGeom facial runner -> %s\n' "$runner"
fi
