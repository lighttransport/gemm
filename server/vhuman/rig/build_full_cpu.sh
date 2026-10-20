#!/bin/sh
# Build just the standalone native facial ABI, without graphics/SDK discovery.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$root"
build="$root/tmp/vhuman-full-native-build"
destination="$root/tmp/vhuman-full-native/examples/lightphysics_vhuman_face"
mkdir -p "$build" "$destination"
export TMPDIR="$build"
cmake -S third_party/LightGeom -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLIGHTGEOM_BUILD_EXAMPLES=ON \
    -DLIGHTGEOM_BUILD_SIM_GUI=OFF -DLIGHTGEOM_GUI_ENABLE_VULKAN=OFF \
    -DLIGHTGEOM_ENABLE_LIGHTPHYSICS=ON -DLIGHTGEOM_ENABLE_MUJOCO=OFF \
    -DLIGHTGEOM_ENABLE_RIGLOGIC=OFF -DLIGHTPHYSICS_ENABLE_CUDA=OFF \
    -DLGPHYS_USE_LIGHTRT_SDF=OFF -DLGPHYS_VDB_CACHE=OFF \
    -DLIGHTGEOM_BUILD_PAMO_TESTS=OFF -DLIGHTGEOM_BUILD_MESH_TOOL_TESTS=OFF \
    -DLIGHTGEOM_BUILD_LGPHYS_TESTS=OFF -DLIGHTPHYSICS_BUILD_SCHEMA_TESTS=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenGL=TRUE -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE
cmake --build "$build" --target lightphysics_vhuman_full --parallel 4
# Rename a fresh inode into place; never truncate a library mapped by a bake.
staged=$(mktemp "$destination/.liblightphysics_vhuman_full.XXXXXX")
trap 'rm -f "$staged"' EXIT HUP INT TERM
cp "$build/examples/lightphysics_vhuman_face/liblightphysics_vhuman_full.so" "$staged"
chmod 755 "$staged"
mv -f "$staged" "$destination/liblightphysics_vhuman_full.so"
