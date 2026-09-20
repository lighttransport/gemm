#!/bin/sh
# Stage native OpenCV development files without modifying the system install.
# On systems with OpenCV development packages already installed, set
# OPENCV_ROOT=/usr when invoking make instead.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export TMPDIR="$project_dir/../../tmp/pixal3d"
mkdir -p "$TMPDIR" "$project_dir/deps/opencv-packages" "$project_dir/deps/opencv"
cd "$project_dir/deps/opencv-packages"
apt-get download libopencv-core-dev libopencv-imgproc-dev libopencv-photo-dev \
    libopencv-core406t64 libopencv-imgproc406t64 libopencv-photo406t64 \
    libtbb12 libtbbbind-2-5 libtbbmalloc2
for package in ./*.deb; do
    dpkg-deb -x "$package" "$project_dir/deps/opencv"
done
