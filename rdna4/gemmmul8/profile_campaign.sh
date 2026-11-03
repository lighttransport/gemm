#!/bin/sh
# Run after profile_build.sh, with exclusive GPU access for meaningful timings.
set -eu
cd "$(dirname "$0")"
PROFILE=../../tmp/gemmmul8/profile/profile_ozaki
"$PROFILE" 1024 1024 1024 int8 15 > phase_timings.csv
"$PROFILE" 1024 1024 1024 fp8 11 | tail -n 1 >> phase_timings.csv
"$PROFILE" 2048 2048 2048 int8 15 | tail -n 1 >> phase_timings.csv
"$PROFILE" 2048 2048 2048 fp8 11 | tail -n 1 >> phase_timings.csv
