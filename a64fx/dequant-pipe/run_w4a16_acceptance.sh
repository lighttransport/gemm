#!/bin/bash
# Three fresh processes per format/path, paired reads on each allocation.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_dir"
mkdir -p tmp/dequant
log_dir=$(mktemp -d "$repo_dir/tmp/dequant/w4a16-acceptance.XXXXXX")
printf 'Logs: %s\n' "$log_dir"
for launch in 1 2 3; do
    for format in int4 fp4; do
        for path in int16 int16x8-full fp16; do
            kernel=opt
            extra=()
            if [[ $path == fp16 ]]; then kernel=opt2; fi
            if [[ $path != int16x8-full ]]; then extra=(--compare-kernels); fi
            taskset -c 12 env \
                LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1 \
                XOS_MMM_L_HPAGE_TYPE=hugetlbfs XOS_MMM_L_HUGETLB_SZ=2M \
                XOS_MMM_L_HUGE_MALLOC=1 XOS_MMM_L_FORCE_MMAP_THRESHOLD=1 \
                XOS_MMM_L_HUGETLB_FALLBACK=0 \
                ./a64fx/dequant-pipe/bench_fused_sdot \
                --format "$format" --path "$path" --kernel "$kernel" \
                --cores 12 --core-base 12 --mib 240 --iterations 10 --trials 5 \
                --paired-baseline "${extra[@]}" \
                > "$log_dir/$format-$path-$launch.log" 2>&1
            awk '/^summary/ || /qualified=/' "$log_dir/$format-$path-$launch.log"
        done
    done
done
# Native halfword SDOT is a comparison; acceptance uses full-range byte recoding.
awk '
    FNR == 1 { files++ }
    /qualified=yes/ { qualified++ }
    /^summary/ && /path=int16x8-full/ {
        for (i=1;i<=NF;i++) if ($i ~ /^packed_GB\/s_median=/) {
            split($i,a,"="); sdot++; if (a[2] <= 200) bad++
        }
    }
    /^summary/ && /path=fp16/ && /kernel=opt2/ {
        for (i=1;i<=NF;i++) if ($i ~ /^packed_GB\/s_median=/) {
            split($i,a,"="); fp16++; if (a[2] <= 150) bad++
        }
    }
    END {
        ok = files == 18 && qualified == 18 && sdot == 6 && fp16 == 6 && !bad
        printf "acceptance=%s qualified=%d/%d SDOT=%d FP16=%d failed_targets=%d\n",
               ok ? "PASS" : "FAIL", qualified, files, sdot, fp16, bad
        exit !ok
    }
' "$log_dir"/*.log
