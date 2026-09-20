#!/bin/bash
# P9: original FP8-byte rate >200 GB/s, stored-byte rate reported separately.
# Each width gets three fresh launches, with five timed trials per launch.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_dir"
mkdir -p tmp/dequant
log_dir=$(mktemp -d "$repo_dir/tmp/dequant/e4-p9-acceptance.XXXXXX")
printf 'Logs: %s\n' "$log_dir"
{
    uname -a
    for cpu in {12..23}; do
        printf 'cpu=%s frequency_khz=' "$cpu"
        cat "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq"
    done
    sha256sum a64fx/dequant-pipe/{bench_w8,bench_w8.c,e4_pack.S,e4_pack.c,e4_pack.h}
} > "$log_dir/host.txt"
for launch in 1 2 3; do
    for bits in 16 32; do
        taskset -c 12 env \
            LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1 \
            XOS_MMM_L_HPAGE_TYPE=hugetlbfs XOS_MMM_L_HUGETLB_SZ=2M \
            XOS_MMM_L_HUGE_MALLOC=1 XOS_MMM_L_FORCE_MMAP_THRESHOLD=1 \
            XOS_MMM_L_HUGETLB_FALLBACK=0 \
            ./a64fx/dequant-pipe/bench_w8 \
            --format e4m3 --bits "$bits" --packing p9 --target-gbps 200 \
            --cores 12 --core-base 12 --mib 240 --iterations 10 --trials 5 \
            > "$log_dir/e4m3-$bits-$launch.log" 2>&1
        awk '/^summary/ || /qualified=/' "$log_dir/e4m3-$bits-$launch.log"
    done
done
awk '
    FNR == 1 { files++ }
    /weight_distribution=all-finite/ { finite++ }
    /packing=p9/ { packed++ }
    /qualified=yes/ { qualified++ }
    /target=PASS/ { passed++ }
    /^summary/ && /path=fma/ {
        for (i=1;i<=NF;i++) if ($i ~ /^GB\/s_median=/) {
            split($i,a,"="); measured++; if (a[2] <= 200) bad++
        }
    }
    END {
        ok = files == 6 && finite == 6 && packed == 6 && qualified == 6 && measured == 6 && passed == 6 && !bad
        printf "acceptance=%s qualified=%d/%d passed_targets=%d/%d threshold_original_GB/s=200 (strictly_greater)\n",
               ok ? "PASS" : "FAIL", qualified, files, passed, measured
        exit !ok
    }
' "$log_dir"/*.log
