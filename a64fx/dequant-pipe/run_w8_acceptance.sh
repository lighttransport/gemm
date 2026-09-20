#!/bin/bash
# Every format / accumulation width must reach 220 GB/s in three fresh launches.
# All-finite weights include subnormals; normal-only diagnostics do not qualify.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_dir"
mkdir -p tmp/dequant
log_dir=$(mktemp -d "$repo_dir/tmp/dequant/w8-acceptance.XXXXXX")
printf 'Logs: %s\n' "$log_dir"
{
    uname -a
    for cpu in {12..23}; do
        printf 'cpu=%s frequency_khz=' "$cpu"
        cat "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq"
    done
    sha256sum a64fx/dequant-pipe/bench_w8 a64fx/dequant-pipe/w8.S
} > "$log_dir/host.txt"
for launch in 1 2 3; do
    for format in int8 e4m3 e5m2; do
        for bits in 16 32; do
            taskset -c 12 env \
                LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1 \
                XOS_MMM_L_HPAGE_TYPE=hugetlbfs XOS_MMM_L_HUGETLB_SZ=2M \
                XOS_MMM_L_HUGE_MALLOC=1 XOS_MMM_L_FORCE_MMAP_THRESHOLD=1 \
                XOS_MMM_L_HUGETLB_FALLBACK=0 \
                ./a64fx/dequant-pipe/bench_w8 \
                --format "$format" --bits "$bits" \
                --cores 12 --core-base 12 --mib 240 --iterations 10 --trials 5 \
                > "$log_dir/$format-$bits-$launch.log" 2>&1
            awk '/^summary/ || /qualified=/' "$log_dir/$format-$bits-$launch.log"
        done
    done
done
awk '
    FNR == 1 { files++ }
    /weight_distribution=all-finite/ { finite++ }
    /qualified=yes/ { qualified++ }
    /^summary/ && /path=fma/ {
        for (i=1;i<=NF;i++) if ($i ~ /^GB\/s_median=/) {
            split($i,a,"="); measured++; if (a[2] < 220) bad++
        }
    }
    END {
        ok = files == 18 && finite == 18 && qualified == 18 && measured == 18 && !bad
        printf "acceptance=%s qualified=%d/%d measured=%d failed_targets=%d threshold_GB/s=220\n",
               ok ? "PASS" : "FAIL", qualified, files, measured, bad
        exit !ok
    }
' "$log_dir"/*.log
