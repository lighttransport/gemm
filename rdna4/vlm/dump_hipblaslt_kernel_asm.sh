#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <code-object-or-kernel-name> [out-prefix] [kernel-symbol]" >&2
    exit 1
fi

target="$1"
out="${2:-/tmp/hipblaslt_kernel}"
symbol="${3:-}"
libdir="${HIPBLASLT_LIBRARY_DIR:-/opt/rocm-7.2.1/lib/hipblaslt/library}"

co=""
if [ -f "$target" ]; then
    co="$target"
else
    while IFS= read -r candidate; do
        if llvm-objdump -t "$candidate" 2>/dev/null | grep -q -- "$target"; then
            co="$candidate"
            break
        fi
    done < <(find "$libdir" -maxdepth 1 -type f \( -name '*.co' -o -name '*.hsaco' \) | sort)
fi

if [ -z "$co" ]; then
    echo "No matching .co/.hsaco found for '$target' under $libdir" >&2
    echo "If hipBLASLt used a lazy Tensile .dat path, rerun the tuner with:" >&2
    echo "  HIPBLASLT_LOG_LEVEL=5 HIPBLASLT_LOG_MASK=255 HIPBLASLT_BENCH_PRINT_COMMAND=1 HIPBLASLT_PRELOAD_KERNELS=1" >&2
    exit 2
fi

echo "# code_object=$co"
obj="$co"
if [ "$(head -c 4 "$co")" = "CCOB" ]; then
    decoded="${out}.decoded.bundle"
    obj="${out}.gfx1201.o"
    tail -c +33 "$co" | zstd -dc > "$decoded"
    /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
        --unbundle --type=o \
        --targets=hipv4-amdgcn-amd-amdhsa--gfx1201,host-x86_64-unknown-linux-gnu- \
        --input="$decoded" --output="$obj" --output="${out}.host.o"
fi

llvm-readelf --notes "$obj" > "${out}.notes.txt"
if [ -n "$symbol" ]; then
    llvm-objdump -d --no-show-raw-insn --disassemble-symbols="$symbol" "$obj" > "${out}.s"
else
    llvm-objdump -d --no-show-raw-insn "$obj" > "${out}.s"
fi

{
    echo "# code_object=$co"
    echo "# decoded_object=$obj"
    if [ -n "$symbol" ]; then echo "# symbol=$symbol"; fi
    echo "# notes=${out}.notes.txt"
    echo "# asm=${out}.s"
    echo
    printf "v_wmma: "; grep -c 'v_wmma' "${out}.s" || true
    printf "v_mfma: "; grep -c 'v_mfma' "${out}.s" || true
    printf "global_load: "; grep -c 'global_load\|buffer_load' "${out}.s" || true
    printf "global_store: "; grep -c 'global_store\|buffer_store' "${out}.s" || true
    printf "lds_load: "; grep -c 'ds_load' "${out}.s" || true
    printf "lds_store: "; grep -c 'ds_store' "${out}.s" || true
    printf "barrier: "; grep -c 's_barrier' "${out}.s" || true
    printf "wait: "; grep -c 's_wait' "${out}.s" || true
} | tee "${out}.summary.txt"
