#!/usr/bin/env bash
# Build a single, pinned oracle. No libraries from older builds are loaded.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="${1:-/mnt/nvme02/work/llama.cpp}"
build_dir="${2:-${root}/tmp/qwen38/reference-build}"
revision=1859b520910af6f682256fd7299797774111a27a
rocm="${ROCM_PATH:-/opt/rocm/core-10.0}"
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || {
    echo "Reference requires llama.cpp ${revision}" >&2; exit 1;
}
[[ -z "$(git -C "$source_dir" status --porcelain --untracked-files=no)" ]] || {
    echo "Reference source has tracked modifications" >&2; exit 1;
}
mkdir -p "$build_dir/scratch"
build_dir="$(cd "$build_dir" && pwd)"
export TMPDIR="$build_dir/scratch"
# The exported source sits inside another Git repository. Never let CMake
# report that unrelated repository's commit as the reference version.
export GIT_CEILING_DIRECTORIES="$build_dir"
# This revision has two C++ expressions in a C CPU header. Export the pinned
# tree and apply only this build fix, never mutate the caller's checkout.
private_source="$build_dir/source"
if [[ ! -d "$private_source" ]]; then
    mkdir -p "$private_source"
    git -C "$source_dir" archive "$revision" | tar -x -C "$private_source"
    python3 - "$private_source/ggml/src/ggml-cpu/vec.h" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
for old, new in [
    ('std::min(GGML_CPU_FP16_TO_FP32(x[i]), limit)', 'fminf(GGML_CPU_FP16_TO_FP32(x[i]), limit)'),
    ('std::clamp(GGML_CPU_FP16_TO_FP32(g[i]), -limit, limit)', 'fminf(fmaxf(GGML_CPU_FP16_TO_FP32(g[i]), -limit), limit)'),
]:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
p.write_text(s)
PY
fi
# Check reused exports too: a stale or manually edited source tree must never
# silently become the pinned oracle. Read the archive incrementally.
python3 - "$source_dir" "$revision" "$private_source" "$build_dir" <<'PY'
import hashlib, json, pathlib, subprocess, sys, tarfile
checkout, revision, source, build = sys.argv[1:]
source, build = pathlib.Path(source), pathlib.Path(build)
patch = [
    (b'std::min(GGML_CPU_FP16_TO_FP32(x[i]), limit)', b'fminf(GGML_CPU_FP16_TO_FP32(x[i]), limit)'),
    (b'std::clamp(GGML_CPU_FP16_TO_FP32(g[i]), -limit, limit)', b'fminf(fmaxf(GGML_CPU_FP16_TO_FP32(g[i]), -limit), limit)'),
]
hashes, changed = {}, []
with subprocess.Popen(['git', '-C', checkout, 'archive', revision], stdout=subprocess.PIPE) as proc:
    with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
        for item in archive:
            if not item.isfile():
                continue
            expected = archive.extractfile(item).read()
            if item.name == 'ggml/src/ggml-cpu/vec.h':
                for before, after in patch:
                    assert expected.count(before) == 1
                    expected = expected.replace(before, after)
            path = source / item.name
            try:
                same = path.is_file() and path.read_bytes() == expected
            except OSError:
                same = False
            if not same:
                changed.append(str(path))
            hashes[item.name] = hashlib.sha256(expected).hexdigest()
    if proc.wait():
        raise SystemExit('Cannot verify pinned reference archive')
if changed:
    raise SystemExit('Pinned reference source differs: ' + ', '.join(changed[:10]))
(build / 'source-sha256.json').write_text(json.dumps(hashes, sort_keys=True, indent=2) + '\n')
PY
cmake -S "$private_source" -B "$build_dir/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$rocm" \
    -DCMAKE_HIP_COMPILER="$rocm/lib/llvm/bin/clang++" \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON \
    -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build "$build_dir/build" --target llama --parallel "${BUILD_JOBS:-8}"
c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic -Werror \
    -I"$private_source/include" -I"$private_source/ggml/include" \
    "$root/rdna4/llm/llama_reference.cpp" -L"$build_dir/build/bin" \
    -Wl,-rpath,"$build_dir/build/bin" -Wl,-rpath-link,"$rocm/lib" \
    -Wl,-rpath,"$rocm/lib" -lllama -lggml -lggml-base -o "$build_dir/llama_reference"
python3 - "$source_dir" "$build_dir" "$revision" <<'PY'
import hashlib, json, pathlib, subprocess, sys
source, build, revision = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
files = sorted({p.resolve() for p in (build / 'build/bin').glob('*.so*')})
files += [build / 'build/CMakeCache.txt', build / 'source/ggml/src/ggml-cpu/vec.h']
files += [build / 'llama_reference']
files += [build / 'source-sha256.json']
manifest = {'revision': revision, 'source': str(source.resolve()),
            'build': str(build.resolve()), 'sha256': {},
            'source_patch': 'vec.h: replace std::min/clamp with fminf/fmaxf in C FP16 SwiGLU clamp'}
for path in files:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    manifest['sha256'][str(path)] = digest.hexdigest()
manifest['compiler'] = subprocess.check_output(['c++', '--version'], text=True)
(build / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
