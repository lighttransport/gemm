#!/bin/bash
# Run under mpiexec: each rank copies only its converted blob to node-local LLIO.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SOURCE="${GLM52_CONVERT_DIR:-$HOME/models/glm52-2bit/a64fx-ep12-v1}"
DEST="${GLM5_STAGE_DIR:-/local/u14346/glm52-2bit-ep12}"
STATUS="${GLM5_STATUS_DIR:-.}"
rank="${GLM52_RANK:-${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${MV2_COMM_WORLD_RANK:-}}}}}"
[ -n "$rank" ] || { echo "stage_glm52: cannot determine MPI rank" >&2; exit 2; }
printf -v rr '%02d' "$rank"
src="$SOURCE/rank$rr.blob"
manifest="$SOURCE/rank$rr.manifest"
[ -s "$src" ] && [ -s "$manifest" ] || {
    echo "stage_glm52: missing converted rank $rr under $SOURCE" >&2; exit 3; }
grep -q '^# glm52-a64fx-ep12-v1' "$manifest" || {
    echo "stage_glm52: incompatible manifest $manifest" >&2; exit 3; }

mkdir -p "$DEST" "$STATUS"
python3 - "$src" "$manifest" "$DEST/rank$rr.blob" "$DEST/rank$rr.manifest" <<'PY'
import hashlib, os, shutil, sys
src, manifest, dst, mdst = sys.argv[1:]
tmp = f"{dst}.tmp.{os.getpid()}"
chunk = 64 << 20
sync_every = 1 << 30
expected = os.stat(src).st_size
if os.path.exists(dst) and os.stat(dst).st_size == expected:
    shutil.copyfile(manifest, mdst)
    print(f"reuse {dst} bytes={expected}")
    raise SystemExit(0)
h = hashlib.sha256()
sfd = os.open(src, os.O_RDONLY)
dfd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
off = 0
next_sync = sync_every
try:
    while off < expected:
        data = os.pread(sfd, min(chunk, expected - off), off)
        if not data:
            raise OSError(f"short read at {off}/{expected}")
        view = memoryview(data)
        done = 0
        while done < len(view):
            done += os.pwrite(dfd, view[done:], off + done)
        h.update(data)
        off += len(data)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(sfd, off-len(data), len(data), os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(dfd, off-len(data), len(data), os.POSIX_FADV_DONTNEED)
        if off >= next_sync:
            os.fdatasync(dfd)
            next_sync += sync_every
    os.fdatasync(dfd)
finally:
    os.close(sfd)
    os.close(dfd)
if off != expected:
    raise OSError(f"size mismatch {off} != {expected}")
os.replace(tmp, dst)
mtmp = f"{mdst}.tmp.{os.getpid()}"
shutil.copyfile(manifest, mtmp)
with open(mtmp, "rb") as f:
    os.fsync(f.fileno())
os.replace(mtmp, mdst)
print(f"staged {dst} bytes={off} sha256={h.hexdigest()}")
PY

avail_kb="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
[ "${avail_kb:-0}" -ge $((2*1024*1024)) ] || {
    echo "stage_glm52: rank $rank MemAvailable below 2 GiB" >&2; exit 4; }
printf 'rank %d staged bytes=%s MemAvailable_kB=%s\n' \
    "$rank" "$(stat -c %s "$DEST/rank$rr.blob")" "$avail_kb" \
    > "$STATUS/glm52_stage_rank$rr.txt"
