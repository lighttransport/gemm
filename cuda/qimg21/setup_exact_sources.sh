#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTORCH_DIR="${PYTORCH_SOURCE:-$ROOT/tmp/pytorch-src}"
FLASH_DIR="${FLASH_ATTN_SOURCE:-$ROOT/tmp/flash-attention-src}"
PYTORCH_REV=08187d9e0fba026dc8217405802ab5381dc88d90
FLASH_REV=14c377950125c70b7a9dabf9c561fca53715ac7d
# FlashAttention's recorded CUTLASS revision predates Blackwell support.  This
# newer pinned CUTLASS revision is the one validated by the SM120 plugins.
CUTLASS_REV=e05f953a5b3d38adc240df2ff928e0421c2abba3
VERIFY_ONLY=0
[[ "${1:-}" == "--verify" ]] && VERIFY_ONLY=1

die() { echo "setup_exact_sources: $*" >&2; exit 1; }

ensure_checkout() {
    local name=$1 url=$2 dir=$3 rev=$4
    if [[ ! -d "$dir/.git" ]]; then
        (( VERIFY_ONLY == 0 )) || die "$name checkout missing: $dir (run make setup-exact-sources)"
        mkdir -p "$(dirname "$dir")"
        git clone --filter=blob:none "$url" "$dir"
    fi
    if ! git -C "$dir" cat-file -e "$rev^{commit}" 2>/dev/null; then
        (( VERIFY_ONLY == 0 )) || die "$name revision $rev is unavailable in $dir"
        git -C "$dir" fetch --depth=1 origin "$rev"
    fi
    local actual
    actual="$(git -C "$dir" rev-parse HEAD)"
    if [[ "$actual" != "$rev" ]]; then
        (( VERIFY_ONLY == 0 )) || die "$name is $actual, expected $rev"
        [[ -z "$(git -C "$dir" status --porcelain)" ]] || die "$name has local changes; refusing checkout in $dir"
        git -C "$dir" checkout --detach "$rev"
    fi
}

ensure_checkout PyTorch https://github.com/pytorch/pytorch.git "$PYTORCH_DIR" "$PYTORCH_REV"
ensure_checkout FlashAttention https://github.com/Dao-AILab/flash-attention.git "$FLASH_DIR" "$FLASH_REV"

git -C "$PYTORCH_DIR" diff --quiet || die "PyTorch checkout has tracked source modifications"
# The FlashAttention CUTLASS gitlink is intentionally replaced below.  Verify
# the attention source independently so that local header edits cannot pass as
# an exact checkout merely because HEAD still names the pinned commit.
git -C "$FLASH_DIR" diff --quiet -- csrc/flash_attn ||
    die "FlashAttention checkout has tracked source modifications"

CUTLASS_DIR="$FLASH_DIR/csrc/cutlass"
if [[ ! -d "$CUTLASS_DIR/.git" && ! -f "$CUTLASS_DIR/.git" ]]; then
    (( VERIFY_ONLY == 0 )) || die "CUTLASS checkout missing: $CUTLASS_DIR"
    git -C "$FLASH_DIR" submodule update --init csrc/cutlass
fi
if ! git -C "$CUTLASS_DIR" cat-file -e "$CUTLASS_REV^{commit}" 2>/dev/null; then
    (( VERIFY_ONLY == 0 )) || die "CUTLASS revision $CUTLASS_REV is unavailable"
    git -C "$CUTLASS_DIR" fetch --depth=1 origin "$CUTLASS_REV"
fi
CUTLASS_ACTUAL="$(git -C "$CUTLASS_DIR" rev-parse HEAD)"
if [[ "$CUTLASS_ACTUAL" != "$CUTLASS_REV" ]]; then
    (( VERIFY_ONLY == 0 )) || die "CUTLASS is $CUTLASS_ACTUAL, expected $CUTLASS_REV"
    [[ -z "$(git -C "$CUTLASS_DIR" status --porcelain)" ]] || die "CUTLASS has local changes; refusing checkout"
    git -C "$CUTLASS_DIR" checkout --detach "$CUTLASS_REV"
fi
git -C "$CUTLASS_DIR" diff --quiet || die "CUTLASS checkout has tracked source modifications"

echo "exact sources verified:"
echo "  PyTorch       $PYTORCH_REV"
echo "  FlashAttention $FLASH_REV"
echo "  CUTLASS       $CUTLASS_REV"
