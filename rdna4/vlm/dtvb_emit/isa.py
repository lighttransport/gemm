"""gfx1201 instruction mnemonic builders + waitcnt helpers.

Each function returns a single-line asm string. Comments may be appended
by the caller via `+ "  ; comment"`.
"""
from __future__ import annotations


# ---- VMEM (buffer_load) ------------------------------------------------------

def buffer_load_b128(vdst_lo: int, voff: int, srd_base: int, soffset: str = "null") -> str:
    """`buffer_load_b128 v[vdst_lo:vdst_lo+3], v<voff>, s[srd_base:srd_base+3], <soffset> offen`.

    soffset is either 'null' (no SGPR offset) or 's<N>'.
    """
    return (f"\tbuffer_load_b128 v[{vdst_lo}:{vdst_lo+3}], v{voff}, "
            f"s[{srd_base}:{srd_base+3}], {soffset} offen")


def global_load_b128(vdst_lo: int, vaddr_lo: int, offset: int = 0) -> str:
    suffix = f" offset:{offset}" if offset else ""
    return f"\tglobal_load_b128 v[{vdst_lo}:{vdst_lo+3}], v[{vaddr_lo}:{vaddr_lo+1}], off{suffix}"


def global_store_b128(vaddr_lo: int, vsrc_lo: int, offset: int = 0) -> str:
    suffix = f" offset:{offset}" if offset else ""
    return f"\tglobal_store_b128 v[{vaddr_lo}:{vaddr_lo+1}], v[{vsrc_lo}:{vsrc_lo+3}], off{suffix}"


# ---- LDS (ds_load / ds_store) ------------------------------------------------

def ds_load_b128(vdst_lo: int, vaddr: int, offset: int = 0) -> str:
    suffix = f" offset:{offset}" if offset else ""
    return f"\tds_load_b128 v[{vdst_lo}:{vdst_lo+3}], v{vaddr}{suffix}"


def ds_store_b128(vaddr: int, vsrc_lo: int, offset: int = 0) -> str:
    suffix = f" offset:{offset}" if offset else ""
    return f"\tds_store_b128 v{vaddr}, v[{vsrc_lo}:{vsrc_lo+3}]{suffix}"


# ---- WMMA --------------------------------------------------------------------

def wmma_f32_16x16x16_bf16(vdst_lo: int, vsrc_b_lo: int, vsrc_a_lo: int) -> str:
    """V_WMMA_F32_16x16x16_BF16 v[D0:D7], v[B0:B3], v[A0:A3], v[D0:D7] (acc as src+dst).

    Matches hipBLASLt operand order: B-fragment first, A-fragment second.
    """
    return (f"\tv_wmma_f32_16x16x16_bf16 "
            f"v[{vdst_lo}:{vdst_lo+7}], "
            f"v[{vsrc_b_lo}:{vsrc_b_lo+3}], "
            f"v[{vsrc_a_lo}:{vsrc_a_lo+3}], "
            f"v[{vdst_lo}:{vdst_lo+7}]")


# ---- waits / barriers --------------------------------------------------------

def s_wait_loadcnt(n: int) -> str:
    """Wait for VMEM load counter ≤ n (i.e. at most n loads still in flight)."""
    assert 0 <= n <= 63
    return f"\ts_wait_loadcnt {hex(n)}"


def s_wait_dscnt(n: int) -> str:
    assert 0 <= n <= 63
    return f"\ts_wait_dscnt {hex(n)}"


def s_wait_kmcnt(n: int) -> str:
    assert 0 <= n <= 63
    return f"\ts_wait_kmcnt {hex(n)}"


def s_barrier_signal(arg: int = -1) -> str:
    return f"\ts_barrier_signal {arg}"


def s_barrier_wait(mask: int = 0xffff) -> str:
    return f"\ts_barrier_wait {hex(mask)}"


# ---- SALU --------------------------------------------------------------------

def s_add_co_u32(dst: int, src0: int, src1: str) -> str:
    return f"\ts_add_co_u32 s{dst}, s{src0}, {src1}"


def s_add_co_ci_u32(dst: int, src0: int, src1: str) -> str:
    return f"\ts_add_co_ci_u32 s{dst}, s{src0}, {src1}"


def s_sub_co_u32(dst: int, src0: int, src1: str) -> str:
    return f"\ts_sub_co_u32 s{dst}, s{src0}, {src1}"


def s_sub_co_ci_u32(dst: int, src0: int, src1: str) -> str:
    return f"\ts_sub_co_ci_u32 s{dst}, s{src0}, {src1}"


def s_cmp_eq_u32(s0: int, s1: str) -> str:
    return f"\ts_cmp_eq_u32 s{s0}, {s1}"


def s_cmp_eq_i32(s0: int, s1: str) -> str:
    return f"\ts_cmp_eq_i32 s{s0}, {s1}"


def s_cselect_b32(dst: int, src0: str, src1: str) -> str:
    return f"\ts_cselect_b32 s{dst}, {src0}, {src1}"


def s_cbranch_scc0(label: str) -> str:
    return f"\ts_cbranch_scc0 {label}"


def s_cbranch_scc1(label: str) -> str:
    return f"\ts_cbranch_scc1 {label}"


def s_branch(label: str) -> str:
    return f"\ts_branch {label}"


def s_mov_b32(dst: int, src: str) -> str:
    return f"\ts_mov_b32 s{dst}, {src}"


def s_mov_b64(dst_lo: int, src: str) -> str:
    return f"\ts_mov_b64 s[{dst_lo}:{dst_lo+1}], {src}"


def s_lshl_b32(dst: int, src: str, shift: int) -> str:
    return f"\ts_lshl_b32 s{dst}, {src}, {shift}"


def s_lshr_b32(dst: int, src: str, shift: int) -> str:
    return f"\ts_lshr_b32 s{dst}, {src}, {shift}"


def s_load_b32(dst: int, base_lo: int, offset: int) -> str:
    return f"\ts_load_b32 s{dst}, s[{base_lo}:{base_lo+1}], {hex(offset)}"


def s_load_b64(dst_lo: int, base_lo: int, offset: int) -> str:
    return f"\ts_load_b64 s[{dst_lo}:{dst_lo+1}], s[{base_lo}:{base_lo+1}], {hex(offset)}"


def s_load_b128(dst_lo: int, base_lo: int, offset: int) -> str:
    return f"\ts_load_b128 s[{dst_lo}:{dst_lo+3}], s[{base_lo}:{base_lo+1}], {hex(offset)}"


def s_load_b256(dst_lo: int, base_lo: int, offset: int) -> str:
    return f"\ts_load_b256 s[{dst_lo}:{dst_lo+7}], s[{base_lo}:{base_lo+1}], {hex(offset)}"


# ---- VALU --------------------------------------------------------------------

def v_mov_b32(dst: int, src: str) -> str:
    return f"\tv_mov_b32_e32 v{dst}, {src}"


def v_dual_mov_b32(d0: int, s0: str, d1: int, s1: str) -> str:
    return f"\tv_dual_mov_b32 v{d0}, {s0} :: v_dual_mov_b32 v{d1}, {s1}"


def v_or_b32(dst: int, src0: str, src1: str) -> str:
    return f"\tv_or_b32_e32 v{dst}, {src0}, {src1}"


def v_xor_b32(dst: int, src0: str, src1: str) -> str:
    return f"\tv_xor_b32_e32 v{dst}, {src0}, {src1}"


def v_lshlrev_b32(dst: int, shift: str, src: str) -> str:
    return f"\tv_lshlrev_b32_e32 v{dst}, {shift}, {src}"


def v_lshrrev_b32(dst: int, shift: str, src: str) -> str:
    return f"\tv_lshrrev_b32_e32 v{dst}, {shift}, {src}"


def v_and_b32(dst: int, src0: str, src1: str) -> str:
    return f"\tv_and_b32_e32 v{dst}, {src0}, {src1}"


def v_mad_co_i64_i32(dst_lo: int, src0: str, src1: str, src2_lo: int) -> str:
    return (f"\tv_mad_co_i64_i32 v[{dst_lo}:{dst_lo+1}], null, "
            f"{src0}, {src1}, s[{src2_lo}:{src2_lo+1}]")


# ---- Pseudo / labels ---------------------------------------------------------

def label(name: str) -> str:
    return f"{name}:"


def comment(text: str) -> str:
    return f"\t; {text}"


# ---- Waitcnt encoder ---------------------------------------------------------

def compute_loadcnt_after_consume(load_idx: int, total_loads: int) -> int:
    """For a consumer that needs load[load_idx] to retire, return the loadcnt
    threshold to wait on, given that all `total_loads` were issued in order
    starting from load_idx=0 most-recent (i.e. queue is LIFO from issue
    perspective — counter decrements as oldest retires).

    AMDGPU convention: `s_wait_loadcnt N` means "wait until at most N loads
    are still outstanding". If we issued `total_loads` and want load[i] to
    have retired (where i is 0-indexed by issue order, oldest first), we need
    `total_loads - i - 1` later loads to still be outstanding. So:
        N = total_loads - 1 - load_idx
    """
    assert 0 <= load_idx < total_loads
    return total_loads - 1 - load_idx
