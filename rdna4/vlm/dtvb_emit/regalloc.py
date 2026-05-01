"""Named VGPR/SGPR register windows for the mm0 DTVB kernel.

Mirrors the layout decoded from hipBLASLt algo-73624 (see
hipblaslt_mm0_alik_bljk_dtvb_groundtruth.s.annot register windows section).

Asserts non-overlap so a typo in any window assignment is caught at
generator-import time, not at kernel runtime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VGPRWindow:
    name: str
    base: int
    size: int

    @property
    def end(self) -> int:
        return self.base + self.size

    def __repr__(self) -> str:
        return f"VGPRWindow({self.name}: v[{self.base}..{self.end - 1}])"


# ---- C accumulator (16 WMMA × 8 fp32 = 128 VGPRs) ----------------------------
# Each WMMA writes 8 fp32 to v[D0:D7]. WMMAs are indexed (m=0..3, n=0..3)
# with (m*4 + n) selecting one of 16; per-K-half they reuse the same C window
# (acc-accumulate pattern: WMMA reads C, multiplies B*A, adds, writes C).
C_ACC = VGPRWindow("C_acc", base=0, size=128)


# ---- A LDS-load destinations (16 VGPRs each for K0 and K1) -------------------
# K0 fragments (used by K-half-0 WMMAs):
A_LDS_K0_FRAGS = [
    VGPRWindow(f"A_lds_K0_{i}", base=132 + 4 * i, size=4) for i in range(4)
]
# K1 fragments (used by K-half-1 WMMAs):
A_LDS_K1_FRAGS = [
    VGPRWindow(f"A_lds_K1_{i}", base=148 + 4 * i, size=4) for i in range(4)
]


# ---- A_next buffer-load destination (4 b128 = 16 VGPRs) ---------------------
A_NEXT_FRAGS = [
    VGPRWindow(f"A_next_{i}", base=166 + 4 * i, size=4) for i in range(4)
]


# ---- B_dir current + next windows (8 b128 each = 32 VGPRs each) -------------
# PGR1+PLR1 software-pipelining via reg renaming: the 2× outer-K-unroll body
# writes B_NEXT in iter-1, and uses it as B_CUR in iter-2 (and vice versa).
# Per K-half each "current" window holds 4 fragments (B0..B3), so 8 fragments per outer K.
B_DIR_A = [VGPRWindow(f"B_dir_A_{i}", base=182 + 4 * i, size=4) for i in range(8)]
B_DIR_B = [VGPRWindow(f"B_dir_B_{i}", base=214 + 4 * i, size=4) for i in range(8)]


# ---- Address VGPRs -----------------------------------------------------------
V_BUFFER_ADDR_A = 128   # offen-addressing offset for A buffer_load
V_BUFFER_ADDR_B = 129   # offen-addressing offset for B buffer_load
V_LDS_STORE_BASE = 130  # XOR-toggled w/ 0x4000 each iter (A double-buffer)
V_LDS_LOAD_BASE = 131   # XOR-toggled w/ 0x4000 each iter (A double-buffer)


# ---- SGPR layout -------------------------------------------------------------
# s[0:1]   = kernarg base ptr (provided by hardware)
# s[2:11]  = kernarg-loaded scalars (bias ptr, M, N, K, ...)
# s[12]    = K-loop counter (decrements per outer K iter)
# s[16:19] = reserved / scratch
# s[20:21] = workgroup ID overrides (used in addressing)
S_KERNARG_BASE = 0   # hardware-provided (s[0:1])

# A buffer SRD (256-bit = 4 SGPRs)
S_A_SRD = 48
# B buffer SRD
S_B_SRD = 52

# Residual K counters for SRD-clamping
S_A_K_RESIDUAL = 56
S_B_K_RESIDUAL = 58

# K-step constants
S_K_STEP_FULL_LO = 60
S_K_STEP_FULL_HI = 61
S_K_STEP_FULL_B_LO = 62
S_K_STEP_FULL_B_HI = 63
S_K_STEP_TAIL_LO = 64
S_K_STEP_TAIL_HI = 65

# Per-load SGPR offsets (replaces v_add_co address arithmetic)
S_A_OFFSET = [66, 67, 68]                  # 3 offsets; load 0 uses null = 0
S_B_OFFSET_GROUP_A = [69, 70, 71]          # 3 offsets; load 0 uses null
S_B_OFFSET_GROUP_B = [72, 73, 74, 75]      # 4 offsets

# Cselect tmps
S_CSEL_TMP_LO = 76
S_CSEL_TMP_HI = 77


# ---- Validation --------------------------------------------------------------

def _all_vgpr_windows() -> list[VGPRWindow]:
    """Returns every VGPRWindow + range of address VGPRs, for overlap checks."""
    out = [C_ACC]
    out.extend(A_LDS_K0_FRAGS)
    out.extend(A_LDS_K1_FRAGS)
    out.extend(A_NEXT_FRAGS)
    out.extend(B_DIR_A)
    out.extend(B_DIR_B)
    out.extend([
        VGPRWindow("v_buffer_addr_A", base=V_BUFFER_ADDR_A, size=1),
        VGPRWindow("v_buffer_addr_B", base=V_BUFFER_ADDR_B, size=1),
        VGPRWindow("v_lds_store_base", base=V_LDS_STORE_BASE, size=1),
        VGPRWindow("v_lds_load_base", base=V_LDS_LOAD_BASE, size=1),
    ])
    return out


def assert_no_overlap() -> None:
    """Run at module import time to catch register-window overlaps."""
    windows = _all_vgpr_windows()
    occupied: dict[int, str] = {}
    for w in windows:
        for vgpr in range(w.base, w.end):
            if vgpr in occupied:
                raise ValueError(
                    f"VGPR overlap at v{vgpr}: {occupied[vgpr]} vs {w.name}"
                )
            occupied[vgpr] = w.name


def total_vgpr_count() -> int:
    """Highest VGPR index used + 1 (for .amdhsa_next_free_vgpr)."""
    return max(w.end for w in _all_vgpr_windows())


# Eagerly validate
assert_no_overlap()
