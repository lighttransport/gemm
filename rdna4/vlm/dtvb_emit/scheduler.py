"""SIA3-style slot scheduler with waitcnt computation.

This module is small on purpose. The mainloop's WMMA order is fixed (it
matches baseline `mm0_bf16_asm_barriersig_early.s` lines 198-239 — a
canonical accumulator-VGPR mapping that we don't reorder), so the
scheduler's job is *not* to permute WMMAs. It is to:

  1. Track ops on two AMDGPU hardware queues — VMEM ('load') and LDS ('ds').
  2. Emit `s_wait_loadcnt` / `s_wait_dscnt` directives for each consumer
     based on how many strictly-later issues of the same queue are still
     outstanding.
  3. Provide a small "Op" abstraction so the mainloop emitter can declare
     issue order + per-WMMA dependencies, and the scheduler does the
     waitcnt arithmetic instead of hand-counting.

The `wait_threshold(producer_idx, total_in_queue)` helper is the
single source-of-truth for `s_wait_*cnt` values:

    N = total_in_queue - 1 - producer_idx

means "after this point, only `N` items issued *after* producer_idx
are still pending; producer_idx itself has retired." The mainloop
emitter calls `consume(idx, queue)` and the scheduler emits the
correct waitcnt — no off-by-one risk in handwritten asm.

Op-graph + greedy-placement features described in the plan are
intentionally not implemented here: they would add a lot of surface
area for a fixed-order kernel. If a later iteration needs to permute
ops we can extend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Op:
    kind: str               # 'vmem'|'ds_load'|'ds_store'|'wmma'|'salu'|'barrier'
    asm: str                # rendered asm line (without trailing newline)
    queue: Optional[str] = None  # 'load' (VMEM) or 'ds' (LDS) or None
    issue_idx: int = -1     # set by Scheduler.issue()
    comment: str = ""

    def render(self) -> str:
        if self.comment:
            return f"{self.asm}  ; {self.comment}"
        return self.asm


@dataclass
class Scheduler:
    """Tracks issue order on the unified VMEM queue and the LDS queue.

    Note: AMDGPU has *one* VMEM counter shared by `buffer_load_*` and
    `global_load_*` — we put both on the 'load' queue.
    """
    load_count: int = 0   # strictly-issued loads so far
    ds_count: int = 0     # strictly-issued LDS ops so far (loads + stores; HW shares dscnt)
    lines: list[str] = field(default_factory=list)

    def issue_vmem(self, op: Op) -> int:
        """Issue a VMEM op; returns its position in load queue (0=oldest)."""
        assert op.queue == "load", op
        op.issue_idx = self.load_count
        self.load_count += 1
        self.lines.append(op.render())
        return op.issue_idx

    def issue_ds(self, op: Op) -> int:
        """Issue an LDS op (load or store); returns position in ds queue."""
        assert op.queue == "ds", op
        op.issue_idx = self.ds_count
        self.ds_count += 1
        self.lines.append(op.render())
        return op.issue_idx

    def issue_other(self, op: Op) -> None:
        """Emit a WMMA / SALU / barrier line — no counter effect."""
        assert op.queue is None, op
        self.lines.append(op.render())

    def wait_load(self, producer_issue_idx: int) -> None:
        """Insert s_wait_loadcnt so that load[producer_issue_idx] is retired.

        Threshold: at most (load_count - 1 - producer_issue_idx) loads
        still outstanding."""
        n = self.load_count - 1 - producer_issue_idx
        assert n >= 0, f"wait_load past end of queue (idx={producer_issue_idx}, count={self.load_count})"
        self.lines.append(f"\ts_wait_loadcnt {hex(n)}")

    def wait_ds(self, producer_issue_idx: int) -> None:
        n = self.ds_count - 1 - producer_issue_idx
        assert n >= 0, f"wait_ds past end of queue (idx={producer_issue_idx}, count={self.ds_count})"
        self.lines.append(f"\ts_wait_dscnt {hex(n)}")

    def wait_load_max(self, n: int) -> None:
        """Direct waitcnt — caller computed N themselves."""
        self.lines.append(f"\ts_wait_loadcnt {hex(n)}")

    def wait_ds_max(self, n: int) -> None:
        self.lines.append(f"\ts_wait_dscnt {hex(n)}")

    def emit_raw(self, line: str) -> None:
        """Drop a literal asm line (labels, comments, directives) into output."""
        self.lines.append(line)

    def emit_comment(self, text: str) -> None:
        self.lines.append(f"\t; {text}")

    def render(self) -> str:
        return "\n".join(self.lines)
