#!/usr/bin/env python3
"""Capacity and admission planning for the Qwen3.8-27B Q38D layout.

This is a planning tool.  It deliberately does not start ``q38d`` or claim
that the current engine can execute the planned layout.  The current source
stores K/V as ``float`` and rejects a TP12 state consumer, so an INT8-KV,
TP12-prefill to TP4-decode plan is normally reported as ``BLOCKED`` even when
the arithmetic capacity estimate fits.

The default cases are the three requested independent-context shapes::

    1M x 8, 512K x 16, and 256K x 32

The decode side is three disjoint TP4 groups over twelve nodes.  Contexts are
balanced across those groups and each TP4 rank is charged for its local KV,
model, SSM state, scratch, and a configurable free-memory reserve.  Prefill
is modelled as one Q38D TP12 context at a time; ``--prefill-in-flight`` can be
used to make a more demanding estimate.

All numbers derived from Q38D source constants are kept here rather than
duplicated in the engine.  The full NVFP4 allocation is a measured value from
``tmp/q38-fast/gen-fp4.log``.  TP4/TP12 resident model sizes are not measured
by the repository, so they must be supplied by the caller.  Missing values
fail closed as ``UNKNOWN``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional


# Q38D source constants (q38d_engine.c).  Keep these explicit so a review can
# compare every capacity term with the engine without importing C headers.
NT = 48
NCMG = 4
PER = 12
EMBD = 5120
NFF = 17408
NLAYER = 64
NHEAD = 24
NKV = 4
HD = 256
QKVD = 10240
DINNER = 6144
DS = 128
NGROUP = 16
NVH = 48
NATTN = 16
KVB = 16
SSM_RING = 1  # ordinary decode; speculative mode is not a TP plan

NODES = 12
PREFILL_TP = 12
DECODE_TP = 4
DECODE_GROUPS = 3
HBM_GIB = 32.0
OUTPUT_RESERVE_TOKENS = 8192
FREE_RESERVE_GIB = 2.0
SCRATCH_GIB = 1.0
LOAD_SCRATCH_GIB = 4.0

# q38-fast/gen-fp4.log: "lowbit: final anonymous allocation 18.212GB".
# The log uses decimal GB; all values exposed by this module use GiB.
FULL_MODEL_ALLOCATION_GB = 18.212
BYTES_PER_GIB = 1 << 30
FULL_MODEL_ALLOCATION_GIB = FULL_MODEL_ALLOCATION_GB * 1_000_000_000 / BYTES_PER_GIB


@dataclass(frozen=True)
class Workload:
    """One context length and the number of independent requests."""

    name: str
    context_tokens: int
    contexts: int


WORKLOADS = (
    Workload("1m_x8", 1_000_000, 8),
    Workload("512k_x16", 512_000, 16),
    Workload("256k_x32", 256_000, 32),
)


@dataclass(frozen=True)
class PlannerConfig:
    """Inputs that affect admission.

    ``tp4_model_gib`` and ``tp12_model_gib`` are intentionally optional.  No
    TP-resident Q38D model measurement is checked into this tree.  Treating
    the 18.212 GB full image as a TP-resident value would be safe but would
    hide the missing measurement and is misleading for tensor parallel runs.
    """

    nodes: int = NODES
    prefill_tp: int = PREFILL_TP
    decode_tp: int = DECODE_TP
    decode_groups: int = DECODE_GROUPS
    hbm_gib: float = HBM_GIB
    output_reserve_tokens: int = OUTPUT_RESERVE_TOKENS
    kv_dtype: str = "int8"
    tp4_model_gib: Optional[float] = None
    tp12_model_gib: Optional[float] = None
    scratch_gib: Optional[float] = SCRATCH_GIB
    load_scratch_gib: Optional[float] = LOAD_SCRATCH_GIB
    free_reserve_gib: float = FREE_RESERVE_GIB
    prefill_in_flight: int = 1
    # These are capability gates, not memory assumptions.  They default to
    # false because the current q38d source does not implement either path.
    engine_int8_kv: bool = False
    engine_tp12_handoff: bool = False


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_config(config: PlannerConfig) -> None:
    if config.nodes != NODES:
        raise ValueError(f"Q38D plan is for {NODES} nodes, got {config.nodes}")
    if config.prefill_tp != PREFILL_TP:
        raise ValueError(f"prefill TP must be {PREFILL_TP}, got {config.prefill_tp}")
    if config.decode_tp != DECODE_TP:
        raise ValueError(f"decode TP must be {DECODE_TP}, got {config.decode_tp}")
    if config.decode_groups != DECODE_GROUPS:
        raise ValueError(f"decode groups must be {DECODE_GROUPS}, got {config.decode_groups}")
    if config.kv_dtype not in {"int8", "fp32"}:
        raise ValueError("kv_dtype must be int8 or fp32")
    if config.output_reserve_tokens < 0:
        raise ValueError("output_reserve_tokens must be non-negative")
    if config.prefill_in_flight < 1:
        raise ValueError("prefill_in_flight must be positive")
    for name in ("hbm_gib", "free_reserve_gib"):
        _require_positive(name, getattr(config, name))
    for name in ("tp4_model_gib", "tp12_model_gib", "scratch_gib", "load_scratch_gib"):
        value = getattr(config, name)
        if value is not None:
            _require_positive(name, value)


def gibibytes(byte_count: int | float) -> float:
    return float(byte_count) / BYTES_PER_GIB


def _dtype_bytes(dtype: str) -> int:
    if dtype == "int8":
        return 1
    if dtype == "fp32":
        return 4
    raise ValueError(f"unsupported KV dtype: {dtype}")


def tp_replicates_mixers(tp: int) -> bool:
    """Mirror q38d_engine.c's ``tp_replicate_mixers`` condition."""

    if tp < 1:
        raise ValueError("TP must be positive")
    return tp > 1 and tp not in (2, 4)


def local_kv_heads(tp: int) -> int:
    """Return K/V heads resident on one rank in the Q38D layout.

    TP12 takes the engine's replicated-mixer branch, which keeps all four KV
    heads on every rank.  TP4 shards the four heads one per rank.  The same
    formula is useful for the TP2 reference path.
    """

    if tp_replicates_mixers(tp):
        return NKV
    if NKV % tp:
        raise ValueError(f"Q38D KV heads ({NKV}) are not divisible by TP{tp}")
    return NKV // tp


def kv_bytes_per_token(tp: int, dtype: str = "int8") -> int:
    """K+V bytes per token per rank, across all sixteen attention layers."""

    return NATTN * local_kv_heads(tp) * HD * 2 * _dtype_bytes(dtype)


def ssm_memory_breakdown(tp: int) -> dict[str, int | float]:
    """Persistent per-rank SSM/convolution state from q38d_engine.c.

    ``conv_state`` is allocated for all 64 layers.  Per-head state is created
    only for SSM layers (48 layers, because ``L->ssm = (l % 4) != 3``) and for
    the rank's local heads.  TP12 replicates mixer state, as in the engine.
    ``cmg_alloc`` rounding and allocator metadata are not represented, so the
    returned total is a lower bound for this component.
    """

    ssm_layers = NLAYER - NATTN
    heads = NVH if tp_replicates_mixers(tp) else NVH // tp
    conv_state = NLAYER * (4 * QKVD) * 4  # calloc(4 * QKVD, sizeof(float))
    ssm_buf = ssm_layers * heads * SSM_RING * (DS * DS + 2 * DS) * 4
    conv_hist = ssm_layers * heads * (8 * 384) * 4
    conv_wl = ssm_layers * heads * (4 * 384) * 4
    total = conv_state + ssm_buf + conv_hist + conv_wl
    return {
        "conv_state_bytes": conv_state,
        "ssm_buf_bytes": ssm_buf,
        "conv_hist_bytes": conv_hist,
        "conv_wl_bytes": conv_wl,
        "total_bytes": total,
        "total_gib": gibibytes(total),
        "ssm_layers": ssm_layers,
        "local_ssm_heads": heads,
        "source": "q38d_engine.c constants; allocator rounding omitted",
    }


def balanced_group_sizes(contexts: int, groups: int = DECODE_GROUPS) -> list[int]:
    """Distribute independent contexts over disjoint decode groups."""

    if contexts < 0 or groups < 1:
        raise ValueError("contexts must be non-negative and groups positive")
    q, r = divmod(contexts, groups)
    return [q + (1 if i < r else 0) for i in range(groups)]


def _optional_add(*values: Optional[float]) -> Optional[float]:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _memory_fit(peak_gib: Optional[float], config: PlannerConfig) -> Optional[bool]:
    if peak_gib is None:
        return None
    return peak_gib <= config.hbm_gib - config.free_reserve_gib


def _capacity_status(
    fit: Optional[bool],
    *,
    capability_ok: bool,
    capability_name: str,
) -> tuple[str, list[str]]:
    """Turn a capacity result and capability gate into an admission result."""

    if fit is None:
        return "UNKNOWN", ["TP-resident model budget is missing"]
    if not fit:
        return "REJECT", ["peak exceeds HBM after free-memory reserve"]
    if not capability_ok:
        return "BLOCKED", [capability_name]
    return "ADMITTED", []


def _model_for_tp(config: PlannerConfig, tp: int) -> Optional[float]:
    if tp == 4:
        return config.tp4_model_gib
    if tp == 12:
        return config.tp12_model_gib
    raise ValueError(f"unsupported Q38D planning TP: {tp}")


def _model_reference() -> dict[str, Any]:
    return {
        "full_allocation_gb_decimal": FULL_MODEL_ALLOCATION_GB,
        "full_allocation_gib": FULL_MODEL_ALLOCATION_GIB,
        "source": "measured tmp/q38-fast/gen-fp4.log",
        "tp_resident_source": "unknown; supply --tp4-model-gib and --tp12-model-gib",
    }


def _handoff_bytes(context_tokens: int, config: PlannerConfig) -> dict[str, Any]:
    """Logical state volume for one context entering a TP4 decode group.

    q38d_state.inc currently writes FP32 K/V.  The INT8 number is a planning
    projection with one byte per K/V value and no scale metadata.  It is thus
    labelled as an estimate and is never silently treated as executable.
    """

    fp32_per_rank = kv_bytes_per_token(DECODE_TP, "fp32")
    int8_per_rank = kv_bytes_per_token(DECODE_TP, "int8")
    ssm_global = int(ssm_memory_breakdown(DECODE_TP)["total_bytes"]) * DECODE_TP
    result: dict[str, Any] = {
        "context_tokens": context_tokens,
        "ssm_bytes_global": ssm_global,
        "fp32_bytes_global": context_tokens * fp32_per_rank * DECODE_TP + ssm_global,
        "fp32_gib_global": gibibytes(context_tokens * fp32_per_rank * DECODE_TP + ssm_global),
        "int8_bytes_global_estimate": context_tokens * int8_per_rank * DECODE_TP + ssm_global,
        "int8_gib_global_estimate": gibibytes(context_tokens * int8_per_rank * DECODE_TP + ssm_global),
        "int8_note": "projection; scale/zero-point metadata and an INT8 state format are not in q38d_state.inc",
    }
    return result


def plan_workload(workload: Workload, config: PlannerConfig) -> dict[str, Any]:
    """Return a JSON-serializable capacity row for one workload."""

    _validate_config(config)
    if workload.context_tokens < 1 or workload.contexts < 1:
        raise ValueError("workload dimensions must be positive")

    groups = balanced_group_sizes(workload.contexts, config.decode_groups)
    max_group_contexts = max(groups)
    decode_tokens_reserved = workload.context_tokens + config.output_reserve_tokens

    decode_kv_bpt = kv_bytes_per_token(config.decode_tp, config.kv_dtype)
    decode_kv_bytes = max_group_contexts * decode_tokens_reserved * decode_kv_bpt
    decode_kv_gib = gibibytes(decode_kv_bytes)
    decode_ssm = ssm_memory_breakdown(config.decode_tp)
    decode_model = _model_for_tp(config, config.decode_tp)
    decode_runtime = _optional_add(
        decode_model,
        float(decode_ssm["total_gib"]),
        config.scratch_gib,
        decode_kv_gib,
    )
    decode_load_peak = _optional_add(
        FULL_MODEL_ALLOCATION_GIB,
        config.load_scratch_gib,
    )
    # During load the full lowbit image is present before TP descriptors are
    # ready.  Include that known peak in every node-local admission check.
    decode_peak = None if decode_runtime is None or decode_load_peak is None else max(decode_runtime, decode_load_peak)
    decode_fit = _memory_fit(decode_peak, config)
    decode_status, decode_gates = _capacity_status(
        decode_fit,
        capability_ok=config.engine_int8_kv or config.kv_dtype != "int8",
        capability_name="current q38d_engine.c has FP32 KV only",
    )

    # Q38D TP12 takes the replicated SSM/attention branch.  Prefill is
    # intentionally one context at a time unless the caller asks for more.
    prefill_kv_bpt = kv_bytes_per_token(config.prefill_tp, config.kv_dtype)
    prefill_kv_bytes = config.prefill_in_flight * workload.context_tokens * prefill_kv_bpt
    prefill_kv_gib = gibibytes(prefill_kv_bytes)
    prefill_ssm = ssm_memory_breakdown(config.prefill_tp)
    prefill_model = _model_for_tp(config, config.prefill_tp)
    prefill_runtime = _optional_add(
        prefill_model,
        float(prefill_ssm["total_gib"]),
        config.scratch_gib,
        prefill_kv_gib,
    )
    prefill_peak = None if prefill_runtime is None or decode_load_peak is None else max(prefill_runtime, decode_load_peak)
    prefill_fit = _memory_fit(prefill_peak, config)
    prefill_status, prefill_gates = _capacity_status(
        prefill_fit,
        capability_ok=(config.engine_int8_kv or config.kv_dtype != "int8") and config.engine_tp12_handoff,
        capability_name="TP12 prefill -> TP4 state handoff is not supported by current q38d state validation",
    )
    if config.kv_dtype == "int8" and not config.engine_int8_kv and prefill_fit:
        prefill_gates.append("current q38d_engine.c allocates float K/V; INT8 KV is a projection")

    row: dict[str, Any] = {
        "name": workload.name,
        "context_tokens": workload.context_tokens,
        "contexts": workload.contexts,
        "decode_group_sizes": groups,
        "max_contexts_per_decode_group": max_group_contexts,
        "output_reserve_tokens_per_context": config.output_reserve_tokens,
        "model": {
            "full_allocation_gib": FULL_MODEL_ALLOCATION_GIB,
            "tp4_resident_gib": decode_model,
            "tp12_resident_gib": prefill_model,
            "load_scratch_gib": config.load_scratch_gib,
        },
        "decode": {
            "topology": "3 independent TP4 groups over 12 nodes",
            "kv_dtype": config.kv_dtype,
            "kv_bytes_per_token_per_rank": decode_kv_bpt,
            "reserved_tokens_per_context": decode_tokens_reserved,
            "kv_bytes_per_node": decode_kv_bytes,
            "kv_gib_per_node": decode_kv_gib,
            "ssm": decode_ssm,
            "scratch_gib": config.scratch_gib,
            "runtime_gib_per_node": decode_runtime,
            "load_peak_gib_per_node": decode_load_peak,
            "peak_gib_per_node": decode_peak,
            "available_gib_after_reserve": config.hbm_gib - config.free_reserve_gib,
            "fits": decode_fit,
            "status": decode_status,
            "gates": decode_gates,
        },
        "prefill": {
            "topology": "one Q38D TP12 context at a time",
            "prefill_in_flight": config.prefill_in_flight,
            "kv_dtype": config.kv_dtype,
            "kv_bytes_per_token_per_rank": prefill_kv_bpt,
            "prompt_tokens_per_context": workload.context_tokens,
            "kv_bytes_per_node": prefill_kv_bytes,
            "kv_gib_per_node": prefill_kv_gib,
            "ssm": prefill_ssm,
            "scratch_gib": config.scratch_gib,
            "runtime_gib_per_node": prefill_runtime,
            "load_peak_gib_per_node": decode_load_peak,
            "peak_gib_per_node": prefill_peak,
            "available_gib_after_reserve": config.hbm_gib - config.free_reserve_gib,
            "fits": prefill_fit,
            "status": prefill_status,
            "gates": prefill_gates,
            "handoff": _handoff_bytes(workload.context_tokens, config),
        },
    }
    return row


def make_plan(config: PlannerConfig = PlannerConfig(), workloads: Iterable[Workload] = WORKLOADS) -> dict[str, Any]:
    """Build the complete plan and source/capability notes."""

    _validate_config(config)
    rows = [plan_workload(workload, config) for workload in workloads]
    return {
        "planner": "q38d_capacity",
        "version": 1,
        "assumptions": {
            "nodes": config.nodes,
            "prefill_tp": config.prefill_tp,
            "decode_tp": config.decode_tp,
            "decode_groups": config.decode_groups,
            "hbm_gib_per_node": config.hbm_gib,
            "free_reserve_gib": config.free_reserve_gib,
            "kv_dtype": config.kv_dtype,
            "output_reserve_tokens": config.output_reserve_tokens,
            "prefill_in_flight": config.prefill_in_flight,
            "scratch_gib": config.scratch_gib,
            "load_scratch_gib": config.load_scratch_gib,
        },
        "model_reference": _model_reference(),
        "source_capabilities": {
            "engine_int8_kv": config.engine_int8_kv,
            "engine_tp12_handoff": config.engine_tp12_handoff,
            "q38d_current_kv_storage": "float32 kcp/vcp",
            "q38d_current_state_format": "FP32 state records",
            "q38d_current_state_consumer_tp": [1, 2, 4],
            "tp12_mixer_layout": "replicated SSM/attention, sharded FFN/head",
            "unknowns": [
                "TP4 resident model size",
                "TP12 resident model size",
                "allocator/CMG rounding and full runtime scratch",
                "INT8 KV scales and state handoff format",
            ],
        },
        "rows": rows,
    }


def _fmt_gib(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:.2f} GiB"


def text_report(plan: dict[str, Any]) -> str:
    """Human-readable report used by the command-line tool."""

    a = plan["assumptions"]
    m = plan["model_reference"]
    lines = [
        "Q38D 12-node capacity plan (planning estimates; no runner started)",
        f"topology: TP{a['prefill_tp']} prefill -> {a['decode_groups']} x TP{a['decode_tp']} decode, {a['nodes']} nodes",
        f"HBM: {a['hbm_gib_per_node']:.2f} GiB/node, free reserve: {a['free_reserve_gib']:.2f} GiB, output reserve: {a['output_reserve_tokens']} tokens/context",
        f"KV accounting: {a['kv_dtype']} (current engine stores FP32; INT8 values are a projection)",
        f"full NVFP4 allocation: {m['full_allocation_gb_decimal']:.3f} GB decimal = {m['full_allocation_gib']:.2f} GiB ({m['source']})",
        "TP4/TP12 resident model sizes: UNKNOWN unless --tp4-model-gib/--tp12-model-gib are supplied",
        "",
    ]
    for row in plan["rows"]:
        d = row["decode"]
        p = row["prefill"]
        lines.append(
            f"{row['name']}: {row['context_tokens']:,} tokens x {row['contexts']} contexts; "
            f"decode groups={row['decode_group_sizes']}"
        )
        lines.append(
            f"  decode TP4: KV={d['kv_gib_per_node']:.2f} GiB/node, "
            f"peak={_fmt_gib(d['peak_gib_per_node'])}, status={d['status']}"
        )
        if d["gates"]:
            lines.append(f"    gate: {'; '.join(d['gates'])}")
        lines.append(
            f"  prefill TP12: KV={p['kv_gib_per_node']:.2f} GiB/node, "
            f"peak={_fmt_gib(p['peak_gib_per_node'])}, status={p['status']}"
        )
        if p["gates"]:
            lines.append(f"    gate: {'; '.join(p['gates'])}")
        h = p["handoff"]
        lines.append(
            f"  handoff volume/context: INT8 estimate={h['int8_gib_global_estimate']:.2f} GiB global, "
            f"current FP32={h['fp32_gib_global']:.2f} GiB global"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--tp4-model-gib", type=float, help="measured TP4 resident model GiB/node")
    parser.add_argument("--tp12-model-gib", type=float, help="measured TP12 resident model GiB/node")
    parser.add_argument("--scratch-gib", type=float, default=SCRATCH_GIB,
                        help=f"runtime scratch GiB/node (default: {SCRATCH_GIB}; estimate)")
    parser.add_argument("--load-scratch-gib", type=float, default=LOAD_SCRATCH_GIB,
                        help=f"model-load scratch GiB/node (default: {LOAD_SCRATCH_GIB}; conservative)")
    parser.add_argument("--free-reserve-gib", type=float, default=FREE_RESERVE_GIB,
                        help=f"unallocated HBM GiB/node (default: {FREE_RESERVE_GIB})")
    parser.add_argument("--output-reserve-tokens", type=int, default=OUTPUT_RESERVE_TOKENS,
                        help=f"decode KV output reserve per context (default: {OUTPUT_RESERVE_TOKENS})")
    parser.add_argument("--prefill-in-flight", type=int, default=1,
                        help="simultaneous TP12 prefill contexts to charge (default: 1)")
    parser.add_argument("--kv-dtype", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--engine-int8-kv", action="store_true",
                        help="explicitly assume a future INT8 KV engine implementation")
    parser.add_argument("--engine-tp12-handoff", action="store_true",
                        help="explicitly assume a future TP12 -> TP4 state handoff implementation")
    parser.add_argument("--require-admitted", action="store_true",
                        help="exit 2 unless every requested row is ADMITTED")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = PlannerConfig(
            kv_dtype=args.kv_dtype,
            tp4_model_gib=args.tp4_model_gib,
            tp12_model_gib=args.tp12_model_gib,
            scratch_gib=args.scratch_gib,
            load_scratch_gib=args.load_scratch_gib,
            free_reserve_gib=args.free_reserve_gib,
            output_reserve_tokens=args.output_reserve_tokens,
            prefill_in_flight=args.prefill_in_flight,
            engine_int8_kv=args.engine_int8_kv,
            engine_tp12_handoff=args.engine_tp12_handoff,
        )
        plan = make_plan(config)
    except ValueError as exc:
        print(f"q38d_capacity: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(text_report(plan))

    if args.require_admitted:
        statuses = [row[phase]["status"] for row in plan["rows"] for phase in ("decode", "prefill")]
        if any(status != "ADMITTED" for status in statuses):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
