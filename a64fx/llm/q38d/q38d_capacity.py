#!/usr/bin/env python3
"""Capacity and admission planning for the Qwen3.8-27B Q38D layout.

This is a memory planning tool; admission is not a throughput or quality
validation. Q38D TP4 decode and Q38P PP12 prefill support packed INT6 K/V,
and the version-2 state format transfers it between PP12 and TP4. The
FP32-prefill/INT6-export path is faster for shorter evaluated prompts.

The default cases are the three requested independent-context shapes::

    1,048,576 x 8, 524,288 x 16, and 262,144 x 32

The decode side is three disjoint TP4 groups over twelve nodes.  Contexts are
balanced across those groups and each TP4 rank is charged for its local KV,
model, SSM state, scratch, and a configurable free-memory reserve.  Prefill
is the Q38P PP12 layer pipeline: each stage owns a contiguous unit range from
the same ``pf_stage_bounds`` cost partition as the C runner.  The default
partition has at most two attention layers on a stage; ``--prefill-in-flight``
can be used to make a more demanding estimate.

All numbers derived from Q38D/Q38P source constants are kept here rather than
duplicated in the engine.  The full NVFP4 allocation is a measured value from
``tmp/q38-fast/gen-fp4.log``.  A separate run measured 13.891 GiB of TP4
source matrix mappings released by ``--prune-model`` and 6.02 GiB post-prune
RSS.  PP12 overhead beyond the full-image measurement remains conservative and
is kept in the scratch budget.
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
PP_SSM_COST = 294.8
PP_ATTN_COST = 326.6
PP_FFN_COST = 566.1
PP_STAGES = 12
KV_SCALE_BYTES = 2 * 4  # one FP32 scale each for K and V
KV_I6_ROW_BYTES = (HD * 6 + 7) // 8  # q38d_kv_i6.h: packed six-bit row

# q38-fast/gen-fp4.log: "lowbit: final anonymous allocation 18.212GB".
# The log uses decimal GB; all values exposed by this module use GiB.
FULL_MODEL_ALLOCATION_GB = 18.212
BYTES_PER_GIB = 1 << 30
FULL_MODEL_ALLOCATION_GIB = FULL_MODEL_ALLOCATION_GB * 1_000_000_000 / BYTES_PER_GIB

# q38d --prune-model measurement: source matrix mappings released per TP4
# rank and post-prune RSS.  The latter is the runtime TP4 model budget; load
# peak remains conservatively based on the full image below.
TP4_PRUNED_SOURCE_MAPPING_GIB = 13.891
TP4_POST_PRUNE_RSS_GIB = 6.02


@dataclass(frozen=True)
class Workload:
    """One context length and the number of independent requests."""

    name: str
    context_tokens: int
    contexts: int


WORKLOADS = (
    Workload("1m_x8", 1_048_576, 8),
    Workload("512k_x16", 524_288, 16),
    Workload("256k_x32", 262_144, 32),
)


@dataclass(frozen=True)
class PlannerConfig:
    """Inputs that affect admission.

    ``tp4_model_gib`` remains optional so callers can fail closed or replace
    the measured post-prune RSS.  The PP12 pipeline is documented as keeping the
    full lowbit image on every node, so ``tp12_model_gib`` defaults to that
    measured allocation.  Descriptor/repack overhead remains in the scratch
    estimate; callers can replace the default with a measured stage RSS.
    """

    nodes: int = NODES
    prefill_tp: int = PREFILL_TP
    prefill_stages: int = PP_STAGES
    prefill_attn_cost: float = PP_ATTN_COST
    decode_tp: int = DECODE_TP
    decode_groups: int = DECODE_GROUPS
    hbm_gib: float = HBM_GIB
    output_reserve_tokens: int = OUTPUT_RESERVE_TOKENS
    kv_dtype: str = "int6"
    tp4_model_gib: Optional[float] = TP4_POST_PRUNE_RSS_GIB
    tp12_model_gib: Optional[float] = FULL_MODEL_ALLOCATION_GIB
    scratch_gib: Optional[float] = SCRATCH_GIB
    load_scratch_gib: Optional[float] = LOAD_SCRATCH_GIB
    free_reserve_gib: float = FREE_RESERVE_GIB
    prefill_in_flight: int = 1
    # These are capability gates, not memory assumptions.  INT6 PP12 cache,
    # version-2 handoff, and grouped TP4 state import have been exercised.
    engine_int8_kv: bool = True
    engine_int6_kv: bool = True
    engine_prefill_compressed_kv: bool = True
    # Compatibility alias for callers of the previous INT8-only planner API.
    engine_prefill_int8_kv: bool = False
    engine_tp12_handoff: bool = True


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_config(config: PlannerConfig) -> None:
    if config.nodes != NODES:
        raise ValueError(f"Q38D plan is for {NODES} nodes, got {config.nodes}")
    if config.prefill_tp != PREFILL_TP:
        raise ValueError(f"prefill TP must be {PREFILL_TP}, got {config.prefill_tp}")
    if config.prefill_stages != PP_STAGES:
        raise ValueError(f"prefill stages must be {PP_STAGES}, got {config.prefill_stages}")
    _require_positive("prefill_attn_cost", config.prefill_attn_cost)
    if config.decode_tp != DECODE_TP:
        raise ValueError(f"decode TP must be {DECODE_TP}, got {config.decode_tp}")
    if config.decode_groups != DECODE_GROUPS:
        raise ValueError(f"decode groups must be {DECODE_GROUPS}, got {config.decode_groups}")
    if config.kv_dtype not in {"int6", "int8", "fp32"}:
        raise ValueError("kv_dtype must be int6, int8, or fp32")
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


def _kv_row_bytes(dtype: str) -> int:
    if dtype == "int6":
        return KV_I6_ROW_BYTES
    if dtype == "int8":
        return HD
    if dtype == "fp32":
        return HD * 4
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


def kv_bytes_per_token(
    tp: int,
    dtype: str = "int8",
    *,
    attention_layers: Optional[int] = None,
    kv_heads: Optional[int] = None,
) -> int:
    """K+V bytes per token for a rank or a PP stage.

    For INT8, q38d --kv-i8 stores ``HD`` bytes for each K/V row and one FP32
    scale for each K/V stream.  For INT6, --kv-i6 stores a packed 192-byte
    row and one FP32 scale for each stream.  Thus both compressed formats use
    ``2 * (row_bytes + 4)`` bytes per local KV head and attention layer.
    FP32 has no additional scale term.

    ``attention_layers`` and ``kv_heads`` let the same accounting describe a
    Q38P pipeline stage, which owns all four KV heads for only its local
    attention layers.
    """

    layers = NATTN if attention_layers is None else attention_layers
    heads = local_kv_heads(tp) if kv_heads is None else kv_heads
    if layers < 0 or heads < 0:
        raise ValueError("attention_layers and kv_heads must be non-negative")
    row_bytes = _kv_row_bytes(dtype)
    scales = KV_SCALE_BYTES if dtype in {"int6", "int8"} else 0
    return layers * heads * (2 * row_bytes + scales)


def _pipeline_units(attn_cost: float) -> list[tuple[str, float]]:
    """Return the 128 mixer/FFN units used by q38p pf_stage_bounds."""

    units: list[tuple[str, float]] = []
    for layer in range(NLAYER):
        if layer % 4 == 3:
            units.append(("attn", attn_cost))
        else:
            units.append(("ssm", PP_SSM_COST))
        units.append(("ffn", PP_FFN_COST))
    return units


def prefill_stage_partition(
    stages: int = PP_STAGES,
    attn_cost: float = PP_ATTN_COST,
) -> dict[str, Any]:
    """Emulate q38p ``pf_stage_bounds`` and count mixer layers per stage.

    The C partition minimizes the largest contiguous stage cost over mixer and
    FFN units.  Cuts may occur between a mixer's unit and its FFN, so counting
    attention units rather than whole layers is intentional.  The default
    result is the prior PP12 shape with attention counts
    ``[1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]`` and worst stage count 2.
    A changed ``Q38P_PP_ATTN_COST`` can change the cuts; pass the same value
    to this planner when evaluating such a run.
    """

    if stages < 1 or stages > len(_pipeline_units(attn_cost)):
        raise ValueError("pipeline stages must be between 1 and 128")
    units = _pipeline_units(attn_cost)
    count = len(units)
    prefix = [0.0]
    for _, weight in units:
        prefix.append(prefix[-1] + weight)

    # This is the same dynamic program and strict tie handling as
    # q38p_partition.h.  Keeping the tie rule makes the reported cuts match
    # pf_stage_bounds for the default costs.
    best = [[math.inf] * (count + 1) for _ in range(stages + 1)]
    split = [[0] * (count + 1) for _ in range(stages + 1)]
    best[0][0] = 0.0
    for k in range(1, stages + 1):
        for end in range(k, count + 1):
            for begin in range(k - 1, end):
                cost = max(best[k - 1][begin], prefix[end] - prefix[begin])
                if cost < best[k][end]:
                    best[k][end] = cost
                    split[k][end] = begin

    cuts = [0] * (stages + 1)
    cuts[stages] = count
    for k in range(stages, 0, -1):
        cuts[k - 1] = split[k][cuts[k]]
    attention_counts = [sum(kind == "attn" for kind, _ in units[a:b])
                        for a, b in zip(cuts, cuts[1:])]
    ssm_counts = [sum(kind == "ssm" for kind, _ in units[a:b])
                  for a, b in zip(cuts, cuts[1:])]
    return {
        "stages": stages,
        "attn_cost": attn_cost,
        "unit_bounds": [[cuts[i], cuts[i + 1]] for i in range(stages)],
        "attention_layers_per_stage": attention_counts,
        "ssm_layers_per_stage": ssm_counts,
        "worst_attention_layers": max(attention_counts),
        "worst_ssm_layers": max(ssm_counts),
        "source": "q38p_prefill.inc pf_stage_bounds + q38p_partition.h",
        "caveat": "PP12 stage-local estimate; full evaluated long-context quality and throughput remain unverified",
    }


def ssm_memory_breakdown(
    tp: int,
    *,
    ssm_layers: Optional[int] = None,
    local_ssm_heads: Optional[int] = None,
) -> dict[str, int | float]:
    """Persistent per-rank SSM/convolution state from q38d_engine.c.

    ``conv_state`` is allocated for all 64 layers.  Per-head state is created
    only for SSM layers (48 layers, because ``L->ssm = (l % 4) != 3``) and for
    the rank's local heads.  TP12 replicates mixer state, as in the engine.
    ``cmg_alloc`` rounding and allocator metadata are not represented, so the
    returned total is a lower bound for this component.
    """

    all_ssm_layers = NLAYER - NATTN
    charged_layers = all_ssm_layers if ssm_layers is None else ssm_layers
    if charged_layers < 0 or charged_layers > all_ssm_layers:
        raise ValueError(f"ssm_layers must be between 0 and {all_ssm_layers}")
    heads = (NVH if tp_replicates_mixers(tp) else NVH // tp) if local_ssm_heads is None else local_ssm_heads
    if heads < 0 or heads > NVH:
        raise ValueError(f"local_ssm_heads must be between 0 and {NVH}")
    conv_state = NLAYER * (4 * QKVD) * 4  # calloc(4 * QKVD, sizeof(float))
    ssm_buf = charged_layers * heads * SSM_RING * (DS * DS + 2 * DS) * 4
    conv_hist = charged_layers * heads * (8 * 384) * 4
    conv_wl = charged_layers * heads * (4 * 384) * 4
    total = conv_state + ssm_buf + conv_hist + conv_wl
    return {
        "conv_state_bytes": conv_state,
        "ssm_buf_bytes": ssm_buf,
        "conv_hist_bytes": conv_hist,
        "conv_wl_bytes": conv_wl,
        "total_bytes": total,
        "total_gib": gibibytes(total),
        "ssm_layers": charged_layers,
        "all_ssm_layers": all_ssm_layers,
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
        "tp4_pruned_source_mapping_released_gib": TP4_PRUNED_SOURCE_MAPPING_GIB,
        "tp4_pruned_source_mapping_source": "measured q38d --prune-model log; six-context hashes identical",
        "tp4_rss_after_prune_gib": TP4_POST_PRUNE_RSS_GIB,
        "tp4_rss_after_prune_source": "measured q38d --prune-model log",
        "pp12_model_default_gib": FULL_MODEL_ALLOCATION_GIB,
        "pp12_model_source": "Q38P README: full lowbit image resident per pipeline node; repack/RSS overhead pending",
        "tp4_resident_source": "measured post-prune RSS; override with --tp4-model-gib if needed",
    }


def _handoff_bytes(context_tokens: int, config: PlannerConfig) -> dict[str, Any]:
    """Logical state volume for one context entering a TP4 decode group.

    Version-2 q38d_state.inc writes packed INT6 K/V plus FP32 scales. These
    are logical state bytes; file metadata and filesystem overhead are excluded.
    """

    fp32_per_rank = kv_bytes_per_token(DECODE_TP, "fp32")
    compressed_per_rank = kv_bytes_per_token(DECODE_TP, config.kv_dtype)
    ssm_global = int(ssm_memory_breakdown(DECODE_TP)["total_bytes"]) * DECODE_TP
    compressed_bytes = context_tokens * compressed_per_rank * DECODE_TP + ssm_global
    result: dict[str, Any] = {
        "context_tokens": context_tokens,
        "ssm_bytes_global": ssm_global,
        "fp32_bytes_global": context_tokens * fp32_per_rank * DECODE_TP + ssm_global,
        "fp32_gib_global": gibibytes(context_tokens * fp32_per_rank * DECODE_TP + ssm_global),
        "compressed_dtype": config.kv_dtype,
        "compressed_bytes_global_estimate": compressed_bytes,
        "compressed_gib_global_estimate": gibibytes(compressed_bytes),
        "compressed_note": (
            "includes 2 FP32 K/V scales per local head/layer/token; INT6 version-2 state format is implemented"
            if config.kv_dtype == "int6" else
            "logical state-size estimate; this dtype does not use the INT6 version-2 state format"
        ),
    }
    # Keep the INT8 aliases for consumers of the first planner revision.
    int8_bytes = context_tokens * kv_bytes_per_token(DECODE_TP, "int8") * DECODE_TP + ssm_global
    result["int8_bytes_global_estimate"] = int8_bytes
    result["int8_gib_global_estimate"] = gibibytes(int8_bytes)
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
    if config.kv_dtype == "int6":
        decode_capability_ok = config.engine_int6_kv
        decode_capability_name = "q38d --kv-i6 is not enabled for this decode plan"
    elif config.kv_dtype == "int8":
        decode_capability_ok = config.engine_int8_kv
        decode_capability_name = "q38d --kv-i8 is not enabled for this decode plan"
    else:
        decode_capability_ok = True
        decode_capability_name = ""
    decode_status, decode_gates = _capacity_status(
        decode_fit,
        capability_ok=decode_capability_ok,
        capability_name=decode_capability_name,
    )

    # PP12 prefill is pipeline parallelism, not Q38D's replicated-mixer TP12
    # branch.  Each stage owns all four KV heads for its attention mixers.  A
    # stage-local cache therefore uses the worst attention-layer count from
    # pf_stage_bounds rather than charging every stage for all sixteen layers.
    prefill_partition = prefill_stage_partition(config.prefill_stages, config.prefill_attn_cost)
    prefill_attn_layers = int(prefill_partition["worst_attention_layers"])
    prefill_ssm_layers = int(prefill_partition["worst_ssm_layers"])
    prefill_kv_bpt = kv_bytes_per_token(
        config.prefill_tp,
        config.kv_dtype,
        attention_layers=prefill_attn_layers,
        kv_heads=NKV,
    )
    prefill_kv_bytes = config.prefill_in_flight * workload.context_tokens * prefill_kv_bpt
    prefill_kv_gib = gibibytes(prefill_kv_bytes)
    # Keep the FP32 alternative beside packed INT6 for choosing a prefill path.
    prefill_current_fp32_kv_bpt = kv_bytes_per_token(
        config.prefill_tp,
        "fp32",
        attention_layers=prefill_attn_layers,
        kv_heads=NKV,
    )
    prefill_current_fp32_kv_bytes = config.prefill_in_flight * workload.context_tokens * prefill_current_fp32_kv_bpt
    prefill_current_fp32_kv_gib = gibibytes(prefill_current_fp32_kv_bytes)
    # Retain a historical all-layer FP32 comparison; current PP12 allocation
    # keeps K/V only for the attention layers owned by each stage.
    prefill_current_alloc_fp32_kv_bpt = kv_bytes_per_token(
        config.prefill_tp,
        "fp32",
        attention_layers=NATTN,
        kv_heads=NKV,
    )
    prefill_current_alloc_fp32_kv_bytes = (
        config.prefill_in_flight * workload.context_tokens * prefill_current_alloc_fp32_kv_bpt
    )
    prefill_current_alloc_fp32_kv_gib = gibibytes(prefill_current_alloc_fp32_kv_bytes)
    prefill_ssm_base = ssm_memory_breakdown(
        config.prefill_tp,
        ssm_layers=prefill_ssm_layers,
        local_ssm_heads=NVH,
    )
    prefill_ssm = dict(prefill_ssm_base)
    if config.prefill_in_flight > 1:
        for key in ("conv_state_bytes", "ssm_buf_bytes", "conv_hist_bytes", "conv_wl_bytes", "total_bytes"):
            prefill_ssm[key] = int(prefill_ssm[key]) * config.prefill_in_flight
        prefill_ssm["total_gib"] = gibibytes(int(prefill_ssm["total_bytes"]))
    prefill_ssm["source"] = str(prefill_ssm["source"]) + "; PP12 worst stage"
    prefill_model = _model_for_tp(config, config.prefill_tp)
    prefill_runtime = _optional_add(
        prefill_model,
        float(prefill_ssm["total_gib"]),
        config.scratch_gib,
        prefill_kv_gib,
    )
    prefill_current_fp32_runtime = _optional_add(
        prefill_model,
        float(prefill_ssm["total_gib"]),
        config.scratch_gib,
        prefill_current_fp32_kv_gib,
    )
    prefill_peak = None if prefill_runtime is None or decode_load_peak is None else max(prefill_runtime, decode_load_peak)
    prefill_current_fp32_peak = (
        None if prefill_current_fp32_runtime is None or decode_load_peak is None
        else max(prefill_current_fp32_runtime, decode_load_peak)
    )
    prefill_fit = _memory_fit(prefill_peak, config)
    prefill_current_fp32_fit = _memory_fit(prefill_current_fp32_peak, config)
    prefill_current_alloc_fp32_runtime = _optional_add(
        prefill_model,
        float(prefill_ssm["total_gib"]),
        config.scratch_gib,
        prefill_current_alloc_fp32_kv_gib,
    )
    prefill_current_alloc_fp32_peak = (
        None if prefill_current_alloc_fp32_runtime is None or decode_load_peak is None
        else max(prefill_current_alloc_fp32_runtime, decode_load_peak)
    )
    prefill_current_alloc_fp32_fit = _memory_fit(prefill_current_alloc_fp32_peak, config)
    prefill_status, prefill_gates = _capacity_status(
        prefill_fit,
        capability_ok=(
            (config.kv_dtype == "int6" and config.engine_prefill_compressed_kv)
            or (config.kv_dtype == "int8" and config.engine_prefill_int8_kv)
            or config.kv_dtype == "fp32"
        ) and config.engine_tp12_handoff,
        capability_name="Q38P PP12 compressed K/V or state handoff disabled in this plan",
    )

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
            "pp12_stage_resident_gib": prefill_model,
            "load_scratch_gib": config.load_scratch_gib,
        },
        "decode": {
            "topology": "3 independent TP4 groups over 12 nodes",
            "kv_dtype": config.kv_dtype,
            "kv_i6_row_bytes": KV_I6_ROW_BYTES,
            "kv_scale_bytes_per_token_head_layer": KV_SCALE_BYTES,
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
            "topology": "Q38P PP12 layer pipeline, one context at a time",
            "stage_partition": prefill_partition,
            "prefill_in_flight": config.prefill_in_flight,
            "kv_dtype": config.kv_dtype,
            "local_kv_heads_per_stage": NKV,
            "worst_stage_attention_layers": prefill_attn_layers,
            "worst_stage_ssm_layers": prefill_ssm_layers,
            "kv_bytes_per_token_per_rank": prefill_kv_bpt,
            "prompt_tokens_per_context": workload.context_tokens,
            "kv_bytes_per_node": prefill_kv_bytes,
            "kv_gib_per_node": prefill_kv_gib,
            "current_fp32_kv_bytes_per_token_per_stage": prefill_current_fp32_kv_bpt,
            "current_fp32_kv_bytes_per_node": prefill_current_fp32_kv_bytes,
            "current_fp32_kv_gib_per_node": prefill_current_fp32_kv_gib,
            "current_alloc_fp32_kv_bytes_per_token_per_node": prefill_current_alloc_fp32_kv_bpt,
            "current_alloc_fp32_kv_bytes_per_node": prefill_current_alloc_fp32_kv_bytes,
            "current_alloc_fp32_kv_gib_per_node": prefill_current_alloc_fp32_kv_gib,
            "ssm": prefill_ssm,
            "scratch_gib": config.scratch_gib,
            "runtime_gib_per_node": prefill_runtime,
            "current_fp32_peak_gib_per_node": prefill_current_fp32_peak,
            "current_fp32_fits": prefill_current_fp32_fit,
            "current_alloc_fp32_peak_gib_per_node": prefill_current_alloc_fp32_peak,
            "current_alloc_fp32_fits": prefill_current_alloc_fp32_fit,
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
            "prefill_stages": config.prefill_stages,
            "prefill_attn_cost": config.prefill_attn_cost,
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
            "engine_int6_kv": config.engine_int6_kv,
            "engine_prefill_compressed_kv": config.engine_prefill_compressed_kv,
            "engine_tp12_handoff": config.engine_tp12_handoff,
            "engine_prefill_int8_kv": config.engine_prefill_int8_kv,
            "q38d_current_kv_storage": "regular TP4 decode: --kv-i6 packed 192-byte rows or --kv-i8 rows, each with one FP32 K/V scale",
            "q38p_current_prefill_kv_storage": "PP-owned FP32 K/V or --prefill-kv-i6 packed K/V; FP32 prefill supports --state-out-i6",
            "q38d_current_state_format": "version 1 FP32 and version 2 packed INT6 K/V with FP32 scales",
            "q38d_current_state_consumer_tp": [1, 2, 4],
            "q38p_pp12_handoff": "PP12 packed INT6 or FP32-to-INT6 export; TP4 grouped context-state import tested at short evaluated depth",
            "q38d_tp12_mixer_layout": "replicated SSM/attention, sharded FFN/head; not used for PP12 prefill",
            "unknowns": [
                "PP12 stage resident overhead beyond full lowbit image",
                "allocator/CMG rounding and full runtime scratch",
                "full evaluated 256K-1M context throughput and output quality",
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
        f"topology: Q38P PP{a['prefill_stages']} prefill -> {a['decode_groups']} x TP{a['decode_tp']} decode, {a['nodes']} nodes",
        f"HBM: {a['hbm_gib_per_node']:.2f} GiB/node, free reserve: {a['free_reserve_gib']:.2f} GiB, output reserve: {a['output_reserve_tokens']} tokens/context",
        f"KV accounting: {a['kv_dtype']} (compressed rows include one FP32 scale for K and V)",
        f"full NVFP4 allocation: {m['full_allocation_gb_decimal']:.3f} GB decimal = {m['full_allocation_gib']:.2f} GiB ({m['source']})",
        f"TP4 --prune-model released source mappings: {m['tp4_pruned_source_mapping_released_gib']:.3f} GiB; post-prune RSS={m['tp4_rss_after_prune_gib']:.2f} GiB",
        f"TP4 resident model: {m['tp4_rss_after_prune_gib']:.2f} GiB measured; PP12 starts from {m['pp12_model_default_gib']:.2f} GiB full-image measurement",
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
            f"  prefill PP12: KV={p['kv_gib_per_node']:.2f} GiB/node (worst stage attention={p['worst_stage_attention_layers']} layers), "
            f"peak={_fmt_gib(p['peak_gib_per_node'])}, status={p['status']}"
        )
        lines.append(
            f"    current FP32 PP12 KV={p['current_fp32_kv_gib_per_node']:.2f} GiB/node, "
            f"peak={_fmt_gib(p['current_fp32_peak_gib_per_node'])}, fits={p['current_fp32_fits']}"
        )
        lines.append(
            f"    historical all-layer FP32 comparison={p['current_alloc_fp32_kv_gib_per_node']:.2f} GiB/node, "
            f"peak={_fmt_gib(p['current_alloc_fp32_peak_gib_per_node'])}, fits={p['current_alloc_fp32_fits']}"
        )
        if p["gates"]:
            lines.append(f"    gate: {'; '.join(p['gates'])}")
        h = p["handoff"]
        lines.append(
            f"  handoff volume/context: {h['compressed_dtype']} estimate={h['compressed_gib_global_estimate']:.2f} GiB global, "
            f"current FP32={h['fp32_gib_global']:.2f} GiB global"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--tp4-model-gib", type=float, default=TP4_POST_PRUNE_RSS_GIB,
                        help=f"measured TP4 resident model GiB/node (default: {TP4_POST_PRUNE_RSS_GIB})")
    parser.add_argument("--tp12-model-gib", type=float, default=FULL_MODEL_ALLOCATION_GIB,
                        help="PP12 stage resident model GiB/node (default: measured full lowbit image)")
    parser.add_argument("--prefill-attn-cost", type=float, default=PP_ATTN_COST,
                        help=f"Q38P_PP_ATTN_COST used by pf_stage_bounds (default: {PP_ATTN_COST})")
    parser.add_argument("--scratch-gib", type=float, default=SCRATCH_GIB,
                        help=f"runtime scratch GiB/node (default: {SCRATCH_GIB}; estimate)")
    parser.add_argument("--load-scratch-gib", type=float, default=LOAD_SCRATCH_GIB,
                        help=f"model-load scratch GiB/node (default: {LOAD_SCRATCH_GIB}; conservative)")
    parser.add_argument("--free-reserve-gib", type=float, default=FREE_RESERVE_GIB,
                        help=f"unallocated HBM GiB/node (default: {FREE_RESERVE_GIB})")
    parser.add_argument("--output-reserve-tokens", type=int, default=OUTPUT_RESERVE_TOKENS,
                        help=f"decode KV output reserve per context (default: {OUTPUT_RESERVE_TOKENS})")
    parser.add_argument("--prefill-in-flight", type=int, default=1,
                        help="simultaneous PP12 prefill contexts to charge (default: 1)")
    parser.add_argument("--kv-dtype", choices=("int6", "int8", "fp32"), default="int6")
    parser.add_argument("--engine-int8-kv", action=argparse.BooleanOptionalAction, default=True,
                        help="regular decode supports q38d --kv-i8 (use --no-engine-int8-kv to model a build without it)")
    parser.add_argument("--engine-int6-kv", action=argparse.BooleanOptionalAction, default=True,
                        help="regular decode supports q38d --kv-i6 (use --no-engine-int6-kv to model a build without it)")
    parser.add_argument("--engine-prefill-compressed-kv", action=argparse.BooleanOptionalAction, default=True,
                        help="model current Q38P PP12 packed INT6 K/V support")
    parser.add_argument("--engine-prefill-int8-kv", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine-tp12-handoff", action=argparse.BooleanOptionalAction, default=True,
                        help="model current PP12 -> TP4 state handoff support")
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
            prefill_attn_cost=args.prefill_attn_cost,
            scratch_gib=args.scratch_gib,
            load_scratch_gib=args.load_scratch_gib,
            free_reserve_gib=args.free_reserve_gib,
            output_reserve_tokens=args.output_reserve_tokens,
            prefill_in_flight=args.prefill_in_flight,
            engine_int8_kv=args.engine_int8_kv,
            engine_int6_kv=args.engine_int6_kv,
            engine_prefill_compressed_kv=args.engine_prefill_compressed_kv,
            engine_prefill_int8_kv=args.engine_prefill_int8_kv,
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
