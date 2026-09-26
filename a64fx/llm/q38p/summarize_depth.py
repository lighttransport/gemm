#!/usr/bin/env python3
"""Summarize measured suffix throughput; prefix fill never enters the numerator."""
import argparse
import json
import pathlib
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('runs', nargs='+', type=pathlib.Path)
args = parser.parse_args()
for run in args.runs:
    text = (run / 'rank0.log').read_text()
    config = re.search(r'q38p_depth_result: depth=(\d+) suffix=(\d+) repeats=(\d+) fill=(\S+) qtile=(\d+) attn_cost=([\d.]+)', text)
    perf = re.search(r'q38p_pp: nodes=(\d+) prompts=(\d+) prompt_tokens=(\d+) end_to_end=([\d.]+) tok/s single_prompt=([\d.]+) s steady=([\d.]+)', text)
    if not config or not perf:
        parser.error(f'incomplete depth benchmark: {run}')
    nodes = int(perf[1])
    logs = [run / f'rank{rank}.log' for rank in range(nodes)]
    if any(not path.exists() or 'q38p_pp: rank=' not in path.read_text() or 'FATAL' in path.read_text() for path in logs):
        parser.error(f'failed or missing rank: {run}')
    value = dict(run=str(run), nodes=nodes, depth=int(config[1]), suffix=int(config[2]),
                 repeats=int(config[3]), fill=config[4], qtile=int(config[5]),
                 attn_cost=float(config[6]), end_to_end_tok_s=float(perf[4]),
                 first_suffix_s=float(perf[5]), steady_tok_s=float(perf[6]))
    extra = re.search(r'kv_tile=(\d+) qk6=(\d+) chunk=(\d+)', text)
    if extra:
        value.update(kv_tile=int(extra[1]), qk6=int(extra[2]), chunk=int(extra[3]))
    packed = re.search(r'gemm=(\d+) key_cache=(\d+)', text)
    if packed:
        value.update(gemm=int(packed[1]), key_cache=int(packed[2]))
    value['steady_tok_s_node'] = value['steady_tok_s'] / nodes
    value['target_130_met'] = value['repeats'] > 1 and value['steady_tok_s_node'] >= 130
    value['residual_hash'] = re.search(r'last_residual_hash=([0-9a-f]+)', text)[1]
    print(json.dumps(value, sort_keys=True))
