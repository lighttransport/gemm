#!/usr/bin/env python3
"""Exact checkpoint byte accounting, without reading tensor payloads.

This emits a runtime ownership plan, not a claim that tensors are resident.
Engram remains on disk, experts use EP, dense layers use layer % 12.
"""
import argparse
import json
import re
from pathlib import Path
from stage_backbone import inventory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    plans = [dict(rank=r, hbm_weight_bytes=0, local_engram_bytes=0,
                  tensors=[]) for r in range(12)]
    for stage_rank in range(12):
        for item in inventory(args.model, stage_rank):
            if item['kind'] == 'engram':
                owner = stage_rank
                key = 'local_engram_bytes'
            elif item['kind'] == 'expert':
                owner = stage_rank
                key = 'hbm_weight_bytes'
            else:
                layer = re.match(r'layers\.(\d+)\.', item['name'])
                # Embed and final projection live at opposite ends of the
                # pipeline. Other global tensors remain on rank zero.
                owner = int(layer.group(1)) % 12 if layer else (
                    11 if item['name'].startswith(('head.', 'norm.')) else 0)
                key = 'hbm_weight_bytes'
            plans[owner][key] += item['bytes']
            plans[owner]['tensors'].append(dict(item, stage_rank=stage_rank))
    # Packed cache accounting: KV sources 3 at ratio2 + 1 at ratio1;
    # Only KV sources own index keys; later index query sources reuse them.
    tokens = 1048576
    cache = int(tokens*2.5*(288+68) + 40*128*528)
    for plan in plans:
        # Conservative replication allowance; actual owner-local KV needs
        # explicit source-consumer placement in the runner.
        plan['replicated_packed_kv_bytes'] = cache
        plan['scratch_reserve_bytes'] = 2*1024**3
        plan['remaining_at_32GiB'] = 32*1024**3-plan['hbm_weight_bytes']-cache-2*1024**3
        print('rank={rank} HBM_weights={hbm_weight_bytes} Engram_local={local_engram_bytes} '
              'headroom_after_KV_2GiB_scratch={remaining_at_32GiB}'.format(**plan))
    with args.output.open('x') as f:
        json.dump(dict(ranks=plans, cache_representation='packed, not dequantized',
                       status='planned, runtime distribution required'), f, indent=2)


if __name__ == '__main__':
    main()
