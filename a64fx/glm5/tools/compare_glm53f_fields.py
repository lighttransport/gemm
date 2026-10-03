#!/usr/bin/env python3
"""Compare diagnostic TP12/PP fields without holding two resident models."""
import argparse
import glob
import json
from pathlib import Path
from glm53f_capture_counts import validate_counts, generated_ids_match
import numpy as np


def read_capture(prefix, require_hidden=True):
    prefix = str(prefix)
    records = glob.glob(prefix + '.rank*.fields')
    if len(records) != 12:
        raise ValueError('capture must contain exactly twelve rank manifests')
    fields, selected, coverage, ranks = {}, {}, {}, set()
    hidden_count = 0
    lengths = set()
    for manifest in sorted(records):
        lines = Path(manifest).read_text().splitlines()
        version, rank, first, end = lines[0].split()
        rank, first, end = int(rank), int(first), int(end)
        if version != 'GLM53F_FIELDS_V1' or rank in ranks or not 0 <= rank < 12 or not 0 <= first < end <= 45:
            raise ValueError('invalid capture metadata')
        ranks.add(rank)
        base = f'{prefix}.rank{rank:02d}'
        seen = set()
        for line in lines[1:]:
            parts = line.split()
            kind = parts[0]
            if kind == 'HIDDEN':
                if hidden_count or len(parts) != 2 or int(parts[1]) < 1:
                    raise ValueError('hidden metadata')
                hidden_count = int(parts[1])
                path = Path(prefix + '.hidden')
                if path.stat().st_size != hidden_count * 16384 * 4:
                    raise ValueError('hidden payload shape')
                hidden = np.memmap(path, dtype='<f4', mode='r', shape=(hidden_count, 4, 4096))
                for t in range(hidden_count):
                    for stream in range(4):
                        fields[f'prompt{t}.stream{stream}'] = hidden[t, stream]
                    fields[f'prompt{t}.mean'] = ((hidden[t, 0] + hidden[t, 1]) + hidden[t, 2] + hidden[t, 3]) * np.float32(0.25)
                continue
            if kind == 'STREAMS':
                data = np.fromfile(base + '.streams', dtype='<f4')
                if data.size != 16384 or Path(base + '.streams').stat().st_size != 16384 * 4:
                    raise ValueError('stream shape')
                for stream in range(4):
                    insert_replica(fields, f'streams.{stream}', data[stream * 4096:(stream + 1) * 4096])
                continue
            layer = int(parts[1])
            if layer in seen or not first <= layer < end:
                raise ValueError('duplicate or unowned layer')
            seen.add(layer)
            if kind == 'KDA':
                if layer % 4 == 3:
                    raise ValueError('KDA layer kind')
                h0, hn = map(int, parts[2:])
                if not 0 <= h0 < h0 + hn <= 64:
                    raise ValueError('head range')
                occupied = coverage.setdefault(layer, set())
                if occupied.intersection(range(h0, h0 + hn)):
                    raise ValueError('duplicate KDA heads')
                occupied.update(range(h0, h0 + hn))
                data = np.fromfile(f'{base}.layer{layer:02d}.kda', dtype='<f4')
                if data.size != hn * (128 * 128 + 3 * 128 * 4) or Path(f'{base}.layer{layer:02d}.kda').stat().st_size != data.size * 4:
                    raise ValueError('KDA payload shape')
                state = data[:hn * 128 * 128].reshape(hn, 128 * 128)
                conv = data[hn * 128 * 128:].reshape(3, hn, 128 * 4)
                for h in range(hn):
                    fields[f'layer{layer}.head{h0 + h}.state'] = state[h]
                    for k in range(3):
                        fields[f'layer{layer}.head{h0 + h}.conv{k}'] = conv[k, h]
            elif kind == 'SPARSE':
                if layer % 4 != 3:
                    raise ValueError('sparse layer kind')
                data = Path(f'{base}.layer{layer:02d}.sparse').read_bytes()
                meta = np.frombuffer(data, dtype='<i4', count=5)
                length, cp, bf16, replicated, hot = map(int, meta)
                lengths.add(length)
                if length < 1 or any((cp, bf16, replicated, hot)):
                    raise ValueError('only replicated FP32 sparse capture is supported')
                insert_replica(fields, f'layer{layer}.metadata', meta.copy())
                offset = 20
                for field, count in [('latent', length * 512), ('key', length * 128),
                                     ('gate', length * 128), ('pool', (length // 4) * 128)]:
                    if offset + count * 4 > len(data):
                        raise ValueError('truncated sparse capture')
                    insert_replica(fields, f'layer{layer}.{field}', np.frombuffer(data, dtype='<f4', count=count, offset=offset).copy())
                    offset += count * 4
                count = min(length // 4, 512) * 4 + length % 4
                if len(data) != offset + count * 4:
                    raise ValueError('sparse selection shape')
                indices = np.frombuffer(data, dtype='<i4', count=count, offset=offset).copy()
                if (indices < 0).any() or (indices >= length).any() or np.unique(indices).size != count:
                    raise ValueError('invalid sparse selection indices')
                insert_replica(selected, f'layer{layer}.selected', indices)
            else:
                raise ValueError('unknown field kind')
        if seen != set(range(first, end)):
            raise ValueError('missing owned layer')
    if any(coverage.get(l) != set(range(64)) for l in range(45) if l % 4 != 3):
        raise ValueError('incomplete KDA head coverage')
    if len(selected) != 11 or not all(f'streams.{s}' in fields for s in range(4)):
        raise ValueError('incomplete sparse or final-stream capture')
    if len(lengths) != 1 or (require_hidden and (not hidden_count or lengths != {hidden_count})):
        raise ValueError("missing full-prompt streams or inconsistent token count")
    return fields, selected


def insert_replica(fields, name, data):
    if name in fields and (fields[name].shape != data.shape or fields[name].tobytes() != data.tobytes()):
        raise ValueError(f'replicated field differs within capture: {name}')
    fields[name] = data


def compare(reference, candidate, tolerance, final=False, bit_exact=False):
    if not 0 <= tolerance <= 1e-3:
        raise ValueError('tolerance must be in [0, 1e-3]')
    suffix = ".decode" if final else ""
    a, a_sel = read_capture(str(reference) + suffix, not final)
    b, b_sel = read_capture(str(candidate) + suffix, not final)
    if a.keys() != b.keys() or a_sel.keys() != b_sel.keys():
        raise ValueError('canonical field sets differ')
    failures, worst = [], {'field': None, 'relative_l2': 0.0}
    for name, ref in a.items():
        actual = b[name]
        if ref.shape != actual.shape or ref.dtype != actual.dtype:
            failures.append({'field': name, 'reason': 'shape/type'})
            continue
        if ref.dtype.kind != 'f':
            if not np.array_equal(ref, actual):
                failures.append({'field': name, 'reason': 'structural_metadata'})
            continue
        if not np.isfinite(ref).all() or not np.isfinite(actual).all():
            failures.append({'field': name, 'reason': 'nonfinite'})
            continue
        if bit_exact and ref.tobytes() != actual.tobytes():
            failures.append({'field': name, 'reason': 'bit_exact'})
        ref64, actual64 = ref.astype(np.float64), actual.astype(np.float64)
        denominator = float(np.dot(ref64, ref64))
        error = float(np.dot(actual64 - ref64, actual64 - ref64))
        if denominator == 0:
            if ref.tobytes() != actual.tobytes():
                failures.append({'field': name, 'reason': 'zero_norm_requires_exact'})
            continue
        relative = (error / denominator) ** 0.5
        if relative > worst['relative_l2']:
            worst = {'field': name, 'relative_l2': relative}
        if relative > tolerance:
            failures.append({'field': name, 'relative_l2': relative})
    selection_changes = {}
    for name, ref in a_sel.items():
        actual = b_sel[name]
        if ref.shape != actual.shape:
            failures.append({'field': name, 'reason': 'selection_shape'})
        else:
            selection_changes[name] = int(np.count_nonzero(ref != actual))
            if bit_exact and selection_changes[name]:
                failures.append({'field': name, 'reason': 'bit_exact_selection'})
    route_changes = {}
    route_type = np.dtype([('ids', '<i4', (8,)), ('weights', '<f4', (8,))])
    for layer in range(3, 45):
        if any(Path(f'{prefix}.layer{layer:02d}.routes').stat().st_size % route_type.itemsize for prefix in (reference, candidate)):
            raise ValueError('truncated route record')
        route_a = np.fromfile(f'{reference}.layer{layer:02d}.routes', dtype=route_type)
        route_b = np.fromfile(f'{candidate}.layer{layer:02d}.routes', dtype=route_type)
        if not route_a.size or route_a.shape != route_b.shape:
            failures.append({'field': f'layer{layer}.routes', 'reason': 'route_shape'})
            continue
        if any(not np.isfinite(r['weights']).all() or (r['ids'] < 0).any() or (r['ids'] >= 288).any() or
               (np.diff(np.sort(r['ids'], axis=1), axis=1) == 0).any() for r in (route_a, route_b)):
            failures.append({'field': f'layer{layer}.routes', 'reason': 'invalid_routes'})
        route_changes[f'layer{layer}'] = int(np.count_nonzero(route_a['ids'] != route_b['ids']))
        if bit_exact and route_a.tobytes() != route_b.tobytes():
            failures.append({'field': f'layer{layer}.routes', 'reason': 'bit_exact_routes'})
    return {'pass': not failures, 'float_and_metadata_fields': len(a), 'worst': worst,
            'sparse_selection_changes': selection_changes, 'expert_route_changes': route_changes, 'failures': failures}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference')
    p.add_argument('candidate')
    p.add_argument('--bit-exact', action='store_true')
    p.add_argument('--phase', choices=['prefill', 'decode'], default='prefill')
    p.add_argument('--tolerance', type=float, default=1e-3)
    p.add_argument('--reference-ids', required=True)
    p.add_argument('--candidate-ids', required=True)
    p.add_argument('--prompt-tokens', required=True, type=int)
    p.add_argument('--output-tokens', required=True, type=int)
    args = p.parse_args()
    for prefix in (args.reference, args.candidate):
        validate_counts(prefix, args.prompt_tokens, args.output_tokens, args.phase)
    result = compare(args.reference, args.candidate, args.tolerance, args.phase == "decode", args.bit_exact)
    result['generated_ids_exact'] = generated_ids_match(args.reference_ids, args.candidate_ids, args.output_tokens)
    result['pass'] &= result['generated_ids_exact']
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
