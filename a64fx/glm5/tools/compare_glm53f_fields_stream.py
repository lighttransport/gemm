#!/usr/bin/env python3
"""Bounded canonical comparison on hosts without NumPy (C stream backend)."""
import argparse
import glob
import json
import re
import struct
import subprocess
from pathlib import Path
from glm53f_capture_counts import validate_counts, generated_ids_match


def capture(prefix, hidden_required):
    fields, selections, replicas, coverage, ranks, lengths = {}, {}, [], {}, set(), set()
    hidden_count = 0
    def field(name, kind, path, offset, count, limit=0):
        value = (kind, str(path), offset, count, limit)
        if name in fields:
            old = fields[name]
            if kind != old[0] or count != old[3] or limit != old[4]:
                raise ValueError('replica shape')
            replicas.append((f'replica.{len(replicas)}', 'B', old[1], old[2], str(path), offset, count * (1 if kind == 'B' else 4), 0))
        else:
            fields[name] = value
    records = sorted(glob.glob(str(prefix) + '.rank*.fields'))
    if len(records) != 12:
        raise ValueError('exactly twelve rank manifests required')
    for record in records:
        lines = Path(record).read_text().splitlines()
        version, rank, first, end = lines[0].split()
        rank, first, end = int(rank), int(first), int(end)
        if version != 'GLM53F_FIELDS_V1' or rank in ranks or not 0 <= rank < 12 or not 0 <= first < end <= 45:
            raise ValueError('invalid capture metadata')
        ranks.add(rank); seen = set(); base = f'{prefix}.rank{rank:02d}'
        for line in lines[1:]:
            parts = line.split(); kind = parts[0]
            if kind == 'HIDDEN':
                if hidden_count or len(parts) != 2 or int(parts[1]) < 1:
                    raise ValueError('hidden metadata')
                hidden_count = int(parts[1]); path = str(prefix) + '.hidden'
                if Path(path).stat().st_size != hidden_count * 16384 * 4:
                    raise ValueError('hidden payload shape')
                for token in range(hidden_count):
                    for stream in range(4):
                        field(f'prompt{token}.stream{stream}', 'F', path, (token * 16384 + stream * 4096) * 4, 4096)
                    field(f'prompt{token}.mean', 'M', path, token * 16384 * 4, 4096)
                continue
            if kind == 'STREAMS':
                path = base + '.streams'
                if len(parts) != 1 or Path(path).stat().st_size != 16384 * 4:
                    raise ValueError('stream shape')
                for stream in range(4):
                    field(f'streams.{stream}', 'F', path, stream * 4096 * 4, 4096)
                continue
            layer = int(parts[1])
            if layer in seen or not first <= layer < end:
                raise ValueError('duplicate or unowned layer')
            seen.add(layer)
            if kind == 'KDA':
                h0, hn = map(int, parts[2:])
                occupied = coverage.setdefault(layer, set())
                if layer % 4 == 3 or not 0 <= h0 < h0 + hn <= 64 or occupied.intersection(range(h0, h0 + hn)):
                    raise ValueError('KDA ownership')
                occupied.update(range(h0, h0 + hn)); path = f'{base}.layer{layer:02d}.kda'
                if Path(path).stat().st_size != hn * (128 * 128 + 3 * 128 * 4) * 4:
                    raise ValueError('KDA payload shape')
                for h in range(hn):
                    field(f'layer{layer}.head{h0+h}.state', 'F', path, h * 128 * 128 * 4, 128 * 128)
                    for k in range(3):
                        field(f'layer{layer}.head{h0+h}.conv{k}', 'F', path, (hn * 128 * 128 + (k * hn + h) * 128 * 4) * 4, 128 * 4)
            elif kind == 'SPARSE':
                if layer % 4 != 3 or len(parts) != 2:
                    raise ValueError('sparse layer kind')
                path = f'{base}.layer{layer:02d}.sparse'
                with open(path, 'rb') as f:
                    meta = struct.unpack('<5i', f.read(20))
                length, cp, bf16, replicated, hot = meta; lengths.add(length)
                if length < 1 or any((cp, bf16, replicated, hot)):
                    raise ValueError('only replicated FP32 sparse capture is supported')
                field(f'layer{layer}.metadata', 'B', path, 0, 20)
                offset = 20
                for name, count in [('latent', length * 512), ('key', length * 128), ('gate', length * 128), ('pool', (length // 4) * 128)]:
                    field(f'layer{layer}.{name}', 'F', path, offset, count); offset += count * 4
                count = min(length // 4, 512) * 4 + length % 4
                if Path(path).stat().st_size != offset + count * 4:
                    raise ValueError('sparse selection shape')
                name = f'layer{layer}.selected'
                value = ('S', path, offset, count, length)
                if name in selections:
                    old = selections[name]
                    if count != old[3] or length != old[4]:
                        raise ValueError('selection replica shape')
                    replicas.append((f'replica.{len(replicas)}', 'B', old[1], old[2], path, offset, count * 4, 0))
                else:
                    selections[name] = value
            else:
                raise ValueError('unknown field kind')
        if seen != set(range(first, end)):
            raise ValueError('missing owned layer')
    if any(coverage.get(layer) != set(range(64)) for layer in range(45) if layer % 4 != 3) or len(selections) != 11 or not all(f'streams.{s}' in fields for s in range(4)):
        raise ValueError('incomplete canonical coverage')
    if len(lengths) != 1 or (hidden_required and (not hidden_count or lengths != {hidden_count})):
        raise ValueError('hidden/sparse token count')
    return fields, selections, replicas


def compare(reference, candidate, checker, final=False, bit_exact=False):
    suffix = '.decode' if final else ''
    a, a_sel, a_replicas = capture(str(reference) + suffix, not final)
    b, b_sel, b_replicas = capture(str(candidate) + suffix, not final)
    if a.keys() != b.keys() or a_sel.keys() != b_sel.keys():
        raise ValueError('canonical field sets differ')
    requests = list(a_replicas)
    requests += [(f'candidate.{r[0]}',) + r[1:] for r in b_replicas]
    for collection, actual in ((a, b), (a_sel, b_sel)):
        for name, value in collection.items():
            kind, path, offset, count, limit = value
            other = actual[name]
            if kind != other[0] or count != other[3] or limit != other[4]:
                raise ValueError('canonical shape/type differs')
            requests.append((name, kind, path, offset, other[1], other[2], count, limit))
    for layer in range(3, 45):
        pa, pb = f'{reference}.layer{layer:02d}.routes', f'{candidate}.layer{layer:02d}.routes'
        size = Path(pa).stat().st_size
        if not size or size % 64 or Path(pb).stat().st_size != size:
            raise ValueError('route payload shape')
        requests.append((f'layer{layer}.routes', 'R', pa, 0, pb, 0, size // 64, 0))
    lines = []
    for name, kind, pa, oa, pb, ob, count, limit in requests:
        values = (kind, name, pa, str(oa), pb, str(ob), str(count), str(limit))
        if any('\t' in value or '\n' in value or '\r' in value for value in values):
            raise ValueError('unsupported request character')
        lines.append('\t'.join(values))
    command = [str(checker)] + (['--bit-exact'] if bit_exact else [])
    result = subprocess.run(command, input='\n'.join(lines)+'\n', universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode not in (0, 1):
        raise ValueError('stream backend failed: ' + result.stderr)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if len(rows) != len(requests) or [r['field'] for r in rows] != [r[0] for r in requests]:
        raise ValueError('incomplete stream backend results')
    canonical = [r for r in rows if r['field'] in a]
    worst = max(canonical, key=lambda r: r['relative_l2'])
    failures = [r for r in rows if not r['pass']]
    if bool(failures) != bool(result.returncode):
        raise ValueError('inconsistent stream backend status')
    layer_worst = {}
    for row in canonical:
        match = re.match(r'(layer\d+)\.', row['field'])
        if match and row['relative_l2'] >= layer_worst.get(match.group(1), {}).get('relative_l2', -1):
            layer_worst[match.group(1)] = {'field': row['field'], 'relative_l2': row['relative_l2']}
    return {'pass': not failures, 'float_and_metadata_fields': len(a), 'worst': {'field': worst['field'], 'relative_l2': worst['relative_l2']}, 'layer_worst': layer_worst,
            'sparse_selection_changes': {r['field']: r['changes'] for r in rows if r['field'] in a_sel},
            'expert_route_changes': {r['field'][:-7]: r['changes'] for r in rows if r['field'].endswith('.routes')}, 'failures': failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference'); parser.add_argument('candidate'); parser.add_argument('--checker', required=True)
    parser.add_argument('--phase', choices=['prefill', 'decode'], default='prefill'); parser.add_argument('--bit-exact', action='store_true')
    parser.add_argument('--reference-ids', required=True); parser.add_argument('--candidate-ids', required=True)
    parser.add_argument('--prompt-tokens', required=True, type=int)
    parser.add_argument('--output-tokens', required=True, type=int)
    args = parser.parse_args()
    for prefix in (args.reference, args.candidate):
        validate_counts(prefix, args.prompt_tokens, args.output_tokens, args.phase)
    result = compare(args.reference, args.candidate, args.checker, args.phase == 'decode', args.bit_exact)
    result['generated_ids_exact'] = generated_ids_match(args.reference_ids, args.candidate_ids, args.output_tokens)
    result['pass'] &= result['generated_ids_exact']
    print(json.dumps(result, indent=2, allow_nan=False)); return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
