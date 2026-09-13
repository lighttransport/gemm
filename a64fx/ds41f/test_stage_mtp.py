#!/usr/bin/env python3
"""Bounded synthetic coverage of every DSpark ownership and row partition."""
from stage_mtp import layout


def item(name, rows, cols=0, dtype='F8_E4M3'):
    shape = [rows, cols] if cols else [rows]
    size = rows * (cols or 1)
    return dict(name=name, shape=shape, bytes=size, source_offset=256, dtype=dtype)


def main():
    items = []
    for stage in range(3):
        prefix = 'mtp.{}.'.format(stage)
        items += [item(prefix + 'attn.attn_sink', 64), item(prefix + 'hc_attn_fn', 24, 20480),
                  item(prefix + 'attn.wq_a.weight', 1280, 5120), item(prefix + 'ffn.gate.weight', 128, 5120)]
        for part, rows, cols in [('attn.wq_b', 32768, 1280), ('attn.wo_a', 8192, 4096),
                                 ('attn.wo_b', 5120, 8192), ('ffn.shared_experts.w1', 2304, 5120)]:
            items += [item(prefix + part + '.weight', rows, cols), item(prefix + part + '.scale', rows//32, cols//32)]
        for expert in range(128):
            for projection in range(1, 4):
                base = prefix + 'ffn.experts.{}.w{}'.format(expert, projection)
                items += [item(base + '.weight', 2304, 2560), item(base + '.scale', 2304, 160)]
    items += [item('mtp.0.main_proj.weight', 5120, 15360), item('mtp.0.main_proj.scale', 160, 480),
              item('mtp.2.markov_head.embed.weight', 129280, 256), item('mtp.2.markov_head.head.weight', 129280, 256)]
    shards = [layout(items, rank) for rank in range(12)]
    for original in items:
        name = original['name']
        copies = [(rank, shard) for rank, rank_items in enumerate(shards) for shard in rank_items if shard['name'] == name]
        if copies[0][1].get('global_shape'):
            ordered = sorted((part['first_row'], part['shape'][0]) for _, part in copies)
            next_row = 0
            for first, rows in ordered:
                assert first == next_row
                next_row += rows
            assert next_row == original['shape'][0]
            assert sum(part['bytes'] for _, part in copies) == original['bytes']
        elif '.ffn.experts.' in name:
            assert [r for r, _ in copies] == [int(name.split('.')[4]) % 12]
        elif name.endswith('.attn.attn_sink'):
            owner = int(name.split('.')[1])*4
            assert [r for r, _ in copies] == list(range(owner, owner+4))
        else:
            assert [r for r, _ in copies] == [int(name.split('.')[1])*4]
    for rank in (-1, 12):
        try:
            layout(items, rank)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid rank accepted')
    print('MTP_STAGE PASS tensors={} all_stages EP128 TP4 Markov_vocab main_projection ranges invalid_ranks'.format(len(items)))


if __name__ == '__main__':
    main()
