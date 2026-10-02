"""Export the local FLUX.2 Qwen3 tokenizer to a vocabulary-only GGUF.

Uses standard-library JSON/struct only. No model weights, tokenizer runtime,
Torch or downloads are needed. The caller records the result in native assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def string(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def export(source, output):
    source, output = Path(source), Path(output)
    doc = json.loads((source / 'tokenizer.json').read_text())
    config = json.loads((source / 'tokenizer_config.json').read_text())
    if doc['model']['type'] != 'BPE' or config.get('tokenizer_class') not in ('Qwen2Tokenizer', 'Qwen2TokenizerFast'):
        raise ValueError('expected the FLUX.2 Qwen3 BPE tokenizer')
    vocab = dict(doc['model']['vocab'])
    special, added = set(), set()
    for token in doc['added_tokens']:
        vocab[token['content']] = token['id']
        added.add(token['id'])
        if token['special']:
            special.add(token['id'])
    if set(vocab.values()) != set(range(len(vocab))):
        raise ValueError('vocabulary ids must be contiguous and unique')
    tokens = [None]*len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    merges = [' '.join(m) if isinstance(m, list) else m for m in doc['model']['merges']]
    if any(len(m.split(' ')) != 2 for m in merges):
        raise ValueError('unsupported merge representation')
    kv = [('tokenizer.ggml.model', 8, string('gpt2')),
          ('tokenizer.ggml.pre', 8, string('qwen2')),
          ('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, len(tokens)) + b''.join(map(string, tokens))),
          ('tokenizer.ggml.token_type', 9, struct.pack('<IQ', 5, len(tokens)) +
           b''.join(struct.pack('<i', 3 if i in special else 4 if i in added else 1) for i in range(len(tokens)))),
          ('tokenizer.ggml.merges', 9, struct.pack('<IQ', 8, len(merges)) + b''.join(map(string, merges)))]
    for name, key in (('eos_token_id', 'eos_token'), ('bos_token_id', 'bos_token'), ('padding_token_id', 'pad_token')):
        token = config.get(key)
        if isinstance(token, dict):
            token = token['content']
        if token is not None:
            kv.append(('tokenizer.ggml.'+name, 4, struct.pack('<I', vocab[token])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        stream.write(struct.pack('<4sIQQ', b'GGUF', 3, 0, len(kv)))
        for key, kind, value in kv:
            stream.write(string(key)+struct.pack('<I', kind)+value)
        stream.write(b'\0'*((-stream.tell()) % 32))
    receipt = dict(format='vhuman.flux2_tokenizer_export.v1', vocabulary=len(tokens), merges=len(merges),
                   source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                  for name in ('tokenizer.json', 'tokenizer_config.json')},
                   output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(export(**vars(parser.parse_args()))))
