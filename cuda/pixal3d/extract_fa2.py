"""Compile-time adapter for the repository's existing CUDA FA2 source string."""
import ast
from pathlib import Path
import re
import sys

source = Path(__file__).resolve().parents[1] / 'fa2/cuda_fa2_kernels.h'
text = source.read_text(encoding='utf-8').split('static const char *k_fa2_attn_src =', 1)[1]
text = text.split('static const char *k_fa2_attn_fp8_src', 1)[0]
lines = []
for line in text.splitlines():
    match = re.match(r'\s*("(?:[^"\\]|\\.)*")', line)
    if match:
        lines.append(ast.literal_eval(match.group(1)))
        if line[match.end():].lstrip().startswith(';'):
            break
body = ''.join(lines)
assert 'void fa2_attn(' in body and body.endswith('}\n')
out = ['/* Generated from cuda/fa2/cuda_fa2_kernels.h; do not edit. */\n']
for dtype in ['bf16', 'fp16']:
    name = 'px_fa2_' + dtype
    out += [f'namespace {name} {{\n']
    if dtype == 'bf16': out += ['#define FA2_BF16 1\n']
    out += [body.replace('fa2_attn', name), '}\n']
    for macro in sorted(set(re.findall(r'^#define\s+(FA2_\w+)', body, re.M)) | {'FA2_BF16'}):
        out += [f'#undef {macro}\n']
path = Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(''.join(out), encoding='utf-8')
