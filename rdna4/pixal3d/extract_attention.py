"""Build adapter for TRELLIS.2 gfx12 WMMA attention, with Pixal3D rounding."""
import ast
from pathlib import Path
import re
import sys

source = Path(__file__).resolve().parents[1] / 'trellis2/hip_trellis2_kernels.h'
text = source.read_text().split('/* ---- flash_attn_sa_wmma_bc32_v2_f32:', 1)[1]
text = text.split('/* ---- flash_attn_sa_wmma_bc32_db_f32:', 1)[0]
body = ''.join(ast.literal_eval(m.group(1)) for line in text.splitlines()
               if (m := re.match(r'\s*("(?:[^"\\]|\\.)*")', line)))
assert '__global__ void flash_attn_sa_wmma_bc32_v2_f32' in body
body = body.replace('flash_attn_sa_wmma_bc32_v2_f32', 'px_trellis_attention')
# AOT host compilation needs the kernel declaration; the WMMA body is device-only.
body = body.replace('#if defined(__gfx1200__) || defined(__gfx1201__)\n', '')
body = body.replace('int qkv_stride) {\n', 'int qkv_stride) {\n#if defined(__gfx1200__) || defined(__gfx1201__)\n')
body = body.replace('}\n#endif\n', '#endif\n}\n')
# TRELLIS truncates P to BF16. Pixal3D keeps its round-to-nearest-even policy.
body = body.replace('(short)(b >> 16)', '(short)((b + 0x7fffU + ((b >> 16) & 1U)) >> 16)')
body = body.replace('if (out) out[(long)qi*out_dim + h*128 + col] = v;',
                    'if (out) out[(long)qi*out_dim + h*128 + col] = round_bfloat(v);')
path = Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('/* Generated from rdna4/trellis2/hip_trellis2_kernels.h. */\n'
                '#define T2_SWZ(v, mask) __shfl_xor((v), (mask))\n'
                'typedef unsigned short t2_bf16_raw;\n' + body + '\n#undef T2_SWZ\n')
