"""Build CPU-correct pbr_voxel oracle from dumped tex_dec intermediates.

Bypasses the ROCm F.linear bug on gfx1201 that corrupts the 822874x64 -> 6
final output_layer for the PyTorch reference. See memory entry
`project_trellis2_tex_dec_outlier.md` for why this is needed.

Inputs (from gen_stage2_ref.py --dump-stages):
  <dir>/stage<L>_pre_out_feats.npy  (last stage's pre-LN feats)
  <dir>/output_layer_weight.npy
  <dir>/output_layer_bias.npy

Output:
  <dir>/pbr_voxel_feats_cpu.npy    (correct reference for HIP validation)
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='dir with dumped intermediates')
    ap.add_argument('--last-stage', type=int, default=4,
                    help='index L of stage<L>_pre_out_feats.npy (default 4)')
    args = ap.parse_args()

    pre_path = os.path.join(args.dir, f'stage{args.last_stage}_pre_out_feats.npy')
    W_path = os.path.join(args.dir, 'output_layer_weight.npy')
    b_path = os.path.join(args.dir, 'output_layer_bias.npy')
    for p in (pre_path, W_path, b_path):
        if not os.path.exists(p):
            sys.exit(f'missing {p} — rerun gen_stage2_ref.py --dump-stages')

    pre = np.load(pre_path)
    W = np.load(W_path)
    b = np.load(b_path)
    t = torch.from_numpy(pre)
    ln = F.layer_norm(t, t.shape[-1:]).numpy()
    pbr = (ln @ W.T + b) * 0.5 + 0.5
    out = os.path.join(args.dir, 'pbr_voxel_feats_cpu.npy')
    np.save(out, pbr.astype(np.float32))
    print(f'wrote {out} shape={pbr.shape} range=[{pbr.min():.3f}, {pbr.max():.3f}]')


if __name__ == '__main__':
    main()
