"""Dump tex_slat_decoder per-stage hidden feats for HIP bisection.

Loads the pipeline, feeds a saved tex_slat (feats + coords) through the decoder
with a forward hook that captures post-stage hidden tensors BEFORE the C2S/up
block (so HIP `--stop-stage N` maps to the same reference).

Saves into the output-dir:
  stage0_pre_c2s_feats.npy   (N_coarse, C_stage0) = after 4 ConvNeXt, before C2S
  stage0_pre_c2s_coords.npy
  stage1_pre_c2s_feats.npy   (N_fine_s0, C_stage1)  etc.
  stage3_pre_out_feats.npy   (N_fine_s3, C_stage3) = after 4 ConvNeXt, before LN+output
"""
import argparse, os, numpy as np, torch
from trellis2.pipelines import Trellis2TexturingPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', default='microsoft/TRELLIS.2-4B')
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    pipe = Trellis2TexturingPipeline.from_pretrained(args.pipeline)
    pipe.cuda()
    dec = pipe.models['tex_slat_decoder']

    feats = torch.from_numpy(np.load(os.path.join(args.input_dir, 'tex_slat_feats.npy'))).cuda()
    coords = torch.from_numpy(np.load(os.path.join(args.input_dir, 'tex_slat_coords.npy'))).cuda().int()
    import trellis2.modules.sparse as sp
    tex_slat = sp.SparseTensor(feats=feats, coords=coords)

    captured = {}

    orig_forward = dec.forward
    def hook_forward(x, guide_subs=None, return_subs=False):
        h = dec.from_latent(x).type(dec.dtype)
        for i, res in enumerate(dec.blocks):
            is_last_stage = (i == len(dec.blocks) - 1)
            for j, block in enumerate(res):
                if not is_last_stage and j == len(res) - 1:
                    captured[f'stage{i}_pre_c2s'] = (h.feats.detach().cpu().numpy(),
                                                    h.coords.detach().cpu().numpy())
                    h = block(h, subdiv=guide_subs[i] if guide_subs else None)
                else:
                    h = block(h)
            if is_last_stage:
                captured[f'stage{i}_pre_out'] = (h.feats.detach().cpu().numpy(),
                                                 h.coords.detach().cpu().numpy())
        from torch.nn import functional as F
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = dec.output_layer(h)
        return h

    with torch.no_grad():
        _ = hook_forward(tex_slat, guide_subs=pipe._guide_subs if hasattr(pipe, '_guide_subs') else None)

    os.makedirs(args.output_dir, exist_ok=True)
    for k, (f, c) in captured.items():
        np.save(os.path.join(args.output_dir, f'{k}_feats.npy'), f)
        np.save(os.path.join(args.output_dir, f'{k}_coords.npy'), c)
        print(f'{k}: feats={f.shape} coords={c.shape}')

if __name__ == '__main__':
    main()
