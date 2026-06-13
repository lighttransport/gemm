#!/usr/bin/env python3
"""
Dump TRELLIS.2 Stage 1 DiT per-block hidden states and block-0 detailed intermediates.

Saves .npy files for layer-by-layer comparison with the HIP implementation.
Use make_comparison.py to compare these against HIP outputs.

Usage:
  python dump_dit_intermediates.py \
    --features /tmp/t2ref/ref_features.npy \
    --noise    /tmp/t2ref/ref_noise_init.npy \
    --stage1   /mnt/disk1/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors \
    --timestep 1000.0 \
    --output-dir /tmp/t2ref
"""
import sys, os, argparse
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../cpu/trellis2/trellis2_repo'))

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def stats(t, name):
    a = t.detach().float().cpu()
    print(f'  {name}: std={a.std():.4f}  [{a.min():.4f}, {a.max():.4f}]  '
          f'first4={a.flatten()[:4].numpy()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features',   required=True, help='ref_features.npy [1029, 1024]')
    parser.add_argument('--noise',      required=True, help='ref_noise_init.npy [1,8,16,16,16]')
    parser.add_argument('--stage1',     required=True, help='Stage 1 DiT safetensors (BF16)')
    parser.add_argument('--timestep',   type=float, default=1000.0, help='t passed to model (0-1000)')
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')

    # ── Load inputs ──────────────────────────────────────────────────────────
    features = torch.from_numpy(np.load(args.features)).float().unsqueeze(0).to(device)
    noise    = torch.from_numpy(np.load(args.noise)).float().to(device)
    print(f'features: {features.shape}  noise: {noise.shape}')

    # ── Load model ───────────────────────────────────────────────────────────
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.modules.utils import manual_cast

    sd = {k: v.float() for k, v in load_file(args.stage1).items()}
    model = SparseStructureFlowModel(
        resolution=16, in_channels=8, model_channels=1536, cond_channels=1024,
        out_channels=8, num_blocks=30, num_head_channels=128,
        mlp_ratio=8192/1536, pe_mode='rope', dtype='float32',
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    del sd
    print('Model loaded.')

    def save(name, tensor):
        arr = tensor.detach().float().cpu().numpy()
        path = os.path.join(args.output_dir, f'{name}.npy')
        np.save(path, arr)
        return arr

    # ── Manual forward with intermediate capture ─────────────────────────────
    with torch.no_grad():
        # Input embedding
        x = noise  # [1, 8, 16, 16, 16]
        h = x.view(1, 8, -1).permute(0, 2, 1).contiguous()  # [1, 4096, 8]
        h = model.input_layer(h)  # [1, 4096, 1536]
        h = manual_cast(h, torch.float32)
        stats(h, 'input_embed')
        save('ref_input_embed', h[0])  # [4096, 1536]

        # Timestep embedding
        t = torch.tensor([args.timestep], device=device)
        t_emb_raw = model.t_embedder(t)                     # [1, 1536]
        t_emb = model.adaLN_modulation(t_emb_raw)           # [1, 6*1536] shared modulation
        t_emb = manual_cast(t_emb, torch.float32)
        stats(t_emb, 't_emb (shared adaLN)')
        save('ref_t_emb', t_emb[0])       # [6*1536]
        save('ref_t_emb_raw', t_emb_raw[0])  # [1536]

        cond = manual_cast(features, torch.float32)
        save('ref_cond_features', cond[0])  # [1029, 1024]

        # ── Block 0 detailed intermediates ────────────────────────────────
        print('\n=== Block 0 detailed intermediates ===')
        block = model.blocks[0]
        mod_bias = block.modulation if hasattr(block, 'modulation') else None

        # adaLN modulation for block 0 (shared + per-block bias)
        if mod_bias is not None:
            t_emb_b0 = t_emb + mod_bias.unsqueeze(0)
        else:
            t_emb_b0 = t_emb
        # The block splits t_emb_b0 into 6 chunks of size 1536
        D = 1536
        shift_sa = t_emb_b0[0, 0*D:1*D]
        scale_sa = t_emb_b0[0, 1*D:2*D]
        gate_sa  = t_emb_b0[0, 2*D:3*D]
        shift_mlp= t_emb_b0[0, 3*D:4*D]
        scale_mlp= t_emb_b0[0, 4*D:5*D]
        gate_mlp = t_emb_b0[0, 5*D:6*D]
        stats(shift_sa, 'b0 shift_sa')
        stats(scale_sa, 'b0 scale_sa')
        stats(gate_sa,  'b0 gate_sa')
        save('ref_b0_adaln_shift_sa', shift_sa)
        save('ref_b0_adaln_scale_sa', scale_sa)
        save('ref_b0_adaln_gate_sa',  gate_sa)
        save('ref_b0_adaln_shift_mlp', shift_mlp)
        save('ref_b0_adaln_scale_mlp', scale_mlp)
        save('ref_b0_adaln_gate_mlp',  gate_mlp)

        # Full block forward (to get per-block output)
        # We need to run one block at a time for per-block saves
        # ── Per-block hidden states ────────────────────────────────────────
        print('\n=== Per-block hidden states ===')
        h_cur = h.clone()
        for bi, block in enumerate(model.blocks):
            h_cur = block(h_cur, t_emb, cond, model.rope_phases)
            s = h_cur.std().item()
            v = h_cur[0, 0, :4].cpu().numpy()
            print(f'  block {bi:2d}: std={s:8.4f}  h[0,0,:4]={v}')
            arr = save(f'ref_block{bi:02d}_hidden', h_cur[0])  # [4096, 1536]

        # Final output
        h_final = manual_cast(h_cur, torch.float32)
        h_final = F.layer_norm(h_final, h_final.shape[-1:])
        out = model.out_layer(h_final)
        out = out.permute(0, 2, 1).view(1, 8, 16, 16, 16)
        stats(out, 'final output')
        save('ref_dit_output', out[0])  # [8, 16, 16, 16]

    print(f'\nSaved all intermediates to {args.output_dir}/')
    print('Key files:')
    print('  ref_input_embed.npy       [4096, 1536]  input layer output')
    print('  ref_t_emb.npy             [9216]        shared adaLN modulation')
    print('  ref_b0_adaln_*.npy        [1536]        block 0 adaLN shift/scale/gate')
    print('  ref_block{00-29}_hidden.npy [4096, 1536]  per-block hidden states')
    print('  ref_dit_output.npy        [8,16,16,16]  final DiT velocity')


if __name__ == '__main__':
    main()
