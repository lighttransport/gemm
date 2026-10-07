"""Diffusers BF16 Edit-2511 with every block linear replaced by its dequantized SVDQuant INT4 weight from our
pack (q*scale + lora_up@lora_down, un-smoothed). Same late-step dump as edit_parity_bf16.py: isolates
quantization error from native-runner modelling errors."""
import sys
import numpy as np, torch
from safetensors import safe_open
from diffusers import QwenImageTransformer2DModel

import os
dump, pack = sys.argv[1], sys.argv[2]
KEEP = [k for k in os.environ.get('FQ_KEEP_BF16', '').split(',') if k]   # linear suffixes left in BF16
d = np.load(dump)
m = QwenImageTransformer2DModel.from_pretrained('/mnt/disk01/models/qwen-image-edit-2511/base', subfolder='transformer',
                                                torch_dtype=torch.bfloat16)
modules = dict(m.named_modules())
with safe_open(pack, 'pt', 'cpu') as f:
    keys = set(f.keys())
    for name in sorted({k.rsplit('.', 1)[0] for k in keys if k.endswith('.qint4')}):
        if any(name.endswith(k) for k in KEEP):
            continue
        q = f.get_tensor(name + '.qint4').to(torch.int16)
        lo, hi = q & 0xF, (q >> 4) & 0xF
        nib = torch.stack((lo, hi), -1).reshape(q.shape[0], -1)
        nib = torch.where(nib >= 8, nib - 16, nib).float()
        ws = f.get_tensor(name + '.wscale').float()
        W = nib * ws.repeat_interleave(nib.shape[1] // ws.shape[1], 1)
        if name + '.smooth' in keys:            # int4 residual consumes x/smooth
            W /= f.get_tensor(name + '.smooth').float()[None]
        if name + '.lora_up' in keys:           # LoRA consumes raw x (lora_down emitted pre-divided)
            W += f.get_tensor(name + '.lora_up').float() @ f.get_tensor(name + '.lora_down').float()
        mod = modules[name]
        assert mod.weight.shape == W.shape, (name, mod.weight.shape, W.shape)
        rel = float((W - mod.weight.float()).norm() / mod.weight.float().norm())
        if name.endswith('blocks.0.attn.to_q') or name.endswith('blocks.30.img_mlp.net.2'): print(name, 'weight rel err', round(rel, 4))
        mod.weight.data.copy_(W.to(torch.bfloat16))
m.enable_group_offload(onload_device=torch.device('cuda'), offload_device=torch.device('cpu'),
                       offload_type='block_level', num_blocks_per_group=1)
x = torch.from_numpy(d['tokens'])[None].to('cuda', torch.bfloat16)
txt = torch.from_numpy(d['txt'])[None].to('cuda', torch.bfloat16)
shapes = [[tuple(int(v) for v in s) for s in d['img_shapes']]]
with torch.no_grad():
    y = m(hidden_states=x, encoder_hidden_states=txt, encoder_hidden_states_mask=torch.ones(txt.shape[:2], device='cuda'),
          timestep=torch.tensor([float(d['timestep'])], device='cuda', dtype=torch.bfloat16), img_shapes=shapes,
          return_dict=False)[0][0].float().cpu().numpy()
ref = np.load(dump.replace('.npz', '_bf16.npy')); n = int(np.prod(shapes[0][0]))
for name, o in (('fakequant (diffusers math, our int4 weights)', y), ('native runner', d['out'])):
    a, b = ref[:n].ravel().astype(np.float64), o[:n].ravel().astype(np.float64)
    print(f'{name}: cos vs bf16 = {a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):.6f} rel_l2 = {np.linalg.norm(a - b) / np.linalg.norm(a):.4f}')
a, b = y[:n].ravel().astype(np.float64), d['out'][:n].ravel().astype(np.float64)
print(f'native vs fakequant: cos = {a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):.6f} rel_l2 = {np.linalg.norm(a - b) / np.linalg.norm(a):.4f}')
