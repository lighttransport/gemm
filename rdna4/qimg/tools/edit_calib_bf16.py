"""Activation calibration for SVDQuant: per-input-channel max|x| of every main block linear, collected with
forward hooks on the BF16 diffusers transformer (group-offloaded) over QIMG_EDIT_DUMP step inputs.
Writes transformer_blocks.<b>.<linear>.amax for svdquant_from_bf16.py --calib."""
import sys
import numpy as np, torch
from safetensors.torch import save_file
from diffusers import QwenImageTransformer2DModel

MAIN = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj", "txt_mlp.net.2"]
out, dumps = sys.argv[1], sys.argv[2:]
m = QwenImageTransformer2DModel.from_pretrained('/mnt/disk01/models/qwen-image-edit-2511/base', subfolder='transformer',
                                                torch_dtype=torch.bfloat16)
amax = {}
def hook(name):
    def fn(mod, args):
        x = args[0].detach().float().abs().reshape(-1, args[0].shape[-1]).amax(0).cpu()
        amax[name] = x if name not in amax else torch.maximum(amax[name], x)
    return fn
for b in range(len(m.transformer_blocks)):
    for suf in MAIN:
        mod = m.get_submodule(f'transformer_blocks.{b}.{suf}')
        mod.register_forward_pre_hook(hook(f'transformer_blocks.{b}.{suf}'))
m.enable_group_offload(onload_device=torch.device('cuda'), offload_device=torch.device('cpu'),
                       offload_type='block_level', num_blocks_per_group=1)
for path in dumps:
    d = np.load(path)
    x = torch.from_numpy(d['tokens'])[None].to('cuda', torch.bfloat16)
    txt = torch.from_numpy(d['txt'])[None].to('cuda', torch.bfloat16)
    shapes = [[tuple(int(v) for v in s) for s in d['img_shapes']]]
    with torch.no_grad():
        m(hidden_states=x, encoder_hidden_states=txt, encoder_hidden_states_mask=torch.ones(txt.shape[:2], device='cuda'),
          timestep=torch.tensor([float(d['timestep'])], device='cuda', dtype=torch.bfloat16), img_shapes=shapes, return_dict=False)
    print('calibrated on', path, 't =', float(d['timestep']), flush=True)
save_file({k + '.amax': v.contiguous() for k, v in amax.items()}, out)
print('wrote', len(amax), 'amax vectors ->', out)
