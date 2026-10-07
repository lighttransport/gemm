"""One-step parity: diffusers GGUF Q4_K_M Edit-2511 transformer vs the native INT4 dump (QIMG_EDIT_DUMP).

Both are quantized differently (Q4_K_M vs SVDQuant int4 r128), so this bounds the end-to-end
error of the native port; a layout/RoPE/zero_cond_t bug shows up as cos << 0.95.
"""
import sys
import numpy as np, torch
from diffusers import QwenImageTransformer2DModel, GGUFQuantizationConfig

d = np.load(sys.argv[1])
root = '/mnt/disk01/models/qwen-image-edit-2511'
m = QwenImageTransformer2DModel.from_single_file(root + '/qwen-image-edit-2511-Q4_K_M.gguf',
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    config=root + '/base', subfolder='transformer', torch_dtype=torch.bfloat16).to('cuda')
x = torch.from_numpy(d['tokens'])[None].to('cuda', torch.bfloat16)
txt = torch.from_numpy(d['txt'])[None].to('cuda', torch.bfloat16)
shapes = [[tuple(int(v) for v in s) for s in d['img_shapes']]]
with torch.no_grad():
    y = m(hidden_states=x, encoder_hidden_states=txt, encoder_hidden_states_mask=torch.ones(txt.shape[:2], device='cuda'),
          timestep=torch.tensor([d['timestep']], device='cuda', dtype=torch.bfloat16), img_shapes=shapes,
          return_dict=False)[0][0].float().cpu().numpy()
n = int(np.prod(shapes[0][0]))
a, b = y[:n].ravel(), d['out'][:n].ravel()
print('noisy tokens', n, 'cos(gguf, native) =', float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
      'rel_l2 =', float(np.linalg.norm(a - b) / np.linalg.norm(a)))
