"""Late-step ground truth: Qwen-Image-Edit-2511 BF16 transformer (41 GB, block-streamed to the GPU with
diffusers group offload) on a QIMG_EDIT_DUMP step; scores the native and GGUF outputs against it."""
import sys
import numpy as np, torch
from diffusers import QwenImageTransformer2DModel

d = np.load(sys.argv[1])
m = QwenImageTransformer2DModel.from_pretrained('/mnt/disk01/models/qwen-image-edit-2511/base', subfolder='transformer',
                                                torch_dtype=torch.bfloat16)
m.enable_group_offload(onload_device=torch.device('cuda'), offload_device=torch.device('cpu'),
                       offload_type='block_level', num_blocks_per_group=1)
x = torch.from_numpy(d['tokens'])[None].to('cuda', torch.bfloat16)
txt = torch.from_numpy(d['txt'])[None].to('cuda', torch.bfloat16)
shapes = [[tuple(int(v) for v in s) for s in d['img_shapes']]]
with torch.no_grad():
    y = m(hidden_states=x, encoder_hidden_states=txt, encoder_hidden_states_mask=torch.ones(txt.shape[:2], device='cuda'),
          timestep=torch.tensor([float(d['timestep'])], device='cuda', dtype=torch.bfloat16), img_shapes=shapes,
          return_dict=False)[0][0].float().cpu().numpy()
np.save(sys.argv[1].replace('.npz', '_bf16.npy'), y)
n = int(np.prod(shapes[0][0])); a = y[:n].ravel().astype(np.float64)
for name, o in [('native', d['out'])] + [(f, np.load(f)) for f in sys.argv[2:]]:
    b = o[:n].ravel().astype(np.float64)
    print(f'{name}: cos(bf16, x) = {a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):.6f} rel_l2 = {np.linalg.norm(a - b) / np.linalg.norm(a):.4f}')
