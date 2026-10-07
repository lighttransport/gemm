"""Qwen-Image-2.1 multi-reference editing (research licence) through diffusers QwenImage21Pipeline.

Unlike the native qimg21 backend (max_references=1, contact-sheet + masked inpaint, which left the
flat fill unchanged), this passes the portrait and the target render as separate references, the
same contract as Qwen-Image-Edit-2511 in qwen_edit_backend: 1024^2 output, target image LAST.
16 GB: DiT weights stored fp8 with bf16 compute (layerwise casting); Qwen3-VL encoder on the CPU.
"""
from pathlib import Path
import time

import numpy as np
from PIL import Image

MODEL=Path('/mnt/disk01/models/qimg-21')
LICENSE='qwen-research'
GENERATOR='Qwen-Image-2.1 multi-reference edit (diffusers, fp8-stored weights)'


class Editor21:
    generator=GENERATOR

    def __init__(self):
        import torch
        from diffusers import QwenImage21Pipeline
        self.torch=torch
        self.pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16)
        self.pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,compute_dtype=torch.bfloat16)
        self.pipe.transformer.to('cuda');self.pipe.vae.to('cuda')
        if hasattr(self.pipe.vae,'enable_tiling'):self.pipe.vae.enable_tiling()
        self.pipe.text_encoder.to('cpu')
        encode=self.pipe._get_qwen_prompt_embeds
        def encode_on_cpu(prompt, image, device=None):
            # Qwen3-VL (17 GB bf16) cannot share the 16 GB GPU with the DiT: encode on CPU, move results.
            return tuple(None if t is None else t.to('cuda') for t in encode(prompt, image, torch.device('cpu')))
        self.pipe._get_qwen_prompt_embeds=encode_on_cpu
        type(self.pipe)._execution_device=property(lambda _:torch.device('cuda'))  # text encoder lives on CPU

    def __call__(self, images, prompt, *, steps=20, seed=317, cfg=4., size=1024,
                 negative='blurry, hair, hat, glasses, text, extra ears, shadows, highlights'):
        torch=self.torch;started=time.time()
        images=[Image.fromarray(i) if isinstance(i,np.ndarray) else i for i in images]
        out=self.pipe(image=images,prompt=prompt,negative_prompt=negative,true_cfg_scale=cfg,height=size,width=size,
            num_inference_steps=steps,generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
        return np.asarray(out.convert('RGB')),time.time()-started
