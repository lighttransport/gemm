"""Qwen-Image-Edit-2511 (Apache-2.0) through diffusers with a GGUF Q4_K_M transformer.

The 13 GB transformer is resident on the 16 GB GPU; the Qwen2.5-VL text/vision
encoder (16.6 GB bf16) runs on the CPU and only its embeddings move to the GPU.
Edit outputs whole images: callers composite photographed pixels back.

Alignment: the pipeline sizes condition latents to ~1 MP, so the output must be 1024^2 and the
target view must be the LAST image; a 512^2 output reproduces only the top-left quarter of the
condition grid (verified on the right view: 512 zooms, 1024 is pixel-aligned).
"""
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT=Path('/mnt/disk01/models/qwen-image-edit-2511')
GGUF=ROOT/'qwen-image-edit-2511-Q4_K_M.gguf'
LICENSE='apache-2.0'
GENERATOR='Qwen-Image-Edit-2511 Q4_K_M (diffusers GGUF)'


class Editor:
    def __init__(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
        self.torch=torch
        transformer=QwenImageTransformer2DModel.from_single_file(str(GGUF),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=str(ROOT/'base'),subfolder='transformer',torch_dtype=torch.bfloat16)
        self.pipe=QwenImageEditPlusPipeline.from_pretrained(str(ROOT/'base'),transformer=transformer,torch_dtype=torch.bfloat16)
        self.pipe.transformer.to('cuda');self.pipe.vae.to('cuda');self.pipe.vae.enable_tiling()
        self.pipe.text_encoder.to('cpu')

    def __call__(self, images, prompt, *, steps=20, seed=317, cfg=4., size=1024,
                 negative='blurry, hair, hat, glasses, text, extra ears, shadows, highlights'):
        torch=self.torch;started=time.time()
        images=[Image.fromarray(i) if isinstance(i,np.ndarray) else i for i in images]
        with torch.no_grad():
            # Encode on CPU (encoder too large to share the GPU with the transformer).
            pos,pos_mask=self.pipe.encode_prompt(prompt=prompt,image=images,device=torch.device('cpu'))
            neg,neg_mask=self.pipe.encode_prompt(prompt=negative,image=images,device=torch.device('cpu'))
        move=lambda t:None if t is None else t.to('cuda')
        out=self.pipe(image=images,prompt_embeds=move(pos),prompt_embeds_mask=move(pos_mask),
            negative_prompt_embeds=move(neg),negative_prompt_embeds_mask=move(neg_mask),
            true_cfg_scale=cfg,height=size,width=size,num_inference_steps=steps,
            generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
        return np.asarray(out.convert('RGB')),time.time()-started
