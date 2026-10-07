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
INT4=ROOT/'edit2511-int4-r128.safetensors'
NATIVE_GENERATOR='Qwen-Image-Edit-2511 SVDQuant INT4 r128 (native RDNA4 DiT)'
REPO=Path(__file__).resolve().parents[3]


def make_editor():
    """Native INT4 DiT when packed (QWEN_EDIT_BACKEND=gguf forces the diffusers GGUF fallback)."""
    import os
    if os.environ.get('QWEN_EDIT_BACKEND','native')=='native' and INT4.exists() and (REPO/'rdna4/qimg/libhip_qimg.so').exists():
        return NativeEditor()
    return Editor()


class NativeEditor:
    generator=NATIVE_GENERATOR

    def __init__(self):
        import sys,torch
        sys.path.insert(0,str(REPO/'rdna4/qimg'))
        from qimg_edit_native import load_pipeline
        self.torch=torch
        self.pipe=load_pipeline(str(ROOT/'base'),str(INT4))

    def __call__(self, images, prompt, *, steps=20, seed=317, cfg=4., size=1024,
                 negative='blurry, hair, hat, glasses, text, extra ears, shadows, highlights'):
        started=time.time();dit0=self.pipe.native.seconds;hits0=self.pipe.vision_cache_hits
        images=[Image.fromarray(i) if isinstance(i,np.ndarray) else i for i in images]
        out=self.pipe(image=images,prompt=prompt,negative_prompt=negative,true_cfg_scale=cfg,height=size,width=size,
            num_inference_steps=steps,generator=self.torch.Generator().manual_seed(seed)).images[0]
        total=time.time()-started;dit=self.pipe.native.seconds-dit0
        print(f'[qwen_edit] total {total:.1f}s native DiT {dit:.1f}s other (encode+VAE+host) {total-dit:.1f}s '
              f'vision-cache hits {self.pipe.vision_cache_hits-hits0}',flush=True)
        return np.asarray(out.convert('RGB')),total


class Editor:
    generator=GENERATOR

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
