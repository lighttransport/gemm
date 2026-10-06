"""MV-Adapter (ig2mv, SDXL) six-view generation conditioned on the fitted GNM.

Code: Apache-2.0 (huanngzh/MV-Adapter, cloned to tmp/MV-Adapter, not vendored).
Weights: adapter Apache-2.0; SDXL base CreativeML OpenRAIL++-M (use-restricted,
evaluation baseline only, not a permissive shipping path).
"""
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
MODELS=Path('/mnt/disk01/models')
LICENSE='openrail++-m (SDXL base) + apache-2.0 (MV-Adapter)'
GENERATOR='MV-Adapter ig2mv SDXL'


def reference_image(portrait, silhouette, size=768):
    """Matte the portrait onto 50% gray, centred at 90% like MV-Adapter preprocess."""
    rgb=np.asarray(Image.open(portrait).convert('RGB'),float)
    alpha=np.asarray(Image.open(silhouette).convert('L').resize(Image.open(portrait).size),float)[...,None]/255
    ys,xs=np.where(alpha[...,0]>.5)
    crop=(rgb*alpha+127.5*(1-alpha))[ys.min():ys.max()+1,xs.min():xs.max()+1]
    h,w=crop.shape[:2];s=.9*size/max(h,w)
    im=Image.fromarray(np.uint8(crop+.5)).resize((max(1,round(w*s)),max(1,round(h*s))),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(size,size),(127,127,127))
    canvas.paste(im,((size-im.width)//2,(size-im.height)//2))
    return canvas


def generate(views, reference, *, steps=30, seed=317, guidance=3., text='a photo of a bald man head, realistic skin, high quality',
             reference_scale=1.):
    import torch
    sys.path.insert(0,str(ROOT/'tmp/MV-Adapter'))
    from diffusers import AutoencoderKL
    from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
    started=time.time()
    pipe=MVAdapterI2MVSDXLPipeline.from_pretrained(str(MODELS/'sdxl-base-1.0'),variant='fp16',torch_dtype=torch.float16,
        vae=AutoencoderKL.from_pretrained(str(MODELS/'sdxl-vae-fp16-fix'),torch_dtype=torch.float16))
    pipe.scheduler=ShiftSNRScheduler.from_scheduler(pipe.scheduler,shift_mode='interpolated',shift_scale=8.,scheduler_class=None)
    pipe.init_custom_adapter(num_views=len(views),self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
    pipe.load_custom_adapter(str(MODELS/'mv-adapter'),weight_name='mvadapter_ig2mv_sdxl.safetensors')
    # 16 GB cards: keep only the active submodule resident (12-image CFG batch at 768px).
    pipe.to(device='cuda',dtype=torch.float16);pipe.cond_encoder.to(device='cuda',dtype=torch.float16)
    encoders=(pipe.text_encoder,pipe.text_encoder_2);encode=pipe.encode_prompt
    for e in encoders:e.to('cpu')
    def encode_on_gpu(*args,**kwargs):
        # Model offload hooks break MV-Adapter's reference-attention cache, so
        # page only the text encoders (1.6 GB) in for prompt encoding.
        for e in encoders:e.to('cuda')
        try:return encode(*args,**kwargs)
        finally:
            for e in encoders:e.to('cpu')
            torch.cuda.empty_cache()
    pipe.encode_prompt=encode_on_gpu
    decode=pipe.vae.decode
    def decode_without_unet(*args,**kwargs):
        # Denoising is done once decoding starts; free the UNet for VAE activations.
        if pipe.unet.device.type=='cuda':pipe.unet.to('cpu');torch.cuda.empty_cache()
        return decode(*args,**kwargs)
    pipe.vae.decode=decode_without_unet
    pipe.vae.enable_slicing()
    control=torch.cat([torch.from_numpy(np.stack([v['position'] for v in views])),
                       torch.from_numpy(np.stack([v['normal'] for v in views]))],-1).permute(0,3,1,2)
    size=views[0]['position'].shape[0]
    images=pipe(text,height=size,width=size,num_inference_steps=steps,guidance_scale=guidance,
        num_images_per_prompt=len(views),control_image=control.to('cuda',torch.float16),control_conditioning_scale=1.,
        reference_image=reference,reference_conditioning_scale=reference_scale,
        negative_prompt='watermark, ugly, deformed, noisy, blurry, low contrast, hair, hat, shadow',
        generator=torch.Generator(device='cuda').manual_seed(seed)).images
    del pipe;torch.cuda.empty_cache()
    return [np.asarray(im.convert('RGB')) for im in images],dict(steps=steps,seed=seed,guidance=guidance,text=text,
        reference_scale=reference_scale,seconds=time.time()-started,generator=GENERATOR,license=LICENSE)
