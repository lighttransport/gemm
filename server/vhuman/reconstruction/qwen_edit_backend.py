"""Qwen-Image-Edit-2511 (Apache-2.0) editors: native RDNA4 DiT (default) or diffusers GGUF Q4_K_M fallback.

Native: mixed INT4/INT8 DiT resident (~11.3 GB), modulation computed exactly on the host, FP32 Qwen2.5-VL
encoder on the CPU (Zen 2 has no native BF16). GGUF: the 13 GB transformer can be resident or block-offloaded;
the encoder runs in FP32 on the CPU. Both use CPU prompt encode and a cuda pipeline + generator.
Edit outputs whole images: callers composite photographed pixels back.

Alignment: the pipeline sizes condition latents to ~1 MP, so the output must be 1024^2 and the
target view must be the LAST image; a 512^2 output reproduces only the top-left quarter of the
condition grid (verified on the right view: 512 zooms, 1024 is pixel-aligned).
"""
from pathlib import Path
import hashlib
import time

import numpy as np
from PIL import Image

ROOT=Path('/mnt/disk01/models/qwen-image-edit-2511')
GGUF=ROOT/'qwen-image-edit-2511-Q4_K_M.gguf'
LICENSE='apache-2.0'
GENERATOR='Qwen-Image-Edit-2511 Q4_K_M (diffusers GGUF)'
INT4=ROOT/'edit2511-int4mix-r128.safetensors'   # mixed: INT8 v-proj/MLP-down, INT4 rest, mods on host
NATIVE_GENERATOR='Qwen-Image-Edit-2511 SVDQuant INT4/INT8 mixed r128 + host BF16 modulation (native RDNA4 DiT)'
REPO=Path(__file__).resolve().parents[3]


def condition_size(size):
    """Edit-Plus resizes every reference to VAE_IMAGE_SIZE (module constant, 1024^2): keep it equal to the
    output size or the output reproduces only part of the condition grid (the 512^2 quarter-zoom)."""
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as plus
    plus.VAE_IMAGE_SIZE=size*size


def reference_areas(areas):
    """Per-image VAE areas for the next pipeline call (targets keep the output grid; references may be smaller).

    Edit-Plus calls calculate_dimensions(VAE_IMAGE_SIZE, ratio) once per image in order; this queue overrides
    those calls. The native DiT handles any per-segment latent shape.
    """
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as plus
    if not hasattr(plus, '_orig_calculate_dimensions'):
        plus._orig_calculate_dimensions = plus.calculate_dimensions
        def calc(target_area, ratio):
            if target_area == plus.VAE_IMAGE_SIZE and plus._area_queue:
                target_area = plus._area_queue.pop(0)
            return plus._orig_calculate_dimensions(target_area, ratio)
        plus.calculate_dimensions = calc
    plus._area_queue = list(areas)


def make_editor(backend=None, model_root=ROOT, offload_blocks=0):
    """Select explicitly, or use native only with a ROCm Torch runtime and its assets."""
    import os
    import torch
    backend=backend or os.environ.get('QWEN_EDIT_BACKEND','auto')
    if backend not in ('auto','native','gguf'):raise ValueError('unknown Edit-2511 backend: '+backend)
    model_root=Path(model_root)
    native=bool(torch.version.hip) and (model_root/INT4.name).is_file() and (REPO/'rdna4/qimg/libhip_qimg.so').is_file()
    if backend=='native' and not native:
        raise ValueError('native Edit-2511 requires ROCm Torch, the HIP library and the mixed INT4 weights')
    if backend=='native' or (backend=='auto' and native):
        if offload_blocks:raise ValueError('block offloading is only supported by the GGUF editor')
        return NativeEditor(model_root)
    return Editor(model_root,offload_blocks=offload_blocks)


class NativeEditor:
    generator=NATIVE_GENERATOR

    def __init__(self, model_root=ROOT):
        import sys,torch
        sys.path.insert(0,str(REPO/'rdna4/qimg'))
        from qimg_edit_native import load_pipeline
        self.torch=torch
        model_root=Path(model_root)
        self.pipe=load_pipeline(str(model_root/'base'),str(model_root/INT4.name))
        self.metadata=dict(backend='native',model_root=str(model_root.resolve()),weights=INT4.name)
        import os
        if os.environ.get('QIMG_ENCODER_DTYPE','fp32')=='fp32':   # identity-checked; 3.8x faster on Zen 2
            self.pipe.text_encoder.to(torch.float32)
        if os.environ.get('QIMG_VISION_CACHE')=='1':
            # positive and negative encodes see the same images: reuse the vision tower output (copies, since
            # transformers mutates pooler_output in place)
            import copy,hashlib
            visual=self.pipe.text_encoder.model.visual;fwd=visual.forward;cache={}
            def cached(pixel_values,grid_thw=None,**kw):
                key=(hashlib.sha1(pixel_values.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                     None if grid_thw is None else tuple(grid_thw.flatten().tolist()))
                if key not in cache:
                    if len(cache)>=4:cache.pop(next(iter(cache)))
                    cache[key]=copy.copy(fwd(pixel_values,grid_thw=grid_thw,**kw))
                return copy.copy(cache[key])
            visual.forward=cached

    def __call__(self, images, prompt, *, steps=20, seed=317, cfg=4., size=1024,
                 negative='blurry, hair, hat, glasses, text, extra ears, shadows, highlights'):
        """Same flow as Editor (GGUF), which preserves identity: CPU prompt encode, cuda pipeline + generator."""
        torch=self.torch;condition_size(size)
        import os
        ref_side=int(os.environ.get('QIMG_REF_SIDE','0'))   # e.g. 512: references at 512^2, target at size^2
        if ref_side:
            reference_areas([ref_side*ref_side]*(len(images)-1)+[size*size])
        started=time.time();dit0=self.pipe.native.seconds
        images=[Image.fromarray(i) if isinstance(i,np.ndarray) else i for i in images]
        with torch.no_grad():
            pos,pos_mask=self.pipe.encode_prompt(prompt=prompt,image=images,device=torch.device('cpu'))
            neg,neg_mask=self.pipe.encode_prompt(prompt=negative,image=images,device=torch.device('cpu'))
        # QIMG_ENCODER_DTYPE=fp32 only changes the encoder arithmetic (Zen 2 has no native BF16); embeddings go back to
        # BF16 so the pipeline sees exactly the dtypes of the identity-verified flow.
        pos,neg=pos.to(torch.bfloat16),neg.to(torch.bfloat16)
        encode=time.time()-started
        move=lambda t:None if t is None else t.to('cuda')
        out=self.pipe(image=images,prompt_embeds=move(pos),prompt_embeds_mask=move(pos_mask),
            negative_prompt_embeds=move(neg),negative_prompt_embeds_mask=move(neg_mask),
            true_cfg_scale=cfg,height=size,width=size,num_inference_steps=steps,
            generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
        total=time.time()-started;dit=self.pipe.native.seconds-dit0
        print(f'[qwen_edit] total {total:.1f}s native DiT {dit:.1f}s prompt_encode {encode:.1f}s other {total-dit-encode:.1f}s',flush=True)
        return np.asarray(out.convert('RGB')),total


class Editor:
    generator=GENERATOR

    def __init__(self, model_root=ROOT, *, offload_blocks=0):
        import torch
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
        from .observations import sha256
        self.torch=torch;self.prompt_cache={};model_root=Path(model_root);weights=model_root/GGUF.name
        if offload_blocks<0:raise ValueError('offload_blocks must be nonnegative')
        self.offload_blocks=offload_blocks
        self.metadata=dict(backend='gguf',model_root=str(model_root.resolve()),weights=weights.name,
                           weights_sha256=sha256(weights),encoder_dtype='float32',offload_blocks=offload_blocks)
        transformer=QwenImageTransformer2DModel.from_single_file(str(weights),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=str(model_root/'base'),subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
        self.pipe=QwenImageEditPlusPipeline.from_pretrained(str(model_root/'base'),transformer=transformer,
                                                        torch_dtype=torch.bfloat16,local_files_only=True)
        self.pipe.vae.to('cuda');self.pipe.vae.enable_tiling()
        self.pipe.text_encoder.to(device='cpu',dtype=torch.float32)
        if offload_blocks:
            transformer.enable_group_offload(onload_device=torch.device('cuda'),offload_device=torch.device('cpu'),
                                             offload_type='block_level',num_blocks_per_group=offload_blocks)
        else:
            # Decode after releasing the resident DiT; reload it for the next view.
            decode=self.pipe.vae.decode
            def decode_without_transformer(*args,**kwargs):
                self.pipe.transformer.to('cpu');torch.cuda.empty_cache()
                return decode(*args,**kwargs)
            self.pipe.vae.decode=decode_without_transformer

    def _encode_prompt(self, prompt, images):
        # Bounded, per-editor CPU cache: the sweep shares negative prompts and a smoke-test recipe.
        key=(prompt,tuple((im.mode,im.size,hashlib.sha256(im.tobytes()).digest()) for im in images))
        if key not in self.prompt_cache:
            print('[qwen_edit] encoding prompt on CPU',flush=True)
            value=self.pipe.encode_prompt(prompt=prompt,image=images,device=self.torch.device('cpu'))
            if len(self.prompt_cache)>=4:self.prompt_cache.pop(next(iter(self.prompt_cache)))
            self.prompt_cache[key]=value
        return self.prompt_cache[key]

    def __call__(self, images, prompt, *, steps=20, seed=317, cfg=4., size=1024,
                 negative='blurry, hair, hat, glasses, text, extra ears, shadows, highlights'):
        torch=self.torch;started=time.time();condition_size(size)
        torch.cuda.reset_peak_memory_stats()
        images=[Image.fromarray(i) if isinstance(i,np.ndarray) else i for i in images]
        with torch.no_grad():
            # Encode on CPU (encoder too large to share the GPU with the transformer).
            pos,pos_mask=self._encode_prompt(prompt,images)
            neg,neg_mask=self._encode_prompt(negative,images)
        pos,neg=pos.to(torch.bfloat16),neg.to(torch.bfloat16)
        encode=time.time()-started
        print(f'[qwen_edit] prompt encode {encode:.1f}s; starting {steps} denoising steps',flush=True)
        if not self.offload_blocks:self.pipe.transformer.to('cuda')
        move=lambda t:None if t is None else t.to('cuda')
        out=self.pipe(image=images,prompt_embeds=move(pos),prompt_embeds_mask=move(pos_mask),
            negative_prompt_embeds=move(neg),negative_prompt_embeds_mask=move(neg_mask),
            true_cfg_scale=cfg,height=size,width=size,num_inference_steps=steps,
            generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
        torch.cuda.synchronize()
        total=time.time()-started
        self.last_metrics=dict(seconds=total,prompt_encode_seconds=encode,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
        print(f'[qwen_edit] {self.last_metrics}',flush=True)
        return np.asarray(out.convert('RGB')),total
