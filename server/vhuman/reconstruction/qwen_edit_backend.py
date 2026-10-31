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
from server.model_paths import MODELS_ROOT
import hashlib
import re
import time

import numpy as np
from PIL import Image

ROOT=MODELS_ROOT / 'qwen-image-edit-2511'
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


def prompt_images(pipe, images):
    """Match Edit-Plus preprocessing before encoding prompts separately on CPU."""
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as plus
    result=[]
    for image in images:
        width,height=plus.calculate_dimensions(plus.CONDITION_IMAGE_SIZE,image.width/image.height)
        result.append(pipe.image_processor.resize(image,height,width))
    return result


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


def parse_lora_spec(spec):
    """'path[:weight[:adapter]]' or dict(path, weight=1.0, adapter=None) -> dict."""
    if isinstance(spec,dict):
        path=Path(spec['path']);weight=float(spec.get('weight',1.));adapter=spec.get('adapter')
    else:
        parts=str(spec).rsplit(':',2)
        # a bare path may contain ':' only in exotic cases; weights/adapters are short tokens
        path=Path(parts[0]);weight=float(parts[1]) if len(parts)>1 and parts[1] else 1.
        adapter=parts[2] if len(parts)>2 and parts[2] else None
    if not path.is_file():raise FileNotFoundError(f'LoRA weights not found: {path}')
    return dict(path=path,weight=weight,adapter=adapter or re.sub(r'[^A-Za-z0-9_]+','_',path.stem))


def host_peak_rss():
    """Peak resident set size of this process in bytes (Linux VmHWM)."""
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmHWM:'):return int(line.split()[1])*1024
    except OSError:pass
    return None


def make_editor(backend=None, model_root=ROOT, offload_blocks=0, *, loras=None, encoder_dtype='bfloat16', encoder_device='cuda'):
    """Select explicitly, or use native only with a ROCm Torch runtime and its assets.

    loras: optional LoRA specs (see parse_lora_spec); only the GGUF editor can apply them.
    encoder_dtype: CPU dtype of the Qwen2.5-VL prompt encoder for the GGUF editor ('bfloat16' halves host RSS)."""
    import os
    import torch
    backend=backend or os.environ.get('QWEN_EDIT_BACKEND','auto')
    if backend not in ('auto','native','gguf'):raise ValueError('unknown Edit-2511 backend: '+backend)
    model_root=Path(model_root)
    native=bool(torch.version.hip) and ((model_root/INT4.name).is_file() or (model_root/NATIVE_PACKS).is_file()) \
        and (REPO/'rdna4/qimg/libhip_qimg.so').is_file()
    if backend=='native' and not native:
        raise ValueError('native Edit-2511 requires ROCm Torch, the HIP library and the mixed INT4 weights')
    if native and backend in ('native','auto') and loras:
        try:
            select_native_pack(model_root,[parse_lora_spec(s) for s in loras])
        except ValueError:
            if backend=='native':raise
            native=False   # auto: no pack with these LoRAs -> GGUF + PEFT adapters
    if backend=='native' or (backend=='auto' and native):
        return NativeEditor(model_root,loras=loras,encoder_device=encoder_device)
    return Editor(model_root,offload_blocks=offload_blocks,loras=loras,encoder_dtype=encoder_dtype,encoder_device=encoder_device)


def linear_patch_embed(visual):
    """Qwen2.5-VL's patch embedding is a Conv3d whose stride equals its kernel (2x14x14): exactly a matmul over the
    flattened (C, T, H, W) patch. Running it as one GEMM skips MIOpen's first-call algorithm search (~2.5 s per process
    on gfx1201) and is numerically the same contraction."""
    import torch
    proj=visual.patch_embed.proj
    if proj.stride!=proj.kernel_size or proj.bias is not None:return False
    def forward(hidden_states):
        weight=proj.weight.reshape(proj.out_channels,-1)   # view of the live conv weight (follows .to() moves)
        return torch.nn.functional.linear(hidden_states.reshape(-1,weight.shape[1]).to(weight.dtype),weight)
    visual.patch_embed.forward=forward
    return True


def stream_decoder_layers(encoder, device):
    """Keep a Qwen2.5-VL encoder's small modules on `device` and stream its decoder layers from pinned host memory."""
    import torch
    language=encoder.model.language_model
    # Prompt embeddings come from the last hidden state; the 152k-vocabulary LM head only produces unused logits.
    encoder.lm_head=torch.nn.Identity()
    encoder._vhuman_resident=(encoder.model.visual,language.embed_tokens,language.norm,language.rotary_emb)
    encoder._vhuman_device=device
    for module in encoder._vhuman_resident:
        module.to(device)
    linear_patch_embed(encoder.model.visual)
    import torch
    host={}
    for layer in language.layers:
        # One pinned buffer per layer: PyTorch's pinned allocator rounds each allocation up to a power of two, so
        # pinning tensors individually costs ~40% extra host RAM.
        tensors=[t for t in list(layer.parameters())+list(layer.buffers())]
        sizes=[t.numel()*t.element_size() for t in tensors]
        flat=torch.empty(sum(sizes),dtype=torch.uint8,pin_memory=True);offset=0;pairs=[]
        for t,size in zip(tensors,sizes):
            view=flat[offset:offset+size].view(t.dtype).view(t.shape);view.copy_(t.data);t.data=view;offset+=size
            pairs.append((t,view))
        host[layer]=pairs
    def onload(module,args,kwargs):
        for t,pinned in host[module]:t.data=pinned.to(device,non_blocking=True)
    def release(module,args,kwargs,output):
        for t,pinned in host[module]:t.data=pinned
    for layer in language.layers:
        layer.register_forward_pre_hook(onload,with_kwargs=True);layer.register_forward_hook(release,with_kwargs=True)
    return host


NATIVE_PACKS='native_packs.json'   # model_root registry: {"bf16_transformer": dir, "packs": [{"file", "sha256", "loras": [{"sha256", "weight"}], ...}]}


def select_native_pack(model_root, specs):
    """The native DiT has no adapter path: LoRAs are merged at pack time, so pick the pack built with exactly these
    LoRAs (by file sha256 and weight). Returns (pack record, registry)."""
    import json
    from .observations import sha256
    registry_file=Path(model_root)/NATIVE_PACKS
    registry=json.loads(registry_file.read_text()) if registry_file.is_file() else dict(packs=[dict(file=INT4.name,loras=[])])
    want=sorted((sha256(s['path']),round(float(s['weight']),4)) for s in specs)
    for pack in registry['packs']:
        have=sorted((l['sha256'],round(float(l['weight']),4)) for l in pack.get('loras',[]))
        if have==want and (Path(model_root)/pack['file']).is_file():return pack,registry
    raise ValueError('no native Edit-2511 pack was built with these LoRAs: '+', '.join(Path(s['path']).name for s in specs))


class NativeEditor:
    """Qwen-Image-Edit-2511 on the native RDNA4 mixed INT4/INT8 DiT (rdna4/qimg). LoRAs are merged into the pack
    (main linears) and into the host BF16 modulation weights; the prompt encoder streams on the GPU like Editor."""
    generator=NATIVE_GENERATOR

    def __init__(self, model_root=ROOT, *, loras=None, encoder_device='cuda'):
        import os,sys,torch
        os.environ.setdefault('QIMG_MIOPEN_CACHE',str(REPO/'tmp/miopen-cache'))
        sys.path.insert(0,str(REPO/'rdna4/qimg'))
        from qimg_edit_native import load_pipeline
        from .observations import sha256
        self.torch=torch;self.device=torch.device('cuda');self.prompt_cache={}
        model_root=Path(model_root);specs=[parse_lora_spec(s) for s in (loras or [])]
        pack,registry=select_native_pack(model_root,specs)
        mod_source=registry.get('bf16_transformer')
        self.pipe=load_pipeline(str(model_root/'base'),str(model_root/pack['file']),mod_source=mod_source,
                                mod_loras=[f"{s['path']}:{s['weight']}" for s in specs])
        self.metadata=dict(backend='native',model_root=str(model_root.resolve()),weights=pack['file'],
                           weights_sha256=pack.get('sha256'),pack_recipe=pack.get('recipe'),encoder_device=encoder_device,
                           host_modulation_source=mod_source,host_modulation_cache=str(getattr(self.pipe.host_mod,'disk','')),
                           loras=[dict(path=str(s['path'].resolve()),sha256=sha256(s['path']),adapter=s['adapter'],weight=s['weight'],
                                       merged='pack + host modulation') for s in specs])
        encoder=self.pipe.text_encoder
        if encoder_device=='cuda':
            encoder.to(device='cpu',dtype=torch.bfloat16);stream_decoder_layers(encoder,self.device)
            self.encoder_device=self.device;self.park_encoder=True   # the resident native DiT needs the room
            for module in encoder._vhuman_resident:module.to('cpu')
        else:
            encoder.to(torch.float32);self.encoder_device=torch.device('cpu');self.park_encoder=False

    def load_loras(self, specs):
        raise NotImplementedError('native packs merge LoRAs at build time; pass loras= to make_editor')

    def __call__(self, images, prompt, *, steps=20, seed=317, cfg=4., size=1024,
                 negative='blurry, hair, hat, glasses, text, extra ears, shadows, highlights'):
        """Same flow as Editor (GGUF), which preserves identity: prompt encode, cuda pipeline + generator."""
        torch=self.torch;condition_size(size)
        import os
        ref_side=int(os.environ.get('QIMG_REF_SIDE','0'))   # e.g. 512: references at 512^2, target at size^2
        if ref_side:
            reference_areas([ref_side*ref_side]*(len(images)-1)+[size*size])
        torch.cuda.reset_peak_memory_stats()
        started=time.time();dit0=self.pipe.native.seconds
        images=[Image.fromarray(i) if isinstance(i,np.ndarray) else i for i in images]
        conditions=prompt_images(self.pipe,images)
        with torch.no_grad():
            pos,pos_mask=self._encode_prompt(prompt,conditions)
            neg,neg_mask=self._encode_prompt(negative,conditions)
        pos,neg=pos.to(torch.bfloat16),neg.to(torch.bfloat16)
        encode=time.time()-started
        move=lambda t:None if t is None else t.to(self.device)
        out=self.pipe(image=images,prompt_embeds=move(pos),prompt_embeds_mask=move(pos_mask),
            negative_prompt_embeds=move(neg),negative_prompt_embeds_mask=move(neg_mask),
            true_cfg_scale=cfg,height=size,width=size,num_inference_steps=steps,
            generator=torch.Generator(device=self.device).manual_seed(seed)).images[0]
        total=time.time()-started;dit=self.pipe.native.seconds-dit0
        mods=getattr(self.pipe,'host_mod',None)
        if mods is not None:mods.release()   # schedule cached in RAM/disk; drop the 13.6 GB BF16 modulation weights
        self.last_metrics=dict(seconds=total,prompt_encode_seconds=encode,native_dit_seconds=dit,
                               host_modulation=None if mods is None else dict(mods.stats,lora_merged_modules=mods.merged),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(),host_peak_rss_bytes=host_peak_rss())
        print(f'[qwen_edit] total {total:.1f}s native DiT {dit:.1f}s prompt_encode {encode:.1f}s other {total-dit-encode:.1f}s',flush=True)
        return np.asarray(out.convert('RGB')),total


class Editor:
    generator=GENERATOR

    def __init__(self, model_root=ROOT, *, offload_blocks=0, loras=None, encoder_dtype='bfloat16', encoder_device='cuda'):
        import torch
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
        from .observations import sha256
        if encoder_dtype not in ('bfloat16','float32'):raise ValueError('encoder_dtype must be bfloat16 or float32')
        if encoder_device not in ('cuda','cpu'):raise ValueError('encoder_device must be cuda or cpu')
        self.torch=torch;self.prompt_cache={};model_root=Path(model_root);weights=model_root/GGUF.name
        if offload_blocks<0:raise ValueError('offload_blocks must be nonnegative')
        self.offload_blocks=offload_blocks;self.device=torch.device('cuda')
        self.metadata=dict(backend='gguf',model_root=str(model_root.resolve()),weights=weights.name,
                           weights_sha256=sha256(weights),encoder_dtype=encoder_dtype,offload_blocks=offload_blocks,
                           prompt_preprocessing='pipeline_condition_images',loras=[],encoder_device=encoder_device)
        transformer=QwenImageTransformer2DModel.from_single_file(str(weights),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=str(model_root/'base'),subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
        self.pipe=QwenImageEditPlusPipeline.from_pretrained(str(model_root/'base'),transformer=transformer,
                                                        torch_dtype=torch.bfloat16,local_files_only=True)
        self.pipe.vae.to(self.device);self.pipe.vae.enable_tiling()
        # Prompt encoder (Qwen2.5-VL 7B, ~16 GB BF16). 'cuda': the vision tower, token embeddings, final norm and
        # LM head live on the GPU (~3.3 GB); each of the 28 decoder layers is uploaded from pinned host memory just
        # before it runs and its parameters re-point at the host copy afterwards (no copy-back). ~1 s per prompt on
        # the RX 9070 XT at ~4.3 GB peak. 'cpu': the old CPU encoder (minutes per prompt without native BF16).
        encoder=self.pipe.text_encoder;encoder.to(device='cpu',dtype=getattr(torch,encoder_dtype))
        self.encoder_device=torch.device('cuda') if encoder_device=='cuda' else torch.device('cpu')
        if encoder_device=='cuda':stream_decoder_layers(encoder,self.encoder_device)
        # With a resident DiT (offload_blocks=0) the encoder's GPU-resident modules are parked in host RAM between
        # encodes so the ~13 GB Q4 transformer and its activations fit on a 16 GB card.
        self.park_encoder=offload_blocks==0
        if self.park_encoder and encoder_device=='cuda':
            for module in encoder._vhuman_resident:module.to('cpu')
        if loras:self.load_loras(loras)   # PEFT injection must precede group-offload hooks
        if offload_blocks:
            transformer.enable_group_offload(onload_device=self.device,offload_device=torch.device('cpu'),
                                             offload_type='block_level',num_blocks_per_group=offload_blocks)
        else:
            # Decode after releasing the resident DiT; reload it for the next view.
            decode=self.pipe.vae.decode
            def decode_without_transformer(*args,**kwargs):
                self.pipe.transformer.to('cpu');torch.cuda.empty_cache()
                return decode(*args,**kwargs)
            self.pipe.vae.decode=decode_without_transformer

    def load_loras(self, specs):
        """Attach LoRA adapters to the GGUF transformer (diffusers dequantizes GGUF weights under PEFT layers).

        Records path, sha256, weight and the number of injected LoRA modules so receipts show whether
        an adapter actually bound to the model (a 2509-trained adapter on 2511 is only accepted if it binds)."""
        from .observations import sha256
        if self.offload_blocks and any(getattr(m,'_diffusers_hook',None) for m in self.pipe.transformer.modules()):
            raise RuntimeError('load LoRAs before enabling group offload')
        names,weights=[],[]
        for spec in specs:
            spec=parse_lora_spec(spec)
            self.pipe.load_lora_weights(str(spec['path']),adapter_name=spec['adapter'])
            # PEFT layers hold one lora_A entry per adapter, so count this adapter's entries.
            before,after=0,sum(1 for n,_ in self.pipe.transformer.named_modules() if n.endswith('lora_A.'+spec['adapter']))
            if after<=before:raise RuntimeError(f"LoRA {spec['path'].name} injected no modules into the transformer")
            names.append(spec['adapter']);weights.append(spec['weight'])
            self.metadata['loras'].append(dict(path=str(spec['path'].resolve()),sha256=sha256(spec['path']),
                                               adapter=spec['adapter'],weight=spec['weight'],lora_modules=after-before))
            print(f"[qwen_edit] LoRA {spec['path'].name}: {after-before} modules, weight {spec['weight']}",flush=True)
        self.pipe.set_adapters(names,weights)
        return self.metadata['loras']

    def _encode_prompt(self, prompt, images):
        # Bounded, per-editor CPU cache: the sweep shares negative prompts and a smoke-test recipe.
        key=(prompt,tuple((im.mode,im.size,hashlib.sha256(im.tobytes()).digest()) for im in images))
        if key not in self.prompt_cache:
            print(f'[qwen_edit] encoding prompt on {self.encoder_device.type}',flush=True)
            encoder=self.pipe.text_encoder;park=getattr(self,'park_encoder',False) and self.encoder_device.type=='cuda'
            if park:
                for module in encoder._vhuman_resident:module.to(self.encoder_device)
            value=self.pipe.encode_prompt(prompt=prompt,image=images,device=self.encoder_device)
            if park:   # free ~3.3 GB for a resident DiT; re-uploaded (~0.5 s) on the next cache miss
                for module in encoder._vhuman_resident:module.to('cpu')
                self.torch.cuda.empty_cache()
            value=tuple(None if v is None else v.to('cpu') for v in value)   # cache on host; moved per call
            if len(self.prompt_cache)>=4:self.prompt_cache.pop(next(iter(self.prompt_cache)))
            self.prompt_cache[key]=value
        return self.prompt_cache[key]

    def __call__(self, images, prompt, *, steps=20, seed=317, cfg=4., size=1024,
                 negative='blurry, hair, hat, glasses, text, extra ears, shadows, highlights'):
        torch=self.torch;started=time.time();condition_size(size)
        torch.cuda.reset_peak_memory_stats()
        images=[Image.fromarray(i) if isinstance(i,np.ndarray) else i for i in images]
        conditions=prompt_images(self.pipe,images)
        with torch.no_grad():
            pos,pos_mask=self._encode_prompt(prompt,conditions)
            neg,neg_mask=self._encode_prompt(negative,conditions)
        pos,neg=pos.to(torch.bfloat16),neg.to(torch.bfloat16)
        encode=time.time()-started
        print(f'[qwen_edit] prompt encode {encode:.1f}s; starting {steps} denoising steps',flush=True)
        if not self.offload_blocks:self.pipe.transformer.to(self.device)
        move=lambda t:None if t is None else t.to(self.device)
        out=self.pipe(image=images,prompt_embeds=move(pos),prompt_embeds_mask=move(pos_mask),
            negative_prompt_embeds=move(neg),negative_prompt_embeds_mask=move(neg_mask),
            true_cfg_scale=cfg,height=size,width=size,num_inference_steps=steps,
            generator=torch.Generator(device=self.device).manual_seed(seed)).images[0]
        torch.cuda.synchronize()
        total=time.time()-started
        self.last_metrics=dict(seconds=total,prompt_encode_seconds=encode,
                               prompt_image_sizes=[list(i.size) for i in conditions],
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                               host_peak_rss_bytes=host_peak_rss())
        print(f'[qwen_edit] {self.last_metrics}',flush=True)
        return np.asarray(out.convert('RGB')),total


NativeEditor._encode_prompt=Editor._encode_prompt   # same cached GPU/CPU prompt encoding


def main(argv=None):
    """Smoke test: one edit with receipts for VRAM and host RSS (budget gates for the 8 GB / 32 GB host)."""
    import argparse,json,os
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('images',nargs='+',help='condition images; the target view is LAST')
    ap.add_argument('--out',required=True,help='output directory (created)')
    ap.add_argument('--prompt',required=True)
    ap.add_argument('--negative',default='blurry, hair, hat, glasses, text, extra ears, shadows, highlights')
    ap.add_argument('--backend',choices=('auto','native','gguf'),default=None)
    ap.add_argument('--offload-blocks',type=int,default=0)
    ap.add_argument('--encoder-dtype',choices=('bfloat16','float32'),default='bfloat16')
    ap.add_argument('--encoder-device',choices=('cuda','cpu'),default='cuda')
    ap.add_argument('--lora',action='append',default=[],help='path[:weight[:adapter]] (repeatable, GGUF only)')
    ap.add_argument('--size',type=int,default=1024);ap.add_argument('--steps',type=int,default=12)
    ap.add_argument('--seed',type=int,default=317);ap.add_argument('--cfg',type=float,default=4.)
    ap.add_argument('--max-vram-mib',type=int,default=8192);ap.add_argument('--max-rss-mib',type=int,default=32768)
    args=ap.parse_args(argv)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    editor=make_editor(args.backend,offload_blocks=args.offload_blocks,loras=args.lora or None,encoder_dtype=args.encoder_dtype,encoder_device=args.encoder_device)
    images=[Image.open(p).convert('RGB') for p in args.images]
    result,seconds=editor(images,args.prompt,steps=args.steps,seed=args.seed,cfg=args.cfg,size=args.size,negative=args.negative)
    Image.fromarray(result).save(out/'edit.png')
    metrics=dict(getattr(editor,'last_metrics',{}));metrics.update(seconds=seconds,metadata=editor.metadata,
        prompt=args.prompt,negative=args.negative,steps=args.steps,seed=args.seed,cfg=args.cfg,size=args.size,
        inputs=[str(Path(p).resolve()) for p in args.images],license=LICENSE,generator=editor.generator)
    vram=metrics.get('peak_reserved_bytes');rss=metrics.get('host_peak_rss_bytes') or host_peak_rss()
    metrics['host_peak_rss_bytes']=rss
    metrics['budget']=dict(max_vram_mib=args.max_vram_mib,max_rss_mib=args.max_rss_mib,
        vram_ok=None if vram is None else vram<=args.max_vram_mib<<20,rss_ok=None if rss is None else rss<=args.max_rss_mib<<20)
    (out/'editor_metrics.json').write_text(json.dumps(metrics,indent=2,default=str)+'\n')
    print(json.dumps(metrics['budget']),flush=True)
    return 0 if all(v is not False for v in metrics['budget'].values()) else 1


if __name__=='__main__':
    raise SystemExit(main())

