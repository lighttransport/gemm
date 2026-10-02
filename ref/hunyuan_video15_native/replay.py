"""Prepare production-shape fixtures or measure their independent PyTorch replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from cuda.hunyuan_video15_native.generate import atomic_json, digest


def save(directory, name, array):
    value = array.detach().float().cpu().numpy() if hasattr(array, 'detach') else array
    value = np.ascontiguousarray(value, dtype='<f4')
    if not value.size or not np.isfinite(value).all():
        raise ValueError('invalid replay fixture/output')
    value.tofile(directory / (name + '.f32'))
    atomic_json(directory / (name + '.json'), dict(shape=list(value.shape), dtype='float32'))


def load(directory, name):
    info = json.loads((directory / (name + '.json')).read_text())
    path=directory/(name+'.f32')
    if info.get('dtype')!='float32' or not info['shape'] or any(d<=0 for d in info['shape']) or path.stat().st_size!=int(np.prod(info['shape']))*4:
        raise ValueError('invalid replay tensor: '+name)
    return np.memmap(path,dtype='<f4',mode='r',shape=tuple(info['shape']))


def compare_arrays(reference,actual):
    if reference.shape!=actual.shape or not reference.size: raise ValueError('mismatched/empty replay output')
    rr=aa=dot=error=0.;equal=True
    iterator=np.nditer([reference,actual],flags=['external_loop','buffered'],
                       op_flags=[['readonly'],['readonly']],order='C',buffersize=1<<18)
    for rv,av in iterator:
        rv,av=rv.astype(np.float64),av.astype(np.float64)
        if not np.isfinite(rv).all() or not np.isfinite(av).all(): raise ValueError('nonfinite replay output')
        delta=rv-av
        rr+=float(np.dot(rv,rv));aa+=float(np.dot(av,av));dot+=float(np.dot(rv,av));error+=float(np.dot(delta,delta))
        equal=equal and np.array_equal(rv,av)
    cosine=float(dot/np.sqrt(rr*aa)) if rr and aa else float(equal)
    relative=float(np.sqrt(error)/max(np.sqrt(rr),1e-30))
    return dict(cosine=cosine,relative_l2=relative,pass_all=cosine>=.9999 and relative<=.02)


def block_state(checkpoint, index):
    import torch
    from safetensors import safe_open
    prefix = f'double_blocks.{index}.'
    state = {}
    with safe_open(checkpoint, framework='pt', device='cpu') as file:
        for name in file.keys():
            if name.startswith(prefix):
                key, value = name[len(prefix):], file.get_tensor(name)
                if '_attn_qkv.' in key:
                    for suffix, tensor in zip(('q','k','v'), value.chunk(3, dim=0)):
                        state[key.replace('_qkv.', f'_{suffix}.')] = tensor
                else:
                    state[key] = value
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'reference', 'compare', 'advance','finish'))
    p.add_argument('--case', choices=('gemm','attention','vae_decode','conv','dit_block','dit_pair','qwen','siglip','byt5'), default='attention')
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--actual', type=Path)
    p.add_argument('--model', type=Path, default=ROOT / 'tmp/hv15-native/model')
    p.add_argument('--captures', type=Path, default=ROOT / 'tmp/hv15-native/reference-performance-v2/fast12/reference')
    p.add_argument('--upstream', type=Path, default=ROOT / 'tmp/hunyuan-video15-upstream')
    p.add_argument('--rows', type=int, default=34138)
    p.add_argument('--n', type=int, default=8192)
    p.add_argument('--k', type=int, default=2048)
    p.add_argument('--heads', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--blocks',type=int,default=1)
    p.add_argument('--negative',action='store_true')
    p.add_argument('--negative-fixture',type=Path)
    p.add_argument('--tile-row',type=int,default=0)
    p.add_argument('--tile-column',type=int,default=0)
    p.add_argument('--profile', default='fast12_i2v', choices=('fast12_i2v','quality_i2v','quality_t2v'))
    p.add_argument('--prefix', default='decoder.up.4.block.0.conv1.conv')
    args = p.parse_args()
    if not 1 <= args.repeats <= 10 or not 1 <= args.blocks <= 16 or not 0 <= args.index < 54 or args.index + args.blocks > 54:
        p.error('replay requires 1..10 repeats and 1..16 blocks within 0..53')
    if args.phase=='prepare' and args.case=='dit_pair' and (args.actual is None or args.negative_fixture is None):
        p.error('dit_pair requires --actual POSITIVE_FIXTURE and --negative-fixture NEGATIVE_FIXTURE')
    if args.phase!='prepare' and args.out is None:p.error('--out is required for this phase')
    if args.phase in ('compare','advance','finish') and args.actual is None:p.error('--actual is required for this phase')
    if args.phase=='prepare' and args.fixture.exists() and any(args.fixture.iterdir()):
        raise ValueError('prepare fixture must be empty/new')
    args.fixture.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.model / 'model.json').read_text())
    if args.phase == 'prepare':
        rng = np.random.default_rng(42)
        case = dict(kind=args.case, repeats=args.repeats, profile=args.profile, index=args.index,
                    blocks=args.blocks, negative=args.negative, model_manifest_sha256=digest(args.model / 'model.json'), height=53, width=30,
                    heads=args.heads, causal=1, prefix=args.prefix)
        if args.case == 'gemm':
            for name, shape in [('x',(args.rows,args.k)), ('w',(args.n,args.k))]:
                save(args.fixture, name, rng.standard_normal(shape, dtype=np.float32) * np.float32(.2))
        elif args.case == 'attention':
            for name in ('q','k','v'):
                save(args.fixture, name, rng.standard_normal((args.rows,args.heads*128), dtype=np.float32) * np.float32(.5))
        elif args.case == 'vae_decode':
            latent = np.load(args.captures / 'latent_final.npy', allow_pickle=False)
            save(args.fixture, 'x', latent[0,:,:,args.tile_row:args.tile_row+8,args.tile_column:args.tile_column+8].transpose(1,2,3,0))
            case['tile_origin']=[args.tile_row,args.tile_column]
            case['checkpoint'] = str((args.model / manifest['components']['vae']).resolve())
        elif args.case == 'conv':
            from safetensors import safe_open
            case['checkpoint'] = str((args.model / manifest['components']['vae']).resolve())
            with safe_open(case['checkpoint'], framework='numpy') as file:
                channels = file.get_slice(args.prefix + '.weight').get_shape()[1]
            save(args.fixture, 'x', rng.standard_normal((81,128,128,channels), dtype=np.float32) * np.float32(.2))
        elif args.case in ('qwen','siglip','byt5'):
            save(args.fixture,'x',load(args.captures,args.case+'_embedding'))
            case['checkpoint']=str((args.model/manifest['components']['vision' if args.case=='siglip' else args.case]).resolve())
            case['fixture_scope']='captured_encoder_embedding'
        elif args.case == 'dit_pair':
            positive=json.loads((args.actual/'case.json').read_text())
            negative=json.loads((args.negative_fixture/'case.json').read_text())
            for key in ('index','profile','checkpoint','height','width'):
                if positive[key]!=negative[key]: raise ValueError('CFG fixture mismatch: '+key)
            case=dict(positive,kind='dit_pair',blocks=args.blocks,repeats=args.repeats,
                      parent_fixtures=[digest(folder/'case.json') for folder in (args.actual,args.negative_fixture)])
            for folder,prefix in ((args.actual,''),(args.negative_fixture,'negative_')):
                for name in ('img','txt','vec'):
                    for suffix in ('.f32','.json'):
                        (args.fixture/(prefix+name+suffix)).hardlink_to(folder/(name+suffix))
            for name in ('freqs_cos','freqs_sin'):
                if digest(args.actual/(name+'.f32'))!=digest(args.negative_fixture/(name+'.f32')):
                    raise ValueError('CFG RoPE fixture mismatch')
                for suffix in ('.f32','.json'):
                    (args.fixture/(name+suffix)).hardlink_to(args.actual/(name+suffix))
        else:
            prepare_prefix(args, manifest, case)
        if 'checkpoint' in case:
            checkpoint=Path(case['checkpoint'])
            relative=str(checkpoint.relative_to(args.model.resolve()))
            case['checkpoint_sha256']=manifest['sources'][relative]['sha256']
            case['checkpoint_bytes']=checkpoint.stat().st_size
            case['checkpoint_mtime_ns']=str(checkpoint.stat().st_mtime_ns)
        case['inputs'] = {file.name:digest(file) for file in args.fixture.glob('*.f32')}
        atomic_json(args.fixture / 'case.json', case)
        return
    case = json.loads((args.fixture / 'case.json').read_text())
    if case['model_manifest_sha256']!=digest(args.model/'model.json'):raise ValueError('stale model manifest receipt')
    if 'checkpoint' in case:
        checkpoint=Path(case['checkpoint'])
        if checkpoint.stat().st_size!=case.get('checkpoint_bytes',checkpoint.stat().st_size) or str(checkpoint.stat().st_mtime_ns)!=case.get('checkpoint_mtime_ns',str(checkpoint.stat().st_mtime_ns)):
            raise ValueError('stale checkpoint receipt')
    for name, expected in case['inputs'].items():
        if digest(args.fixture / name) != expected:
            raise ValueError('stale fixture: ' + name)
    if args.phase in ('advance','finish'):
        if args.out.exists(): raise ValueError('advance output must be new')
        output_names=['output','text_output']+(['negative_output','negative_text_output'] if case['kind']=='dit_pair' else [])
        next_case=dict(case,index=case['index']+case.get('blocks',1),
                       parent_fixture_sha256=digest(args.fixture/'case.json'),
                       parent_outputs={name:digest(args.actual/(name+'.f32')) for name in output_names})
        if args.phase=='advance' and next_case['index']>=54: raise ValueError('all blocks already replayed')
        if args.phase=='finish' and next_case['index']!=54: raise ValueError('finish requires all 54 blocks')
        next_case['blocks']=min(case.get('blocks',1),54-next_case['index'])
        args.out.mkdir(parents=True)
        pairs=[('output','img'),('text_output','txt')]+([('negative_output','negative_img'),('negative_text_output','negative_txt')] if case['kind']=='dit_pair' else [])
        for old,new in pairs:
            for suffix in ('.f32','.json'):
                (args.out/(new+suffix)).hardlink_to(args.actual/(old+suffix))
        for name in ('vec','freqs_cos','freqs_sin')+ (('negative_vec',) if case['kind']=='dit_pair' else ()):
            for suffix in ('.f32','.json'):
                (args.out/(name+suffix)).hardlink_to(args.fixture/(name+suffix))
        if args.phase=='finish':
            next_case.update(kind='dit_pair_final' if case['kind']=='dit_pair' else 'dit_final',blocks=1,repeats=args.repeats)
            noise=np.load(args.captures/'noise_input.npy',allow_pickle=False)
            save(args.out,'latent',noise[0].transpose(1,2,3,0))
            steps,shift=(12,7) if case['profile']=='fast12_i2v' else (50,5)
            t=np.float32(1)-np.float32(1)/np.float32(steps)
            next_case['delta']=float(np.float32(shift)*t/(np.float32(1)+np.float32(shift-1)*t)-np.float32(1))
        next_case['inputs']={file.name:digest(file) for file in args.out.glob('*.f32')}
        atomic_json(args.out/'case.json',next_case)
        return
    if args.phase=='reference' and args.out.exists() and any(args.out.iterdir()):
        raise ValueError('reference output must be empty/new')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'compare':
        names = ['output'] + (['text_output'] if case['kind'] in ('dit_block','dit_pair') else [])
        if case['kind']=='dit_pair': names+=['negative_output','negative_text_output']
        if case['kind'] in ('dit_final','dit_pair_final'):
            names+=['guided_output','updated_latent']
            if case['kind']=='dit_pair_final':names+=['negative_output']
        result={}
        for name in names:
            values=[load(folder,name) for folder in (args.out,args.actual)]
            if case['kind'] in ('attention','gemm'):
                values=[v[None] if v.ndim==2 else v for v in values]
            entry=compare_arrays(*values)
            result[name]={**{k:v for k,v in entry.items() if k!='pass_all'},'pass':entry['pass_all']}
        frames=[]
        if case['kind']=='vae_decode':
            r=load(args.out,'output');a=load(args.actual,'output')
            for i in range(r.shape[0]):
                frames.append(dict(frame=i,**compare_arrays(r[i],a[i])))
        atomic_json(args.actual / 'parity.json', dict(results=result, frames=frames,pass_all=all(v['pass'] for v in result.values()) and all(v['pass_all'] for v in frames),
                    scope='bounded_production_shape_replay', full_pipeline_acceptance=False,
                    fixture_sha256=digest(args.fixture / 'case.json'),
                    outputs={name:{backend:digest(folder/(name+'.f32')) for backend,folder in (('native',args.actual),('reference',args.out))} for name in names},
                    output_metadata={name:{backend:digest(folder/(name+'.json')) for backend,folder in (('native',args.actual),('reference',args.out))} for name in names},
                    timing_sha256={backend:digest(folder/'timing.json') if (folder/'timing.json').is_file() else None for backend,folder in (('native',args.actual),('reference',args.out))},
                    receipt_sha256={backend:digest(folder.with_suffix('.json')) if folder.with_suffix('.json').is_file() else None for backend,folder in (('native',args.actual),('reference',args.out))}))
        print(json.dumps(result))
        if not all(v['pass'] for v in result.values()) or not all(v['pass_all'] for v in frames):
            raise ValueError('replay parity failed')
        return
    reference(args, case, manifest)


def prepare_prefix(args, manifest, case):
    import torch
    from safetensors import safe_open
    from ref.hunyuan_video15_native.verify import source
    source(args.upstream)
    from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
    checkpoint = args.model / manifest['checkpoints'][args.profile]
    config = {k:v for k,v in json.loads((args.model / manifest['reference_configs'][args.profile]).read_text()).items()
              if not k.startswith('_')}
    config['attn_mode'] = 'torch'
    with torch.device('meta'):
        model = HunyuanVideo_1_5_DiffusionTransformer(**config)
    state = {}
    with safe_open(checkpoint, framework='pt', device='cpu') as file:
        for name in file.keys():
            if not name.startswith(('double_blocks.', 'single_blocks.', 'final_layer.')):
                state[name] = file.get_tensor(name)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected or any(not name.startswith(('double_blocks.', 'single_blocks.', 'final_layer.')) for name in missing):
        raise ValueError('prefix checkpoint mismatch')
    del state
    for name, child in model.named_children():
        if name not in ('double_blocks','single_blocks','final_layer'):
            child.to('cuda', dtype=torch.float16)
    model.eval()
    class Captured(Exception):
        pass
    def capture(module, inputs, kwargs):
        for i,name in enumerate(('img','txt','vec')):
            save(args.fixture, name, inputs[i] if len(inputs)>i else kwargs[name])
        cosine, sine = kwargs['freqs_cis']
        save(args.fixture, 'freqs_cos', cosine)
        save(args.fixture, 'freqs_sin', sine)
        raise Captured()
    model.double_blocks[0].register_forward_pre_hook(capture, with_kwargs=True)
    def reference_tensor(name):
        return torch.from_numpy(np.load(args.captures / (name + '.npy'), allow_pickle=False).copy()).cuda()
    latent = reference_tensor('noise_input')
    condition = torch.zeros_like(latent)
    mask = torch.zeros_like(latent[:,:1])
    if args.profile.endswith('_i2v'):
        condition[:,:,:1] = reference_tensor('vae_encoded')
        mask[:,:,0] = 1
    text, glyph = reference_tensor('qwen_negative_hidden' if args.negative else 'qwen_hidden'), reference_tensor('byt5_hidden')
    vision = reference_tensor('siglip_hidden') if args.profile.endswith('_i2v') else None
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        try:
            model(torch.cat((latent,condition,mask),1), torch.tensor([1000.],device='cuda'),text,None,
                  torch.ones(text.shape[:2],dtype=torch.int64,device='cuda'),
                  timestep_r=torch.tensor([float(np.float32(7)*np.float32(11/12)/(1+np.float32(6)*np.float32(11/12))*1000)],device='cuda') if args.profile=='fast12_i2v' else None,
                  vision_states=vision,mask_type='i2v' if vision is not None else 't2v',
                  extra_kwargs={'byt5_text_states':glyph,'byt5_text_mask':torch.zeros(glyph.shape[:2],dtype=torch.int64,device='cuda')},return_dict=False)
        except Captured:
            pass
    if not (args.fixture / 'img.f32').is_file():
        raise ValueError('prefix hook did not run')
    case['checkpoint'] = str(checkpoint.resolve())
    case['upstream_revision'] = '60783e704160023913bee78f0b47036d393d4dfa'
    case['fixture_scope'] = 'official_first_block_inputs'


def reference(args, case, manifest):
    import torch
    from ref.hunyuan_video15_native import verify
    verify.source(args.upstream)
    torch.set_num_threads(16)
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    tick = time.monotonic()
    def tensor(name):
        return torch.from_numpy(load(args.fixture, name).copy()).to('cuda', dtype=torch.float16)
    kind = case['kind']
    cpu_encoder=kind in ('qwen','siglip','byt5')
    if kind == 'gemm':
        x, w = tensor('x'), tensor('w')
        action = lambda: torch.nn.functional.linear(x,w)
    elif cpu_encoder:
        action=encoder_reference(args,case,manifest)
    elif kind == 'attention':
        from hyvideo.models.transformers.modules.attention import parallel_attention
        q,k,v = [tensor(n).reshape(1,-1,case['heads'],128) for n in ('q','k','v')]
        empty = q[:,:0]
        mask = torch.ones((1,0),dtype=torch.int64,device='cuda')
        action = lambda: parallel_attention((q,empty),(k,empty),(v,empty),q.shape[1],q.shape[1],attn_mode='torch',text_mask=mask).reshape(-1,case['heads']*128)
    elif kind in ('dit_final','dit_pair_final'):
        from safetensors import safe_open
        from hyvideo.models.transformers.modules.mlp_layers import FinalLayer
        with torch.device('meta'):model=FinalLayer(2048,[1,1,1],32,torch.nn.SiLU)
        with safe_open(case['checkpoint'],framework='pt',device='cpu') as file:
            state={name[len('final_layer.'):]:file.get_tensor(name) for name in file.keys() if name.startswith('final_layer.')}
        model.load_state_dict(state,strict=True,assign=True)
        model.to('cuda',dtype=torch.float16).eval().requires_grad_(False)
        image,vector=tensor('img'),tensor('vec')
        latent=torch.from_numpy(load(args.fixture,'latent').copy()).cuda()
        if kind=='dit_pair_final':negative_image,negative_vector=tensor('negative_img'),tensor('negative_vec')
        def final():
            positive=model(image,vector).reshape(latent.shape)
            negative=model(negative_image,negative_vector).reshape(latent.shape) if kind=='dit_pair_final' else None
            guided=negative+6*(positive-negative) if negative is not None else positive
            return positive,negative,guided,latent+guided.float()*case['delta']
        action=final
    elif kind in ('dit_block','dit_pair'):
        from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import MMDoubleStreamBlock
        with torch.device('meta'):
            model = MMDoubleStreamBlock(hidden_size=2048,heads_num=16,mlp_width_ratio=4,
                                        attn_mode='torch',qkv_bias=True)
        model.load_state_dict(block_state(case['checkpoint'],case['index']),strict=True,assign=True)
        model.to('cuda',dtype=torch.float16).eval().requires_grad_(False)
        img,txt,vec = [tensor(n) for n in ('img','txt','vec')]
        freqs = tuple(torch.from_numpy(load(args.fixture,n)).cuda() for n in ('freqs_cos','freqs_sin'))
        mask = torch.ones(txt.shape[:2],dtype=torch.int64,device='cuda')
        action = lambda: model(img,txt,vec,freqs_cis=freqs,text_mask=mask)
        if case.get('blocks',1)>1 or kind=='dit_pair':
            models=[]
            for index in range(case['index'],case['index']+case['blocks']):
                with torch.device('meta'):
                    child=MMDoubleStreamBlock(hidden_size=2048,heads_num=16,mlp_width_ratio=4,attn_mode='torch',qkv_bias=True)
                child.load_state_dict(block_state(case['checkpoint'],index),strict=True,assign=True)
                child.eval().requires_grad_(False)
                models.append(child)
            model.to('cpu');del model
            inputs=[(img,txt,vec,mask)]
            if kind=='dit_pair':
                ni,nt,nv=[tensor(n) for n in ('negative_img','negative_txt','negative_vec')]
                inputs.append((ni,nt,nv,torch.ones(nt.shape[:2],dtype=torch.int64,device='cuda')))
            def chain():
                states=list(inputs)
                for child in models:
                    child.to('cuda',dtype=torch.float16)
                    states=[(*child(image,text,vector,freqs_cis=freqs,text_mask=text_mask),vector,text_mask)
                            for image,text,vector,text_mask in states]
                    child.to('cpu')
                return tuple(value for state in states for value in state[:2])
            action=chain
    elif kind == 'vae_decode':
        model = verify.vae_model(args.model,manifest,args.upstream)
        # Match the pipeline's FP32 scaling boundary before autocast convolutions.
        x = torch.from_numpy(load(args.fixture,'x').copy()).cuda().permute(3,0,1,2)[None]
        action = lambda: model.decode(x/model.scaling_factor).sample[0].permute(1,2,3,0)
    else:
        from safetensors import safe_open
        with safe_open(case['checkpoint'],framework='pt',device='cpu') as file:
            weight = file.get_tensor(case['prefix']+'.weight').cuda()
            bias = file.get_tensor(case['prefix']+'.bias').cuda() if case['prefix']+'.bias' in file.keys() else None
        x = tensor('x').permute(3,0,1,2)[None]
        kt,kh,kw = weight.shape[2:]
        padding = (kw//2,kw//2,kh//2,kh//2,kt-1,0) if case['causal'] else (kw//2,kw//2,kh//2,kh//2,0,0)
        padded = torch.nn.functional.pad(x,padding,mode='replicate' if case['causal'] else 'constant')
        action = lambda: torch.nn.functional.conv3d(padded,weight,bias)[0].permute(1,2,3,0)
    if not cpu_encoder: torch.cuda.synchronize()
    setup = time.monotonic()-tick
    device,wall = [],[]
    if not cpu_encoder: torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=not cpu_encoder):
        for i in range(case['repeats']):
            if not cpu_encoder: begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tick = time.monotonic()
            if not cpu_encoder: begin.record()
            output = action()
            if not cpu_encoder:
                end.record();torch.cuda.synchronize()
                device.append(begin.elapsed_time(end)/1000)
            wall.append(time.monotonic()-tick)
    if kind in ('dit_final','dit_pair_final'):
        save(args.out,'output',output[0]);save(args.out,'guided_output',output[2]);save(args.out,'updated_latent',output[3])
        if output[1] is not None:save(args.out,'negative_output',output[1])
    elif kind in ('dit_block','dit_pair'):
        save(args.out,'output',output[0]);save(args.out,'text_output',output[1])
        if kind=='dit_pair':
            save(args.out,'negative_output',output[2]);save(args.out,'negative_text_output',output[3])
    else:
        save(args.out,'output',output[None] if kind in ('gemm','attention') else output)
    atomic_json(args.out / 'timing.json',dict(setup_seconds=setup,device_seconds=device,wall_seconds=wall,
                peak_torch_allocated_mib=0 if cpu_encoder else torch.cuda.max_memory_allocated()/1048576,
                device='cpu_fp32' if cpu_encoder else 'cuda_fp16',
                upstream_revision=verify.UPSTREAM,torch_version=torch.__version__,full_pipeline_acceptance=False))
    print(json.dumps(dict(kind=kind,wall_seconds=wall,setup_seconds=setup)))


def encoder_reference(args,case,manifest):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig,SiglipVisionConfig,T5Config
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer,Qwen2_5_VLRotaryEmbedding
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
    from transformers.models.t5.modeling_t5 import T5Block
    kind,index=case['kind'],case['index']
    if kind=='qwen':
        config=AutoConfig.from_pretrained(args.model,local_files_only=True).text_config
        config._attn_implementation='eager'
        constructor=lambda:Qwen2_5_VLDecoderLayer(config,index)
        prefix=f'model.layers.{index}.'
    elif kind=='siglip':
        config=SiglipVisionConfig.from_dict(json.loads((args.model/'google_siglip/config.json').read_text())['vision_config'])
        config._attn_implementation='eager'
        constructor=lambda:SiglipEncoderLayer(config)
        prefix=f'vision_model.encoder.layers.{index}.'
    else:
        config=T5Config.from_json_file(args.model/manifest['reference_configs']['byt5'])
        config.is_decoder=False
        config._attn_implementation='eager'
        constructor=lambda:T5Block(config,has_relative_attention_bias=True,layer_idx=index)
        prefix=f'encoder.block.{index}.'
    with torch.device('meta'):model=constructor()
    with safe_open(case['checkpoint'],framework='pt',device='cpu') as file:
        state={name[len(prefix):]:file.get_tensor(name) for name in file.keys() if name.startswith(prefix)}
        if kind=='byt5' and index:
            state['layer.0.SelfAttention.relative_attention_bias.weight']=file.get_tensor('encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight')
    model.load_state_dict(state,strict=True,assign=True)
    model.to(dtype=torch.float32,device='cpu').eval().requires_grad_(False)
    x=torch.from_numpy(load(args.fixture,'x').copy())
    if x.ndim==2:x=x[None]
    if kind=='qwen':
        positions=torch.arange(x.shape[1]).reshape(1,1,-1).expand(3,1,-1)
        freqs=Qwen2_5_VLRotaryEmbedding(config,device='cpu')(x,positions)
        mask=torch.full((x.shape[1],x.shape[1]),float('-inf')).triu(1)[None,None]
        return lambda:model(x,attention_mask=mask,position_embeddings=freqs)
    if kind=='siglip':return lambda:model(x,attention_mask=None)
    return lambda:model(x)[0]


if __name__ == '__main__':
    main()
