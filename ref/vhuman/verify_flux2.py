"""Offline independent FLUX.2 Klein references; never imported by inference."""
import argparse
import json
from pathlib import Path
import numpy as np
if __package__:
    from .export_flux2_weights import digest
else:
    from export_flux2_weights import digest


def metric(actual, reference):
    a=np.asarray(actual,np.float64);b=np.asarray(reference,np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite tensors or different shapes')
    return dict(max_error=float(np.abs(a-b).max()),mean_error=float(np.abs(a-b).mean()),
                relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)))


def text_reference(model, prompt, output, threads=4, actual=None):
    import torch
    from transformers import AutoTokenizer, Qwen3Model
    torch.set_num_threads(threads)
    tokenizer=AutoTokenizer.from_pretrained(model/'tokenizer',local_files_only=True)
    text=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,
                                        add_generation_prompt=True,enable_thinking=False)
    inputs=tokenizer(text,return_tensors='pt',padding='max_length',truncation=True,max_length=512)
    encoder=Qwen3Model.from_pretrained(model/'text_encoder',dtype=torch.float32,
                                      local_files_only=True,attn_implementation='eager').eval()
    # Intermediate hidden states do not depend on later blocks; retain only 27.
    encoder.layers=encoder.layers[:27]
    # hidden_states[-1] has final RMSNorm. Capture block26 directly instead.
    captured={}
    hook=encoder.layers[26].register_forward_hook(lambda module,args,out: captured.update(last=out[0] if isinstance(out,tuple) else out))
    with torch.inference_mode():
        result=encoder(**inputs,output_hidden_states=True,use_cache=False)
    hook.remove()
    value=torch.cat([result.hidden_states[9],result.hidden_states[18],captured['last']],-1)[0].numpy()
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/'text-reference.npy',value)
    np.savez(output/'text-inputs.npz',ids=inputs['input_ids'].numpy(),mask=inputs['attention_mask'].numpy())
    report=dict(prompt=prompt,padding_side=tokenizer.padding_side,valid_tokens=int(inputs['attention_mask'].sum()),dtype='float32')
    if actual:
        report.update(metric(np.load(actual),value));report['passed']=report['relative_l2']<.002
    (output/'text-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report.get('passed', True)


def pipeline_reference(model, actual, output, text, image=None, threads=4, **unused):
    import gc
    import torch
    from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline as P, compute_empirical_mu
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True,exist_ok=True)
    noise=np.load(actual/'cuda_flux2_noise.npy')
    h,w=noise.shape[-2:];count=(h//2)*(w//2)
    device='cuda'
    vae=AutoencoderKLFlux2.from_pretrained(model/'vae',torch_dtype=torch.float32,local_files_only=True).eval()
    reference=None
    reports={}
    def record(name,value,actual_name=None):
        a=value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
        np.save(output/(name+'.npy'),a)
        path=actual/(actual_name or 'cuda_flux2_'+name+'.npy')
        if path.is_file():
            reports[name]=metric(np.load(path),a)
            reports[name]['passed']=reports[name]['relative_l2']<=.02
        else:reports[name]=dict(passed=False,error='missing native capture')
        print(name,reports[name],flush=True)
    def unpack(tokens):
        packed=tokens.transpose(1,2).reshape(1,128,h//2,w//2)
        return P._unpatchify_latents(packed)
    hidden=torch.from_numpy(np.load(text))[None].to(device)
    record('text_hidden',hidden[0])
    with torch.inference_mode():
        if image:
            vae.to(device)
            pixels=torch.from_numpy(np.fromfile(image,dtype='<f4').reshape(1,3,h*8,w*8)).to(device)
            encoded=vae.encode(pixels).latent_dist.mode()
            packed=P._patchify_latents(encoded)
            packed=(packed-vae.bn.running_mean[None,:,None,None])/torch.sqrt(vae.bn.running_var[None,:,None,None]+vae.config.batch_norm_eps)
            reference=P._pack_latents(packed)
            record('reference_latent',P._unpatchify_latents(packed)[0])
            vae.cpu();del pixels,encoded,packed;torch.cuda.empty_cache()
        dit=Flux2Transformer2DModel.from_pretrained(model/'transformer',torch_dtype=torch.float32,local_files_only=True).eval()
        blocks={'transformer_blocks','single_transformer_blocks'}
        for name,module in dit.named_children():
            if name not in blocks:module.to(device)
        hooks=[]
        def upload(module,args):module.to(device)
        def offload(module,args,result):module.cpu()
        for block in list(dit.transformer_blocks)+list(dit.single_transformer_blocks):
            hooks.extend([block.register_forward_pre_hook(upload),block.register_forward_hook(offload)])
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model/'scheduler',local_files_only=True)
        scheduler.set_timesteps(4,device=device,sigmas=np.linspace(1,1/4,4),mu=compute_empirical_mu(count,4))
        latents=P._pack_latents(P._patchify_latents(torch.from_numpy(noise)[None].to(device)))
        ids=P._prepare_latent_ids(torch.empty(1,128,h//2,w//2)).to(device)
        text_ids=P._prepare_text_ids(hidden).to(device)
        if reference is not None:
            ref_ids=P._prepare_image_ids([torch.empty(1,128,h//2,w//2)]).to(device)
            ids=torch.cat([ids,ref_ids],dim=1)
        np.save(output/'sigmas.npy',scheduler.sigmas.cpu().numpy())
        for i,t in enumerate(scheduler.timesteps):
            inputs=torch.cat([latents,reference],dim=1) if reference is not None else latents
            velocity=dit(hidden_states=inputs,timestep=t.expand(1)/1000,guidance=None,
                         encoder_hidden_states=hidden,txt_ids=text_ids,img_ids=ids,return_dict=False)[0][:,:count]
            record('vel'+str(i),unpack(velocity)[0])
            latents=scheduler.step(velocity,t,latents,return_dict=False)[0]
            record('step'+str(i),unpack(latents)[0])
        record('latent_final',unpack(latents)[0])
        for hook in hooks:hook.remove()
        del dit,inputs,velocity;gc.collect();torch.cuda.empty_cache()
        vae.to(device)
        packed=latents.transpose(1,2).reshape(1,128,h//2,w//2)
        raw=packed*torch.sqrt(vae.bn.running_var[None,:,None,None]+vae.config.batch_norm_eps)+vae.bn.running_mean[None,:,None,None]
        decoded=vae.decode(P._unpatchify_latents(raw),return_dict=False)[0][0]
        record('decoded',decoded)
    import diffusers
    report=dict(scope='full_four_step_component_chain',reference_dtype='float32_original_bf16_weights',
                shared_inputs=['noise','prepared_reference_image'] if image else ['noise'],
                reference_text=str(text),components=reports,passed=all(v['passed'] for v in reports.values()),
                torch_version=torch.__version__,diffusers_version=diffusers.__version__,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                provenance=dict(reference_script_sha256=digest(__file__),text_sha256=digest(text),
                    image_sha256=digest(image) if image else None,
                    weights={name:digest(model/name/'diffusion_pytorch_model.safetensors') for name in ('vae','transformer')},
                    native_captures={p.name:digest(p) for p in actual.glob('cuda_flux2_*.npy')},
                    reference_captures={p.name:digest(p) for p in output.glob('*.npy')}))
    (output/'pipeline-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--phase',choices=['text','pipeline'],default='text')
    p.add_argument('--prompt',default='The same person smiles gently, fixed frontal camera.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--actual',type=Path)
    p.add_argument('--text',type=Path)
    p.add_argument('--image',type=Path)
    args=vars(p.parse_args());phase=args.pop('phase')
    if phase=='text':
        args.pop('text');args.pop('image');raise SystemExit(0 if text_reference(**args) else 1)
    else:
        if not args['actual'] or not args['text']:p.error('pipeline requires --actual and independent --text')
        raise SystemExit(0 if pipeline_reference(**args) else 1)
