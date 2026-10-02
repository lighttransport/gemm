"""Native training bundle writer and optional legacy Torch conversion."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def save_training(checkpoint, data, tensors):
    """Write safetensors directly; file-shaped outputs are JSON receipts.

    Directory outputs are native bundles. File outputs (including old .pt CLI
    paths) keep the existing sibling-.native runtime resolution. No pickle or
    framework serialization is used by native training.
    """
    from ....rig import safetensors
    checkpoint = Path(checkpoint)
    directory = not checkpoint.suffix or checkpoint.suffix == '.native' or checkpoint.is_dir()
    bundle = checkpoint if directory else checkpoint.with_suffix('.native')
    bundle.mkdir(parents=True, exist_ok=True)
    source = bundle/'training.json' if directory else checkpoint
    source.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data, format='vhuman.native_motion_training.v1', backend='repository_cpu_gemm')
    safetensors.save(bundle/'motion.safetensors', tensors)
    data['weights_sha256'] = sha256(bundle/'motion.safetensors')
    partial = source.with_name(source.name+'.partial')
    partial.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    partial.replace(source)
    spec = {k:data[k] for k in ('trained','purpose','hidden_size','names','ranges','tts_revision')}
    spec.update(format='vhuman.native_motion.v1', text_feed=data.get('text_feed','incremental'),
                weights_sha256=data['weights_sha256'], source_sha256=sha256(source), training_backend=data['backend'])
    partial = bundle/'native.json.partial'
    partial.write_text(json.dumps(spec,indent=2,allow_nan=False)+'\n')
    partial.replace(bundle/'native.json')
    return bundle


def export(checkpoint, output=None):
    import torch
    from .causal_model import CausalMotion
    from ....rig import safetensors
    checkpoint=Path(checkpoint);output=Path(output) if output else checkpoint.with_suffix('.native')
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if data.get('format')!='vhuman.tts_motion.v1' or data.get('trained') is not True:
        raise ValueError('trained motion checkpoint required')
    model=CausalMotion(data['hidden_size'],data['names'],data['ranges'])
    state=dict(data['state_dict'])
    state.setdefault('hidden_mean',model.hidden_mean);state.setdefault('hidden_scale',model.hidden_scale)
    model.load_state_dict(state,strict=True)
    if not torch.equal(model.bounds,torch.tensor(data['ranges'],dtype=torch.float32)):
        raise ValueError('checkpoint bounds disagree with metadata')
    output.mkdir(parents=True,exist_ok=True)
    safetensors.save(output/'motion.safetensors',{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    spec={k:data[k] for k in ('trained','purpose','hidden_size','names','ranges','tts_revision')}
    spec.update(format='vhuman.native_motion.v1',text_feed=data.get('text_feed','incremental'),
                weights_sha256=sha256(output/'motion.safetensors'),source_sha256=sha256(checkpoint))
    temp=output/'native.json.partial';temp.write_text(json.dumps(spec,indent=2)+'\n');temp.replace(output/'native.json')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();print(export(a.checkpoint,a.out))
