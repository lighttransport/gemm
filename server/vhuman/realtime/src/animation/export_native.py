"""Offline conversion of trained motion adapters; Torch is used only here."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


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
