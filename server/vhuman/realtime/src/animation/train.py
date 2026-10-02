"""Native CPU causal-adapter training on independently cleared takes."""
import json
from pathlib import Path
import numpy as np
from .native_training import MotionTrainer
from .export_native import save_training
from ..avatar.provenance import validate_receipts, sha256


def train(manifest, output, epochs=20, device='cpu', seed=7, threads=4):
    if type(threads) is not int or not 1 <= threads <= 64 or type(epochs) is not int or epochs < 1:
        raise ValueError('threads must be 1..64 and epochs must be a positive integer')
    if device != 'cpu':
        raise ValueError('native motion training currently supports CPU')
    manifest = Path(manifest)
    spec = json.loads(manifest.read_text())
    if spec.get('format') != 'vhuman.motion_corpus.v1': raise ValueError('unsupported corpus')
    validate_receipts(spec['provenance'], 'motion-training')
    names, bounds = spec['names'], np.asarray(spec['ranges'],np.float32)
    if (not isinstance(names,list) or not names or len(set(names)) != len(names) or
            any(not isinstance(n,str) or not n for n in names) or bounds.shape != (len(names),2) or
            not np.isfinite(bounds).all() or (bounds[:,0] > bounds[:,1]).any()):
        raise ValueError('invalid motion control names/ranges')
    text_feed = spec.get('text_feed','incremental')
    if text_feed not in ('incremental','full'): raise ValueError('invalid motion text feed')
    root = manifest.parent.resolve()
    receipts = {r.get('path'):r for r in spec['provenance']}
    takes, seen = [], set()
    for item in spec['takes']:
        relative, split = item['path'], item['split']
        if split not in ('train','validation') or relative in seen:
            raise ValueError('invalid corpus split or duplicate take')
        if relative not in receipts: raise ValueError('take missing artifact receipt')
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != receipts[relative]['sha256']:
            raise ValueError('take checksum/path mismatch')
        with np.load(path,allow_pickle=False) as z:
            hidden, codes, target = z['hidden'].copy(), z['codes'].copy(), z['controls'].copy()
        if (hidden.ndim != 2 or codes.shape != (len(hidden),16) or
                target.shape != (len(hidden),8,len(names))):
            raise ValueError('take shapes must be hidden[T,H], codes[T,16], controls[T,8,C]')
        if (codes.dtype.kind not in 'iu' or (codes < 0).any() or (codes >= 2048).any() or
                not np.isfinite(hidden).all() or not np.isfinite(target).all()):
            raise ValueError('invalid corpus features/targets')
        if not len(hidden) or (takes and hidden.shape[1] != takes[0][1].shape[1]):
            raise ValueError('empty take or inconsistent hidden size')
        seen.add(relative)
        takes.append((split,np.asarray(hidden,np.float32),codes,np.asarray(target,np.float32)))
    if not any(t[0] == 'train' for t in takes) or not any(t[0] == 'validation' for t in takes):
        raise ValueError('independent train and validation takes required')
    train_hidden = np.concatenate([t[1] for t in takes if t[0] == 'train'])
    train_target = np.concatenate([t[3] for t in takes if t[0] == 'train'])
    mean = train_hidden.mean(0)
    scale = np.maximum(train_hidden.std(0),.1)
    activity = np.max(np.stack([abs(t[3]).max(axis=(0,1)) for t in takes if t[0] == 'train']),axis=0)
    weights = np.where(activity > .05,4.,1.).astype(np.float32)
    reports = []
    with MotionTrainer(train_hidden.shape[1],len(names),seed,threads) as model:
        tensors = model.state_dict()
        probability = np.clip((train_target.mean(axis=(0,1))-bounds[:,0])/np.maximum(bounds[:,1]-bounds[:,0],1e-6),.01,.99)
        tensors['output.bias'][:] = np.tile(np.log(probability/(1-probability)),8)
        model.load_state_dict(tensors)
        for epoch in range(epochs):
            losses = {'train':[],'validation':[]}
            for split, hidden, codes, target in takes:
                state = None  # Never carry recurrent state between utterances.
                for start in range(0,len(hidden),32):
                    normalized = (hidden[start:start+32]-mean)/scale
                    loss, _, state = model.compute(normalized,codes[start:start+32],target[start:start+32],
                                                  bounds,weights,state,update=split == 'train')
                    # Returned state has no gradient history: truncated BPTT.
                    losses[split].append(loss)
            reports.append(dict(epoch=epoch+1,**{k:float(np.mean(v)) for k,v in losses.items()}))
        tensors = model.state_dict()
    tensors.update(bounds=bounds,hidden_mean=mean,hidden_scale=scale)
    data = dict(trained=True,purpose=spec.get('purpose','diagnostic'),hidden_size=train_hidden.shape[1],
                names=names,ranges=spec['ranges'],tts_revision=spec['tts_revision'],text_feed=text_feed,
                reports=reports,provenance=spec['provenance'],corpus_sha256=sha256(manifest),
                hidden_normalization='training-only per-channel mean/std, floor0.1',
                objective='active-control weighted Huber plus temporal velocity',
                active_controls=[n for n,a in zip(names,activity) if a > .05],
                optimizer='native AdamW lr0.001 decay0.01 global gradient clip1',
                training_device='cpu',thread_budget=threads)
    save_training(output,data,tensors)
    return reports[-1]
