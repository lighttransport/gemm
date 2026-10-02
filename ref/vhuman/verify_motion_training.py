"""Optional CPU Torch oracle for native GRU gradients and one AdamW step."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def verify(output):
    import torch
    from server.vhuman.realtime.src.animation.causal_model import CausalMotion
    from server.vhuman.realtime.src.animation.native_training import MotionTrainer
    from server.vhuman.realtime.src.avatar.provenance import sha256
    torch.set_num_threads(1)
    rng=np.random.default_rng(31)
    hidden=rng.normal(size=(5,12)).astype(np.float32)
    codes=rng.integers(0,16,(5,16),dtype=np.int32)
    target=rng.uniform(-.1,.7,(5,8,2)).astype(np.float32)
    initial=rng.normal(0,.1,(2,128)).astype(np.float32)
    bounds=np.array([[-.2,.8],[0,1]],np.float32)
    weights=np.array([1,4],np.float32)
    reports={}
    with MotionTrainer(12,2,seed=19,threads=1) as native:
        model=CausalMotion(12,['jawOpen','mouthSmileLeft'],bounds).cpu()
        original=native.state_dict()
        state=dict(model.state_dict())
        state.update({k:torch.tensor(v) for k,v in original.items()})
        model.load_state_dict(state)
        prediction,final=model(torch.tensor(hidden)[None],torch.tensor(codes,dtype=torch.long)[None],torch.tensor(initial)[:,None])
        truth=torch.tensor(target)[None];w=torch.tensor(weights)
        loss=(torch.nn.functional.smooth_l1_loss(prediction,truth,beta=.1,reduction='none')*w).mean()
        flattened,flat_truth=prediction.flatten(1,2),truth.flatten(1,2)
        loss+=.05*((flattened[:,1:]-flattened[:,:-1]-flat_truth[:,1:]+flat_truth[:,:-1]).abs()*w).mean()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        loss.backward()
        native_loss,actual,actual_state=native.compute(hidden,codes,target,bounds,weights,initial,backward=True)
        gradients=native.gradients()
        reports['prediction']=float(abs(actual-prediction.detach().numpy()[0]).max())
        reports['state']=float(abs(actual_state-final.detach().numpy()[:,0]).max())
        reports['loss']=abs(native_loss-float(loss.detach()))
        reports['gradients']={name:float(abs(gradients[name]-p.grad.numpy()).max()) for name,p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        native.compute(hidden,codes,target,bounds,weights,initial,update=True)
        trained=native.state_dict()
        reports['adamw']={name:float(abs(trained[name]-p.detach().numpy()).max()) for name,p in model.named_parameters()}
        # AdamW amplifies tiny differences near zero gradients through epsilon.
        # Separately verify optimizer math using byte-identical input gradients.
        for name,p in model.named_parameters():
            p.data.copy_(torch.tensor(original[name]));p.grad=torch.tensor(gradients[name])
        identical_optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        identical_optimizer.step()
        reports['adamw_identical_gradients']={name:float(abs(trained[name]-p.detach().numpy()).max()) for name,p in model.named_parameters()}
    passed=(all(reports[k]<3e-6 for k in ('prediction','state','loss')) and
            max(reports['gradients'].values())<3e-6 and max(reports['adamw'].values())<2e-5 and
            max(reports['adamw_identical_gradients'].values())<3e-6)
    result=dict(passed=passed,scope='CPU math parity; quality/GPU validation deferred',metrics=reports,
                torch_version=torch.__version__,native_library_sha256=sha256(ROOT/'cpu/vhuman/libvhuman_training.so'))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return passed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if verify(args.output) else 1)
