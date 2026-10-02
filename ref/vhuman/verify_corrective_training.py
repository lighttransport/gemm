"""Optional CPU Torch oracle for native corrective rig, solve and MLP math."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def verify(output):
    import torch
    from ref.vhuman import corrective_torch_reference as reference
    from server.vhuman.test_native_corrective import fixture
    from server.vhuman.rig import mldeformer_training as implementation
    from server.vhuman.rig.native_corrective import MLP2
    from server.vhuman.rig.torchrig import TorchRig
    from server.vhuman.realtime.src.avatar.provenance import sha256
    torch.set_num_threads(1);rng=np.random.default_rng(23)
    tmpl,rig,contacts,feat,skel,teeth,shapes,tongue=fixture()
    oracle=TorchRig(rig.d,rig.rest,shapes,rig.jn,rig.w,device='cpu')
    ref_contacts=reference.Contacts(tmpl,feat,skel,teeth,'cpu',rig.rest,tongue=tongue)
    controls=rng.uniform(0,1,(2,len(rig.controls))).astype(np.float32)
    original=oracle(torch.tensor(controls));actual=rig(controls)
    reports={name:float(abs(actual[name]-original[name].detach().numpy()).max()) for name in ('pos','skin','blend','inputs','pre')}
    # Same positions/transforms for contact derivatives, to isolate kernel math.
    x=original['pos'].detach()*1000;x.requires_grad_(True)
    e,stats,depth=ref_contacts.energy(x,original['skin'].detach(),per_vertex=True);e.sum().backward()
    energy,counts,gradient,actual_depth=contacts.compute(x.detach().numpy(),contacts.posed(original['skin'].detach().numpy()),True)
    reports['contact_energy_relative']=float(abs(energy-e.detach().numpy()).max()/max(1,float(e.max().detach())))
    reports['contact_gradient']=float(abs(gradient-x.grad.numpy()).max())
    reports['contact_depth']=float(abs(actual_depth-depth.detach().numpy()).max())
    reports['contact_counts_equal']=all(np.array_equal(counts[key],value.numpy()) for key,value in stats.items())
    native_solver=implementation.Solver(rig,tmpl,contacts,iters=3)
    ref_solver=reference.Solver(oracle,tmpl,ref_contacts,iters=3)
    # Both algorithms deliberately use twelve Newton iterations, even on thin fans.
    reference_rotation=ref_solver.rotations(x.detach()).numpy()
    actual_rotation=native_solver.rotations(x.detach().numpy())
    reports['rotation']=float(abs(actual_rotation-reference_rotation).max())
    solved=native_solver.solve(controls);ref_solved=ref_solver.solve(torch.tensor(controls))
    reports['solve_offset_mm']=float(abs(solved['offset_mm']-ref_solved['offset_mm'].numpy()).max())
    reports['solve_residual_mm']=float(abs(solved['residual_mm']-ref_solved['residual_mm'].numpy()).max())
    # Exercise coefficient and posed lip objective, all MLP parameter gradients,
    # cosine-scheduled AdamW over multiple steps using exactly the same weights.
    n,k,p=6,3,5;model=MLP2(5,7,k,seed=31)
    mlp=reference.MLP2(5,7,k);mlp.load_state_dict({key:torch.tensor(value) for key,value in model.state_dict().items()})
    inputs=rng.normal(size=(n,5)).astype(np.float32);target=rng.normal(size=(n,k)).astype(np.float32)
    cs=np.array([.4,1.4,2],np.float32);sw=np.array([[1],[5],[1],[5],[1],[5]],np.float32)
    lips=(rng.normal(size=(k,p,3)).astype(np.float32),rng.normal(size=(k,p,3)).astype(np.float32),
        rng.normal(size=(p,3)).astype(np.float32),rng.normal(size=(p,3)).astype(np.float32),
        rng.normal(size=(n,p,3)).astype(np.float32),rng.normal(size=(n,p,3)).astype(np.float32),
        rng.normal(size=(n,p)).astype(np.float32),np.full(p,2,np.float32),np.array([.1,.2,.3],np.float32))
    bu,bl,mu,ml,au,al,separation,floor,cm=[torch.tensor(v) for v in lips]
    optimizer=torch.optim.AdamW(mlp.parameters(),lr=.003,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,10)
    maximum_forward,maximum_loss,maximum_gradient=0.,0.,0.
    for step in range(10):
        optimizer.zero_grad();pred=mlp(torch.tensor(inputs))
        coefficient=pred*torch.tensor(cs)+cm
        du=torch.einsum('nk,kpc->npc',coefficient,bu)+mu
        dl=torch.einsum('nk,kpc->npc',coefficient,bl)+ml
        sep=separation+(au*du).sum(-1)-(al*dl).sum(-1)
        loss=(((pred-torch.tensor(target))*torch.tensor(cs/cs.max()))**2*torch.tensor(sw)).mean()+2/float(cs.max())**2*(torch.relu(floor-sep)**2).mean()
        loss.backward()
        prediction=model(inputs);native_loss,upstream=implementation.objective(prediction,target,cs,sw,np.arange(n),lips,2)
        _,g=model.compute(inputs,upstream)
        expected_gradients=np.concatenate([v.grad.detach().numpy().ravel() for v in mlp.parameters()])
        maximum_forward=max(maximum_forward,float(abs(prediction-pred.detach().numpy()).max()))
        maximum_loss=max(maximum_loss,abs(native_loss-float(loss.detach())))
        maximum_gradient=max(maximum_gradient,float(abs(g-expected_gradients).max()))
        model.optimizer.step(g,lr=optimizer.param_groups[0]['lr']);optimizer.step();scheduler.step()
    reports.update(mlp_forward=maximum_forward,objective=maximum_loss,mlp_gradients=maximum_gradient,
                   adamw=float(abs(model.parameters-np.concatenate([v.detach().numpy().ravel() for v in mlp.parameters()])).max()))
    passed=(all(reports[key]<2e-6 for key in ('pos','skin','blend','inputs','pre')) and
        reports['contact_energy_relative']<2e-6 and reports['contact_gradient']<2e-3 and reports['contact_depth']<2e-5 and
        reports['contact_counts_equal'] and reports['rotation']<2e-3 and reports['solve_offset_mm']<2e-4 and
        reports['solve_residual_mm']<2e-4 and all(reports[key]<2e-4 for key in ('mlp_forward','objective','mlp_gradients','adamw')))
    report=dict(scope='CPU algebra only; visual quality and GPU validation deferred',passed=passed,errors=reports,
                native_library_sha256=sha256(ROOT/'cpu/vhuman/libvhuman_training.so'))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return passed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();sys.exit(0 if verify(args.output) else 1)
