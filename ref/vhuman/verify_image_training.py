"""Optional CPU Torch oracle for native cue CNN and Gaussian fitting math."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def appearance_reference(parameters,model,vertices,controls,view,intrinsics,size,truth,mask):
    import torch as t
    from server.vhuman.realtime.src.avatar.geometry import deform_torch
    width,height=size;p=parameters[:model.n*32].reshape(model.n,32);expr=parameters[model.n*32:].reshape(model.c,8)
    tensor=lambda x:t.tensor(x,dtype=t.float64)
    limits=tensor([1,1,.002]);scales=t.minimum(p[:,4:7].exp().clamp_min(1e-5),limits)
    arrays=dict(triangle=t.arange(model.n),barycentric=tensor(model.bary),normal_offset=.005*p[:,7].tanh(),
                covariance_local=t.diag_embed(scales.square()),opacity=p[:,3].sigmoid(),rgb=p[:,:3].sigmoid(),
                color_basis=p[:,8:].reshape(model.n,8,3),expression_matrix=expr)
    points,cov,opacity,color=deform_torch(tensor(vertices),t.tensor(model.attachments,dtype=t.long),arrays,tensor(controls),'trace-v1')
    view=tensor(view);k=tensor(intrinsics);camera=points@view[:3,:3].T+view[:3,3];cam_cov=view[:3,:3]@cov@view[:3,:3].T
    z=camera[:,2];fx,fy,cx,cy=k[0,0],k[1,1],k[0,2],k[1,2]
    tx=(camera[:,0]/z).clamp(float(-(cx+.15*width)/fx),float((width-cx+.15*width)/fx))
    ty=(camera[:,1]/z).clamp(float(-(cy+.15*height)/fy),float((height-cy+.15*height)/fy))
    zeros=t.zeros_like(z);jac=t.stack((fx/z,zeros,-fx*tx/z,zeros,fy/z,-fy*ty/z),1).reshape(-1,2,3)
    screen_cov=jac@cam_cov@jac.transpose(1,2)+t.eye(2,dtype=t.float64)*.3
    conic=t.linalg.inv(screen_cov);means=t.stack((fx*camera[:,0]/z+cx,fy*camera[:,1]/z+cy),1)
    extent=t.minimum(t.full_like(opacity,3.33),(2*(opacity*255).log()).sqrt())
    radius=t.ceil(extent[:,None]*screen_cov.diagonal(dim1=-2,dim2=-1).sqrt())
    order=t.argsort(z.detach(),stable=True).tolist();projection=[]
    for i in order:
        x,y=means[i].detach().tolist();rx,ry=radius[i].detach().tolist()
        valid=float(z[i].detach())>=.01 and float(z[i].detach())<=1e10 and float(opacity[i].detach())>=1/255 and x+rx>0 and x-rx<width and y+ry>0 and y-ry<height
        if valid:projection.append((i,max(0,int(np.floor((x-rx)/16))),min((width+15)//16,int(np.ceil((x+rx)/16))),
                                    max(0,int(np.floor((y-ry)/16))),min((height+15)//16,int(np.ceil((y+ry)/16)))))
    pixels=[]
    for y in range(height):
        for x in range(width):
            trans=t.ones((),dtype=t.float64);rgb=t.zeros(3,dtype=t.float64)
            for i,x0,x1,y0,y1 in projection:
                if not (x0<=x//16<x1 and y0<=y//16<y1):continue
                delta=means[i]-tensor([x+.5,y+.5]);sigma=.5*(delta@conic[i]@delta)
                a=t.minimum(t.tensor(.99,dtype=t.float64),opacity[i]*(-sigma).exp())
                if float(sigma.detach())<0 or float(a.detach())<1/255:continue
                following=trans*(1-a)
                if float(following.detach())<=1e-4:break
                rgb=rgb+trans*a*color[i];trans=following
            pixels.append(t.cat((rgb,(1-trans)[None])))
    rgba=t.stack(pixels).reshape(height,width,4)
    l1=(rgba[:,:,:3]-tensor(truth)).abs().mean()
    loss=l1+.05*(rgba[:,:,3]-tensor(mask)).abs().mean()+1e-4*p[:,8:].square().mean()
    return rgba,loss,l1


def verify(output):
    import torch
    from ref.vhuman.cue_torch_reference import network
    from server.vhuman.test_native_image_training import cue_fixture,appearance_fixture
    from server.vhuman.realtime.src.avatar.provenance import sha256
    torch.set_num_threads(1)
    model,rgb,prior,truth,mask=cue_fixture();reference=network(9)
    reference.load_state_dict({key:torch.tensor(value) for key,value in model.state_dict().items()})
    normal,logit=reference(torch.tensor(rgb),torch.tensor(prior));m=torch.tensor(mask[:,None]);y=torch.tensor(truth);pr=torch.tensor(prior)
    loss=((1-(normal*y).sum(1,keepdim=True))*m).sum()/m.sum().clamp_min(1)+.15*torch.nn.functional.binary_cross_entropy_with_logits(logit,m)
    loss+=.02*((normal-pr)**2*m).sum()/m.sum().clamp_min(1);loss.backward()
    actual,actual_logits,actual_loss,gradient=model.compute(rgb,prior,truth,mask)
    expected=np.concatenate([p.grad.numpy().ravel() for p in reference.parameters()])
    cue=dict(normals=float(abs(actual-normal.detach().numpy()).max()),logits=float(abs(actual_logits-logit.detach().numpy()).max()),
             loss=abs(actual_loss-float(loss.detach())),gradients=float(abs(gradient-expected).max()))
    default_normal,default_logits=reference(torch.tensor(rgb));native_normal,native_logits=model(rgb)
    cue['default_prior']=max(float(abs(native_normal-default_normal.detach().numpy()).max()),float(abs(native_logits-default_logits.detach().numpy()).max()))
    optimizer=torch.optim.AdamW(reference.parameters(),lr=.003,weight_decay=1e-4);optimizer.step();model.optimizer.step(gradient)
    cue['adamw']=float(abs(model.parameters-np.concatenate([p.detach().numpy().ravel() for p in reference.parameters()])).max())
    # Double precision independent projection/compositing oracle; production stores F32 parameters.
    model,avatar,v,tri,ctrl,view,k,target,mask=appearance_fixture()
    parameters=torch.tensor(model.parameters.astype(np.float64),requires_grad=True)
    rgba,loss,l1=appearance_reference(parameters,model,v,ctrl,view,k,(15,13),target,mask);loss.backward()
    actual,actual_loss,gradient=model.compute(v,ctrl,view,k,(15,13),target,mask)
    appearance=dict(rgba=float(abs(actual-rgba.detach().numpy()).max()),loss=abs(actual_loss[0]-float(loss.detach())),
                    gradients=float(abs(gradient-parameters.grad.numpy()).max()))
    optimizer=torch.optim.Adam([parameters],lr=.01);optimizer.step();model.optimizer.step(gradient)
    appearance['adam']=float(abs(model.parameters-parameters.detach().numpy()).max())
    # Exercise several tiles, off-center projection clamps, saturation and visibility.
    checks=[]
    for size,offset,opacity,mode in [((35,33),-.015,1,'tiles_fov'),((15,13),0,9,'opacity_cap'),
                                   ((15,13),0,-9,'invisible'),((15,13),0,1,'color_cap'),
                                   ((15,13),0,1,'degenerate'),((15,13),0,1,'behind_camera')]:
        model,_,v,_,ctrl,view,k,_,_=appearance_fixture();model.local[:,3]=opacity
        v=v.copy();v[:,0]+=offset
        if mode=='tiles_fov':k[0,0]=500;k[1,1]=490
        if mode=='opacity_cap':
            points=v[model.attachments[0]].astype(np.float64);normal=np.cross(points[1]-points[0],points[2]-points[0]);normal/=np.linalg.norm(normal)
            center=(points*model.bary[0,:,None]).sum(0)+normal*.005*np.tanh(float(model.local[0,7]))
            k[0,2]=size[0]//2+.5-k[0,0]*center[0]/center[2];k[1,2]=size[1]//2+.5-k[1,1]*center[1]/center[2]
        if mode=='color_cap':model.local[:,8:]=2;model.expression[:]=1
        if mode=='degenerate':v[2]=v[0]
        if mode=='behind_camera':v[:,2]=-1
        target=np.full((size[1],size[0],3),.12,np.float32);mask=np.full(size[::-1],.2,np.float32)
        parameters=torch.tensor(model.parameters.astype(np.float64),requires_grad=True)
        expected_rgba,expected_loss,_=appearance_reference(parameters,model,v,ctrl,view,k,size,target,mask)
        expected_loss.backward();actual,actual_loss,g=model.compute(v,ctrl,view,k,size,target,mask)
        checks.append(dict(mode=mode,size=list(size),opacity_raw=opacity,rgba=float(abs(actual-expected_rgba.detach().numpy()).max()),
            loss=abs(actual_loss[0]-float(expected_loss.detach())),gradients=float(abs(g-parameters.grad.numpy()).max())))
    appearance['cases']=checks
    passed=(max(cue[key] for key in ('normals','logits','loss','gradients','default_prior'))<3e-6 and cue['adamw']<2e-5 and
            max(appearance[key] for key in ('rgba','loss','gradients'))<3e-6 and appearance['adam']<2e-5 and
            all(max(row[key] for key in ('rgba','loss','gradients'))<3e-6 for row in checks))
    report=dict(passed=passed,scope='CPU math only; GPU, gsplat kernel parity and visual quality deferred',cue=cue,appearance=appearance,
                library_sha256=sha256(ROOT/'cpu/vhuman/libvhuman_training.so'))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return passed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();sys.exit(0 if verify(args.output) else 1)
