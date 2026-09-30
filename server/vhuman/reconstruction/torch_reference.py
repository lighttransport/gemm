"""Differentiable BRDF reference, independent of optional rasterization backend."""


def ggx(albedo, normal, view, light, roughness, f0):
    import torch
    def unit(x):
        return x / torch.linalg.vector_norm(x,dim=-1,keepdim=True).clamp_min(1e-12)
    n,v,l = unit(normal),unit(view),unit(light)
    h = unit(v+l)
    nv=(n*v).sum(-1).clamp_min(1e-5)
    nl=(n*l).sum(-1).clamp_min(0)
    nh=(n*h).sum(-1).clamp_min(0)
    vh=(v*h).sum(-1).clamp_min(0)
    a2=roughness.clamp_min(.04)**4
    d=a2/(torch.pi*(nh*nh*(a2-1)+1)**2)
    def smith(x):
        return 2*x/(x+torch.sqrt(a2+(1-a2)*x*x)).clamp_min(1e-9)
    fresnel=f0+(1-f0)*(1-vh)**5
    spec=d*smith(nv)*smith(nl)*fresnel/(4*nv*nl).clamp_min(1e-8)
    diffuse=albedo/torch.pi*(1-fresnel)[...,None]*nl[...,None]
    return diffuse,spec[...,None]*nl[...,None]


def backend_probe():
    """Do not downgrade torch or silently build an untested CUDA extension."""
    import torch
    result=dict(torch=torch.__version__,cuda=torch.cuda.is_available(),reference='numpy',pytorch3d=False)
    try:
        import pytorch3d
        from pytorch3d.renderer import MeshRasterizer
        result.update(pytorch3d=True,pytorch3d_version=getattr(pytorch3d,'__version__','unknown'))
    except (ImportError,OSError) as exc:
        result['reason']=str(exc)
    return result
