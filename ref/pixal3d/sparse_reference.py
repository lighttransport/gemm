"""Portable PyTorch submanifold convolution for upstream Pixal3D decoders.

The upstream model, norms, subdivisions and weights remain unchanged. This
backend replaces only the optional compiled sparse convolution with a bounded
gather + torch.nn.functional.linear operation, on CPU, CUDA or ROCm.
"""
import sys
import types
import torch
import torch.nn.functional as F
from upstream_import import prepare

def install():
    prepare()
    name="pixal3d.modules.sparse.conv.conv_none"
    module=types.ModuleType(name)
    def init(self,ci,co,kernel,stride,dilation,padding,bias,indice_key):
        assert kernel==3 and stride==1 and dilation==1
        self.weight=torch.nn.Parameter(torch.empty(co,3,3,3,ci))
        self.bias=torch.nn.Parameter(torch.empty(co)) if bias else None
        self.ci,self.co=ci,co
    def forward(self,x):
        cache=x.get_spatial_cache("pixal3d_reference_neighbors")
        if cache is None:
            coords=x.coords[:,1:].long()
            encode=lambda c:c[:,0]+(c[:,1]<<21)+(c[:,2]<<42)
            keys,order=encode(coords).sort()
            offsets=torch.cartesian_prod(*[torch.arange(-1,2,device=coords.device)]*3)
            query=coords[:,None,:]+offsets[None,:,:]
            qkeys=encode(query.reshape(-1,3))
            idx=torch.searchsorted(keys,qkeys).clamp(max=len(keys)-1)
            valid=(keys[idx]==qkeys)&(query.reshape(-1,3)>=0).all(-1)
            cache=(order[idx].reshape(-1,27),valid.reshape(-1,27))
            x.register_spatial_cache("pixal3d_reference_neighbors",cache)
        neighbors,valid=cache
        out=torch.empty(len(x.feats),self.co,device=x.device,dtype=x.dtype)
        for start in range(0,len(x.feats),256):
            gathered=x.feats[neighbors[start:start+256]]
            gathered=gathered*valid[start:start+256,:,None]
            out[start:start+256]=F.linear(gathered.flatten(1),self.weight.flatten(1),self.bias)
        return x.replace(out)
    module.sparse_conv3d_init=init
    module.sparse_conv3d_forward=forward
    sys.modules[name]=module
