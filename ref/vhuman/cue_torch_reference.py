"""Optional Torch CueNet v3 oracle; never imported by native training."""
def network(side=64):
    import torch
    from torch import nn
    class CueNet(nn.Module):
        def __init__(self):
            super().__init__()
            prior=torch.zeros((1,3,side,side));prior[:,2]=1
            self.register_buffer('prior',prior)
            self.encoder=nn.Sequential(nn.Conv2d(6,16,5,padding=2),nn.SiLU(),
                nn.Conv2d(16,24,3,stride=2,padding=1),nn.SiLU(),
                nn.Conv2d(24,32,3,stride=2,padding=1),nn.SiLU())
            self.decoder=nn.Sequential(nn.Conv2d(32,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,4,1))
        def forward(self,image,geometry_prior=None):
            prior=nn.functional.interpolate(self.prior if geometry_prior is None else geometry_prior,size=image.shape[-2:],mode='bilinear',align_corners=False).expand(len(image),-1,-1,-1)
            raw=self.decoder(self.encoder(torch.cat((image,prior),dim=1)))
            raw=nn.functional.interpolate(raw,size=image.shape[-2:],mode='bilinear',align_corners=False)
            normals=nn.functional.normalize(prior+torch.tanh(raw[:,:3])*.15,dim=1,eps=1e-6)
            return normals,raw[:,3:4]
    return CueNet()
