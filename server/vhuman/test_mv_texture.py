"""Multiview texture conditioning frames, projection visibility and fusion."""
import unittest
import numpy as np

from .reconstruction import mv_conditioning as cond
from .reconstruction.mv_texture import fuse


def sphere(n=24):
    u,v=np.meshgrid(np.linspace(0,np.pi,n),np.linspace(0,2*np.pi,2*n,endpoint=False),indexing='ij')
    p=np.stack((np.sin(u)*np.cos(v),np.cos(u),np.sin(u)*np.sin(v)),-1).reshape(-1,3)*.1
    tri=[]
    for i in range(n-1):
        for j in range(2*n):
            a,b=i*2*n+j,i*2*n+(j+1)%(2*n);c,d=a+2*n,b+2*n
            tri+=[(a,c,b),(b,c,d)]
    tri=np.array(tri,np.int32)
    uv=np.random.default_rng(0).random((len(tri),3,2)).astype(np.float32)
    return dict(captured=p[None].astype(np.float32),triangles=tri,triangle_uvs=uv)


class MultiviewTextureTests(unittest.TestCase):
    def test_frame_maps_gnm_front_to_mvadapter_front_camera(self):
        frame=cond.Frame(np.zeros(3),1.)
        front=cond.ortho_camera(0,-90,64)
        # GNM faces +Z; the MV-Adapter front camera must look at it head-on.
        self.assertAlmostEqual(float(frame.direction([0,0,1])@front.rotation[2]),1.,6)
        top=cond.ortho_camera(89.99,90,64)
        self.assertGreater(float(frame.direction([0,1,0])@top.rotation[2]),.999)

    def test_conditions_are_deterministic_and_framed(self):
        g=sphere();atlas=np.full((16,16,3),.3);known=np.zeros((16,16))
        _,a=cond.render_conditions(g,atlas,known,48);_,b=cond.render_conditions(g,atlas,known,48)
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x['position'],y['position']);np.testing.assert_array_equal(x['rgb'],y['rgb'])
        self.assertEqual([v['name'] for v in a],['front','right','back','left','top','bottom'])
        self.assertTrue(all(v['valid'][24,24] and not v['valid'][0,0] for v in a))

    def test_projection_sees_only_front_hemisphere(self):
        g=sphere(32);p=g['captured'][0].astype(float);n=p/np.linalg.norm(p,axis=1,keepdims=True)
        frame,views=cond.render_conditions(g,np.full((8,8,3),.5),np.zeros((8,8)),96)
        image=np.full((96,96,3),200,np.uint8)
        _,w=cond.project_views(frame,views[:1],[image],p,n)
        self.assertTrue((w[0][p[:,2]<-.02]==0).all())
        self.assertTrue((w[0][p[:,2]>.08]>0).mean()>.9)

    def test_fuse_reports_disagreement(self):
        colors=np.array([[[.2,.2,.2],[.2,.2,.2]],[[.2,.2,.2],[.6,.6,.6]]])
        mean,support,spread,count=fuse(colors,np.ones((2,2)))
        np.testing.assert_allclose(mean[1],.4);self.assertEqual(spread[0],0);self.assertGreater(spread[1],.3)
        np.testing.assert_array_equal(count,[2,2])


    def test_delight_removes_directional_shading_but_keeps_detail(self):
        from .reconstruction.mv_delight import delight
        from .reconstruction.reference import srgb_to_linear,linear_to_srgb
        r=64;yy,xx=np.mgrid[:r,:r];x=(xx+.5)/r*2-1;y=1-(yy+.5)/r*2;z=np.sqrt(np.clip(1-x*x-y*y,0,1))
        valid=x*x+y*y<.9;n=np.stack((x,y,z),-1)
        albedo=.4+.04*((xx//4+yy//4)%2)  # checker detail
        lit=albedo*(.35+.65*np.clip(.7*x+.7*z,0,1))[...,]
        img=np.uint8(np.clip(linear_to_srgb(np.repeat(lit[...,None],3,2))*255+.5,0,255))
        out,_=delight(img,n/2+.5,valid,blob_strength=0)
        before=srgb_to_linear(img[valid][:,0]/255);after=srgb_to_linear(out[valid][:,0]/255)
        self.assertLess(np.std(np.log(after)),.5*np.std(np.log(before)))
        # checker contrast survives
        a=srgb_to_linear(out[...,0]/255);self.assertGreater(abs(a[32,30]-a[32,34]),.01)

if __name__=='__main__':unittest.main()
