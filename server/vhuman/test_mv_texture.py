"""Multiview texture conditioning frames, projection visibility and fusion."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image

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

    def test_chroma_delight_removes_hue_blotch_but_keeps_detail(self):
        from .reconstruction.mv_delight import delight
        from .reconstruction.reference import srgb_to_linear,linear_to_srgb
        r=96;yy,xx=np.mgrid[:r,:r];valid=np.ones((r,r),bool);n=np.zeros((r,r,3));n[...,2]=1
        base=np.stack([np.full((r,r),.40),np.full((r,r),.30),np.full((r,r),.22)],-1)
        blob=np.exp(-((xx-30)**2+(yy-40)**2)/(2*12.**2))
        lin=base*np.stack([1+.5*blob,1-.1*blob,1-.1*blob],-1)*(1+.08*((xx//3+yy//3)%2))[...,None]
        img=np.uint8(np.clip(linear_to_srgb(lin)*255+.5,0,255))
        hue=lambda im:(lambda l:np.log(l[...,0]/np.maximum(l[...,1],1e-4)))(srgb_to_linear(im/255))
        out,_=delight(img,n/2+.5,valid,blob_strength=1.,chroma=True)
        self.assertLess(abs(hue(out)[40,30]-hue(out)[40,85]),.5*abs(hue(img)[40,30]-hue(img)[40,85]))
        o=srgb_to_linear(out[...,1]/255);self.assertGreater(abs(o[60,60]-o[60,63]),.005)   # checker survives

class TextureContinuationTests(unittest.TestCase):
    def test_nvidia_auto_never_selects_hip_and_paths_are_explicit(self):
        from .reconstruction import qwen_edit_backend as edit
        with patch.dict('sys.modules',{'torch':SimpleNamespace(version=SimpleNamespace(hip=None))}), \
             patch.object(edit,'Editor') as gguf,patch.object(edit,'NativeEditor') as native:
            edit.make_editor('auto','/models/edit',offload_blocks=1)
            gguf.assert_called_once_with(Path('/models/edit'),offload_blocks=1)
            native.assert_not_called()
            with self.assertRaisesRegex(ValueError,'ROCm'):
                edit.make_editor('native','/models/edit')
            with self.assertRaisesRegex(ValueError,'unknown'):
                edit.make_editor('typo','/models/edit')

    def test_selected_views_keep_inputs_and_record_actual_recipe(self):
        from .reconstruction import mv_qwen as q
        calls=[]
        class Editor:
            generator='test';metadata={'backend':'test'};last_metrics={'peak_allocated_bytes':123}
            def __call__(self,images,prompt,**kwargs):
                calls.append((images,prompt,kwargs))
                return np.full((1024,1024,3),200,np.uint8),1.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(16,16),(120,90,60)).save(root/'portrait.png')
            rgb=np.full((16,16,3),100,np.uint8);known=np.zeros((16,16));known[:8]=1
            views=[dict(name=n,rgb=rgb.copy(),valid=np.ones((16,16),bool),known=known) for n in ('right','left')]
            result,info=q.selected(root,views,root/'run',Editor(),portrait_mode='raw',prompt_recipe='original')
            for image in result:np.testing.assert_array_equal(image[:8],rgb[:8])
            np.testing.assert_array_equal(calls[0][0][-1],calls[1][0][-1])
            self.assertEqual([c[2]['seed'] for c in calls],[318,319])
            self.assertEqual(info['prompt'],q.ORIGINAL_EDIT_PROMPT)
            self.assertFalse(info['atlas_feedback']);self.assertFalse(info['chained'])
            self.assertEqual(info['view_runs']['right']['metrics']['peak_allocated_bytes'],123)
            self.assertEqual(len(info['reference_sha256']),64)

    def test_prompt_cache_is_bounded_and_distinguishes_images_and_text(self):
        from .reconstruction.qwen_edit_backend import Editor
        from unittest.mock import Mock
        editor=Editor.__new__(Editor);editor.prompt_cache={}
        editor.torch=SimpleNamespace(device=lambda value:value)
        editor.pipe=SimpleNamespace(encode_prompt=Mock(side_effect=lambda **kw:object()))
        images=[Image.new('RGB',(2,2),'red')]
        first=editor._encode_prompt('one',images)
        self.assertIs(editor._encode_prompt('one',images),first)
        self.assertIsNot(editor._encode_prompt('two',images),first)
        images[0].putpixel((0,0),(0,0,0))
        self.assertIsNot(editor._encode_prompt('one',images),first)
        for i in range(8):editor._encode_prompt(str(i),images)
        self.assertEqual(len(editor.prompt_cache),4)
        self.assertEqual(editor.pipe.encode_prompt.call_count,11)

    def test_requested_matte_does_not_silently_fall_back(self):
        from .reconstruction.mv_qwen import matted_portrait
        with tempfile.TemporaryDirectory() as tmp:
            Image.new('RGB',(16,16)).save(Path(tmp)/'portrait.png')
            with patch('server.vhuman.reconstruction.mv_qwen._foreground_alpha',side_effect=RuntimeError('unavailable')):
                with self.assertRaisesRegex(RuntimeError,'matte failed'):
                    matted_portrait(tmp,strict=True)

    def test_compose_requires_complete_verified_sources(self):
        from .reconstruction import mv_texture as mv
        from .reconstruction.observations import sha256
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);source=work/'edit';(source/'front').mkdir(parents=True)
            image=source/'front/edited.png';Image.new('RGB',(16,16)).save(image)
            info=dict(generator='test',license='apache-2.0',view_runs={'front':{'edited_sha256':sha256(image)}})
            (source/'generation.json').write_text(json.dumps(info))
            record=dict(resolution=16,views=[dict(name='front'),dict(name='back')],geometry_sha256='geometry',basecolor_sha256='base')
            with patch.object(mv,'check',return_value=record):
                with self.assertRaisesRegex(ValueError,'every view'):mv.compose(work,'front=edit:raw')
                self.assertFalse((work/'hybrid').exists())
                record['views']=[dict(name='front')]
                combined=mv.compose(work,'front=edit:raw')
                self.assertEqual(combined['sources']['front']['sha256'],sha256(image))
                Image.new('RGB',(16,16),'red').save(image)
                with self.assertRaisesRegex(ValueError,'checksum'):mv.compose(work,'front=edit:raw','hybrid2')
                self.assertFalse((work/'hybrid2').exists())

    def test_bake_preserves_photographed_texels_and_geometry(self):
        from .reconstruction import mv_texture as mv
        from .reconstruction.observations import sha256
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);candidate=work/'candidate';candidate.mkdir();source=work/'generated';source.mkdir()
            geometry=sphere(8);np.savez(candidate/'geometry.npz',**geometry)
            base=np.full((16,16,3),(130,100,80),np.uint8);observed=np.zeros((16,16),np.uint8);observed[:8]=255
            Image.fromarray(base).save(candidate/'skin_basecolor.png');Image.fromarray(observed).save(candidate/'skin_coverage.png')
            Image.new('RGB',(16,16)).save(candidate/'portrait.png')
            manifest=dict(format='vhuman.reconstruction.v1',face_model='gnm_v3',material={},
                          geometry_sha256=sha256(candidate/'geometry.npz'),portrait_sha256=sha256(candidate/'portrait.png'))
            (candidate/'manifest.json').write_text(json.dumps(manifest))
            frame,views=cond.render_conditions(geometry,base/255,observed/255,48)
            for v in views:Image.new('RGB',(48,48),(170,120,90)).save(source/f"view_{v['name']}.png")
            info=dict(generator='test',license='test',views={p.name:sha256(p) for p in source.glob('*.png')})
            (source/'generation.json').write_text(json.dumps(info))
            record=dict(candidate=str(candidate),resolution=48,geometry_sha256=manifest['geometry_sha256'])
            with patch.object(mv,'check',return_value=record),patch.object(mv,'conditions',return_value=(frame,views)):
                report=mv.bake(work,'generated',work/'baked',exclude_clothing=False,delight=False)
            result=np.asarray(Image.open(work/'baked/skin_basecolor.png'))
            np.testing.assert_array_equal(result[observed>0],base[observed>0])
            self.assertEqual(report['photographed_texels_changed'],0)
            self.assertEqual(sha256(work/'baked/geometry.npz'),manifest['geometry_sha256'])


if __name__=='__main__':unittest.main()
