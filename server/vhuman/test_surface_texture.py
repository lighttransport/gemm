"""Surface-space color protection and portable Blender bridge contracts."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from .reconstruction.surface_texture import tone_field


class SurfaceToneTests(unittest.TestCase):
    def plane(self):
        x,y=np.meshgrid(np.arange(16)*.002,np.arange(12)*.002)
        points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
        normals=np.tile([0.,0.,1.],(len(points),1));protected=points[:,0]<.008
        colors=np.full(points.shape,.3);colors[points[:,0]>.016]=.5
        return points,normals,colors,protected

    def test_tone_jump_falls_with_exact_photo_protection(self):
        p,n,c,seen=self.plane();out,report=tone_field(p,n,c,seen)
        np.testing.assert_array_equal(out[seen],c[seen])
        self.assertLess(np.mean(abs(out[~seen]-.3)),np.mean(abs(c[~seen]-.3)))
        self.assertTrue(np.isfinite(out).all());self.assertGreater(report['nodes'],0)
        self.assertLessEqual(np.max(abs(np.log(out/c))),.500001)

    def test_uniform_albedo_is_fixed_point(self):
        p,n,c,seen=self.plane();c[:]=[.25,.2,.15]
        out,_=tone_field(p,n,c,seen)
        np.testing.assert_allclose(out,c,atol=1e-10)

    def test_opposing_sheets_do_not_exchange_color(self):
        p,n,c,seen=self.plane();c[:]=.2
        q=p.copy();q[:,2]+=.0001
        colors=np.concatenate((c,np.full_like(c,.65)))
        out,_=tone_field(np.concatenate((p,q)),np.concatenate((n,-n)),colors,np.concatenate((seen,np.zeros_like(seen))))
        np.testing.assert_allclose(out[len(p):],.65,atol=1e-10)

    def test_invalid_or_unanchored_samples_fail(self):
        p,n,c,seen=self.plane()
        with self.assertRaises(ValueError):tone_field(p,n,c,np.zeros_like(seen))
        for kwargs in ({'strength':0},{'prior':float('nan')},{'spacing':-1}):
            with self.assertRaises(ValueError):tone_field(p,n,c,seen,**kwargs)
        c[0,0]=float('nan')
        with self.assertRaises(ValueError):tone_field(p,n,c,seen)


class EditGuardTests(unittest.TestCase):
    def test_clothing_count_is_limited_to_visible_geometry(self):
        from .reconstruction.edit_guard import clothing_fraction
        from .face_parsing import LABELS
        labels=np.full((4,4),LABELS.index('clothes'));confidence=np.ones((4,4));valid=np.zeros((4,4),bool)
        valid[:2]=True;labels[:2]=LABELS.index('skin')
        self.assertEqual(clothing_fraction(labels,confidence,valid),0)
        labels[0,0]=LABELS.index('clothes')
        self.assertEqual(clothing_fraction(labels,confidence,valid),1/8)
        with self.assertRaises(ValueError):clothing_fraction(labels,confidence,~np.ones((4,4),bool))

    def test_guard_receipt_rejects_changed_or_unapproved_edit(self):
        from .reconstruction.edit_guard import require_approved
        from .reconstruction.observations import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'generation.json').write_text('{}')
            record=dict(geometry_sha256='geom',basecolor_sha256='base')
            receipt=dict(schema='vhuman.edit_guard.v1',generation_sha256=sha256(root/'generation.json'),
                source_geometry_sha256='geom',source_basecolor_sha256='base',views={'right':dict(approved=True,edited_sha256='edit')})
            (root/'quality.json').write_text(json.dumps(receipt))
            require_approved(root,'right','edit',record)
            with self.assertRaises(ValueError):require_approved(root,'right','changed',record)
            with self.assertRaises(ValueError):require_approved(root,'left','edit',record)
            (root/'generation.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'provenance'):require_approved(root,'right','edit',record)


if __name__=='__main__':unittest.main()
