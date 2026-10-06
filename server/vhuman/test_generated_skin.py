"""Synthetic completion protects observations and rejects unstable video evidence."""
import unittest
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from .reconstruction.generated_skin import (band_detail,unseen_weight,triplanar_detail,
    temporal_consistency,render_plate,orbit_camera,apply_detail)
from .reconstruction.multiview_skin import fuse_views,contact_sheet,verify_inputs,detail_quality,crop_camera
from .reconstruction.reference import Camera
from .reconstruction.wrinkle_skin import material_field, review, _verified_material_receipt
from .reconstruction.provenance import validate_candidate
from .mobile.export import export
from .reconstruction.observations import sha256
from PIL import Image


class CompletionProvenanceTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2]/'tmp/vhuman-generated-skin-tests'
        parent.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.work = self.root/'work'
        self.work.mkdir()
        for name, color in (('input.png', 'green'), ('edited.png', 'blue'), ('mask.png', 'white')):
            Image.new('RGB', (32, 32), color).save(self.work/name)
        self.receipt = dict(sha256=sha256(self.work/'edited.png'), request=dict(
            reference_sha256=sha256(self.work/'input.png'),
            init_image_sha256=sha256(self.work/'input.png'),
            mask_sha256=sha256(self.work/'mask.png'), seed=412))
        self.write_receipt(self.receipt)
        self.source = self.root/'source'
        self.completed = self.root/'completed'
        for folder in (self.source, self.completed):
            folder.mkdir()
            (folder/'geometry.npz').write_bytes(b'geometry fixture; must not render invalid reviews')
            Image.new('RGB', (32, 32), 'brown').save(folder/'portrait.png')
            Image.new('RGB', (32, 32), 'brown').save(folder/'skin_basecolor.png')
        self.manifest = dict(format='vhuman.reconstruction.v1', face_model='gnm_v3',
            geometry_sha256=sha256(self.source/'geometry.npz'),
            portrait_sha256=sha256(self.source/'portrait.png'), material={})
        (self.source/'manifest.json').write_text(json.dumps(self.manifest))
        self.report = dict(schema='vhuman.synthetic_skin_completion.v1',
            basecolor_sha256=sha256(self.completed/'skin_basecolor.png'),
            source_basecolor_sha256=sha256(self.source/'skin_basecolor.png'),
            source_geometry_sha256=self.manifest['geometry_sha256'],
            material_edit=self.receipt, license='qwen-research')
        self.manifest['material']['synthetic_completion'] = self.report
        self.write_completion()

    def write_receipt(self, receipt):
        (self.work/'edited.json').write_text(json.dumps(receipt))

    def write_completion(self):
        (self.completed/'manifest.json').write_text(json.dumps(self.manifest))
        (self.completed/'generated_skin.json').write_text(json.dumps(self.report))

    def assert_review_rejected(self):
        with patch('server.vhuman.reconstruction.wrinkle_skin.skin.render_plate') as render:
            with self.assertRaisesRegex(ValueError, 'receipt'):
                review(self.source, self.work, self.completed)
            render.assert_not_called()
        self.assertFalse((self.work/'review.json').exists())

    def test_valid_and_legacy_candidates_still_validate(self):
        validate_candidate(self.source)
        validate_candidate(self.completed)
        self.assertEqual(_verified_material_receipt(
            self.work, expected_receipt=self.report['material_edit']), self.receipt)
        # Earlier completion schemas did not store the optional geometry hash.
        del self.report['source_geometry_sha256']
        self.write_completion()
        validate_candidate(self.completed)

    def test_review_rejects_missing_or_malformed_receipt(self):
        (self.work/'edited.json').unlink()
        self.assert_review_rejected()
        for contents in ('{broken', '{}', 'null'):
            (self.work/'edited.json').write_text(contents)
            self.assert_review_rejected()

    def test_review_rejects_changed_material_images(self):
        for name in ('input.png', 'edited.png', 'mask.png'):
            with self.subTest(name=name):
                path = self.work/name
                original = path.read_bytes()
                Image.new('RGB', (32, 32), 'red').save(path)
                self.assert_review_rejected()
                path.write_bytes(original)

    def test_review_rejects_self_consistent_receipt_from_other_generation(self):
        other = copy.deepcopy(self.receipt)
        Image.new('RGB', (32, 32), 'red').save(self.work/'edited.png')
        other['sha256'] = sha256(self.work/'edited.png')
        other['request']['seed'] += 1
        self.write_receipt(other)
        self.assertEqual(_verified_material_receipt(self.work), other)
        self.assert_review_rejected()

    def test_export_rejects_stale_atlas_before_writing_output(self):
        Image.new('RGB', (32, 32), 'red').save(self.completed/'skin_basecolor.png')
        out = self.root/'export'
        # A nonexistent scene also proves the completion check runs first.
        with self.assertRaisesRegex(ValueError, 'basecolor hash mismatch'):
            export(self.completed, out, scene=self.root/'missing-scene')
        self.assertFalse(out.exists())

    def test_export_rejects_missing_or_inconsistent_completion_report(self):
        report_path = self.completed/'generated_skin.json'
        for contents in (None, '{broken', json.dumps(dict(self.report, license='wrong'))):
            with self.subTest(contents=contents):
                if contents is None:
                    report_path.unlink()
                else:
                    report_path.write_text(contents)
                with self.assertRaisesRegex(ValueError, 'synthetic completion'):
                    export(self.completed, self.root/'export', scene=self.root/'missing-scene')
                self.assertFalse((self.root/'export').exists())

    def test_completion_cannot_claim_another_geometry(self):
        self.report['source_geometry_sha256'] = '0'*64
        self.write_completion()
        with self.assertRaisesRegex(ValueError, 'geometry hash mismatch'):
            validate_candidate(self.completed)


class GeneratedSkinTests(unittest.TestCase):
    def test_wrinkle_material_is_continuous_in_world_space_and_tapers_off_on_scalp(self):
        yy,xx=np.mgrid[:64,:64]
        patch=np.repeat((.3+.07*np.sin(yy*.4+np.sin(xx*.1)))[...,None],3,axis=2)
        points=np.array([[.02,-.03,.01],[.02,-.03,.01],[.02,.08,.01],
                         [.1-1e-8,-.03,.01],[.1+1e-8,-.03,.01]])
        normals=np.tile([0.,0.,1.],(len(points),1))
        values=material_field(points,normals,patch)
        self.assertEqual(values[0],values[1]);self.assertEqual(values[2],0)
        self.assertAlmostEqual(values[3],values[4],places=7)
        self.assertLessEqual(abs(values).max(),.08)
        with self.assertRaises(ValueError):material_field(points,normals,patch,period=0)

    def test_flat_completion_gate_rejects_tiny_changes_and_accepts_resolved_creases(self):
        original=np.full((128,128,3),160,np.uint8);mask=np.ones((128,128),bool)
        noise=np.random.default_rng(7).integers(-2,3,original.shape)
        weak=np.uint8(original.astype(int)+noise)
        self.assertFalse(detail_quality(original,original,mask)['passed'])
        self.assertFalse(detail_quality(original,weak,mask)['passed'])
        yy,xx=np.mgrid[:128,:128]
        creases=sum(22*np.exp(-((yy-row-5*np.sin(xx/23))/2)**2) for row in (25,50,75,100))
        detailed=np.uint8(np.clip(original.astype(float)-creases[...,None],0,255))
        report=detail_quality(original,detailed,mask)
        self.assertTrue(report['passed']);self.assertFalse(report['proves_wrinkle_accuracy'])

    def test_closeup_camera_retains_mesh_projection_calibration(self):
        points=np.array([[-.1,-.1,-.1],[.1,.1,.1],[0.,0.,0.]])
        camera=orbit_camera(points,-85,pitch=10);box=(90,190,410,510)
        before,z=camera.project(points);after,depth=crop_camera(camera,box).project(points)
        np.testing.assert_allclose(after,(before-np.array(box[:2]))*1.6)
        np.testing.assert_array_equal(z,depth)

    def test_wrinkle_budget_retains_visible_contrast_but_protects_observations(self):
        base=np.full((8,8,3),160,np.uint8);valid=np.ones((8,8),bool);observed=np.zeros((8,8),bool);observed[0]=True
        delta=np.full((64,3),-.07)
        subtle=apply_detail(base,valid,delta,observed)
        detailed=apply_detail(base,valid,delta,observed,limit=.08)
        self.assertTrue((detailed[~observed]<subtle[~observed]).all())
        np.testing.assert_array_equal(detailed[observed],base[observed])

    def test_multiview_input_and_review_reject_tampering(self):
        from .mobile.browser import build
        work=Path(__file__).resolve().parents[2]/'tmp/vhuman-generated-skin-tests'
        work.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as directory:
            root=Path(directory);image=root/'reference.png';image.write_bytes(b'original reference')
            record={'inputs':{'reference.png':sha256(image)}}
            verify_inputs(root,record)
            image.write_bytes(b'changed reference')
            with self.assertRaises(ValueError):verify_inputs(root,record)
            with self.assertRaises(ValueError):verify_inputs(root,{'inputs':{'../outside':'wrong'}})
            (root/'review.html').write_text('review')
            review=dict(schema='vhuman.multiview_skin_review.v1',geometry_sha256='geometry',basecolor_sha256='bake',
                        files={'review.html':sha256(root/'review.html')})
            manifest=dict(source_geometry_sha256='geometry',material={'synthetic_completion':{'basecolor_sha256':'other bake'}})
            (root/'review.json').write_text(json.dumps(review))
            with patch('server.vhuman.mobile.browser.validate_package',return_value=manifest):
                with self.assertRaisesRegex(ValueError,'another baked material'):
                    build(root,root/'out',root,skin_review=root)
                manifest['material']['synthetic_completion']['basecolor_sha256']='bake'
                (root/'review.html').write_text('altered review')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    build(root,root/'out',root,skin_review=root)

    def test_elevated_camera_sees_crown_and_roundtrips(self):
        points=np.array([[-.1,-.1,-.1],[.1,.1,.1]])
        for pitch in (0,65,75,90):
            camera=Camera.from_dict(orbit_camera(points,35,pitch=pitch).as_dict())
            xy,z=camera.project(np.array([[0.,0.,0.]]))
            np.testing.assert_allclose(xy,[[256,256]],atol=1e-8)
            self.assertGreater(z[0],0)
            if pitch:self.assertGreater(camera.origin[1],0)

    def test_multiview_rejects_disagreement_and_keeps_corroborated_detail(self):
        colors=np.array([[[.01,0,0],[.02,0,0],[.01,0,0]],
                         [[.011,0,0],[-.02,0,0],[.02,0,0]]])
        weights=np.array([[1.,1.,1.],[1.,1.,0.]])
        result,support,report=fuse_views(colors,weights)
        np.testing.assert_allclose(result[0],[.0105,0,0])
        np.testing.assert_array_equal(result[1],[0,0,0])
        np.testing.assert_allclose(result[2],[.01,0,0])
        np.testing.assert_allclose(support,[2,0,.075])
        self.assertEqual(report['agreeing_overlap_texels'],1)
        self.assertEqual(report['rejected_overlap_texels'],1)
        with self.assertRaises(ValueError):fuse_views(colors,-weights)

    def test_multiview_outlier_does_not_bias_consensus(self):
        colors=np.array([[[.01,.005,0]],[[.011,.004,0]],[[-.025,-.025,.025]]])
        result,support,_=fuse_views(colors,np.ones((3,1)))
        np.testing.assert_allclose(result,[[.0105,.0045,0]])
        np.testing.assert_array_equal(support,[2])
        result,support,_=fuse_views(colors,np.zeros((3,1)))
        np.testing.assert_array_equal(result,np.zeros((1,3)))
        np.testing.assert_array_equal(support,[0])

    def test_contact_sheet_keeps_target_and_guides_in_fixed_quadrants(self):
        colors=((255,0,0),(0,255,0),(0,0,255),(255,255,0))
        sheet=np.asarray(contact_sheet(*(Image.new('RGB',(512,512),c) for c in colors)))
        for xy,color in zip(((256,256),(768,256),(256,768),(768,768)),colors):
            np.testing.assert_array_equal(sheet[xy[1],xy[0]],color)

    def test_final_bake_keeps_photographed_bytes_exactly(self):
        base=np.random.default_rng(12).integers(0,256,(8,8,3),dtype=np.uint8)
        valid=np.ones((8,8),bool);valid[0]=False
        observed=np.zeros((8,8),bool);observed[:,2:4]=True
        result=apply_detail(base,valid,np.full((valid.sum(),3),.02),observed)
        np.testing.assert_array_equal(result[observed],base[observed])
        np.testing.assert_array_equal(result[~valid],base[~valid])
        self.assertTrue(np.any(result[valid&~observed]!=base[valid&~observed]))

    def test_world_space_feather_protects_observed_and_nearby_skin(self):
        points=np.array([[0.,0,0],[.001,0,0],[.003,0,0],[.007,0,0]])
        blend=unseen_weight(points,[True,False,False,False])
        self.assertEqual(blend[0],0);self.assertEqual(blend[-1],1)
        self.assertTrue((np.diff(blend)>0).all())
        with self.assertRaises(ValueError):unseen_weight(points,[False]*4)

    def test_detail_removes_constant_light_and_bounds_contrast(self):
        np.testing.assert_allclose(band_detail(np.full((64,64,3),.6)),0,atol=1e-12)
        image=np.random.default_rng(5).random((64,64,3))
        self.assertLessEqual(abs(band_detail(image)).max(),.025)

    def test_duplicate_uv_vertices_receive_same_spatial_detail(self):
        plate=np.random.default_rng(7).random((64,64,3))
        points=np.array([[.04,.03,-.02],[.04,.03,-.02]])
        normals=np.array([[0.,0,1],[0,0,1]])
        result=triplanar_detail(points,normals,plate)
        np.testing.assert_array_equal(result[0],result[1])
        self.assertLessEqual(abs(result).max(),.012+1e-12)

    def test_i2v_agreement_does_not_accept_photometric_drift(self):
        rng=np.random.default_rng(9)
        image=rng.integers(40,120,(64,64,3),dtype=np.uint8)
        stable,report=temporal_consistency(image,[image]*3)
        _,_,consensus=temporal_consistency(image,[image]*3,return_consensus=True)
        # Dense flow interpolation can round a few static edge pixels by 1 LSB.
        self.assertLessEqual(abs(consensus.astype(int)-image.astype(int)).max(),1)
        self.assertGreater((stable>0).mean(),.9)
        drift,_=temporal_consistency(image,[image+100]*3)
        self.assertLess((drift>0).mean(),.01)
        self.assertFalse(report['proves_observed_accuracy'])
        with self.assertRaises(ValueError):temporal_consistency(image,[image])

    def test_render_mask_excludes_photographed_surface_and_silhouette(self):
        points=np.array([[-.1,-.1,0],[.1,-.1,0],[0,.1,0]])
        geometry=dict(captured=points[None],triangles=np.array([[0,1,2]]),
                      triangle_uvs=np.array([[[0,0],[1,0],[.5,1]]]))
        camera=orbit_camera(points,0,64)
        json.dumps(orbit_camera(points.astype(np.float32),0,64).as_dict())
        _,mask,depth=render_plate(geometry,np.full((32,32,3),.3),np.zeros((32,32)),camera,64)
        self.assertFalse(mask.any());self.assertTrue(np.isfinite(depth).any())
        _,mask,depth=render_plate(geometry,np.full((32,32,3),.3),np.ones((32,32)),camera,64)
        self.assertTrue(mask.any());self.assertFalse(mask[~np.isfinite(depth)].any())


if __name__=='__main__':unittest.main()
