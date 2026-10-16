"""Session isolation, HTTP handoff and synthetic camera/texture regression tests."""
import io
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
import zipfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from . import gpu, portrait
from .service import EyeService, ROOT, ServiceError
from .reconstruction import fitting, materials, multiview
from .reconstruction.reference import Camera

SCRATCH = ROOT/'tmp/test-portrait-workspace'


class SessionTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.service = EyeService(Path(self.temp.name))
        self.cancel = threading.Event()
        self.progress = lambda *_: None
        self.execution = gpu.execution('cpu')
        self.execution.__enter__()

    def tearDown(self):
        self.execution.__exit__(None, None, None)
        self.temp.cleanup()

    def generate(self, **kwargs):
        return portrait.generate_job(self.service, dict(subject='test person', **kwargs),
                                     self.progress, self.cancel, mock=True)['session']

    def approve(self, doc, ids=None):
        return portrait.review(self.service, doc['id'], dict(revision=doc['active_revision'],
                               portrait_reviewed=True, view_ids=ids or []))

    def test_review_revision_and_both_view_methods(self):
        doc = self.generate()
        req = dict(session_id=doc['id'], revision=doc['active_revision'])
        with self.assertRaisesRegex(ServiceError, 'approve'):
            portrait.views_job(self.service, req, self.progress, self.cancel, mock=True)
        doc = self.approve(doc)
        for method, size in [('individual', 1024), ('sheet', 512)]:
            result = portrait.views_job(self.service, dict(req, method=method), self.progress, self.cancel, mock=True)
            run = result['session']['view_runs'][-1]
            self.assertEqual([v['size'] for v in run['views']], [[size, size]]*4)
            self.assertEqual(run['reference_sha256'], doc['portraits'][0]['sha256'])
        new = self.generate(session_id=doc['id'], seed=12)
        self.assertFalse(new['selection']['portrait_reviewed'])
        self.assertTrue(all(r['stale'] for r in new['view_runs']))
        with self.assertRaisesRegex(ServiceError, 'revision changed'):
            portrait.snapshot(self.service, dict(req, view_ids=[]))

    def test_selected_snapshot_export_and_path_rejection(self):
        doc = self.approve(self.generate())
        req = dict(session_id=doc['id'], revision=doc['active_revision'])
        doc = portrait.views_job(self.service, req, self.progress, self.cancel, mock=True)['session']
        views = doc['view_runs'][0]['views']
        with self.assertRaisesRegex(ServiceError, 'review selected'):
            portrait.snapshot(self.service, dict(req, view_ids=[views[0]['id']]))
        self.approve(doc, [views[0]['id'], views[1]['id']])
        from .reconstruction.upload import server_request
        request = server_request(self.service, dict(req, view_ids=[views[0]['id']]), direct=True)
        data = json.loads(Path(request['generated_input']).read_text())
        self.assertTrue(data['synthetic'])
        self.assertEqual(len(data['views']), 2)
        self.assertEqual(data['views'][1]['generation']['method'], 'individual')
        self.assertEqual(data['views'][1]['generation']['reference_sha256'], data['views'][0]['sha256'])
        with zipfile.ZipFile(io.BytesIO(portrait.bundle(self.service, doc['id']))) as archive:
            self.assertEqual(len([n for n in archive.namelist() if n.endswith('.png')]), 3)
            exported = json.loads(archive.read('session.json'))
            self.assertTrue(all(v['file'] in archive.namelist() for r in exported['view_runs'] for v in r['views']))
        for field in ('portrait', 'observations', 'generated_input'):
            with self.assertRaises(ServiceError):
                server_request(self.service, dict(req, **{field: '/etc/passwd'}), direct=True)
        for path in ('../session.json', '/etc/passwd', views[0]['file']+'/../session.json'):
            with self.assertRaises(ServiceError):
                portrait.artifact(self.service, doc['id'], path)
        # Published input hashes, not merely the filenames, govern reconstruction.
        portrait.artifact(self.service, doc['id'], views[0]['file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ServiceError, 'checksum'):
            portrait.snapshot(self.service, dict(req, view_ids=[views[0]['id']]))

    def test_cancel_does_not_publish_revision(self):
        self.cancel.set()
        with self.assertRaises(gpu.Cancelled):
            self.generate()
        self.assertEqual(portrait.listing(self.service), [])
        self.assertFalse(list(self.service.work.rglob('*.partial')))

    def test_low8_rejects_competing_gpu_work_before_loading_model(self):
        with patch.object(portrait.qwen, 'availability', return_value={'available': True}), \
             patch.object(gpu, 'gpu_status', return_value=dict(free_mib=7600)), \
             patch.object(portrait.qwen, 'make_backend') as backend:
            with self.assertRaisesRegex(gpu.GpuBusy, '7680 MiB needed'):
                portrait.generate_job(self.service, dict(subject='test person', qwen_preset='low8'),
                                      self.progress, self.cancel, python=sys.executable)
            backend.assert_not_called()

    def test_invalid_angles_and_duplicate_selected_angle(self):
        doc = self.approve(self.generate())
        req = dict(session_id=doc['id'], revision=doc['active_revision'])
        with self.assertRaises(ServiceError):
            portrait.views_job(self.service, dict(req, angles=[0]), self.progress, self.cancel, mock=True)
        with self.assertRaises(ServiceError):
            portrait.views_job(self.service, dict(req, method='sheet', angles=[45]), self.progress, self.cancel, mock=True)
        for _ in range(2):
            doc = portrait.views_job(self.service, dict(req, angles=[45]), self.progress, self.cancel, mock=True)['session']
        with self.assertRaisesRegex(ServiceError, 'one view per angle'):
            self.approve(doc, [r['views'][0]['id'] for r in doc['view_runs']])


def sphere():
    from scipy.spatial import ConvexHull
    i = np.arange(320)
    y = 1-2*(i+.5)/len(i)
    phi = i*np.pi*(3-np.sqrt(5))
    p = np.column_stack((np.sqrt(1-y*y)*np.cos(phi)*.07, y*.09, np.sqrt(1-y*y)*np.sin(phi)*.08))
    tri = ConvexHull(p).simplices.copy()
    a, b, c = p[tri].transpose(1, 0, 2)
    flip = np.sum(np.cross(b-a, c-a)*(a+b+c), axis=1) < 0
    tri[flip] = tri[flip][:, ::-1]
    return p, tri


class CameraTests(unittest.TestCase):
    def test_camera_recovery_outliers_visibility_and_wrong_side(self):
        from scipy.spatial.transform import Rotation
        p, tri = sphere()
        focal = 512/np.tan(np.deg2rad(10))
        front = Camera(focal, 512, 512, np.array([0., 0., 1.]), np.eye(3))
        orbit = Rotation.from_euler('y', 48, degrees=True).as_matrix()
        actual = Camera(focal, 512, 512, orbit@front.origin, orbit.T)
        xy, _ = actual.project(p)
        xy[::7] += [120, -80]
        def view():
            return dict(size=[1024, 1024], anchors={f'mp_{i:03d}': dict(vertex=i, xy=q.tolist(), weight=1.) for i, q in enumerate(xy)})
        v = view()
        report = multiview.estimate_camera(p, tri, v, front, 45)
        self.assertAlmostEqual(report['estimated_yaw_deg'], 48, delta=1.)
        self.assertLess(report['median_error_1024px'], 1.)
        self.assertLess(len(v['anchors']), len(xy))
        self.assertTrue(v['synthetic'])
        with self.assertRaises(multiview.MultiviewRejected):
            multiview.estimate_camera(p, tri, view(), front, -45)

    def test_front_consistency_rejects_regression(self):
        p, _ = sphere()
        camera = Camera(1000, 512, 512, np.array([0., 0., 1.]), np.eye(3))
        xy, _ = camera.project(p)
        view = dict(size=[1024, 1024], anchors={f'mp_{i:03d}': dict(vertex=i, xy=q.tolist()) for i, q in enumerate(xy)})
        report = multiview.check_front(p, camera, view, dict(baseline_error_1024px=0.))
        self.assertTrue(report['accepted'])
        with self.assertRaises(multiview.MultiviewRejected):
            multiview.check_front(p+[.01, 0, 0], camera, view, dict(baseline_error_1024px=0.))

    def test_synthetic_views_do_not_enable_170_modes(self):
        p, _ = sphere()
        camera = Camera(1000, 512, 512, np.array([0., 0., 1.]), np.eye(3))
        xy, _ = camera.project(p)
        anchors = {f'mp_{i:03d}': dict(vertex=i, xy=q.tolist()) for i, q in enumerate(xy)}
        views = [dict(camera=camera.as_dict(), size=[1024, 1024], anchors=anchors, synthetic=True) for _ in range(3)]
        calls = []
        def fake(source, initial, train, max_modes, **kwargs):
            calls.append(max_modes)
            return p.copy(), np.array([p+[.01/max_modes, 0, 0]]*3), [camera]*3, {}
        with patch.object(fitting, 'fit', side_effect=fake):
            fitting.fit_staged(None, p, views, seed={})
        self.assertEqual(calls, [32, 64])

    def test_front_pose_is_fixed_while_side_pose_can_move(self):
        p = np.array([[-.03, -.03, .01], [.03, -.03, .02], [.03, .03, .01], [-.03, .03, .03]])
        tri = np.array([[0, 1, 2], [0, 2, 3]])
        source = SimpleNamespace(triangles=tri, identity_basis=None, expression_basis=None, eye_centers=None)
        cam = Camera(1000, 256, 256, np.array([0., 0., 1.]), np.eye(3))
        views = []
        for offset in (0., .004):
            xy, _ = cam.project(p+[offset, 0, 0])
            views.append(dict(camera=cam.as_dict(), size=[512, 512], anchors={str(i): dict(vertex=i, xy=q.tolist()) for i, q in enumerate(xy)}))
        _, _, cams, report = fitting.fit(source, p, views, fixed_pose_views=(0,), iterations=50)
        np.testing.assert_allclose(cams[0].origin, cam.origin)
        self.assertGreater(np.linalg.norm(cams[1].origin-cam.origin), .001)
        self.assertEqual(report['fixed_pose_views'], [0])


class TextureTests(unittest.TestCase):
    def test_side_fill_preserves_front_bytes_and_support(self):
        p, tri = sphere()
        # UV coordinates shared with the model's spherical parameterization.
        uv = np.column_stack((np.arctan2(p[:, 2], p[:, 0])/(2*np.pi)+.5, p[:, 1]/.18+.5))[tri]
        from scipy.spatial.transform import Rotation
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root = Path(temp)
            views, cams = [], []
            for i, (yaw, colour) in enumerate(((0, (180, 120, 90, 255)), (75, (140, 100, 80, 255)))):
                path = root/f'{i}.png'
                Image.new('RGBA', (128, 128), colour).save(path)
                orbit = Rotation.from_euler('y', yaw, degrees=True).as_matrix()
                cams.append(Camera(500, 64, 64, orbit@np.array([0., 0., 1.]), orbit.T))
                views.append(dict(size=[128, 128], image_path=str(path), synthetic=True))
            materials.bake_portrait(p, tri, uv, views[:1], cams[:1], root/'front', res=64)
            report = materials.bake_portrait(p, tri, uv, views, cams, root/'joint', res=64, protect_front=True)
            a = np.array(Image.open(root/'front/skin_basecolor.png'))
            b = np.array(Image.open(root/'joint/skin_basecolor.png'))
            mask = np.array(Image.open(root/'joint/skin_front_protected.png')) > 0
            generated = np.array(Image.open(root/'joint/skin_generated_support.png')) > 0
            np.testing.assert_array_equal(a[mask], b[mask])
            self.assertTrue(mask.any() and generated.any())
            self.assertFalse((mask & generated).any())
            self.assertTrue(report['synthetic_inputs']['front_preserved'])


class MultiviewIsolationTests(unittest.TestCase):
    def test_joint_rejection_keeps_baseline_and_records_input_provenance(self):
        import copy
        from .reconstruction import pipeline, observations
        SCRATCH.mkdir(parents=True, exist_ok=True)
        p, tri = sphere()
        camera = Camera(1000, 512, 512, np.array([0., 0., 1.]), np.eye(3))
        xy, _ = camera.project(p)
        anchors = {f'mp_{i:03d}': dict(vertex=i, xy=q.tolist(), weight=.8) for i, q in enumerate(xy)}
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root = Path(temp)
            inputs = root/'inputs'; inputs.mkdir()
            for name in ('front.png', 'side.png'):
                Image.new('RGBA', (1024, 1024), (170, 120, 90, 255)).save(inputs/name)
            source = dict(format='vhuman.generated_face_inputs.v1', synthetic=True, views=[
                dict(id='front', file='front.png', sha256=observations.sha256(inputs/'front.png'), yaw_deg=0),
                dict(id='side', file='side.png', sha256=observations.sha256(inputs/'side.png'), yaw_deg=45)])
            (inputs/'inputs.json').write_text(json.dumps(source))
            view = dict(image='view_0.png', sha256=observations.sha256(inputs/'front.png'), size=[1024, 1024],
                        anchors=anchors, camera=camera.as_dict())
            def fake_run(folder, run_id, **kwargs):
                if 'fit_seed' in kwargs:
                    self.assertEqual(kwargs['fixed_pose_views'], (0,))
                    self.assertTrue(kwargs['protect_front'])
                    self.assertTrue(kwargs['input_provenance']['synthetic'])
                    raise multiview.MultiviewRejected('deliberate frontal regression')
                out = folder/'reconstruction'/run_id; out.mkdir(parents=True)
                (out/'view_0.png').write_bytes((inputs/'front.png').read_bytes())
                np.savez(out/'geometry.npz', captured=p[None], triangles=tri)
                (out/'observations.json').write_text(json.dumps(dict(format=observations.FORMAT, views=[view])))
                return dict(geometry=dict(fitted_cameras=[camera.as_dict()], expression_coefficients=[[]],
                                          expression_modes=[], identity_coefficients=[]))
            def camera_fit(vertices, triangles, side, front_camera, yaw):
                side.update(camera=camera.as_dict(), camera_kind='estimated', synthetic=True)
                return dict(estimated_yaw_deg=yaw, inliers=32)
            with patch.object(pipeline, 'run', side_effect=fake_run), \
                 patch.object(observations, 'observe', return_value=dict(views=[copy.deepcopy(view)])), \
                 patch.object(multiview, 'estimate_camera', side_effect=camera_fit):
                result = multiview.run_generated(root/'head', inputs/'inputs.json', face_model='gnm_v3', run_id='baseline')
            review = result['generated_multiview']
            self.assertEqual(review['selected_run_id'], 'baseline')
            self.assertEqual(review['status'], 'rejected')
            self.assertIn('frontal regression', review['reason'])
            self.assertTrue((root/'head/reconstruction/baseline/geometry.npz').is_file())


class NativeCancelTests(unittest.TestCase):
    def test_cancellation_kills_driver_process_group(self):
        from .qwen import _import_qimg21
        _import_qimg21()
        from qimg21_i23d.native import NativeBackend
        cancel = threading.Event()
        backend = NativeBackend(python=sys.executable, resident=False, cancel=cancel)
        backend.command = lambda *_: [sys.executable, '-c', 'import time; time.sleep(60)']
        timer = threading.Timer(.3, cancel.set)
        timer.start()
        try:
            with self.assertRaises(InterruptedError):
                backend._run(None, [])
        finally:
            timer.cancel()
            backend.close()


class ExportProvenanceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('pxr'), 'optional USD SDK unavailable')
    def test_generated_input_survives_glb_and_usd_roundtrip(self):
        from .rig import gltf, usd
        from .test_rig import _toy_rig
        from .eye.glb import GLB
        from pxr import Sdf
        definition = _toy_rig()
        provenance = dict(format='vhuman.generated_face_inputs.v1', synthetic=True,
                          subject='quoted "identity" with Unicode: \u9854',
                          views=[dict(sha256='a'*64, license='qwen-research')])
        definition['input_provenance'] = provenance
        asset = SimpleNamespace(rig=definition, skeleton=dict(joints=definition['joints']),
                                parts=[], materials={}, info={})
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            folder = Path(temp)
            gltf.write(asset, folder/'rig.glb')
            usd.write(asset, folder)
            self.assertEqual(GLB.load(folder/'rig.glb').doc['asset']['extras']['vhuman_generated_input'], provenance)
            layer = Sdf.Layer.FindOrOpen(str(folder/'rig.usda'))
            self.assertIsNotNone(layer)
            self.assertEqual(json.loads(layer.customLayerData['vhumanGeneratedInput']), provenance)


class HTTPTests(unittest.TestCase):
    def test_http_generation_review_export_and_reconstruction_handoff(self):
        from http.server import ThreadingHTTPServer
        from .app import App, make_handler
        from .reconstruction.upload import server_request
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            app = App(SimpleNamespace(work=temp, mock=True, qwen_python=sys.executable, rig_python=sys.executable,
                                      inference_backend='cpu', device=0, models_root='/mnt/disk1/models'))
            def reconstruct(req, progress, cancel):
                resolved = server_request(app.service, req, direct=True)
                doc = json.loads(Path(resolved['generated_input']).read_text())
                return dict(input_count=len(doc['views']), synthetic=doc['synthetic'])
            app.jobs.handlers['portrait_reconstruct'] = reconstruct
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app, quiet=True))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def request(path, body=None):
                req = Request(base+path, data=json.dumps(body).encode() if body is not None else None,
                              headers={'Content-Type': 'application/json'})
                with urlopen(req, timeout=10) as response:
                    return json.load(response)
            def job(body):
                jid = request('/v1/jobs', body)['id']
                deadline = time.monotonic()+30
                while time.monotonic() < deadline:
                    result = request('/v1/jobs/'+jid)
                    if result['state'] in ('done', 'failed', 'cancelled'):
                        self.assertEqual(result['state'], 'done', result)
                        return result['result']
                    time.sleep(.05)
                self.fail('HTTP job timeout')
            try:
                doc = job(dict(kind='portrait_generate', subject='HTTP test person'))['session']
                sid, rev = doc['id'], doc['active_revision']
                self.assertEqual(request('/v1/portrait-sessions')['sessions'][0]['id'], sid)
                request(f'/v1/portrait-sessions/{sid}/review', dict(revision=rev, portrait_reviewed=True, view_ids=[]))
                for method in ('individual', 'sheet'):
                    doc = job(dict(kind='portrait_views', session_id=sid, revision=rev, method=method))['session']
                ids = [v['id'] for v in doc['view_runs'][-1]['views']]
                request(f'/v1/portrait-sessions/{sid}/review', dict(revision=rev, portrait_reviewed=True, view_ids=ids))
                result = job(dict(kind='portrait_reconstruct', session_id=sid, revision=rev, view_ids=ids))
                self.assertEqual(result['input_count'], 5)
                self.assertTrue(result['synthetic'])
                with urlopen(base+doc['portraits'][0]['url']) as response:
                    self.assertEqual(response.headers['Content-Type'], 'image/png')
                with urlopen(base+doc['bundle_url']) as response:
                    self.assertIn('session.json', zipfile.ZipFile(io.BytesIO(response.read())).namelist())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
