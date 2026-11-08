"""Metric contact-field invariants, independent of Blender and external assets."""
import unittest
import numpy as np
from .reconstruction.periocular import distance_to_segments, build_contact_atlas


class ContactDistanceTests(unittest.TestCase):
    def test_segments_and_endpoints(self):
        np.testing.assert_allclose(distance_to_segments([[.5,1,0],[2,0,0]], [[0,0,0],[1,0,0]]), [1,1])

    def test_rigid_motion_and_duplicate_points(self):
        p=np.array([[.5,.2,0],[1,1,0]])
        line=np.array([[0,0,0],[0,0,0],[1,0,0]])
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(distance_to_segments(p,line),
                                  distance_to_segments(p@rotation+3,line@rotation+3))

    def test_invalid_geometry(self):
        for path in ([[0,0,0]], [[0,0,0],[float('nan'),0,0]]):
            with self.assertRaises(ValueError):
                distance_to_segments([[0,0,0]],path)


class ContactAtlasTests(unittest.TestCase):
    def test_metric_texel_centres_and_cache_invalidation(self):
        import tempfile
        from pathlib import Path
        root = Path(__file__).resolve().parents[2] / 'tmp'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            base = Path(tmp)
            points = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [.01, .01, 0]])
            tri = np.array([[0, 1, 2], [1, 3, 2]])
            uv = points[:, :2] / .01
            lower = np.array([[0, .005, 0], [.01, .005, 0]])
            upper = lower + [0, .003, 0]
            def save():
                np.savez(base / 'scene_assets.npz', skin_positions=points, skin_triangles=tri, skin_uvs=uv[tri],
                         left_tearline_positions=np.repeat(lower[:, None], 8, axis=1).reshape(-1, 3),
                         left_tearline_upper_positions=np.repeat(upper[:, None], 8, axis=1).reshape(-1, 3))
            save()
            path = build_contact_atlas(base, 32)
            with np.load(path) as z:
                expected = np.abs((np.arange(32) + .5) / 32 * .01 - .005)
                np.testing.assert_allclose(z['lower_distance_m'], np.broadcast_to(expected[:, None], (32, 32)), atol=1e-8)
                before = z['lower_distance_m'].copy()
            stamp = path.stat().st_mtime_ns
            build_contact_atlas(base, 32)
            self.assertEqual(path.stat().st_mtime_ns, stamp)
            lower += [0, .001, 0]; save(); build_contact_atlas(base, 32)
            with np.load(path) as z:
                self.assertGreater(float(np.abs(z['lower_distance_m'] - before).max()), .0009)

    def test_mirrored_uv_charts_share_metric_distance(self):
        import tempfile
        from pathlib import Path
        root = Path(__file__).resolve().parents[2] / 'tmp'
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            base = Path(tmp)
            points = np.array([[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]])
            tri = np.array([[0,1,2],[1,3,2]])
            uv = points[:, :2] / .01
            left = uv[tri].copy(); left[..., 0] *= .5
            right = uv[tri].copy(); right[..., 0] = 1 - .5 * right[..., 0]
            lower = np.array([[.003,0,0],[.003,.01,0]])
            upper = lower + [.005,0,0]
            np.savez(base/'scene_assets.npz', skin_positions=points, skin_triangles=np.tile(tri,(2,1)),
                     skin_uvs=np.concatenate((left,right)),
                     left_tearline_positions=np.repeat(lower[:,None],8,axis=1).reshape(-1,3),
                     left_tearline_upper_positions=np.repeat(upper[:,None],8,axis=1).reshape(-1,3))
            with np.load(build_contact_atlas(base, 64)) as atlas:
                for name in atlas.files:
                    np.testing.assert_allclose(atlas[name][:,:32], atlas[name][:,32:][:,::-1], atol=1e-8)


class NativePreservationProfileTests(unittest.TestCase):
    def test_ear_and_face_clusters_with_eye_exclusion(self):
        from .reconstruction.periocular import native_preservation_profile
        cluster=np.array([[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]])
        points=np.concatenate((cluster,cluster+[1,0,0]))
        data=dict(skin_positions=points,skin_native_ids=np.arange(8))
        excluded=np.array([1,1,1,1,0,0,0,0])
        surface=np.array([[[.005,.005,0],[1.005,.005,0],[1.005,.005,0]]])
        result=native_preservation_profile(data,excluded,surface,np.array([[1,1,0]]))
        np.testing.assert_array_equal(result,[[0,1,0]])
