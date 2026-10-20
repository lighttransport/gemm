"""CPU ocular contact fitting: fixed eyes, retained topology, bounded lids."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from .rig import geometry_fit
from .rig.biomechanics_native import LIBRARY

HAS_SCIPY = importlib.util.find_spec('scipy') is not None


def ocular_fixture():
    from scipy.spatial import ConvexHull
    angle = np.arange(32)*2*np.pi/32
    rings = [np.column_stack((np.sin(t)*np.cos(angle), np.sin(t)*np.sin(angle),
                             np.full(32, np.cos(t)))) for t in np.linspace(.1, np.pi-.1, 20)]
    eye = .01*np.concatenate(([[0, 0, 1]], *rings, [[0, 0, -1]]))
    hull = ConvexHull(eye)
    eye_faces = hull.simplices.copy()
    tri = eye[eye_faces]
    flipped = (np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])*tri.mean(1)).sum(1) < 0
    eye_faces[flipped] = eye_faces[flipped, ::-1]
    # A shallow original cup closes in the eyeball. The outer rim stays fixed.
    cup = [np.column_stack((r*np.cos(angle), r*np.sin(angle), np.full(32, .0078)))
           for r in .0102*np.sin(.7)*np.array([1., .75, .5, .25])]
    cup = np.concatenate((*cup, [[0, 0, .0078]]))
    faces = []
    for ring in range(3):
        for i in range(32):
            a, b, c, d = ring*32+i, ring*32+(i+1)%32, (ring+1)*32+i, (ring+1)*32+(i+1)%32
            faces.extend(((a, b, c), (b, d, c)))
    faces.extend((96+i, 96+(i+1)%32, 128) for i in range(32))
    hidden = np.arange(len(cup)) >= 32
    return cup, np.array(faces, np.int32), hidden, ~hidden, eye, eye_faces


@unittest.skipUnless(HAS_SCIPY, 'CPU SciPy fitting dependency')
class GeometryFitTests(unittest.TestCase):
    @unittest.skipUnless(LIBRARY.is_file(), 'native CPU surface BVH')
    def test_crossing_witnesses_match_analytic_edges_and_preserve_input(self):
        vertices=np.array([[-2,-2,0],[2,-2,0],[0,2,0],[0,-1,-1],[0,-1,1],[0,1,1]],float)
        faces=np.array([[0,1,2],[3,4,5]],np.int32);original=vertices.copy()
        with geometry_fit.CrossingCounter(vertices,faces) as counter:
            records=counter.intersections(vertices)
            np.testing.assert_array_equal(records,[[3,4,0,1,2],[3,5,0,1,2]])
            self.assertEqual(counter.count(vertices),len(records))
            np.testing.assert_array_equal(vertices,original)
            with self.assertRaisesRegex(ValueError,'limit exceeded'):
                counter.intersections(vertices,limit=1)
            separate=vertices.copy();separate[3:,2]+=3
            self.assertEqual(counter.intersections(separate).shape,(0,5))
            with self.assertRaisesRegex(ValueError,'query'):
                counter.intersections(np.full_like(vertices,np.nan))

    def test_disk_chart_has_strictly_interior_coordinates(self):
        vertices, faces, hidden, _, _, _ = ocular_fixture()
        ids, loop, _, uv = geometry_fit.disk_chart(vertices, faces, hidden)
        np.testing.assert_array_equal(ids, np.flatnonzero(hidden))
        self.assertEqual(len(loop), 32)
        self.assertLess(np.linalg.norm(uv, axis=1).max(), 1.)
        self.assertLess(np.linalg.norm(uv[-1]), 1e-12)

    def test_disk_chart_rejects_disconnected_boundaries_and_bad_masks(self):
        vertices, faces, hidden, _, _, _ = ocular_fixture()
        duplicate = np.concatenate((vertices, vertices+[1, 0, 0]))
        two = np.concatenate((faces, faces+len(vertices)))
        with self.assertRaisesRegex(ValueError, 'multiple boundary'):
            geometry_fit.disk_chart(duplicate, two, np.tile(hidden, 2))
        hidden[0] = True
        with self.assertRaisesRegex(ValueError, 'labels'):
            geometry_fit.disk_chart(vertices, faces, hidden)

    def test_bounds_and_malformed_geometry_fail_before_native_calls(self):
        vertices, faces, hidden, visible, eye, _ = ocular_fixture()
        with self.assertRaisesRegex(ValueError, 'displacement bounds'):
            geometry_fit.fit_socket(vertices, faces, hidden, visible, eye, hidden_limit=.001)
        with self.assertRaisesRegex(ValueError, 'inputs'):
            geometry_fit.fit_socket(vertices, faces, hidden, visible, eye[:, :2])
        with patch.object(geometry_fit, 'library', side_effect=AssertionError('native call')):
            with self.assertRaisesRegex(ValueError, 'surface geometry'):
                geometry_fit.CrossingCounter(vertices, faces-1000)
        bad = faces.copy(); bad[0, 0] = len(vertices)
        with self.assertRaisesRegex(ValueError, 'surface geometry'):
            geometry_fit.fit_socket(vertices, bad, hidden, visible, eye)
        vertices[visible] *= .9
        with self.assertRaisesRegex(ValueError, 'displacement bounds'):
            geometry_fit.fit_socket(vertices, faces, hidden, visible, eye, visible_limit=.00001)

    def test_tangential_lid_constraint_recovers_a_reversed_triangle(self):
        original = np.array([[-.001, 0, .01], [.001, 0, .01], [0, .001, .01]])
        result = original.copy();result[2, 1] = -.0001
        faces = np.array([[0, 1, 2]], np.int32)
        geometry_fit._preserve_visible_faces(original, result, faces, np.ones(3, bool), np.zeros(3))
        normal = np.cross(original[1]-original[0], original[2]-original[0])
        new = np.cross(result[1]-result[0], result[2]-result[0])
        self.assertGreaterEqual(float(new@normal/(normal@normal)), .149)
        self.assertLess(np.linalg.norm(result-original, axis=1).max(), .0015)

    @unittest.skipUnless(LIBRARY.is_file(), 'native CPU surface BVH')
    def test_fitted_cup_clears_eye_without_changing_topology_or_eye(self):
        vertices, faces, hidden, visible, eye, eye_faces = ocular_fixture()
        before_eye, before_vertices, before_faces = eye.copy(), vertices.copy(), faces.copy()
        fitted, report = geometry_fit.fit_socket(vertices, faces, hidden, visible, eye)
        whole = np.concatenate((vertices, eye))
        combined = np.concatenate((faces, eye_faces+len(vertices)))
        with geometry_fit.CrossingCounter(whole, combined) as counter:
            self.assertGreater(counter.count(whole), 0)
            self.assertEqual(counter.count(np.concatenate((fitted, eye))), 0)
        with geometry_fit.CrossingCounter(vertices, faces) as counter:
            self.assertEqual(counter.count(fitted), 0)
        self.assertLess(fitted[hidden, 2].min(), -.01)
        self.assertLessEqual(report['visible_max_mm'], 1.5)
        self.assertLessEqual(report['hidden_max_mm'], 30.)
        np.testing.assert_array_equal(eye, before_eye)
        np.testing.assert_array_equal(vertices, before_vertices)
        np.testing.assert_array_equal(faces, before_faces)

    @unittest.skipUnless(LIBRARY.is_file(), 'native CPU surface BVH')
    def test_already_clear_sockets_keep_exact_source_geometry(self):
        from types import SimpleNamespace
        cup, faces, hidden, visible, eye, eye_faces = ocular_fixture()
        cup, _ = geometry_fit.fit_socket(cup, faces, hidden, visible, eye)
        nc, ne = len(cup), len(eye)
        extra = np.array([[-.1, 0, 0], [-.09, 0, 0], [-.1, .01, 0]])
        vertices = np.concatenate((cup, cup+[.04, 0, 0], extra, eye, eye+[.04, 0, 0]))
        skin_end = 2*nc+3
        triangles = np.concatenate((faces, faces+nc, [[2*nc, 2*nc+1, 2*nc+2]],
                                    eye_faces+skin_end, eye_faces+skin_end+ne)).astype(np.int32)
        index = np.arange(len(vertices))
        skin = index < skin_end
        left_eye = (index >= skin_end) & (index < skin_end+ne)
        right_eye = index >= skin_end+ne
        groups = dict(skin=skin, eye_exteriors=~skin, left_eye=left_eye, right_eye=right_eye,
                      skin_exterior=np.r_[visible, visible, np.ones(3, bool), np.zeros(2*ne, bool)],
                      eye_sockets=np.r_[hidden, hidden, np.zeros(3+2*ne, bool)],
                      left=(index < nc) | left_eye, right=((index >= nc) & (index < 2*nc)) | right_eye)
        model = SimpleNamespace(data={'triangles':triangles}, group=lambda name:groups[name])
        parts = [('skin', skin, 0), ('eye_left', left_eye, -1), ('eye_right', right_eye, -1)]
        with patch('server.vhuman.rig.anatomy.part_masks', return_value=parts), \
             patch.object(geometry_fit, 'fit_socket', side_effect=AssertionError('unnecessary geometry edit')):
            fitted, report = geometry_fit.fit_sockets(model, vertices)
        np.testing.assert_array_equal(fitted, vertices)
        self.assertFalse(report['applied'])
        self.assertTrue(report['accepted'])
        self.assertEqual(report['after_crossings'], 0)


if __name__ == '__main__':
    unittest.main()
