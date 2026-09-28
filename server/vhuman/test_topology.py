"""Connectivity invariants for local reconstruction boundary selection."""
import unittest
import numpy as np

from .head.topology import boundary_loops, projected_winding


class BoundaryTests(unittest.TestCase):
    def test_uv_seams_and_existing_hole(self):
        # Four quads surround a pre-existing square hole. Each triangle has
        # separate vertices, as in a UV atlas; index-only tracing fails here.
        p = np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0],
                      [-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], float)
        t = []
        for i in range(4):
            j = (i+1) % 4
            t.extend([[i,j,i+4],[j,j+4,i+4]])
        t = np.array(t)
        split = p[t.ravel()]
        split_tri = np.arange(len(split)).reshape(-1,3)
        before = split.copy()
        loops, info = boundary_loops(split, split_tri)
        self.assertEqual(sorted(map(len, loops)), [4,4])
        self.assertEqual(info, {"rejected_components": 0, "nonmanifold_edges": 0})
        np.testing.assert_array_equal(split, before)
        # Only one newly clipped vertex is required to identify its complete
        # boundary, including older hole edges (not just new-new edges).
        selected, _ = boundary_loops(split, split_tri, [2])
        self.assertEqual(len(selected), 1)
        self.assertEqual({tuple(x) for x in split[selected[0]]}, {tuple(x) for x in p[4:]})

    def test_nonmanifold_join_is_rejected(self):
        p = np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]], float)
        loops, info = boundary_loops(p, np.array([[0,1,2],[1,0,3],[0,1,4]]))
        self.assertEqual(loops, [])
        self.assertEqual(info['nonmanifold_edges'], 1)
        self.assertEqual(info['rejected_components'], 1)
        split, split_info = boundary_loops(p, np.array([[0,1,2],[1,0,3],[0,1,4]]),
                                           split_vertex_fans=True)
        self.assertEqual(split, [])
        self.assertEqual(split_info, info)

    def test_touching_fans_follow_existing_edges(self):
        # Two disks touch at one vertex. Geometry-only boundary degree is
        # four, but the two triangle fans identify unambiguous pairings.
        p = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                      [-1,0,0],[-1,-1,0],[0,-1,0]], float)
        t = np.array([[0,1,2],[0,2,3],[0,4,5],[0,5,6]])
        self.assertEqual(boundary_loops(p, t)[0], [])
        loops, info = boundary_loops(p, t, split_vertex_fans=True)
        self.assertEqual({frozenset(loop) for loop in loops},
                         {frozenset([0,1,2,3]), frozenset([0,4,5,6])})
        self.assertEqual(info['rejected_components'], 0)
        # A required vertex on one fan must not select the touching fan.
        selected, _ = boundary_loops(p, t, [1], split_vertex_fans=True)
        self.assertEqual(len(selected), 1)
        self.assertEqual(set(selected[0]), {0,1,2,3})
        # UV-seam duplicates and arbitrary triangle order/winding must not
        # alter geometric fan connectivity or move source positions.
        split = p[t[::-1, ::-1].ravel()]
        before = split.copy()
        seams, _ = boundary_loops(split, np.arange(len(split)).reshape(-1,3),
                                  split_vertex_fans=True)
        self.assertEqual(sorted(map(len, seams)), [4,4])
        np.testing.assert_array_equal(split, before)

    def test_winding_selects_enclosing_cut_not_nearest_small_hole(self):
        ring = np.array([[-2,-1],[2,-1],[2,1],[-2,1]], float)
        self.assertAlmostEqual(projected_winding(ring, [0,0]), 1)
        self.assertAlmostEqual(projected_winding(ring[::-1], [0,0]), -1)
        self.assertAlmostEqual(projected_winding(ring*.1+[0,2], [0,0]), 0)
        with self.assertRaises(ValueError):
            projected_winding([[0,0],[1,0],[0,1]], [0,0])
        with self.assertRaises(ValueError):
            projected_winding([[-1,0],[1,0],[0,1]], [0,0])

    def test_rejection_diagnostics_preserve_strict_acceptance(self):
        p = np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]], float)
        t = np.array([[0,1,2],[1,0,3],[0,1,4]])
        loops, info = boundary_loops(p, t, diagnostics=True)
        self.assertEqual(loops, [])
        self.assertEqual(info['rejections'], [
            {'reason': 'nonmanifold_touch', 'vertices': [0, 1]}])
        # Two separate fans have no unambiguous pairing until fan splitting
        # is requested. Diagnostics must not enable that behavior implicitly.
        t = np.array([[0,1,2],[0,3,4]])
        loops, info = boundary_loops(p, t, diagnostics=True)
        self.assertEqual(loops, [])
        self.assertEqual(info['rejections'], [
            {'reason': 'unresolved_vertex_fan', 'vertices': [0]}])
        loops, info = boundary_loops(p, t, split_vertex_fans=True, diagnostics=True)
        self.assertEqual(len(loops), 2)
        self.assertEqual(info['rejections'], [])

    def test_self_touching_walk_is_diagnostic_only(self):
        # A disk has two nonadjacent perimeter vertices at the same position.
        # Their incident fans share no edges. Tracing follows the original
        # perimeter, but welding makes it visit the same vertex twice.
        p = np.array([[0,0,0],[1,0,0],[2,1,0],[1,2,0],
                      [0,0,0],[-1,2,0],[-2,1,0],[-1,0,0]], float)
        t = np.array([[0,1,7],[1,2,7],[2,6,7],
                      [2,3,6],[3,5,6],[3,4,5]])
        loops, info = boundary_loops(p, t, split_vertex_fans=True, diagnostics=True)
        self.assertEqual(loops, [])
        self.assertEqual(info['rejected_components'], 1)
        self.assertEqual(info['nonmanifold_edges'], 0)
        self.assertEqual(info['rejections'][0]['reason'], 'self_touching_walk')
        walk = info['rejections'][0]['vertices']
        self.assertEqual(len(walk), 8)
        self.assertEqual(len(set(walk)), 7)
        self.assertEqual(walk.count(0), 2)
        self.assertEqual(info['rejected_walks'], [walk])

        # A third fan touches the same component. The strict result still
        # rejects the component, but diagnostics must finish tracing it
        # instead of hiding the second walk after the first failure.
        p = np.concatenate([p, [[0,-1,0], [-1,-1,0]]])
        t = np.concatenate([t, [[0,8,9]]])
        loops, info = boundary_loops(p, t, split_vertex_fans=True, diagnostics=True)
        self.assertEqual(loops, [])
        self.assertEqual(sorted(map(len, info['rejected_walks'])), [3, 8])
        self.assertEqual(info['rejected_components'], 1)
        self.assertEqual(boundary_loops(p, t, split_vertex_fans=True)[0], [])

    def test_bad_indices_and_collapsed_weld(self):
        p = np.array([[0,0,0],[1e-9,0,0],[0,1,0]])
        for t in (np.array([[0,1,2]]), np.array([[0,1,9]])):
            with self.assertRaises(ValueError):
                boundary_loops(p, t)


if __name__ == '__main__':
    unittest.main()
