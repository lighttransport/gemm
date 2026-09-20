import unittest

import numpy as np

from mesh_components import component_diagnostics


class ComponentDiagnosticsTest(unittest.TestCase):
    def test_edge_connected_components_are_ranked_deterministically(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [3, 0, 0], [4, 0, 0], [3, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.uint32)
        result = component_diagnostics(vertices, faces)
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["largest_face_fraction"], 2 / 3)
        self.assertEqual([item["faces"] for item in result["top"]], [2, 1])
        self.assertEqual(result["top"][0]["bounds"], [[0.0, 0.0, 0.0],
                                                      [1.0, 1.0, 0.0]])

    def test_vertex_only_contact_does_not_join_components(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
            dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.uint32)
        self.assertEqual(component_diagnostics(vertices, faces)["count"], 2)


if __name__ == "__main__":
    unittest.main()
