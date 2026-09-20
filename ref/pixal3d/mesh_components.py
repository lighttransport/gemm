"""Deterministic edge-connected triangle-component diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def component_diagnostics(vertices: np.ndarray, faces: np.ndarray,
                          face_areas: np.ndarray | None = None,
                          top: int = 8) -> dict:
    """Summarize face components joined by complete mesh edges.

    The sort and sparse graph avoid Python edge dictionaries, which are
    prohibitively large for Pixal3D's one-million-triangle output.
    """
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must be a non-empty Nx3 array")
    # GLB UV seams duplicate vertices. Weld exact exported positions before
    # building the face graph so seams do not appear as detached components.
    welded_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    welded_faces = inverse[faces]
    edges = np.concatenate((welded_faces[:, (0, 1)],
                            welded_faces[:, (1, 2)],
                            welded_faces[:, (2, 0)]), axis=0)
    edges.sort(axis=1)
    owners = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_owners = owners[order]
    shared = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    rows = sorted_owners[:-1][shared]
    columns = sorted_owners[1:][shared]
    graph = coo_matrix(
        (np.ones(len(rows) * 2, dtype=np.uint8),
         (np.concatenate((rows, columns)), np.concatenate((columns, rows)))),
        shape=(len(faces), len(faces))).tocsr()
    count, labels = connected_components(graph, directed=False)
    face_counts = np.bincount(labels, minlength=count)
    if face_areas is None:
        triangles = vertices[faces]
        face_areas = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    areas = np.bincount(labels, weights=np.asarray(face_areas, dtype=np.float64),
                        minlength=count)
    first_face = np.full(count, len(faces), dtype=np.int64)
    np.minimum.at(first_face, labels, np.arange(len(faces), dtype=np.int64))
    ranked = np.lexsort((first_face, -areas, -face_counts))
    total_area = float(areas.sum())
    items = []
    for rank, label in enumerate(ranked[:top], 1):
        component_faces = faces[labels == label]
        points = vertices[np.unique(component_faces)]
        items.append({
            "rank": rank,
            "faces": int(face_counts[label]),
            "face_fraction": float(face_counts[label] / len(faces)),
            "area": float(areas[label]),
            "area_fraction": float(areas[label] / max(total_area, 1e-30)),
            "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        })
    return {
        "connectivity": "faces-sharing-an-edge-after-exact-position-weld",
        "welded_vertices": int(len(welded_vertices)),
        "count": int(count),
        "largest_face_fraction": items[0]["face_fraction"],
        "largest_area_fraction": items[0]["area_fraction"],
        "top": items,
    }
