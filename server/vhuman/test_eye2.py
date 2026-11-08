"""Layered eye (server/vhuman/eye2): geometry consistency, vessel topology, determinism."""
import math

import numpy as np
import pytest

from server.vhuman.eye2 import geometry as G
from server.vhuman.eye2 import textures as T

PARAMS = dict(optics=dict(sclera_radius=.0146, limbus_radius=.006, cornea_radius=.0085, chamber_depth=.0035),
              structure=dict(seed=1, fibers=.5, crypts=.5, furrows=.5, freckles=.2, collarette=.5))


@pytest.fixture(scope='module')
def dims():
    return G.dims_from_params(PARAMS, .26)


def test_cornea_meets_sclera_at_limbus(dims):
    zc = dims.cornea_center_z
    assert math.isclose(math.hypot(dims.L, dims.z_limbus - zc), dims.Rc, rel_tol=1e-9)
    assert math.isclose(math.hypot(dims.L, dims.z_limbus), dims.R, rel_tol=1e-9)
    assert dims.cornea_center_z < dims.iris_z < dims.z_limbus < dims.apex_z


def test_cornea_volume_closed_and_outward(dims):
    part = G.cornea(dims)
    pos, tri = G.weld(part.positions, part.triangles, 1e-8)
    assert G.boundary_edges(tri) == 0
    n = G.face_normals(pos, tri)
    centre = pos.mean(0)
    assert ((n * (pos[tri].mean(1) - centre)).sum(1) > 0).mean() > .99


def test_sclera_outward_and_open_only_at_limbus(dims):
    part = G.sclera(dims)
    pos, tri = G.weld(part.positions, part.triangles, 1e-8)
    n = G.face_normals(pos, tri)
    assert ((n * pos[tri].mean(1)).sum(1) > 0).all()
    # boundary vertices all lie on the limbus circle
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    u, c = np.unique(e, axis=0, return_counts=True)
    b = np.unique(u[c == 1])
    edge = dims.R * math.sin(dims.theta_limbus - G.LIMBUS_FADE * dims.scale / dims.R)
    assert len(b) and np.allclose(np.hypot(pos[b, 0], pos[b, 1]), edge, rtol=1e-6)
    # opaque from the limbus outward, transparent at the inner fringe edge (soft limbus)
    alpha = part.extra['alpha']
    r = np.hypot(part.positions[:, 0], part.positions[:, 1])
    front = part.positions[:, 2] > 0
    body = front & (r >= dims.R * math.sin(dims.theta_limbus + .31 * G.LIMBUS_FADE * dims.scale / dims.R))
    assert body.any() and (alpha[body] > .999).all()
    assert np.allclose(alpha[front & np.isclose(r, edge, rtol=1e-6)], 0.)


def test_iris_inside_chamber(dims):
    height = np.random.default_rng(0).uniform(-1, 1, (64, 256))
    part = G.iris(dims, height, .00006 * dims.scale)
    p = part.positions
    r = np.hypot(p[:, 0], p[:, 1])
    vis = r <= dims.L
    # visible iris stays behind the cornea and in front of the closing disc
    zc = dims.cornea_center_z
    assert (np.linalg.norm(p[vis] - [0, 0, zc], axis=1) < dims.Rc).all()
    assert (p[:, 2] > G.cornea(dims).extra['z_back']).all()
    # outer skirt reaches but does not cross the sclera wall
    assert (np.linalg.norm(p, axis=1) < dims.R).all()
    assert (G.face_normals(*G.weld(p, part.triangles, 1e-9))[:, 2] > 0).mean() > .95


def test_vessels_forest_without_crossings(dims):
    vs = T.grow_vessels(dims, 3, vascularity=.5, coverage=.5)
    assert len(vs) > 40
    assert T.crossings(vs) <= 2          # stop-before-contact; rare tangential grazes only
    for v in vs:
        assert np.allclose(np.linalg.norm(v.pts, axis=1), 1., atol=1e-6)
        assert (np.diff(v.width) <= 1e-12).all()


def test_vessel_coverage_band(dims):
    alb, hgt, deep, sup, vs = T.sclera_maps(dims, T.ScleraColors(np.array([.33, .25, .23]), np.array([.36, .27, .25]),
                                                                np.array([.85, .89, 1.])), 3, h=384, w=1024)
    front = slice(int(.25 * len(alb)), None)            # visible front half of the map
    cov = float((sup[front] > .3).mean() + (deep[front] > .3).mean())
    assert .005 < cov < .12
    assert alb.min() >= 0 and alb.max() <= 1 and 0 <= hgt.min() and hgt.max() <= 1


def test_deterministic(dims):
    a = T.grow_vessels(dims, 7)
    b = T.grow_vessels(dims, 7)
    assert len(a) == len(b) and all(np.array_equal(x.pts, y.pts) for x, y in zip(a, b))
    ia, ih = T.iris_maps(T.IrisColors(np.array([.03, .015, .002]), np.array([.03, .013, .002]), .26), 7, h=64, w=256)
    ib, _ = T.iris_maps(T.IrisColors(np.array([.03, .015, .002]), np.array([.03, .013, .002]), .26), 7, h=64, w=256)
    assert np.array_equal(ia, ib)


def test_iris_zone_calibration():
    inner, outer = np.array([.05, .03, .01]), np.array([.03, .02, .008])
    a, _ = T.iris_maps(T.IrisColors(inner, outer, .26), 2, h=192, w=1024)
    rn = ((np.arange(192) + .5) / 192 * G.IRIS_V_SPAN)[:, None] * np.ones((1, 1024))
    med_out = np.median(a[(rn > .5) & (rn < .75)], 0)
    assert np.allclose(med_out, outer, rtol=.25)


def test_uv_orientation_matches_texture_rows(dims):
    """Blender UV v = 1 - row/(H-1): the sclera limbus samples the last texture row, the iris pupil the first."""
    sc = G.sclera(dims)
    r = np.hypot(sc.positions[:, 0], sc.positions[:, 1])
    limbus = np.isclose(r, dims.L, rtol=1e-6) & (sc.positions[:, 2] > 0)
    assert np.allclose(sc.uvs[limbus, 1], 0.)
    dirs, theta = T.sclera_dirs(dims, 64, 128)
    assert theta[-1, 0] < theta[0, 0]                     # last row = near the limbus
    ir = G.iris(dims)
    pupil = np.isclose(np.hypot(ir.positions[:, 0], ir.positions[:, 1]), dims.pupil, rtol=1e-6)
    assert np.allclose(ir.uvs[pupil, 1], 1.)              # first row of the polar iris map


def test_shell_smooth_closed_and_outward(dims):
    part = G.shell(dims)
    pos, tri = G.weld(part.positions, part.triangles, 1e-9)
    assert G.boundary_edges(tri) == 0
    n = G.face_normals(pos, tri)
    assert ((n * pos[tri].mean(1)).sum(1) > 0).all()
    th = np.linspace(0., 1.4, 2000)
    r = G.shell_radius(th, dims)
    assert math.isclose(r[0], dims.apex_z, rel_tol=1e-9) and np.allclose(r[th > .9], dims.R)
    slope = np.diff(r) / np.diff(th)
    assert np.abs(np.diff(slope)).max() < 20 * np.abs(np.diff(slope)).mean() + 1e-9   # no crease at the limbus
    a = part.extra['alpha']; theta = np.arccos(np.clip(part.positions[:, 2] / np.linalg.norm(part.positions, axis=1), -1, 1))
    assert (a[theta < .5 * dims.theta_limbus] == 0).all() and (a[theta > 1.2 * dims.theta_limbus] == 1).all()


def test_chamber_meets_shell_and_encloses_iris(dims):
    ch = G.chamber(dims)
    top = ch.positions[np.isclose(ch.positions[:, 2], ch.extra['z_top'])]
    rho = np.linalg.norm(top, axis=1)
    th = np.arccos(top[:, 2] / rho)
    assert np.allclose(rho, G.shell_radius(th, dims), rtol=1e-6)
    ir = G.iris(dims).positions
    vis = np.hypot(ir[:, 0], ir[:, 1]) < dims.L
    assert (ir[vis, 2] > ch.extra['z_back']).all() and (ir[vis, 2] < ch.extra['z_top']).all()
