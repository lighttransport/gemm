"""Skin chart, atlas interpolation and surface-detail invariants."""
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from PIL import Image
import numpy as np

from .head import skin


class SkinTests(unittest.TestCase):
    def test_portrait_brows_follow_roll_and_relative_exposure(self):
        from .eye import optics
        root = Path(__file__).resolve().parents[2] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            path = Path(temp) / 'portrait.png'
            yy, xx = np.mgrid[:256, :256]
            for angle in (0., .35):
                rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
                xy = (np.stack([xx, yy], -1)-[128, 120]) @ rotation / 80
                x, y = xy[..., 0], xy[..., 1]
                brow = (np.abs(np.abs(x)-.5) < .22) & (np.abs(y+.31) < .055)
                for exposure in (.25, 1.):
                    linear = np.full((256, 256, 3), [.32, .17, .10]) * exposure
                    linear[brow] *= .10
                    Image.fromarray(np.round(optics.linear_to_srgb(linear)*255).astype('u1')).save(path)
                    centers = np.array([[-40, 0], [40, 0]]) @ rotation.T + [128, 120]
                    eyes = [SimpleNamespace(cx=a, cy=b) for a, b in centers]
                    mask = skin.portrait_brow_exclusion(path, eyes)
                    probes = np.array([[-.5, -.31], [.5, -.31], [-.5, -.1], [0, -.65], [.6, .5]])
                    values = skin.sample_portrait_mask(mask, probes*80 @ rotation.T + [128, 120])
                    self.assertTrue((values[:2] > .95).all(), values)
                    self.assertTrue((values[2:] < .001).all(), values)
                    self.assertEqual(skin.sample_portrait_mask(mask, [[-1, 20]])[0], 0)
            Image.new('RGB', (256, 256), (100, 65, 40)).save(path)
            self.assertLess(skin.portrait_brow_exclusion(path, eyes).max(), 1e-10)

    def test_validation(self):
        p = skin.validate({"regions": {"cheeks": {"redness": .3}}, "freckles": {"density": .2}})
        self.assertEqual(p["regions"]["cheeks"]["redness"], .3)
        self.assertEqual(p["regions"]["nose"]["redness"], 0.)
        for params in ({"tone_u": float("nan")}, {"roughness": -1}, {"seed": 1.2},
                       {"delight_strength": 1.1}, {"delight_strength": True},
                       {"freckles": {"bad": 1}}, {"regions": {"bad": {}}}, {"enabled": 1}):
            with self.assertRaises(ValueError):
                skin.validate(params)

    def test_tone_chart(self):
        self.assertTrue((skin.tone_color(.5, 0) > skin.tone_color(.5, 1)).all())
        self.assertGreater(skin.tone_color(1, .5)[0], skin.tone_color(0, .5)[0])
        self.assertLess(skin.tone_color(1, .5)[2], skin.tone_color(0, .5)[2])

    def test_atlas_barycentrics(self):
        uv = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        tri = np.array([[0, 1, 2], [1, 3, 2]])
        covered = np.zeros((16, 32), bool)
        for y, x, ids, weights in skin.atlas_samples(uv, tri, covered.shape):
            interp = (uv[ids] * weights[..., None]).sum(1)
            np.testing.assert_allclose(interp, np.stack([(x + .5) / 32, (y + .5) / 16], 1), atol=1e-12)
            covered[y, x] = True
        self.assertTrue(covered.all())
        self.assertEqual(sum(len(y) for y, *_ in skin.atlas_samples(uv, np.array([[0, 0, 0]]), (16, 32))), 0)

    def test_detail_gradient_and_seams(self):
        rng = np.random.default_rng(21)
        xyz = rng.uniform(-.02, .02, (200, 3))
        value, gradient = skin.cellular(xyz, .003, 11)
        self.assertGreater(value.max(), .1)
        self.assertTrue((value >= 0).all())
        eps = 1e-8
        for axis in range(3):
            d = np.eye(3)[axis] * eps
            numerical = (skin.cellular(xyz + d, .003, 11)[0] - skin.cellular(xyz - d, .003, 11)[0]) / (2 * eps)
            np.testing.assert_allclose(gradient[:, axis], numerical, atol=1e-4)
        # Geometrically coincident points on independent UV charts agree.
        np.testing.assert_array_equal(value, skin.cellular(xyz.copy(), .003, 11)[0])
        self.assertTrue((skin.cellular(xyz, .003, 11, 0)[0] == 0).all())

    def test_bake_preserves_color_and_mirrored_normals(self):
        root = Path(__file__).resolve().parents[2] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            out = Path(temp)
            image = Image.new("RGB", (64, 64), (180, 125, 100))
            image.save(out / "portrait.png")
            buf = io.BytesIO(); image.save(buf, format="PNG")
            # Same plane in two independent charts, second chart mirrored.
            pos = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.]])
            uv = np.array([[0., 0.], [.5, 0.], [0., 1.], [1., 0.], [.5, 0.], [1., 1.]])
            mesh = {"positions": np.tile(pos, (2, 1)), "normals": np.tile([0., 0., 1.], (6, 1)), "uvs": uv}
            tri = np.array([[0, 1, 2], [3, 4, 5]])
            cam = SimpleNamespace(origin=np.array([.01, .01, 1.]), project=lambda p: p[:, :2] * 1600 + 16)
            eyes = [SimpleNamespace(cx=20., cy=20.), SimpleNamespace(cx=44., cy=20.)]
            poses = [SimpleNamespace(units_per_m=1.)] * 2
            info, tangents = skin.bake(mesh, tri, buf.getvalue(), None, out / "portrait.png", eyes, poses, cam, {}, out)
            color = np.asarray(Image.open(out / "skin_basecolor.png"))
            mask = np.asarray(Image.open(out / "skin_mask.png")) > 127
            np.testing.assert_array_equal(color[mask], np.tile([180, 125, 100], (mask.sum(), 1)))
            normal = np.asarray(Image.open(out / "skin_normal.png"), float) / 255 * 2 - 1
            # Mirror charts carry opposite tangent handedness and X slopes.
            self.assertNotEqual(tangents[0, 3], tangents[3, 3])
            left, right = normal[:, :32], normal[:, 32:][:, ::-1]
            valid = mask[:, :32] & mask[:, 32:][:, ::-1]
            np.testing.assert_allclose(left[valid, 0], -right[valid, 0], atol=2/255 + 1e-6)
            np.testing.assert_allclose(left[valid, 1:], right[valid, 1:], atol=2/255 + 1e-6)
            self.assertGreater(np.std(normal[..., 0][mask]), .005)
            before = color.copy()
            unsupported, _ = skin.bake(mesh, tri, buf.getvalue(), None, out / "portrait.png", eyes, poses, cam,
                                       {"delight_strength": 1.}, out)
            self.assertNotEqual(unsupported["illumination"]["status"], "estimated")
            np.testing.assert_array_equal(np.asarray(Image.open(out / "skin_basecolor.png"))[mask], before[mask])
            # Project this same surface onto the mouth: pores and freckles
            # are suppressed even though its colour is confidently skin.
            mouth_cam = SimpleNamespace(origin=cam.origin,
                project=lambda p: np.tile([32., 45.2], (len(p), 1)))
            skin.bake(mesh, tri, buf.getvalue(), None, out / "portrait.png", eyes, poses, mouth_cam,
                      {"freckles": {"density": 1., "strength": 1., "mask": "full_face"}}, out)
            mouth_normal = np.asarray(Image.open(out / "skin_normal.png"), float)
            self.assertTrue((np.abs(mouth_normal[mask, :2] - 127.5) <= .5).all())
            np.testing.assert_array_equal(np.asarray(Image.open(out / "skin_basecolor.png"))[mask], before[mask])
            skin.bake(mesh, tri, buf.getvalue(), None, out / "portrait.png", eyes, poses, cam,
                      {"tone_strength": .7, "tone_v": .9, "roughness": 1.5}, out)
            after = np.asarray(Image.open(out / "skin_basecolor.png"))
            self.assertLess(after[mask].mean(), before[mask].mean())
            self.assertGreater(info["covered_texels"], 1000)
            # Non-skin pixels retain source PBR factors when scalar factors
            # are folded into the new material's maps.
            blue = Image.new("RGB", (64, 64), (20, 60, 200))
            buf2 = io.BytesIO(); blue.save(buf2, format="PNG")
            skin.bake(mesh, tri, buf2.getvalue(), None, out / "portrait.png", eyes, poses, cam, {}, out,
                      roughness_factor=.5, metallic_factor=.2)
            orm = np.asarray(Image.open(out / "skin_orm.png"))
            np.testing.assert_array_equal(orm[mask], np.tile([255, 128, 51], (mask.sum(), 1)))

    def test_regions_and_detail_follow_portrait_roll(self):
        eyes = [SimpleNamespace(cx=-50., cy=0., r=10.), SimpleNamespace(cx=50., cy=0., r=10.)]
        pix = np.array([[0., 105.], [-50., 0.], [50., 0.], [-65., 55.], [65., 55.], [0., -65.]])
        regions = skin.region_masks(pix, eyes)
        detail = skin.detail_mask(pix, eyes, regions)
        np.testing.assert_array_equal(detail[:3], 0)
        self.assertTrue((detail[3:] > .99).all())
        angle = .3
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        offset = np.array([210., 350.])
        transformed = pix @ rotation.T + offset
        eye_pos = np.array([[e.cx, e.cy] for e in eyes]) @ rotation.T + offset
        tilted = [SimpleNamespace(cx=x, cy=y, r=10.) for x, y in eye_pos]
        after = skin.region_masks(transformed, tilted)
        for key in regions:
            np.testing.assert_allclose(regions[key], after[key], atol=1e-14)
        np.testing.assert_allclose(detail, skin.detail_mask(transformed, tilted, after), atol=1e-14)
        # Eye ordering is not an anatomical change.
        for key, values in skin.region_masks(transformed, tilted[::-1]).items():
            np.testing.assert_array_equal(values, after[key])

    def test_skin_gate(self):
        reference = np.array([.4, .23, .16])
        colors = np.array([reference, reference * .7, [.005, .004, .003], [.8, .8, .8], [.05, .1, .4]])
        m = skin.skin_mask(colors, reference)
        self.assertGreater(m[0], .99)
        self.assertGreater(m[1], .99)
        self.assertLess(m[2], .01)
        self.assertLess(m[3], .2)
        self.assertLess(m[4], .01)

    def test_measured_lips_follow_location_roll_and_exposure(self):
        root = Path(__file__).resolve().parents[2] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            path = Path(temp) / "portrait.png"
            yy, xx = np.mgrid[:384, :384]
            delta = np.stack([xx, yy], -1) - [192, 105]
            for angle, exposure in ((0., 1.), (.35, .65), (-.25, .85)):
                right = np.array([np.cos(angle), np.sin(angle)])
                down = np.array([-right[1], right[0]])
                x, y = delta @ right / 100, delta @ down / 100
                lip = ((x-.04)/.4)**2 + ((y-1.28)/.105)**2 < 1
                rgb = np.full((384, 384, 3), [190., 145., 115.])
                rgb[lip] = [170, 82, 96]
                # Dark beard-like distractor has no lip chroma support.
                rgb[(np.abs(x)<.45) & (y>1.42)] = [12, 10, 9]
                Image.fromarray(np.rint(rgb*exposure).astype('u1')).save(path)
                eye_pos = np.array([192,105]) + np.array([[-50],[50]])*right
                eyes = [SimpleNamespace(cx=a, cy=b, r=10.) for a,b in eye_pos]
                features = skin.portrait_regions(path, eyes)
                self.assertEqual(features['method'], 'portrait lip chroma')
                np.testing.assert_allclose(features['lips'][:2], [.04,1.28], atol=.025)
                point = np.array([[192,105]]) + (right*.04+down*1.28)*100
                regions = skin.region_masks(point, eyes, features)
                self.assertGreater(regions['lips'][0], .97)
                self.assertEqual(skin.detail_mask(point, eyes, regions)[0], 0.)
                envelope = np.array([[-.31,1.28],[.39,1.28],[.04,1.19],[.04,1.37]])
                samples = [192,105] + (envelope[:, :1]*right+envelope[:, 1:]*down)*100
                measured = skin.region_masks(samples, eyes, features)
                np.testing.assert_array_equal(skin.detail_mask(samples, eyes, measured), 0.)
            Image.new('RGB', (384,384), (190,145,115)).save(path)
            fallback = skin.portrait_regions(path, eyes)
            self.assertEqual(fallback['method'], 'anatomical fallback')
            self.assertEqual(fallback['lips'], [0.,1.05,.4,.16])


if __name__ == "__main__":
    unittest.main()
