"""Material interchange regressions, evaluated through the authored USD graph."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from .rig.usd import _material_usd


@unittest.skipUnless(importlib.util.find_spec('pxr'), 'optional USD SDK unavailable')
class USDMaterialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'tmp')
        self.root = Path(self.tmp.name)
        (self.root/'textures').mkdir()
        self.addCleanup(self.tmp.cleanup)

    def graph(self, material, images=None):
        from pxr import Sdf, Usd, UsdShade
        text = _material_usd('eye', dict(gltf=material, images=images or {}), self.root/'textures', 'textures')
        layer = Sdf.Layer.CreateAnonymous()
        layer.ImportFromString('#usda 1.0\ndef Xform "Character" {\n def Scope "Materials" {\n'+text+'\n}\n}')
        stage = Usd.Stage.Open(layer)
        return stage, UsdShade.Shader(stage.GetPrimAtPath('/Character/Materials/eye/Surface'))

    def test_sclera_opaque_cornea_clear_and_ior_preserved(self):
        from pxr import UsdShade
        # Only red carries transmission; green/blue deliberately disagree.
        pixels = np.array([[[0, 255, 255], [255, 0, 0]]], np.uint8)
        png = io.BytesIO(); Image.fromarray(pixels).save(png, format='PNG')
        stage, surface = self.graph(dict(extensions={
            'KHR_materials_transmission': dict(transmissionFactor=.8, transmissionTexture=dict(index=3)),
            'KHR_materials_ior': dict(ior=1.336)}), {3: png.getvalue()})
        source, channel, _ = surface.GetInput('opacity').GetConnectedSource()
        self.assertEqual(channel, 'r')
        texture = UsdShade.Shader(source.GetPrim())
        self.assertEqual(texture.GetInput('sourceColorSpace').Get(), 'raw')
        scale, bias = texture.GetInput('scale').Get(), texture.GetInput('bias').Get()
        path = self.root/texture.GetInput('file').Get().path
        opacity = np.asarray(Image.open(path))[..., 0]/255*scale[0]+bias[0]
        np.testing.assert_allclose(opacity, [[1., .2]], atol=1e-6)
        self.assertEqual(path.read_bytes(), png.getvalue())
        self.assertAlmostEqual(surface.GetInput('ior').Get(), 1.336, places=5)

    def test_constant_transmission_is_not_replaced_with_fixed_opacity(self):
        stage, surface = self.graph(dict(extensions={'KHR_materials_transmission': dict(transmissionFactor=.7)}))
        self.assertFalse(surface.GetInput('opacity').HasConnectedSource())
        self.assertAlmostEqual(surface.GetInput('opacity').Get(), .3, places=6)

    def test_normal_strength_scales_tangent_xy_only(self):
        from pxr import UsdShade
        stage, surface = self.graph(dict(normalTexture=dict(index=0, scale=.6)), {0: b'normal'})
        source, _, _ = surface.GetInput('normal').GetConnectedSource()
        texture = UsdShade.Shader(source.GetPrim())
        scale, bias = texture.GetInput('scale').Get(), texture.GetInput('bias').Get()
        vector = np.array([.75, .25, 1.])*np.asarray(scale)[:3]+np.asarray(bias)[:3]
        np.testing.assert_allclose(vector, [.3, -.3, 1.], atol=1e-6)

    def test_alpha_blended_lid_keeps_alpha_connection(self):
        stage, surface = self.graph(dict(alphaMode='BLEND', pbrMetallicRoughness=dict(
            baseColorTexture=dict(index=0))), {0: b'rgba'})
        source, channel, _ = surface.GetInput('opacity').GetConnectedSource()
        self.assertEqual(channel, 'a')
        self.assertFalse(surface.GetInput('ior'))


if __name__ == '__main__':
    unittest.main()
