"""Run material roundtrip and tamper checks inside Blender's Python runtime."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import bpy
except ImportError:
    bpy = None
from server.vhuman.reconstruction.usd_materials import capture, restore, verify


@unittest.skipUnless(bpy is not None, 'requires Blender Python')
class USDMaterialTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', 'tmp'))
        self.root = Path(self.temp.name)
        self.material = bpy.data.materials.new('test_material')
        self.material.use_nodes = True
        self.shader = self.material.node_tree.nodes.get('Principled BSDF')
        self.shader.inputs['Subsurface Weight'].default_value = .12
        image = bpy.data.images.new('test_texture', width=2, height=2)
        image.pixels[:] = [.4, .2, .1, 1.]*4
        image.filepath_raw = str(self.root/'original.png')
        image.file_format = 'PNG'
        image.save()
        image.pack()
        node = self.material.node_tree.nodes.new('ShaderNodeTexImage')
        node.image = image
        self.material.node_tree.links.new(node.outputs['Color'], self.shader.inputs['Base Color'])
        self.record = capture([self.material], self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_restored_values_links_images_and_tampering(self):
        self.material.node_tree.nodes.clear()
        restore(self.root)
        report = verify(self.root)
        self.assertEqual(report['materials'], 1)
        self.assertEqual(report['images'], 1)
        shader = self.material.node_tree.nodes.get('Principled BSDF')
        self.assertAlmostEqual(shader.inputs['Subsurface Weight'].default_value, .12, places=6)
        shader.inputs['Subsurface Weight'].default_value = .9
        with self.assertRaisesRegex(ValueError, 'socket value'):
            verify(self.root)

    def test_asset_corruption_and_external_paths_rejected(self):
        image = next(iter(self.record['images'].values()))
        path = self.root/image['path']
        original = path.read_bytes()
        path.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'path/hash'):
            restore(self.root)
        path.write_bytes(original)
        image['path'] = '../outside.png'
        (self.root/'blender_materials.json').write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, 'path/hash'):
            restore(self.root)


@unittest.skipUnless(bpy is not None, 'requires Blender Python')
class MaterialRampTests(unittest.TestCase):
    def test_ramp_roundtrip_and_tampering(self):
        import bpy
        from server.vhuman.reconstruction.usd_materials import capture, restore, verify, color_ramp_record
        root=Path(__file__).resolve().parents[2]/'tmp'
        root.mkdir(exist_ok=True)
        material=bpy.data.materials.new('vhuman_ramp_test')
        try:
            material.use_nodes=True
            node=material.node_tree.nodes.new('ShaderNodeValToRGB')
            ramp=node.color_ramp
            ramp.color_mode='HSV';ramp.hue_interpolation='CW';ramp.interpolation='EASE'
            ramp.elements[0].position=.1;ramp.elements[0].color=(.12,.23,.34,.45)
            ramp.elements[1].position=.9;ramp.elements[1].color=(.87,.76,.65,.54)
            ramp.elements.new(.3).color=(.2,.8,.1,.9)
            ramp.elements.new(.55).color=(.4,.1,.7,.6)
            material.node_tree.links.new(node.outputs['Color'],
                material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
            expected=color_ramp_record(ramp)
            with tempfile.TemporaryDirectory(dir=root) as directory:
                capture([material],directory)
                restore(directory);verify(directory)
                actual=next(n for n in material.node_tree.nodes if n.type=='VALTORGB').color_ramp
                self.assertEqual(color_ramp_record(actual),expected)
                actual.elements[1].color=(1,0,0,1)
                with self.assertRaisesRegex(ValueError,'color ramp'):
                    verify(directory)
                restore(directory)
                actual=next(n for n in material.node_tree.nodes if n.type=='VALTORGB').color_ramp
                actual.interpolation='CONSTANT'
                with self.assertRaisesRegex(ValueError,'color ramp'):
                    verify(directory)
                restore(directory);verify(directory)
        finally:
            bpy.data.materials.remove(material,do_unlink=True)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
