"""Run with Blender --background --python server/vhuman/test_usd_materials.py."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('bpy'), 'requires Blender Python')
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


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    unittest.main(argv=[__file__],verbosity=2)
