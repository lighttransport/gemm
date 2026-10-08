"""Run inside Blender to validate sampled shader animation without scene drivers."""
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
from server.vhuman.reconstruction import usd_materials, usd_material_animation as animation


@unittest.skipUnless(bpy is not None, 'requires Blender Python')
class MaterialAnimationTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', 'tmp'))
        self.root = Path(self.temp.name)
        self.scene = bpy.context.scene
        self.material = bpy.data.materials.new('animated_skin')
        self.material.use_nodes = True
        bpy.ops.mesh.primitive_plane_add()
        bpy.context.object.data.materials.append(self.material)
        self.math = self.material.node_tree.nodes.new('ShaderNodeMath')
        self.math.name = 'nonlinear_wrinkle'
        self.scene['strength'] = .1
        for frame, value in ((1, .1), (2, .8), (3, .2)):
            self.scene['strength'] = value
            self.scene.keyframe_insert('["strength"]', frame=frame)
        driver = self.math.inputs[1].driver_add('default_value').driver
        driver.expression = 'strength * strength'
        variable = driver.variables.new()
        variable.name = 'strength'
        variable.targets[0].id_type = 'SCENE'
        variable.targets[0].id = self.scene
        variable.targets[0].data_path = '["strength"]'
        self.color = self.material.node_tree.nodes.new('ShaderNodeRGB')
        self.color.name = 'animated_color'
        for frame, value in ((1, (.1, .2, .3, 1)), (3, (.8, .6, .4, 1))):
            self.color.outputs[0].default_value = value
            self.color.outputs[0].keyframe_insert('default_value', frame=frame)
        self.scene.frame_set(1)
        usd_materials.capture([self.material], self.root)
        self.frames = [1, 1.5, 2, 2.5, 3]

    def tearDown(self):
        self.temp.cleanup()

    def bake(self):
        report = animation.capture([self.material], self.root, self.frames)
        self.scene.animation_data_clear()
        del self.scene['strength']
        usd_materials.restore(self.root)
        animation.restore(self.root)
        return report

    def test_nonlinear_driver_and_vector_output_survive_without_source(self):
        report = self.bake()
        self.assertEqual(len(report['channels']), 2)
        result = animation.verify(self.root)
        self.assertEqual(result['checks'], 10)
        self.assertTrue(result['original_driver_dependencies_removed'])
        scalar = next(c for c in report['channels'] if c['node'] == 'nonlinear_wrinkle')
        self.assertAlmostEqual(scalar['values'][2], .64, places=6)
        self.assertGreater(max(scalar['values']) - min(scalar['values']), .6)
        # An uncaptured point uses the advertised linear sampled interpolation.
        self.scene.frame_set(1, subframe=.25)
        self.assertAlmostEqual(self.material.node_tree.nodes['nonlinear_wrinkle'].inputs[1].default_value,
                               .5*(scalar['values'][0]+scalar['values'][1]), places=6)

    def test_changed_keys_and_interpolation_are_rejected(self):
        self.bake()
        curve = next(animation.action_curves(self.material.node_tree.animation_data))
        curve.keyframe_points[0].co.y += .2
        with self.assertRaisesRegex(ValueError, 'animation differs'):
            animation.verify(self.root)
        animation.restore(self.root)
        curve = next(animation.action_curves(self.material.node_tree.animation_data))
        curve.keyframe_points[0].interpolation = 'CONSTANT'
        with self.assertRaisesRegex(ValueError, 'interpolation'):
            animation.verify(self.root)

    def test_frame_driver_is_sampled_before_export_time_mapping(self):
        driver = self.material.node_tree.animation_data.drivers[0].driver
        driver.expression = 'frame * frame'
        report = animation.capture([self.material], self.root, [10, 10.5, 11],
                                   output_frames=[1, 2, 3])
        scalar = next(c for c in report['channels'] if c['node'] == 'nonlinear_wrinkle')
        self.assertEqual(scalar['values'], [100., 110.25, 121.])
        usd_materials.restore(self.root)
        animation.restore(self.root)
        self.assertEqual(animation.verify(self.root)['frames'], 3)

    def test_unsupported_property_and_invalid_samples_rejected(self):
        self.material.node_tree['unsupported'] = 0.
        self.material.node_tree.keyframe_insert('["unsupported"]', frame=1)
        with self.assertRaisesRegex(ValueError, 'unsupported animated material property'):
            animation.capture([self.material], self.root, self.frames)
        with self.assertRaisesRegex(ValueError, 'increase strictly'):
            animation.capture([self.material], self.root, [1, 1])

    def test_invalid_sidecar_values_rejected(self):
        self.bake()
        path = self.root / animation.FILENAME
        report = json.loads(path.read_text())
        report['channels'][0]['values'][0] = float('nan')
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'invalid material animation values'):
            animation.restore(self.root)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
