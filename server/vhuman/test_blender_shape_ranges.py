"""Run in Blender to check imported corrective weights beyond the 0..1 UI range."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import bpy
except ImportError:
    bpy = None
from server.vhuman.rig.blender_usd import restore_shape_ranges
from server.vhuman.reconstruction.usd_material_animation import action_curves


@unittest.skipUnless(bpy, 'requires Blender Python')
class ShapeRangeTest(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh = bpy.data.meshes.new('triangle')
        mesh.from_pydata([(0,0,0), (1,0,0), (0,1,0)], [], [(0,1,2)])
        self.obj = bpy.data.objects.new('skin', mesh)
        bpy.context.collection.objects.link(self.obj)
        self.obj.shape_key_add(name='Basis')
        self.key = self.obj.shape_key_add(name='ml_signed')
        self.key.data[0].co.z = 1
        self.key.slider_min = -3; self.key.slider_max = 3
        for frame, value in ((0,-2.), (6,1.5)):
            self.key.value = value; self.key.keyframe_insert('value', frame=frame)
        for curve in action_curves(self.obj.data.shape_keys.animation_data):
            for p in curve.keyframe_points:p.interpolation = 'LINEAR'
        self.key.slider_min = 0; self.key.slider_max = 1

    def test_signed_and_large_weights_restore_evaluated_geometry_and_keep_curves(self):
        curves = list(action_curves(self.obj.data.shape_keys.animation_data))
        before = [[tuple(p.co) for p in c.keyframe_points] for c in curves]
        bpy.context.scene.frame_set(0)
        self.assertEqual(self.key.value, 0)
        report = restore_shape_ranges([self.obj])
        self.assertEqual(report['changed_shapes'], 1)
        for frame, value in ((0,-2.), (3,-.25), (6,1.5)):
            bpy.context.scene.frame_set(frame)
            ev = self.obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            self.assertAlmostEqual(ev.data.vertices[0].co.z, value, places=6)
        self.assertEqual(before, [[tuple(p.co) for p in c.keyframe_points] for c in curves])
        self.assertEqual(restore_shape_ranges([self.obj])['changed_shapes'], 0)

    def test_mixed_interpolation_uses_the_next_keys_incoming_handle(self):
        curve = next(action_curves(self.obj.data.shape_keys.animation_data))
        points = list(curve.keyframe_points)
        left, right = points
        for interpolation in ('LINEAR', 'CONSTANT'):
            for sign in (-1, 1):
                with self.subTest(interpolation=interpolation, sign=sign):
                    left.co = (0, 0); right.co = (6, 0)
                    left.interpolation = 'BEZIER'; right.interpolation = interpolation
                    for point in points:
                        point.handle_left_type = point.handle_right_type = 'FREE'
                    left.handle_left = (-2, 0); left.handle_right = (2, 0)
                    right.handle_left = (4, sign*8); right.handle_right = (8, 0)
                    self.key.slider_min = 0; self.key.slider_max = 1
                    before = [(tuple(p.co), tuple(p.handle_left), tuple(p.handle_right), p.interpolation)
                              for p in points]
                    expected = sign*3.
                    self.assertAlmostEqual(curve.evaluate(3), expected, places=6)
                    bpy.context.scene.frame_set(3)
                    evaluated = self.obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                    self.assertAlmostEqual(evaluated.data.vertices[0].co.z, 0 if sign < 0 else 1)
                    report = restore_shape_ranges([self.obj])
                    self.assertEqual(report['changed_shapes'], 1)
                    for frame in (1, 3, 5):
                        bpy.context.scene.frame_set(frame)
                        evaluated = self.obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                        self.assertAlmostEqual(evaluated.data.vertices[0].co.z, curve.evaluate(frame), places=6)
                    self.assertEqual(before, [(tuple(p.co), tuple(p.handle_left), tuple(p.handle_right), p.interpolation)
                                              for p in points])
                    self.assertEqual(restore_shape_ranges([self.obj])['changed_shapes'], 0)

    def test_unused_endpoint_handles_do_not_expand_or_reject_ranges(self):
        curve = next(action_curves(self.obj.data.shape_keys.animation_data))
        left, right = curve.keyframe_points
        left.interpolation = 'BEZIER'; right.interpolation = 'LINEAR'
        for point in (left, right):
            point.handle_left_type = point.handle_right_type = 'FREE'
        left.handle_left = (-2, -100); left.handle_right = (2, -2)
        right.handle_left = (4, 1.5); right.handle_right = (8, 100)
        report = restore_shape_ranges([self.obj])
        self.assertEqual(report['changed_shapes'], 1)
        self.assertEqual((self.key.slider_min, self.key.slider_max), (-2, 1.5))
        for frame in (0, 3, 6):
            bpy.context.scene.frame_set(frame)
            evaluated = self.obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            self.assertAlmostEqual(evaluated.data.vertices[0].co.z, curve.evaluate(frame), places=6)

    def test_unsupported_incoming_bezier_handle_is_rejected(self):
        curve = next(action_curves(self.obj.data.shape_keys.animation_data))
        left, right = curve.keyframe_points
        left.interpolation = 'BEZIER'; right.interpolation = 'LINEAR'
        right.handle_left_type = 'FREE'; right.handle_left = (4, 100)
        with self.assertRaisesRegex(ValueError, 'limits'):
            restore_shape_ranges([self.obj])

    def test_unsupported_weight_range_is_rejected(self):
        curve = next(action_curves(self.obj.data.shape_keys.animation_data))
        curve.keyframe_points[0].co.y = -100
        with self.assertRaisesRegex(ValueError, 'limits'):
            restore_shape_ranges([self.obj])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
