"""Joint Edit grids keep camera order, measured pixels and raw-view provenance."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from .reconstruction import mv_qwen
from .reconstruction.observations import sha256


class TextureGridTests(unittest.TestCase):
    def test_one_joint_generation_splits_views_without_modifying_protected_pixels(self):
        order = ('front', 'right', 'back', 'left')
        colors = [(180, 110, 90), (160, 120, 100), (140, 90, 70), (170, 130, 110)]
        views = []
        for name in reversed(order):
            valid = np.zeros((32, 32), bool); valid[3:-3, 3:-3] = True
            known = np.zeros((32, 32)); known[:, :16] = 1
            views.append(dict(name=name, rgb=np.full((32, 32, 3), 128, np.uint8),
                              valid=valid, known=known, depth=np.ones((32, 32))))
        class Editor:
            generator = 'fixture'; metadata = {}; last_metrics = {}
            calls = 0
            def __call__(self, images, prompt, **options):
                self.calls += 1
                self.input = images[-1].copy()
                result = np.zeros((1024, 1024, 3), np.uint8)
                for i, color in enumerate(colors):
                    y, x = divmod(i, 2)
                    result[y*512:(y+1)*512, x*512:(x+1)*512] = color
                return result, .5
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'tmp') as tmp:
            root = Path(tmp); Image.new('RGB', (40, 48), (100, 90, 80)).save(root/'portrait.png')
            editor = Editor()
            images, receipt = mv_qwen.grid(root, None, views, root/'generation', editor=editor, steps=40)
            self.assertEqual(editor.calls, 1)
            self.assertEqual(editor.input.shape, (1024, 1024, 3))
            for view, image in zip(views, images):
                editable = view['valid'] & (view['known'] < .5)
                np.testing.assert_array_equal(image[~editable], view['rgb'][~editable])
                np.testing.assert_array_equal(image[editable], np.tile(colors[order.index(view['name'])], (editable.sum(), 1)))
                self.assertTrue((root/'generation'/view['name']/'edited.png').is_file())
            self.assertTrue(receipt['shared_generation'])
            self.assertEqual(receipt['grid_order'], list(order))
            self.assertEqual(receipt['input_sha256'], sha256(root/'generation/grid_input.png'))


if __name__ == '__main__':
    unittest.main()
