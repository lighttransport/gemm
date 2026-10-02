import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from .export_flux2_tokenizer import export

ROOT = Path(__file__).resolve().parents[2]


class FluxTokenizerTests(unittest.TestCase):
    def test_literal_added_tokens_and_merges_in_native_loader(self):
        probe = ROOT / 'cpu/vhuman/tokenizer_probe'
        if not probe.is_file():
            self.skipTest('make -C cpu/vhuman tokenizer_probe')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            folder = Path(folder)
            spec = dict(model=dict(type='BPE', vocab={'a': 0, 'b': 1, 'ab': 2}, merges=[['a', 'b']]),
                        added_tokens=[dict(id=3, content='<think>', special=False),
                                      dict(id=4, content='<|endoftext|>', special=True)])
            (folder / 'tokenizer.json').write_text(json.dumps(spec))
            (folder / 'tokenizer_config.json').write_text(json.dumps(
                dict(tokenizer_class='Qwen2Tokenizer', eos_token='<|endoftext|>', pad_token='<|endoftext|>')))
            output = folder / 'tokenizer.gguf'
            receipt = export(folder, output)
            self.assertEqual(receipt['vocabulary'], 5)
            actual = subprocess.run([str(probe), str(output), 'ab<think><|endoftext|>'],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(actual.stdout), [2, 3, 4])
            spec['model']['vocab']['a'] = 8
            (folder / 'tokenizer.json').write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                export(folder, output)


if __name__ == '__main__':
    unittest.main()
