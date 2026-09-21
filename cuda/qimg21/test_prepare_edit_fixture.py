"""CPU tests for the captured editing-input/native-layout boundary."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from prepare_edit_fixture import prepare


class EditFixtureTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2] / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.output = self.root / "native"
        self.metadata = {"img_shapes": [[[1, 2, 2], [1, 2, 4]]],
                         "text_slots": 3, "target_tokens": 8}
        self.write_metadata()
        np.save(self.reference / "positive_img_mask.npy", [[0, 1, 0, 1, 1]])
        np.save(self.reference / "positive_encoder_hidden_states_mask.npy", [[1, 1, 1]])
        self.prompt = np.arange(3 * 4096, dtype=np.float32).reshape(1, 3, 4096)
        self.latents = np.arange(12 * 64, dtype=np.float32).reshape(1, 12, 64)
        np.save(self.reference / "prompt_embeds.npy", self.prompt)
        np.save(self.reference / "input_000.npy", self.latents)
        np.save(self.reference / "timestep_000.npy", np.array([0.5], dtype=np.float32))

    def write_metadata(self):
        (self.reference / "positive_layout.json").write_text(json.dumps(self.metadata))

    def test_round_trip(self):
        result = prepare(self.reference, self.output)
        self.assertEqual(result["condition_tokens"], 4)
        self.assertFalse(result["native_editing_validated"])
        np.testing.assert_array_equal(np.load(self.output / "condition_latents.npy"), self.latents[0, :4])
        np.testing.assert_array_equal(np.load(self.output / "target_latents.npy"), self.latents[0, 4:])
        np.testing.assert_array_equal(np.load(self.output / "prompt_embeds.npy"), self.prompt[0])
        binary = Path(__file__).with_name("test_joint_layout")
        layout = self.output / "layout.txt"
        file_result = subprocess.run([str(binary), str(layout)], capture_output=True, check=True)
        stdin_result = subprocess.run([str(binary)], input=layout.read_bytes(), capture_output=True, check=True)
        self.assertEqual(file_result.stdout, stdin_result.stdout)
        with self.assertRaises(FileExistsError):
            prepare(self.reference, self.output)

    def test_reject_bad_inputs_before_output(self):
        path = self.reference / "input_000.npy"
        for bad in (np.full_like(self.latents, np.nan), self.latents[:, :-1],
                    np.full(self.latents.shape, 1e100), self.latents.astype(np.complex64)):
            np.save(path, bad)
            with self.assertRaises(ValueError):
                prepare(self.reference, self.output)
            self.assertFalse(self.output.exists())

    def test_reject_padding_and_bad_metadata(self):
        path = self.reference / "positive_encoder_hidden_states_mask.npy"
        np.save(path, [[1, 0, 1]])
        with self.assertRaises(ValueError):
            prepare(self.reference, self.output)
        np.save(path, [[1, 1, 1]])
        self.metadata["target_tokens"] = 7
        self.write_metadata()
        with self.assertRaises(ValueError):
            prepare(self.reference, self.output)
        self.assertFalse(self.output.exists())

    def test_native_file_parser_rejects_malformed(self):
        binary = Path(__file__).with_name("test_joint_layout")
        layout = self.root / "invalid.txt"
        for data in ("", "9999999999999999999999 1 1", "2 1 1 0 1 2 2 extra",
                     "2x 1 1 0 1 2 2", "2 1 1 0 1 2", "2 1 1 0 0 2 2"):
            layout.write_text(data)
            result = subprocess.run([str(binary), str(layout)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
