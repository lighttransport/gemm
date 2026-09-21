"""Check native tokenizer fixtures against the official pipeline, CPU only."""
import argparse
from pathlib import Path
import tempfile

from diffusers import QwenImage21Pipeline
import numpy as np
from transformers import Qwen3VLProcessor

from native_text import prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    args = ap.parse_args()
    processor = Qwen3VLProcessor.from_pretrained(str(args.model / "processor"), local_files_only=True)
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=None,
                              processor=processor, transformer=None)
    root = Path(__file__).resolve().parents[2]
    (root / "tmp").mkdir(exist_ok=True)
    for prompt in ("a red apple on a white table", "", "猫、青い空\ntext: café"):
        with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="qimg21-tokens-") as work:
            folder = Path(work)
            drop = prepare(args.model, prompt, folder)
            expected = processor(text=[pipe.prompt_template_t2i.format(prompt or " ")],
                                 padding=True, padding_side="left", return_tensors="pt").input_ids.numpy()
            np.testing.assert_array_equal(np.load(folder / "input_ids.npy"), expected)
            np.testing.assert_array_equal(np.loadtxt(folder / "tokens.txt", dtype=np.int64), expected[0])
            assert drop == pipe._drop_idx, (drop, pipe._drop_idx)
            print(f"PASS prompt={prompt!r}: {expected.shape[1]} tokens, drop={drop}")


if __name__ == "__main__":
    main()
