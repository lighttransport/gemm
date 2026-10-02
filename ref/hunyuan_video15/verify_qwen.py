"""Compare HunyuanVideo's Qwen selected layer with a bounded FP32 CPU reference.

Uses public pinned config/tokenizer and equal FP16 weights. CPU avoids keeping
an FP32 7B reference on the consumer GPU. Native activations are FP32.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, PreTrainedTokenizerFast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ref.hunyuan_video15.compare import compare
from ref.hunyuan_video15.convert_native import convert
SYSTEM = ("You are a helpful assistant. Describe the video by detailing the following aspects:         "
          "1. The main content and theme of the video.         "
          "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         "
          "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         "
          "4. background environment, light, style and atmosphere.         "
          "5. camera angles, movements, and transitions used in the video.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--generation-manifest", type=Path, help="use the exact prompt of a saved native run")
    ap.add_argument("--actual", type=Path, help="compare saved native conditioning without another GPU run")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--reference-dtype", choices=("float32", "float16"), default="float32")
    args = ap.parse_args()
    generation = json.loads(args.generation_manifest.read_text()) if args.generation_manifest else None
    if generation is not None:
        if args.prompt is not None and args.prompt != generation['prompt']:
            raise ValueError('prompt disagrees with generation manifest')
        args.prompt = generation['prompt']
    args.prompt = args.prompt if args.prompt is not None else 'A person smiles naturally.'
    if (not isinstance(args.prompt,str) or len(args.prompt.encode()) > (4096 if args.actual else 256)
        or args.threads < 1):
        raise ValueError("invalid prompt length or thread count")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference, actual = out / "reference", out / "native"
    reference.mkdir(); actual.mkdir()
    model_dir = args.model.resolve()
    weights = model_dir / "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"
    if weights.with_suffix(weights.suffix + ".aria2").exists():
        raise ValueError("Qwen checkpoint download is incomplete")
    torch.set_num_threads(args.threads)
    config = AutoConfig.from_pretrained(args.config, local_files_only=True).text_config
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen2_5_VLTextModel(config)
    state = {}
    with safe_open(str(weights), framework="pt", device="cpu") as source:
        for key in source.keys():
            if key.startswith("model."):
                state[key.removeprefix("model.")] = source.get_tensor(key)
    model.load_state_dict(state, strict=True, assign=True)
    model.rotary_emb = type(model.rotary_emb)(config, device="cpu")
    model = model.eval().to(dtype=getattr(torch, args.reference_dtype), device="cpu")
    del state
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))
    prefix = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
    text = prefix + args.prompt + "<|im_end|>\n<|im_start|>assistant\n"
    crop = len(tokenizer.encode(prefix, add_special_tokens=False))
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=crop+1000, return_tensors="pt").input_ids
    with torch.inference_mode():
        value = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[26]
    np.save(reference / "qwen_hidden.npy", value[:,crop:,:].float().numpy())
    del model, value
    if args.actual:
        actual = args.actual.resolve()
    else:
        subprocess.run([str(ROOT / "cuda/hunyuan_video15/test_cuda_hunyuan_video15_qwen"),
            str(weights), str(model_dir / "tokenizer.json"), str(actual), args.prompt], check=True)
    convert(actual)
    result = compare(reference, actual, ["qwen_hidden"])
    report = {"reference_dtype":args.reference_dtype, "reference_device":"cpu", "weights":"float16",
              "selected_hidden_state":26, "prefix_tokens":crop,
              "prompt_sha256":hashlib.sha256(args.prompt.encode()).hexdigest(),
              "generation_manifest_sha256":hashlib.sha256(args.generation_manifest.read_bytes()).hexdigest() if args.generation_manifest else None,
              "scope":"qwen_conditioning_with_exact_generation_prompt" if generation else "bounded_qwen_conditioning",
              "results":result}
    (out / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if result["qwen_hidden"]["pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
