"""Compare the native SigLIP probe with Transformers using Google-only weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
from transformers import SiglipVisionModel
try:
    from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cuda.hunyuan_video15.native_generate import prepare_portrait
from ref.hunyuan_video15.compare import compare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-dtype", choices=("float32", "float16"), default="float32",
                    help="FP32 matches native activation precision; FP16 checks the upstream inference dtype")
    ap.add_argument("--reference-device", choices=("cpu","cuda"), default="cuda")
    ap.add_argument("--actual", type=Path, help="compare a saved native siglip_hidden dump")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference, actual = out / "reference", out / "native"
    reference.mkdir(); actual.mkdir()
    prepared, pixels = prepare_portrait(args.image, out, 480, 848)
    google = args.model.resolve() / "google_siglip/reference_vision"
    processor = SiglipImageProcessor.from_pretrained(google / "feature_extractor", local_files_only=True)
    with Image.open(prepared) as image:
        tensor = processor(images=image, return_tensors="pt").pixel_values
    native_pixels = np.fromfile(pixels, dtype="<f4").reshape(1, 3, 384, 384)
    if not np.allclose(tensor.numpy(), native_pixels, atol=1e-7, rtol=0):
        raise ValueError("native SigLIP preprocessing differs from the Google processor")
    dtype = getattr(torch, args.reference_dtype)
    model = SiglipVisionModel.from_pretrained(google / "image_encoder", local_files_only=True,
                                            dtype=dtype, attn_implementation="eager").eval().to(args.reference_device)
    with torch.inference_mode():
        hidden = model(pixel_values=tensor.to(args.reference_device, dtype=dtype)).last_hidden_state.float().cpu().numpy()
    np.save(reference / "siglip_hidden.npy", hidden)
    del model
    if args.reference_device=='cuda':
        torch.cuda.empty_cache()
    if args.actual:
        actual=args.actual.resolve()
    else:
        subprocess.run([str(ROOT / "cuda/hunyuan_video15/test_cuda_hunyuan_video15_siglip"),
            str(args.model.resolve() / "google_siglip/vision_fp16.safetensors"), str(pixels),
            str(actual / "siglip_hidden.f32")], check=True)
    value = np.fromfile(actual / "siglip_hidden.f32", dtype="<f4").reshape(1, 729, 1152)
    np.save(actual / "siglip_hidden.npy", value)
    result = compare(reference, actual, ["siglip_hidden"])
    report = {"reference_dtype": args.reference_dtype, "reference_device":args.reference_device,
              "image_sha256":hashlib.sha256(prepared.read_bytes()).hexdigest(),"native_activations": "float32",
              "weights": "float16_google_vision", "results": result}
    (out / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["siglip_hidden"]["pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
