"""Run the pinned official generator with component capture hooks.

Example: python capture_ref.py --upstream tmp/hunyuan-video15-upstream
 --dump-dir tmp/hv15-reference -- --model_path CKPTS --image_path portrait.png
 --resolution 480p --video_length 81 --seed 42 --offloading true --rewrite false

This requires the upstream environment and its official checkpoint layout.
No checkpoint layout conversions or alternate vision weights are inferred.
"""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np
PIN = "60783e704160023913bee78f0b47036d393d4dfa"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--vision-model", type=Path, required=True, help="Google-only exported reference_vision directory")
    args, forwarded = ap.parse_known_args()
    upstream = args.upstream.resolve()
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError(f"reference source must be pinned to {PIN}")
    dump = args.dump_dir.resolve()
    dump.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(upstream))
    from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline as Pipeline
    import torch
    # Explicitly replace the upstream vision source before constructing a pipeline.
    from hyvideo.models.vision_encoder import VisionEncoder
    def load_google_vision(pretrained_model_name_or_path, device):
        encoder = VisionEncoder(vision_encoder_type="siglip", vision_encoder_precision="fp16",
            vision_encoder_path=str(args.vision_model.resolve()), device=device)
        try:
            from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil
            encoder.processor = SiglipImageProcessorPil.from_pretrained(
                args.vision_model.resolve() / "feature_extractor", local_files_only=True)
        except ImportError:
            pass  # Transformers 4.57.1 already uses the trained PIL recipe.
        return encoder
    Pipeline._load_vision_encoder = staticmethod(load_google_vision)
    original_create = Pipeline.create_pipeline
    captured = set()
    def save(name, tensor, overwrite=False):
        if tensor is None or (name in captured and not overwrite):
            return
        value = tensor.detach().float().cpu().numpy()
        np.save(dump / f"{name}.npy", value, allow_pickle=False)
        captured.add(name)
        if name == "noise_input":
            value.astype("<f4").tofile(dump / "noise_input.f32")
    def hook_method(obj, name, after):
        original = getattr(obj, name)
        def wrapped(*a, **kw):
            value = original(*a, **kw)
            after(value)
            return value
        setattr(obj, name, wrapped)
    def create(*a, **kw):
        pipe = original_create(*a, **kw)
        def qwen(value):
            hidden, _, mask, _ = value
            # Native removes masked padding rather than carrying it through DiT.
            active = mask[0].bool()
            save("qwen_hidden", hidden[:, active, :])
            save("qwen_mask", mask)
        hook_method(pipe, "encode_prompt", qwen)
        hook_method(pipe, "prepare_latents", lambda v: save("noise_input", v))
        hook_method(pipe, "get_image_condition_latents", lambda v: save("vae_encoded", v))
        def vision(value):
            # CFG duplicates vision features; the native branch uses one sample.
            save("siglip_hidden", value[:1] if value is not None else None)
        hook_method(pipe, "_prepare_vision_states", vision)
        def byt5(value):
            save("byt5_hidden", value["byt5_text_states"][-1:])
            save("byt5_mask", value["byt5_text_mask"][-1:])
        hook_method(pipe, "_prepare_byt5_embeddings", byt5)
        hook_method(pipe.transformer, "forward", lambda v: save("dit_first", v[0][-1:]))
        hook_method(pipe.scheduler, "step", lambda v: save("latent_final", v[0], overwrite=True))
        hook_method(pipe.vae, "decode", lambda v: save("vae_decoded", ((v[0] / 2 + 0.5).clamp(0, 1))))
        return pipe
    Pipeline.create_pipeline = create
    forwarded = forwarded[1:] if forwarded[:1] == ["--"] else forwarded
    sys.argv = [str(upstream / "generate.py"), *forwarded]
    with torch.inference_mode():
        runpy.run_path(str(upstream / "generate.py"), run_name="__main__")
    (dump / "capture.json").write_text(json.dumps({"reference_revision": PIN,
        "arguments": forwarded, "vision_profile": "google_siglip_so400m_14_384", "components": sorted(captured)}, indent=2) + "\n")

if __name__ == "__main__":
    main()
