"""Load Wan's local T2V/I2V pipelines and encode a prompt without GPU access."""
import argparse
import json
from pathlib import Path
import torch
from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig, WanPipeline,
                       WanImageToVideoPipeline, WanTransformer3DModel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/mnt/disk01/models/wan22"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    path = args.model / "pipeline"
    transformer = WanTransformer3DModel.from_single_file(
        str(args.model / "gguf/Wan2.2-TI2V-5B-Q8_0.gguf"), config=str(path), subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
        torch_dtype=torch.float16, local_files_only=True)
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32,
                                         local_files_only=True)
    pipe = WanPipeline.from_pretrained(path, transformer=transformer, vae=vae,
                                       torch_dtype=torch.bfloat16, local_files_only=True)
    image_pipe = WanImageToVideoPipeline.from_pretrained(
        path, transformer=transformer, vae=vae, text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer, scheduler=pipe.scheduler, torch_dtype=torch.bfloat16,
        local_files_only=True)
    assert pipe.config.expand_timesteps and image_pipe.config.expand_timesteps
    assert vae.config.z_dim == transformer.config.in_channels == 48
    assert pipe.vae_scale_factor_spatial == 16 and pipe.vae_scale_factor_temporal == 4
    with torch.inference_mode():
        embeds, negative = pipe.encode_prompt("A red ball rolling on a wooden table.",
                                              negative_prompt="", device="cpu", dtype=torch.float16)
        decoded = vae.decode(torch.zeros(1, 48, 2, 4, 4)).sample
    assert embeds.shape == negative.shape == (1, 226, 4096)
    assert torch.isfinite(embeds).all() and torch.isfinite(negative).all()
    assert decoded.shape == (1, 3, 5, 64, 64) and torch.isfinite(decoded).all()
    result = {"t2v_loaded": True, "i2v_loaded": True, "prompt_shape": list(embeds.shape),
              "prompt_finite": True, "vae_channels": vae.config.z_dim,
              "vae_dtype": str(vae.dtype), "transformer_dtype": str(transformer.dtype),
              "text_dtype": str(pipe.text_encoder.dtype), "decoded_shape": list(decoded.shape),
              "decoded_finite": True, "execution": "cpu", "torch": torch.__version__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
