#!/usr/bin/env python3
"""Compare native HIP velocity with BFL's official DiT on CPU.

Use an independent checkout of black-forest-labs/flux-3-image-klein with
--reference-dir. Real prompt features and deterministic BF16 image/condition inputs exercise
all three modes, including nonzero fourth-axis condition IDs. FP8 scaled_mm is emulated on CPU using
F32 accumulation over the same quantized activations and per-row scales.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("/mnt/nvme02/model/flux3-klein/variants/fp8r/model.safetensors"))
    p.add_argument("--output-dir", type=Path, default=Path("tmp/flux3/verify"))
    p.add_argument("--size", type=int, default=256, help="square image size, multiple of 16")
    p.add_argument("--text", type=Path, required=True, help="prepared F32 [tokens,20480] Qwen3-VL text features")
    p.add_argument("--trace-dir", type=Path, help="save reference/native intermediate activations for debugging")
    p.add_argument("--no-condition", action="store_true", help="also exercise the empty condition stream used by T2I")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-relative-rmse", type=float, default=0.04)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("TMPDIR", str(root / "tmp"))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    torch.set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location("flux3_official_dit", args.reference_dir / "src/flux3/dit.py")
    dit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dit
    spec.loader.exec_module(dit)
    state = load_file(str(args.checkpoint), device="cpu")
    with torch.device("meta"):
        model = dit.DiT(flow_map=True).to(torch.bfloat16)
    class FP8Linear(torch.nn.Module):
        def __init__(self, weight, scale):
            super().__init__()
            self.weight = weight
            self.scale = scale
        def forward(self, x):
            x = x.to(torch.bfloat16)
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            scale = flat.float().abs().amax(-1, keepdim=True).div(448).clamp_min(1e-12)
            quant = (flat.float() / scale).to(torch.float8_e4m3fn).float()
            with torch.autocast("cpu", enabled=False):
                out = (quant @ self.weight.float().T) * scale * self.scale[None]
            return out.to(torch.bfloat16).reshape(*shape[:-1], self.weight.shape[0])
    # Replace quantized Linear modules before loading the remaining BF16 state.
    fp8 = []
    for name, module in list(model.named_modules()):
        key = name + ".weight"
        if isinstance(module, torch.nn.Linear) and state[key].dtype == torch.float8_e4m3fn:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            setattr(parent, child, FP8Linear(state.pop(key), state.pop(name + ".weight_scale")))
            fp8.append(name)
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    if args.trace_dir:
        for sub in ("reference", "native"):
            (args.trace_dir / sub).mkdir(parents=True, exist_ok=True)
        def save(name, value):
            value.detach().float().numpy().astype("<f4").tofile(args.trace_dir / "reference" / (name + ".f32"))
        def hook(name):
            def capture(module, inputs, output):
                if isinstance(module, (torch.nn.Linear, FP8Linear)):
                    save(name, output)
                    save(name + ".input", inputs[0])
                if isinstance(module, dit.ModeModulation):
                    save(name + ".input", inputs[0])
                if isinstance(module, dit.QKNorm):
                    save(name + ".q", output[0][0].transpose(0, 1).reshape(-1, 3072))
                    save(name + ".k", output[1][0].transpose(0, 1).reshape(-1, 3072))
            return capture
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, FP8Linear, dit.ModeModulation, dit.QKNorm)):
                module.register_forward_hook(hook(name))
    rng = torch.Generator().manual_seed(17)
    if args.size < 16 or args.size % 16:
        p.error("--size must be a positive multiple of 16")
    side=args.size//16
    ni, nc = side*side, (0 if args.no_condition else 2)
    array = np.fromfile(args.text, dtype="<f4").reshape(-1, 20480).copy()
    nt = len(array)
    if nt < 1 or nt > 4096 or not np.isfinite(array).all():
        p.error("--text must contain 1..4096 finite feature rows")
    data = {"image": torch.randn(ni, 128, generator=rng).bfloat16(),
            "image_cond": torch.randn(nc, 128, generator=rng).bfloat16(),
            "text": torch.from_numpy(array).bfloat16()}
    yy,xx=torch.meshgrid(torch.arange(side),torch.arange(side),indexing="ij")
    image_ids=torch.stack([torch.zeros_like(yy),yy*.5,xx*.5,torch.zeros_like(yy)],dim=-1).reshape(ni,4)
    ids = {"image": image_ids,
           "image_cond": torch.tensor([[0, 0, 0, -1], [0, .5, .5, -2]])[:nc],
           "text": torch.stack([torch.zeros(nt),torch.zeros(nt),torch.zeros(nt),torch.arange(nt)],dim=-1)}
    packed = dit.Packed(data=data, ids=ids,
                        timesteps={m: torch.full((len(x),), .75 if m == "image" else 0, dtype=torch.bfloat16) for m, x in data.items()},
                        mode_indices={m: [(0, len(x))] for m, x in data.items()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def dump(name, x):
        path = args.output_dir / name
        x.float().numpy().astype("<f4").tofile(path)
        return str(path)
    image = dump("image.f32", data["image"])
    text = dump("text.f32", data["text"])
    cond = dump("condition.f32", data["image_cond"])
    condids = dump("condition_ids.f32", ids["image_cond"])
    with torch.inference_mode():
        expected = model(packed, torch.tensor([.5], dtype=torch.bfloat16)).float()
    dump("reference.f32", expected)
    del model, state
    output = args.output_dir / "native.f32"
    command = [str(Path(__file__).resolve().parent / "test_hip_flux3"), "--dit", str(args.checkpoint),
                    "--width", str(args.size), "--height", str(args.size), "--text", text, "--ntxt", str(nt),
                    "--latent", image,
                    "--step", ".75", ".5", "--dump", str(output), "--device", str(args.device)]
    if args.trace_dir:
        command += ["--trace-dir", str(args.trace_dir / "native")]
    if nc:
        command += ["--condition", cond, "--condition-ids", condids, "--ncond", str(nc)]
    subprocess.run(command, check=True)
    actual = np.fromfile(output, dtype="<f4").reshape(ni, 128)
    ref = expected.numpy()
    delta = actual - ref
    relative_rmse = float(np.sqrt(np.mean(delta ** 2)) / np.sqrt(np.mean(ref ** 2)))
    cosine = float(np.sum(actual * ref) / np.sqrt(np.sum(actual ** 2) * np.sum(ref ** 2)))
    result = {"relative_rmse": relative_rmse, "cosine": cosine, "max_abs": float(np.max(np.abs(delta))),
              "fp8_layers": len(fp8), "condition_tokens": nc, "image_tokens": ni, "pass": relative_rmse <= args.max_relative_rmse}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
