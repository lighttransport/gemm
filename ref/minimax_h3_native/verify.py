"""Independent CPU PyTorch checks for H3 checkpoint projections.

This never calls native kernels to compute the expected result. INT8 sums use
int32 arithmetic and the rotation uses a dense Kronecker Hadamard matrix.
Full pipeline acceptance compares every supplied independent latent/frame;
component checks alone cannot mark a generated video verified.
"""
from __future__ import annotations
import argparse
import importlib.util
import hashlib
import json
import math
import sys
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ref.minimax_h3_native import captures

UPSTREAM = "2472a20bd291451acc303917059ab14dfc380478"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def comparison(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64).ravel(), np.asarray(expected, dtype=np.float64).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("shape mismatch or nonfinite reference")
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    cosine = float(a @ b / (norm_a * norm_b)) if norm_a and norm_b else float(np.array_equal(a, b))
    relative = float(np.linalg.norm(a - b) / max(norm_b, 1e-30))
    return {"cosine": cosine, "relative_l2": relative, "pass": cosine >= .9999 and relative <= .02}


def capture(directory, name):
    meta = json.loads((directory / (name + ".json")).read_text())
    raw = captures.read_f32(directory, name, math.prod(meta["shape"]))
    if meta["dtype"] != "float32":
        raise ValueError("unsupported native capture type")
    return raw.reshape(meta["shape"])


def components(args):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(16)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = Path(args.model).resolve()
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    rotation = h4
    for _ in range(3):
        rotation = torch.kron(rotation, h4)
    rotation /= 16
    checks = [
        ("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "model.layers.0.self_attn.q_proj", 1),
        ("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "model.layers.0.mlp.down_proj", 1),
        ("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "blocks.0.attn.qkv_proj", 1),
        ("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "blocks.0.mlp.fc2", 1),
        ("vae/minimax_h3_video_vae_fp16.safetensors", "decoder.transformer_blocks.0.attn.to_qkv", 2),
    ]
    report = {}
    rng = np.random.default_rng(42)
    for i, (filename, prefix, kind) in enumerate(checks):
        directory = output / str(i)
        with safe_open(model / filename, framework="pt", device="cpu") as weights:
            weight = weights.get_tensor(prefix + ".weight")
            rows, features = 19, weight.shape[1]
            dtype = torch.bfloat16 if kind == 1 else torch.float16
            x = torch.from_numpy(rng.standard_normal((rows, features)).astype(np.float32)).to(dtype)
            path = output / f"input_{i}.f32"
            x.float().numpy().tofile(path)
            subprocess.run([args.probe, str(model), "linear", filename, prefix, str(path), str(rows), str(directory), str(kind)], check=True,
                           env=__import__("os").environ | {"TMPDIR": str(ROOT / "tmp/video-rocm")})
            if weight.dtype == torch.int8:
                metadata = json.loads(bytes(weights.get_tensor(prefix + ".comfy_quant").tolist()))
                if metadata != {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}:
                    raise ValueError("unsupported reference quantization metadata")
                rotated = (x.float().reshape(rows, -1, 256) @ rotation).to(dtype).reshape(rows, features)
                scale = (rotated.abs().amax(-1, keepdim=True).float() / 127).clamp(min=1e-30)
                math_scale = scale.to(dtype)
                math_scale = torch.where(math_scale == 0, torch.full_like(math_scale, torch.finfo(dtype).tiny), math_scale)
                quant = (rotated / math_scale).round().clamp(-128, 127).to(torch.int32)
                # Chunk output channels to avoid a full expanded checkpoint matrix.
                expected = torch.empty((rows, weight.shape[0]), dtype=torch.float32)
                for c in range(0, weight.shape[0], 1024):
                    sums = quant @ weight[c:c + 1024].to(torch.int32).T
                    ws = weights.get_tensor(prefix + ".weight_scale")[c:c + 1024].float().T
                    expected[:, c:c + 1024] = (sums.float() * (scale * ws)).to(dtype).float()
            else:
                expected = x.float() @ weight.float().T
                if prefix + ".bias" in weights.keys():
                    expected = expected + weights.get_tensor(prefix + ".bias").float()
                expected = expected.to(dtype).float()
            report[prefix] = comparison(capture(directory, "output"), expected.numpy())
            print(prefix, report[prefix], flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(item["pass"] for item in report.values()):
        raise RuntimeError("independent checkpoint projection comparison failed")


def qwen(args):
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    torch.set_num_threads(16)
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Path(args.model)
    ids = Tokenizer.from_file(str(model / "tokenizer/tokenizer.json")).encode(args.prompt).ids
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    rotation = h4
    for _ in range(3):
        rotation = torch.kron(rotation, h4)
    rotation = (rotation / 16).to(device)
    dtype = torch.bfloat16
    rows = len(ids)
    conditioned = Path(args.conditioning_dir) if getattr(args, 'conditioning_dir', None) else None
    if conditioned:
        rows = json.loads((conditioned/'manifest.json').read_text())['text_rows']
    with safe_open(model / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", framework="pt", device="cpu") as weights:
        def norm(x, name):
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), weights.get_tensor(name + ".weight").to(device), 1e-6)
        def linear(x, name):
            # ConvRot multiplies in the activation dtype in the pinned upstream.
            rotated = (x.reshape(rows, -1, 256) @ rotation.to(dtype)).reshape(rows, -1)
            scale = (rotated.abs().amax(-1, keepdim=True).float() / 127).clamp(min=1e-30)
            divisor = scale.to(dtype)
            divisor = torch.where(divisor == 0, torch.full_like(divisor, torch.finfo(dtype).tiny), divisor)
            quant = (rotated / divisor).round().clamp(-128, 127).to(torch.int8)
            # Float32 BLAS on integral inputs is an independent, efficient CPU reference.
            # Strict INT32 accumulation is tested separately by `components`.
            weight = weights.get_tensor(name + ".weight")
            if device == "cpu":
                product = quant.float() @ weight.float().T
            else:
                padded = torch.nn.functional.pad(quant, (0, 0, 0, max(0, 32 - rows)))
                product = torch._int_mm(padded, weight.to(device).T.contiguous())[:rows].float()
            return (product * (scale * weights.get_tensor(name + ".weight_scale").float().T.to(device))).to(dtype)
        x = weights.get_tensor("model.embed_tokens.weight")[ids].to(device)
        positions = torch.arange(rows, dtype=torch.float32, device=device)[:, None]
        frequencies = 1 / 5000000 ** (torch.arange(64, dtype=torch.float32, device=device) / 64)
        angles = positions * frequencies
        cosine, sine = angles.cos(), angles.sin()
        if conditioned:
            x = torch.from_numpy(np.fromfile(conditioned/'qwen_inputs.f32', '<f4').reshape(rows,5120)).to(device,dtype)
            rotary = torch.from_numpy(np.fromfile(conditioned/'qwen_rotation.f32','<f4').reshape(rows,64,2)).to(device)
            cosine, sine = rotary[...,0], rotary[...,1]
        def rope(x):
            left, right = x.float().chunk(2, -1)
            # The pinned Qwen rotary matrix stays FP32, including both products.
            return torch.cat((left * cosine[:, None] - right * sine[:, None],
                              right * cosine[:, None] + left * sine[:, None]), -1).to(dtype)
        for layer in range(50):
            p = f"model.layers.{layer}"
            z = norm(x, p + ".input_layernorm")
            q = norm(linear(z, p + ".self_attn.q_proj").reshape(rows, 64, 128), p + ".self_attn.q_norm")
            k = norm(linear(z, p + ".self_attn.k_proj").reshape(rows, 8, 128), p + ".self_attn.k_norm")
            v = linear(z, p + ".self_attn.v_proj").reshape(rows, 8, 128)
            q, k = rope(q).transpose(0, 1), rope(k).transpose(0, 1)
            v = v.transpose(0, 1)
            attention = torch.nn.functional.scaled_dot_product_attention(q.float(), k.repeat_interleave(8, 0).float(),
                         v.repeat_interleave(8, 0).float(), is_causal=True).to(dtype).transpose(0, 1).reshape(rows, 8192)
            x = x + linear(attention, p + ".self_attn.o_proj")
            z = norm(x, p + ".post_attention_layernorm")
            z = torch.nn.functional.silu(linear(z, p + ".mlp.gate_proj")) * linear(z, p + ".mlp.up_proj")
            x = x + linear(z, p + ".mlp.down_proj")
            if conditioned and layer < 3:
                ds = np.fromfile(conditioned/f'deepstack_{layer}.f32','<f4').reshape(rows,5120)
                x = x + torch.from_numpy(ds).to(device,dtype)
            print(f"reference Qwen layer {layer + 1}/50", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "qwen_hidden.npy", x.float().cpu().numpy()[None], allow_pickle=False)
    result = comparison(capture(Path(args.native), "qwen_hidden"), x.float().cpu().numpy())
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps({
        "prompt": args.prompt, "token_ids": ids, "device": device, "upstream_revision": UPSTREAM,
        "reference_source_sha256": digest(__file__),
        "capture_reader_sha256": digest(captures.__file__),
        "qwen_hidden_sha256": digest(out / "qwen_hidden.npy"),
        "conditioning_manifest_sha256": digest(conditioned/'manifest.json') if conditioned else None,
        "components": {name: digest(model / name) for name in (
            "tokenizer/tokenizer.json", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")},
    }, indent=2) + "\n")
    print(result, flush=True)
    if not result["pass"]:
        raise RuntimeError("independent Qwen layer-50 parity failed")


def pipeline(args):
    native, reference = Path(args.native), Path(args.reference)
    generation = json.loads(Path(args.manifest).read_text())
    if generation.get('conditioning'):
        raise ValueError('use ref/minimax_h3_native/conditioning.py for conditioned diagnostics')
    if generation.get("backend") not in ("minimax_h3_rocm_experimental", "minimax_h3_cuda_experimental") or generation.get("parity") != "unverified":
        raise ValueError("expected a complete native H3 generation")
    diagnostic = args.mode == "diagnostic"
    if not diagnostic and (generation["width"], generation["height"], generation["frames"], generation["fps"]) != (1344, 768, 124, 24):
        raise ValueError("bounded diagnostic shapes cannot pass full pipeline acceptance")
    if not diagnostic and (generation["sigma_grid_points"], generation["euler_updates"], generation["seed"], generation["video_shift"], generation["audio_shift"]) != (40, 39, 42, 12, 3):
        raise ValueError("full pipeline acceptance requires the prescribed 40-point schedule and seed")
    if generation["metrics"].get("memory_fit") != "pass":
        raise ValueError("pipeline acceptance requires measured process VRAM within budget")
    receipts = json.loads((reference / "receipts.json").read_text())
    report = {}
    spec = importlib.util.spec_from_file_location("video_tools", ROOT / "cuda/hunyuan_video15_native/generate.py")
    video = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video)
    provenance = video.digest(args.manifest)
    names = ["qwen_hidden", "refined_text", "noise_video", "noise_audio"]
    for step in range(generation["euler_updates"]):
        names += [f"latent_video_{step:03d}", f"latent_audio_{step:03d}"]
    for name in names:
        receipt = receipts.get(name, {})
        storage = receipt.get("storage", name + ".npy")
        if storage not in (name + ".npy", name + ".npy.gz"):
            raise ValueError("noncanonical reference storage path")
        path = reference / storage
        if (receipt.get("generation_sha256") != provenance or receipt.get("sha256") != video.digest(path)
                or receipt.get("upstream_revision") != UPSTREAM
                or receipt.get("reference_source_sha256") != digest(Path(__file__).with_name("reference.py"))
                or receipt.get("capture_reader_sha256") != digest(captures.__file__)
                or receipt.get("shared_input") != name.startswith("noise_")):
            raise ValueError(f"missing independent reference receipt: {name}")
        actual, expected = capture(native, name), captures.read_npy(path)
        if actual.shape != expected.shape or list(expected.shape) != receipt.get("shape"):
            raise ValueError(f"noncanonical reference shape: {name}")
        report[name] = comparison(actual, expected)
    for index in range(generation["frames"]):
        name = f"frame_{index:03d}"
        receipt = receipts.get(name, {})
        storage = receipt.get("storage", name + ".npy")
        if storage not in (name + ".npy", name + ".npy.gz"):
            raise ValueError("noncanonical reference storage path")
        path = reference / storage
        if (receipt.get("generation_sha256") != provenance or receipt.get("sha256") != video.digest(path)
                or receipt.get("upstream_revision") != UPSTREAM
                or receipt.get("reference_source_sha256") != digest(Path(__file__).with_name("reference.py"))
                or receipt.get("capture_reader_sha256") != digest(captures.__file__)
                or receipt.get("shared_input") is not False):
            raise ValueError(f"missing independent frame receipt: {name}")
        raw = captures.read_f32(native, name, generation["height"] * generation["width"] * 3).reshape(generation["height"], generation["width"], 3)
        expected = captures.read_npy(path)
        if raw.shape != expected.shape or list(expected.shape) != receipt.get("shape"):
            raise ValueError(f"noncanonical reference shape: {name}")
        report[name] = comparison(raw, expected)
    Path(args.out).write_text(json.dumps({"scope": "bounded_diagnostic" if diagnostic else "full_pipeline", "comparisons": report,
                                        "pass": all(x["pass"] for x in report.values())}, indent=2) + "\n")
    if not all(x["pass"] for x in report.values()):
        raise RuntimeError("H3 diagnostic parity failed" if diagnostic else "full H3 pipeline parity failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("components")
    c.add_argument("--model", default="/mnt/disk01/models/h3/weights")
    c.add_argument("--probe", default=str(ROOT / "tmp/video-rocm/h3-build/component_probe"))
    c.add_argument("--out", required=True)
    c.set_defaults(run=components)
    c = sub.add_parser("qwen")
    c.add_argument('--conditioning-dir')
    c.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    c.add_argument("--model", default="/mnt/disk01/models/h3/weights")
    for name in ("prompt", "native", "out"):
        c.add_argument("--" + name, required=True)
    c.set_defaults(run=qwen)
    for mode in ("pipeline", "diagnostic"):
        c = sub.add_parser(mode)
        for name in ("native", "reference", "manifest", "out"):
            c.add_argument("--" + name, required=True)
        c.set_defaults(run=pipeline)
    args = p.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
