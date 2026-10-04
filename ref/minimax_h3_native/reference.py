"""Independent PyTorch H3 DiT/VAE reference, with block weight offload.

Only initial noise is shared with the native capture. Conditioning comes from
the independent CPU Qwen reference; no native intermediate or output is used.
The formulas follow the pinned ComfyUI H3 graph, using torch INT32 matmul and
SDPA rather than any kernels from the native port.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ref.minimax_h3_native import captures

UPSTREAM = "2472a20bd291451acc303917059ab14dfc380478"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Reference:
    def __init__(self, weights, device):
        self.weights, self.device, self.cache = weights, device, {}
        h = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        self.rotation = h
        for _ in range(3):
            self.rotation = torch.kron(self.rotation, h)
        self.rotation = (self.rotation / 16).to(device)

    def weight(self, name, dtype=None):
        key = (name, dtype)
        if key not in self.cache:
            self.cache[key] = self.weights.get_tensor(name).to(device=self.device, dtype=dtype)
        return self.cache[key]

    def clear(self):
        self.cache.clear()

    def linear(self, x, name, fp32=False):
        weight = self.weight(name + ".weight")
        dtype = torch.float32 if fp32 else x.dtype
        bias = self.weight(name + ".bias", dtype) if name + ".bias" in self.weights.keys() else None
        if weight.dtype != torch.int8:
            return F.linear(x.to(dtype), weight.to(dtype), bias)
        metadata = json.loads(bytes(self.weights.get_tensor(name + ".comfy_quant").tolist()))
        if metadata != {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}:
            raise ValueError("unsupported checkpoint quantization")
        matrix_key = (name, "int8_transpose")
        if matrix_key not in self.cache:
            self.cache[matrix_key] = weight.T.contiguous()
        matrix = self.cache[matrix_key]
        scales = self.weight(name + ".weight_scale", torch.float32).reshape(1, -1)
        result = torch.empty((x.shape[0], weight.shape[0]), dtype=dtype, device=self.device)
        # Bound the INT32 output and rotation temporaries, including short M tails.
        for start in range(0, len(x), 512):
            part = x[start:start + 512]
            rotated = (part.reshape(-1, 256) @ self.rotation.to(part.dtype)).reshape_as(part)
            scale = (rotated.abs().amax(-1, keepdim=True).float() / 127).clamp(min=1e-30)
            divisor = scale.to(part.dtype)
            divisor = torch.where(divisor == 0, torch.full_like(divisor, torch.finfo(part.dtype).tiny), divisor)
            quant = (rotated / divisor).round().clamp(-128, 127).to(torch.int8)
            # ROCm's integer GEMM requires M >= 17; zero rows are ignored on writeback.
            count = len(quant)
            if count < 32:
                quant = F.pad(quant, (0, 0, 0, 32 - count))
            sums = torch._int_mm(quant, matrix)[:count]
            result[start:start + count] = (sums.float() * (scale * scales)).to(dtype)
        if bias is not None:
            result += bias
        return result

    def norm(self, x, name=None, eps=1e-5):
        weight = self.weight(name + ".weight", x.dtype) if name else None
        return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)

    @staticmethod
    def rope(x, rotation):
        cosine, sine = rotation
        pairs = cosine.shape[-1]
        # The pinned eager split-half path rounds the first product, then addcmul.
        a, b = x[..., :pairs], x[..., pairs:2 * pairs]
        left = torch.addcmul(a * cosine[:, None], b, -sine[:, None])
        right = torch.addcmul(b * cosine[:, None], a, sine[:, None])
        return torch.cat((left, right, x[..., 2 * pairs:]), dim=-1)

    def attention(self, x, name, rotation=None, vae=False):
        heads, dim = (32, 64) if vae else (56, 128)
        qkv = self.linear(x, name + (".to_qkv" if vae else ".qkv_proj"))
        if vae:
            q, k, v = qkv.reshape(len(x), heads, 3 * dim).chunk(3, -1)
        else:
            q, k, v = (p.reshape(len(x), heads, dim) for p in qkv.chunk(3, -1))
        q = self.norm(q, None if vae else name + ".q_norm")
        k = self.norm(k, None if vae else name + ".k_norm")
        if rotation is not None:
            q, k = self.rope(q, rotation), self.rope(k, rotation)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            attended = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                                      v.transpose(0, 1)[None])
        attended = attended[0].transpose(0, 1).reshape(len(x), heads * dim)
        return self.linear(attended, name + (".to_out" if vae else ".out_proj"))

    def ffn(self, x, name, vae=False):
        out = torch.empty_like(x)
        for start in range(0, len(x), 256):
            gate, up = self.linear(x[start:start + 256], name + (".w1" if vae else ".fc1")).chunk(2, -1)
            out[start:start + 256] = self.linear(F.silu(gate) * up, name + (".w2" if vae else ".fc2"))
        return out

    def refine(self, x):
        x = self.linear(x, "condition_proj")
        for i in range(2):
            p = f"token_refiner.blocks.{i}"
            x = x + self.attention(self.norm(x, p + ".norm1"), p + ".attn")
            x = x + self.ffn(self.norm(x, p + ".norm2"), p + ".mlp")
            self.clear()
        x = self.norm(x, "token_refiner.final_norm")
        self.clear()
        return x

    def dit_rope(self, text, audio, geometry):
        t, h, w = geometry
        area = math.sqrt(h * w)
        def axis(dim):
            ratio = dim / area
            return (torch.arange(dim // 2, dtype=torch.float64) * ratio / (dim // 2) + (1 - ratio) / 2) * 32
        yy, xx = torch.meshgrid(axis(h), axis(w), indexing="ij")
        times = torch.tensor([1 if i % 5 == 0 else 4 for i in range(t)], dtype=torch.float64) * (5 / 3)
        times = text + torch.cat((torch.zeros(1), times[:-1].cumsum(0)))
        video = torch.stack((times[:, None, None].expand(t, h // 2, w // 2),
                             yy.expand(t, -1, -1), xx.expand(t, -1, -1)), -1).reshape(-1, 3)
        prefix = torch.zeros((text, 3), dtype=torch.float64)
        prefix[:, 0] = torch.arange(text)
        stereo = torch.zeros((2, audio, 3), dtype=torch.float64)
        stereo[..., 0] = text + torch.arange(audio)
        stereo[0, :, 2], stereo[1, :, 2] = axis(w)[0], axis(w)[-1]
        ids = torch.cat((prefix, stereo.reshape(-1, 3), video)).float().to(self.device)
        angles = (ids[..., None] * self.weight("rope.inv_freq")).reshape(-1, 48)
        return angles.cos().bfloat16(), angles.sin().bfloat16()

    def time_embed(self, tv, ta):
        table = self.weight("adaln_t_table", torch.float32)
        pos = torch.tensor([tv, ta], device=self.device, dtype=torch.float32).clamp(0, 1) * 1024
        low = pos.floor().long().clamp(max=1023)
        return torch.lerp(table[low], table[low + 1], (pos - low)[:, None])

    @staticmethod
    def modulate(x, mod, segments, chunk):
        out = x.clone()
        for start, stop, row in segments:
            out[start:stop].mul_(1 + mod[row, chunk + 1].to(x.dtype)).add_(mod[row, chunk].to(x.dtype))
        return out

    @staticmethod
    def gated(x, delta, mod, segments, chunk):
        for start, stop, row in segments:
            x[start:stop].addcmul_(delta[start:stop], mod[row, chunk].to(x.dtype))

    def denoise(self, text, video, audio, rotation, tv, ta, step):
        t, h, w, _ = video.shape
        patches = video.reshape(t, h // 2, 2, w // 2, 2, 24).permute(0, 1, 3, 5, 2, 4).reshape(-1, 96)
        vi = self.linear(patches, "video_patch_proj", True).bfloat16()
        au = self.linear(audio, "audio_patch_proj", True).bfloat16()
        x = torch.cat((text, au, vi))
        nt, na = len(text), len(audio)
        segments = ((0, nt, 1), (nt, nt + na, 5), (nt + na, len(x), 0))
        emb = self.time_embed(tv, ta)
        self.clear()
        for i in range(50):
            p = f"blocks.{i}"
            mod = self.linear(emb, p + ".adaln_proj.linear", True).reshape(6, 6, 5376)
            z = self.modulate(self.norm(x, p + ".norm1"), mod, segments, 0)
            self.gated(x, self.attention(z, p + ".attn", rotation), mod, segments, 2)
            z = self.modulate(self.norm(x, p + ".norm2"), mod, segments, 3)
            self.gated(x, self.ffn(z, p + ".mlp"), mod, segments, 5)
            self.clear()
            if i % 10 == 9:
                print(f"reference step {step} block {i + 1}/50", flush=True)
        final = self.linear(emb, "final_layer.adaln_proj.linear", True).reshape(2, 2, 5376)
        def finish(part, row, name):
            part = self.norm(part, "final_layer.norm").float() * (1 + final[row, 1]) + final[row, 0]
            return self.linear(part, name, True)
        av = finish(x[nt:nt + na], 1, "final_layer.audio_out")
        vv = finish(x[nt + na:], 0, "final_layer.video_out")
        vv = vv.reshape(t, h // 2, w // 2, 24, 2, 2).permute(0, 1, 4, 2, 5, 3).reshape_as(video)
        self.clear()
        return vv, av

    def decode_tile(self, z):
        t, h, w, _ = z.shape
        z = z.reshape(-1, 24).to(self.device, torch.float16)
        post = self.weight("post_quant_conv.weight").reshape(24, 24)
        z = F.linear(z, post, self.weight("post_quant_conv.bias"))
        x = self.linear(z, "decoder.x_embedder")
        x = torch.cat((x, self.weight("decoder.register_tokens").reshape(4, 2048), torch.zeros_like(x[:1])))
        coords = torch.stack(torch.meshgrid(*[(torch.arange(d, dtype=torch.float16) + .5) / d * 2 - 1
                                            for d in (t, h, w)], indexing="ij"), -1).reshape(-1, 3)
        coords = torch.cat((coords, torch.zeros((5, 3), dtype=coords.dtype))).float().to(self.device)
        freq = 1 / 100 ** (torch.arange(8, device=self.device, dtype=torch.float32) / 8)
        angles = ((coords * (2 * math.pi))[..., None] * freq).reshape(-1, 24)
        rotation = angles.cos().half(), angles.sin().half()
        self.clear()
        for i in range(36):
            p = f"decoder.transformer_blocks.{i}"
            delta = self.attention(self.norm(x, p + ".norm1"), p + ".attn", rotation, True)
            x = torch.addcmul(x, delta, self.weight(p + ".scale1"))
            delta = self.ffn(self.norm(x, p + ".norm2"), p + ".ff", True)
            x = torch.addcmul(x, delta, self.weight(p + ".scale2"))
            self.clear()
        norm = F.layer_norm(x, (2048,), self.weight("decoder.norm_out.weight"), self.weight("decoder.norm_out.bias"), 1e-5)
        patch = self.linear(norm, "decoder.proj_out")[:t * h * w]
        pixels = patch.reshape(t, h, w, 3, 4, 16, 16).permute(0, 4, 1, 5, 2, 6, 3).reshape(t * 4, h * 16, w * 16, 3)
        self.clear()
        return pixels.cpu()

    @staticmethod
    def tiles(n):
        if n <= 256:
            return [0], [n], []
        count = math.ceil(n / 256)
        while 256 * count - 64 * (count - 1) < n:
            count += 1
        overlap = [64] * (count - 1)
        for i in range((256 * count - 64 * (count - 1) - n) // 16):
            overlap[i % (count - 1)] += 16
        starts = [0]
        for value in overlap:
            starts.append(starts[-1] + 256 - value)
        return starts, [256] * count, overlap

    @staticmethod
    def blend(a, b, dim, extent):
        shape = [1] * b.ndim
        shape[dim] = extent
        fraction = (torch.arange(extent, dtype=b.dtype) / extent).reshape(shape)
        first, tail = [slice(None)] * b.ndim, [slice(None)] * a.ndim
        first[dim], tail[dim] = slice(0, extent), slice(-extent, None)
        b[tuple(first)] = a[tuple(tail)] * (1 - fraction) + b[tuple(first)] * fraction
        return b

    def decode_spatial(self, z):
        t, h, w, _ = z.shape
        ys, heights, yo = self.tiles(h * 16)
        xs, widths, xo = self.tiles(w * 16)
        canvas = torch.empty((t * 4, h * 16, w * 16, 3), dtype=torch.float16)
        strip = None
        for i, y in enumerate(ys):
            new_strip = torch.empty((t * 4, yo[i], w * 16, 3), dtype=torch.float16) if i < len(yo) else None
            left = None
            for j, x in enumerate(xs):
                tile = self.decode_tile(z[:, y // 16:(y + heights[i]) // 16, x // 16:(x + widths[j]) // 16])
                if i:
                    tile = self.blend(strip[:, :, x:x + widths[j]], tile, 1, yo[i - 1])
                if j:
                    tile = self.blend(left, tile, 2, xo[j - 1])
                left = tile[:, :, -xo[j]:].clone() if j < len(xo) else None
                keep_w = widths[j] - (xo[j] if j < len(xo) else 0)
                keep_h = heights[i] - (yo[i] if i < len(yo) else 0)
                canvas[:, y:y + keep_h, x:x + keep_w] = tile[:, :keep_h, :keep_w]
                if new_strip is not None:
                    new_strip[:, :, x:x + keep_w] = tile[:, -yo[i]:, :keep_w]
                print(f"reference VAE tile {i + 1},{j + 1}", flush=True)
            strip = new_strip
        return canvas

    def decode(self, video, requested):
        latent = video.float().cpu()
        mean = self.weights.get_tensor("latents_mean").float()
        std = self.weights.get_tensor("latents_std").float()
        latent = latent * std + mean
        t = len(latent)
        pad = (-(t + 3)) % 5
        chunks = (t + 3 + pad) // 5 - 1
        if chunks < 1:
            pad += 5
            chunks += 1
        if pad:
            latent = torch.cat((latent, latent[-1:].expand(pad, -1, -1, -1)))
        overlap, emitted = None, 0
        for i in range(chunks):
            pixels = self.decode_spatial(latent[i * 5:i * 5 + 7])
            main = pixels[3:20]
            if overlap is not None:
                main = self.blend(overlap, main, 0, min(5, len(overlap), len(main)))
            overlap = pixels[23:]
            parts = (main, overlap) if i == chunks - 1 else (main,)
            for part in parts:
                for frame in part:
                    if emitted >= requested:
                        return
                    yield (frame * torch.tensor([.229, .224, .225]) + torch.tensor([.485, .456, .406])).clamp(0, 1).numpy()
                    emitted += 1
        if emitted != requested:
            raise RuntimeError("reference VAE frame count mismatch")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/disk01/models/h3/weights")
    for name in ("manifest", "native", "qwen-reference", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--compress-output", action="store_true")
    args = p.parse_args()
    manifest, native, out = Path(args.manifest), Path(args.native), Path(args.out)
    generation = json.loads(manifest.read_text())
    if generation["backend"] != "minimax_h3_rocm_experimental":
        raise ValueError("wrong generation backend")
    model = Path(args.model)
    for name, receipt in generation["verified_components"].items():
        path = (model / name).resolve()
        if not path.is_relative_to(model.resolve()) or path.stat().st_size != receipt["bytes"] or digest(path) != receipt["sha256"]:
            raise ValueError(f"reference checkpoint differs from native generation: {name}")
    qwen_dir = Path(args.qwen_reference)
    qwen_path = qwen_dir / "qwen_hidden.npy"
    qwen_report = json.loads((qwen_dir / "report.json").read_text())
    qwen_provenance = json.loads((qwen_dir / "provenance.json").read_text())
    if (not qwen_report.get("pass") or qwen_provenance["prompt"] != generation["prompt"]
            or qwen_provenance["upstream_revision"] != UPSTREAM
            or qwen_provenance["reference_source_sha256"] != digest(Path(__file__).with_name("verify.py"))
            or qwen_provenance.get("capture_reader_sha256") != digest(captures.__file__)
            or qwen_provenance["qwen_hidden_sha256"] != digest(qwen_path)
            or any(value != generation["verified_components"][name]["sha256"]
                   for name, value in qwen_provenance["components"].items())):
        raise ValueError("independent Qwen reference provenance does not match generation")
    if out.exists():
        raise FileExistsError("reference output must be new")
    out.mkdir(parents=True)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    receipts = {}
    provenance = digest(manifest)
    def save(name, value, shared=False):
        if isinstance(value, torch.Tensor):
            value = value.float().cpu().numpy()
        path = captures.write_npy(out / (name + ".npy"), value, args.compress_output)
        receipts[name] = {"sha256": digest(path), "generation_sha256": provenance,
                          "upstream_revision": UPSTREAM, "shared_input": shared,
                          "shape": list(value.shape), "reference_source_sha256": digest(__file__),
                          "capture_reader_sha256": digest(captures.__file__), "storage": path.name}
        (out / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    def noise(name):
        meta = json.loads((native / (name + ".json")).read_text())
        value = captures.read_f32(native, name, math.prod(meta["shape"])).reshape(meta["shape"])
        save(name, value, True)
        tensor = torch.from_numpy(value).to(device)
        return tensor.reshape(-1, 32) if name == "noise_audio" else tensor
    video, audio = noise("noise_video"), noise("noise_audio")
    qwen = np.load(qwen_path, allow_pickle=False)
    save("qwen_hidden", qwen)
    text = torch.from_numpy(qwen.reshape(-1, 5120)).to(device, torch.bfloat16)
    with safe_open(model / "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", framework="pt", device="cpu") as weights:
        ref = Reference(weights, device)
        text = ref.refine(text)
        save("refined_text", text[None])
        rotation = ref.dit_rope(len(text), len(audio) // 2, video.shape[:3])
        points = generation["sigma_grid_points"]
        grid = 1 - torch.arange(points, dtype=torch.float32) / (points - 1)
        vs, aus = generation["video_shift"], generation["audio_shift"]
        sv, sa = vs * grid / (1 + (vs - 1) * grid), aus * grid / (1 + (aus - 1) * grid)
        for step in range(len(grid) - 1):
            vv, av = ref.denoise(text, video, audio, rotation, float(1 - sv[step]), float(1 - sa[step]), step + 1)
            video = video + (sv[step] - sv[step + 1]).to(device) * vv
            audio = audio + (sa[step] - sa[step + 1]).to(device) * av
            save(f"latent_video_{step:03d}", video)
            save(f"latent_audio_{step:03d}", audio[None])
        del text, rotation, vv, av, audio, ref
    torch.cuda.empty_cache()
    with safe_open(model / "vae/minimax_h3_video_vae_fp16.safetensors", framework="pt", device="cpu") as weights:
        ref = Reference(weights, device)
        for index, frame in enumerate(ref.decode(video, generation["frames"])):
            save(f"frame_{index:03d}", frame)
    print(json.dumps({"reference": "independent_pytorch", "captures": len(receipts),
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576}), flush=True)


if __name__ == "__main__":
    main()
