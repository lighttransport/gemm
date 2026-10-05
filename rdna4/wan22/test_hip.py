"""Independent Q8_0 decode + PyTorch GEMM comparison on the RX 9070 XT."""
import torch
from gguf import GGMLQuantizationType
from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
from hip_runner import HipRunner


def main():
    runner = HipRunner()
    torch.manual_seed(42)
    for m, n, k in [(1, 17, 32), (17, 67, 96), (129, 257, 256), (256, 256, 1024)]:
        blocks = n * k // 32
        scales = torch.rand(blocks, 1, dtype=torch.float16) * .025
        quant = torch.randint(-128, 128, (blocks, 32), dtype=torch.int8)
        packed = torch.cat((scales.view(torch.uint8), quant.view(torch.uint8)), dim=1)
        packed = packed.reshape(n, k // 32 * 34).cuda()
        weight = GGUFParameter(packed, quant_type=GGMLQuantizationType.Q8_0)
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        bias = torch.randn(n, device="cuda", dtype=torch.float16)
        # Exercise nondefault stream and tails; reference decodes independently.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = runner.linear(x, weight, bias)
            dense = dequantize_gguf_tensor(weight).half()
            expected = torch.nn.functional.linear(x, dense) + bias
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual, expected, rtol=.002, atol=.03125)
        rel = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(expected.float())
        assert rel < .001, rel
        print(f"PASS {m}x{n}x{k}: relative L2={float(rel):.7f}")
    assert runner.calls == 4
    # Exercise the actual Wan graph's normalization, expanded timestep, RoPE,
    # self/cross-attention and FFN around quantized projections.
    from diffusers import WanTransformer3DModel
    from diffusers.quantizers.gguf.utils import GGUFLinear
    model = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=32,
                                  in_channels=48, out_channels=48, text_dim=64,
                                  freq_dim=32, ffn_dim=128, num_layers=2).half()
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, torch.nn.Linear) or name.startswith("condition_embedder"):
            continue
        dense = layer.weight.detach().reshape(-1, 32).float()
        scale = (dense.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8).half()
        quant = (dense / scale.float()).round().clamp(-127, 127).to(torch.int8)
        packed = torch.cat((scale.view(torch.uint8), quant.view(torch.uint8)), dim=1)
        packed = packed.reshape(layer.out_features, layer.in_features // 32 * 34)
        replacement = GGUFLinear(layer.in_features, layer.out_features,
                                  layer.bias is not None, compute_dtype=torch.float16)
        replacement.weight = GGUFParameter(packed, quant_type=GGMLQuantizationType.Q8_0)
        replacement.bias = layer.bias
        parent_name, _, key = name.rpartition(".")
        setattr(model.get_submodule(parent_name), key, replacement)
    model = model.cuda()
    noise = torch.randn(1, 48, 2, 4, 4, device="cuda", dtype=torch.float16)
    text = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)
    timestep = torch.full((1, 8), 500., device="cuda")
    with torch.inference_mode():
        expected = model(noise, timestep, text).sample
        installed = runner.install(model)
        actual = model(noise, timestep, text).sample
    rel = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(expected.float())
    torch.testing.assert_close(actual, expected, rtol=.005, atol=.005)
    assert rel < .002, rel
    print(f"PASS Wan graph: {installed} HIP modules, relative L2={float(rel):.7f}")


if __name__ == "__main__":
    main()
