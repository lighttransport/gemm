"""Synchronized Q8 projection costs for Wan attention and FFN matrix shapes."""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path
import torch
from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
from gguf import GGMLQuantizationType
from hip_runner import HipRunner, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    spec = importlib.util.spec_from_file_location("video_tools", ROOT / "cuda/hunyuan_video15_native/generate.py")
    video = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video)
    with video.device_lock(0), torch.inference_mode():
        torch.cuda.set_per_process_memory_fraction(.85)
        torch.manual_seed(42)
        runners = {name: HipRunner(gemm=name) for name in ("wmma", "blaslt")}
        results = []
        for m, n, k in [(226, 3072, 3072), (1170, 3072, 3072),
                        (1170, 14336, 3072), (1170, 3072, 14336),
                        (8190, 3072, 3072), (8190, 14336, 3072), (8190, 3072, 14336)]:
            blocks = n * k // 32
            scales = torch.rand(blocks, 1, device="cuda", dtype=torch.float16) * .001
            quant = torch.randint(-128, 128, (blocks, 32), device="cuda", dtype=torch.int8)
            packed = torch.cat((scales.view(torch.uint8), quant.view(torch.uint8)), dim=1)
            weight = GGUFParameter(packed.reshape(n, k // 32 * 34), quant_type=GGMLQuantizationType.Q8_0)
            del scales, quant, packed
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            bias = torch.randn(n, device="cuda", dtype=torch.float16)
            def reference():
                return torch.nn.functional.linear(x, dequantize_gguf_tensor(weight).half(), bias)
            funcs = {"pytorch": reference,
                     **{name: (lambda runner=runner: runner.linear(x, weight, bias))
                        for name, runner in runners.items()}}
            expected = reference()
            timings = {name: [] for name in funcs}
            errors = {}
            for name, func in funcs.items():
                actual = func()
                relative = float(torch.linalg.vector_norm(actual.float() - expected.float()) /
                                 torch.linalg.vector_norm(expected.float()))
                assert torch.isfinite(actual).all() and relative < .001, (name, relative)
                errors[name] = relative
                del actual
            del expected
            torch.cuda.synchronize()
            for repeat in range(args.repeats):
                order = list(funcs) if repeat % 2 == 0 else list(reversed(funcs))
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    actual = funcs[name]()
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end))
                    del actual
            result = {"m": m, "n": n, "k": k, "milliseconds": timings,
                      "median_ms": {name: statistics.median(values) for name, values in timings.items()},
                      "relative_l2": errors}
            results.append(result)
            print(json.dumps(result), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"scope": "Projection including Q8 decode, GEMM and bias; synthetic weights",
                                       "torch": torch.__version__, "rocm": torch.version.hip,
                                       "device": torch.cuda.get_device_name(), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
