"""Explicit inference configuration and Python subprocess entry point."""
from pathlib import Path
from . import gpu

def add_arguments(parser):
    parser.add_argument("--backend", dest="inference_backend", choices=("auto", "cpu", "cuda", "rocm"), default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--models-root", default="/mnt/disk1/models")

def configure_args(args):
    import os
    root = Path(__file__).resolve().parents[2]
    temp = root / "tmp/vhuman-runtime"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp)
    cache = os.environ.setdefault("XDG_CACHE_HOME", str(root / "tmp/vhuman-cache"))
    Path(cache).mkdir(parents=True, exist_ok=True)
    gpu.configure(getattr(args, "inference_backend", "auto"), getattr(args, "device", 0),
                  getattr(args, "models_root", "/mnt/disk1/models"))
    for key, relative in (("sam3d_body_model", "sam3d-body"), ("sam3_model", "sam3/sam3.model.safetensors"),
                          ("clip_bpe", "clip-bpe"), ("tts_model", "speech/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
                          ("aligner", "speech/japanese-wav2vec2-large-hiragana-ctc/ja_align.safetensors")):
        if not getattr(args, key, None):
            setattr(args, key, str(gpu.model_path(relative)))
    selected = gpu.backend()
    defaults = {"qwen_python": "tmp/qimg21-ref-venv/bin/python",
                "rig_python": "tmp/vhuman-rig-venv/bin/python"}
    for key, relative in defaults.items():
        if not getattr(args, key, None):
            path = root / ("tmp/vhuman-rocm-venv/bin/python" if selected == "rocm" else relative)
            if path.exists():
                setattr(args, key, str(path))

def python_command(command):
    """Keep backend/device selection when a job starts a fresh interpreter."""
    if len(command) >= 3 and command[1] == "-m" and command[2].startswith("server.vhuman."):
        return [command[0], "-m", "server.vhuman.runtime", "--backend", gpu.backend(),
                "--device", str(gpu.device_index()), "--models-root", str(gpu.model_path("")),
                "--module", command[2], *command[3:]]
    return command

def torch_device(torch):
    selected = gpu.backend()
    if selected == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(f"{selected} PyTorch device unavailable")
    if bool(torch.version.hip) != (selected == "rocm"):
        raise RuntimeError(f"wrong PyTorch build for {selected}; select the matching interpreter")
    device = gpu.device_index()
    count = torch.cuda.device_count()
    if device >= count:
        raise RuntimeError(f"{selected} device {device} unavailable; detected {count} device(s)")
    torch.cuda.set_device(device)
    return f"cuda:{device}"

def main():
    import argparse, runpy, sys, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", dest="inference_backend", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--module", required=True)
    args, rest = parser.parse_known_args()
    if args.inference_backend == "rocm" and "/opt/rocm/core/lib" not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/rocm/core/lib:" + env.get("LD_LIBRARY_PATH", "")
        os.execve(sys.executable, [sys.executable, "-m", "server.vhuman.runtime", *sys.argv[1:]], env)
    configure_args(args)
    sys.argv = [args.module, *rest]
    runpy.run_module(args.module, run_name="__main__")

if __name__ == "__main__":
    main()
