"""Native HunyuanVideo generation with the ROCm backend selected by default."""
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hv15_video_tools", ROOT / "cuda/hunyuan_video15_native/generate.py")
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)

if __name__ == "__main__":
    video.main(default_backend="rocm")
