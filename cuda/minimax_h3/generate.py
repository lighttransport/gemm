"""Native MiniMax H3 generation with the CUDA backend selected by default."""
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("h3_video_tools", ROOT / "rdna4/minimax_h3/generate.py")
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)

if __name__ == "__main__":
    h3.main(default_backend="cuda")
