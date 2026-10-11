"""Export GenIA's separate SLAT conditioning fuser from existing checkpoints.

The existing native sam3d_cond_fuser is exported from SS, not SLAT. This tool
does not overwrite it or download weights. Defaults write into repo tmp/.
Use the offline Torch/safetensors interpreter; production needs neither package.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "_base_models.condition_embedder."


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=Path("/mnt/nvme01/models/sam3d/checkpoints"))
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/genia-native-models")
    args = parser.parse_args()
    import torch
    def fuser(name):
        state = torch.load(args.checkpoints / name, map_location="cpu", weights_only=True, mmap=True)
        state = state.get("state_dict", state)
        return {key[len(PREFIX):]: value for key, value in state.items()
                if key.startswith(PREFIX) and ("projection_nets." in key or key == PREFIX + "idx_emb")}
    ss = fuser("ss_generator.ckpt")
    slat = fuser("slat_generator.ckpt")
    if not slat or "idx_emb" not in slat or len(slat) != 11:
        raise ValueError("unexpected SLAT fuser layout; do not substitute SS weights")
    shared = sorted(set(ss) & set(slat))
    same = [key for key in shared if torch.equal(ss[key], slat[key])]
    tensors = {key: value.float().contiguous() for key, value in slat.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "sam3d_slat_cond_fuser.safetensors"
    if path.exists():
        raise ValueError("refusing to overwrite an existing conditioning export")
    partial = path.with_suffix(".partial")
    # The restricted native export contains only contiguous little-endian F32.
    # Write the standard safetensors header/offset layout without changing the
    # caller's offline environment to install another serialization package.
    header = {"__metadata__": {"source": "slat_generator.ckpt", "component": "slat_condition_embedder"}}
    offset = 0
    arrays = {}
    for key, tensor in tensors.items():
        array = tensor.numpy().astype("<f4", copy=False)
        arrays[key] = array
        header[key] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset+array.nbytes]}
        offset += array.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with partial.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for array in arrays.values():
            array.tofile(stream)
    partial.rename(path)
    manifest = {"format": "gemm.genia_conditioning_export.v1", "torch": torch.__version__,
                "source": str(args.checkpoints / "slat_generator.ckpt"),
                "source_sha256": digest(args.checkpoints / "slat_generator.ckpt"),
                "file": path.name, "sha256": digest(path),
                "shared_keys": len(shared), "identical_shared_keys": same,
                "tensors": {key: list(value.shape) for key, value in tensors.items()}}
    (args.output / "conditioning.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(path), "bytes": path.stat().st_size,
                      "shared_keys": len(shared), "identical_shared_keys": len(same)}))


if __name__ == "__main__":
    main()
