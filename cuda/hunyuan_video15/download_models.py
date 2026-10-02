"""Download only a selected preset with pinned sources and SHA256 receipts.

Requires huggingface_hub, safetensors and numpy. Uses public Google SigLIP.
The manifest identifies this alternate vision component explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, hf_hub_url, HfApi

COMFY = ("Comfy-Org/HunyuanVideo_1.5_repackaged", "1405a2b738c29b890cf6d03fc038d8d3e92eeed7")
QWEN = ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5")
OFFICIAL = ("tencent/HunyuanVideo-1.5", "9b49404b3f5df2a8f0b31df27a0c7ab872e7b038")
VISION = ("google/siglip-so400m-patch14-384", "9fdffc58afc957d1a03a25b10dba0329ab15c2a3")
CHECKPOINTS = {
    "quality_i2v": "hunyuanvideo1.5_480p_i2v_fp16.safetensors",
    "quality_t2v": "hunyuanvideo1.5_480p_t2v_fp16.safetensors",
    "fast12_i2v": "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors",
}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", choices=CHECKPOINTS, default="fast12_i2v")
    ap.add_argument("--reference-configs", action="store_true", help="also stage pinned official VAE/DiT and Qwen reference configs")
    ap.add_argument("--transport", choices=("hub", "aria2"), default="hub")
    ap.add_argument("--parallel-files", type=int, choices=(1, 2, 3), default=1)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = [
        (VISION, "model.safetensors", "apache-2.0"),
        (VISION, "config.json", "apache-2.0"),
        (VISION, "preprocessor_config.json", "apache-2.0"),
        (COMFY, "split_files/vae/hunyuanvideo15_vae_fp16.safetensors", "tencent-hunyuan-community"),
        (COMFY, "split_files/text_encoders/qwen_2.5_vl_7b.safetensors", "apache-2.0"),
        (COMFY, "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", "Glyph-SDXL-v2; verify component terms"),
        (QWEN, "tokenizer.json", "apache-2.0"),
        (COMFY, "split_files/diffusion_models/" + CHECKPOINTS[args.checkpoint], "tencent-hunyuan-community"),
    ]
    reference_configs = {}
    if args.reference_configs:
        task_folder = {"quality_i2v": "480p_i2v", "quality_t2v": "480p_t2v",
                       "fast12_i2v": "480p_i2v_step_distilled"}[args.checkpoint]
        reference_configs = {"vae": "vae/config.json", "qwen": "config.json",
                             args.checkpoint: f"transformer/{task_folder}/config.json"}
        files.extend([(OFFICIAL, reference_configs["vae"], "tencent-hunyuan-community"),
                      (OFFICIAL, reference_configs[args.checkpoint], "tencent-hunyuan-community"),
                      (QWEN, reference_configs["qwen"], "apache-2.0")])
    existing = out / "model.json"
    manifest = json.loads(existing.read_text()) if existing.exists() else {
        "schema": "hunyuan_video15.model.v1", "checkpoints": {}, "components": {}, "sources": {}}
    catalogs = {}
    for repo, revision in dict.fromkeys(item[0] for item in files):
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        catalogs[repo, revision] = {item.rfilename: item for item in info.siblings}
    required_free = 2 * 1024**3  # headroom for conversion and packaged clips
    for (repo, revision), filename, _ in files:
        local = out / "google_siglip" if repo == VISION[0] else out
        path = local / filename
        # aria2 files are sparse. Allocated blocks, not logical size, measure
        # consumed storage during resume. Completed small files are bounded too.
        allocated = path.stat().st_blocks * 512 if path.exists() else 0
        required_free += max(0, catalogs[repo, revision][filename].size - allocated)
    if not (out / "google_siglip/vision_fp16.safetensors").exists():
        required_free += 900 * 1024**2
    if shutil.disk_usage(out).free < required_free:
        raise RuntimeError(f"allow {required_free / 1024**3:.1f} GiB free disk to stage the remaining profile")
    def fetch(repo, revision, filename):
        local = out / "google_siglip" if repo == VISION[0] else out
        if args.transport == "hub":
            return Path(hf_hub_download(repo, filename, revision=revision, local_dir=local))
        if not shutil.which("aria2c"):
            raise RuntimeError("aria2c is required for --transport aria2")
        item = catalogs[repo, revision][filename]
        path = local / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = item.lfs.sha256 if item.lfs else None
        if path.exists() and path.stat().st_size == item.size and not path.with_suffix(path.suffix + ".aria2").exists():
            if not expected_hash or digest(path) == expected_hash:
                return path
        cmd = ["aria2c", "--continue=true", "--file-allocation=none", "--split=16",
            "--max-connection-per-server=16", "--min-split-size=8M", "--summary-interval=30",
            "--console-log-level=warn", "--download-result=hide", "--auto-file-renaming=false",
            f"--dir={path.parent}", f"--out={path.name}"]
        if expected_hash:
            cmd.append(f"--checksum=sha-256={expected_hash}")
        cmd.append(hf_hub_url(repo, filename, revision=revision) + "?download=true")
        subprocess.run(cmd, check=True)
        if path.stat().st_size != item.size or (expected_hash and digest(path) != expected_hash):
            raise RuntimeError(f"download checksum/size mismatch: {filename}")
        return path
    def stage(item):
        (repo, revision), filename, license_name = item
        print(f"Downloading {repo}/{filename}", flush=True)
        path = fetch(repo, revision, filename)
        return str(path.relative_to(out)), {"repo": repo, "revision": revision, "file": filename,
            "license": license_name, "sha256": digest(path), "bytes": path.stat().st_size}
    with ThreadPoolExecutor(max_workers=args.parallel_files) as workers:
        for name, receipt in workers.map(stage, files):
            manifest["sources"][name] = receipt
    from export_siglip import export
    vision = export(out / "google_siglip")
    manifest["vision_profile"] = "google_siglip_so400m_14_384"
    manifest["sources"][str(vision.relative_to(out))] = {"repo": VISION[0], "revision": VISION[1],
        "conversion": "vision_model tensors only, F32 to F16", "license": "apache-2.0",
        "sha256": digest(vision), "bytes": vision.stat().st_size}
    manifest["components"] = {"vision": "google_siglip/vision_fp16.safetensors",
        "vae": "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
        "qwen": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
        "byt5": "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", "tokenizer": "tokenizer.json"}
    manifest["checkpoints"][args.checkpoint] = "split_files/diffusion_models/" + CHECKPOINTS[args.checkpoint]
    if reference_configs:
        manifest.setdefault("reference_configs", {}).update(reference_configs)
    partial = out / "model.json.partial"
    partial.write_text(json.dumps(manifest, indent=2) + "\n")
    partial.replace(existing)

if __name__ == "__main__":
    main()
