"""Stage immutable, pinned native profiles without changing another port's files."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from generate import atomic_json, digest

COMFY = ("Comfy-Org/HunyuanVideo_1.5_repackaged", "1405a2b738c29b890cf6d03fc038d8d3e92eeed7")
QWEN = ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5")
OFFICIAL = ("tencent/HunyuanVideo-1.5", "9b49404b3f5df2a8f0b31df27a0c7ab872e7b038")
VISION = ("google/siglip-so400m-patch14-384", "9fdffc58afc957d1a03a25b10dba0329ab15c2a3")
BYT5 = ("google/byt5-small", "68377bdc18a2ffec8a0533fef03b1c513a4dd49d")
CHECKPOINTS = {"quality_i2v": "hunyuanvideo1.5_480p_i2v_fp16.safetensors",
               "quality_t2v": "hunyuanvideo1.5_480p_t2v_fp16.safetensors",
               "fast12_i2v": "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors"}
FOLDERS = {"quality_i2v": "480p_i2v", "quality_t2v": "480p_t2v", "fast12_i2v": "480p_i2v_step_distilled"}


def stage(out, profiles, reuse=None, transport="hub"):
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    out = Path(out).resolve()
    reuse = Path(reuse).resolve() if reuse else None
    if reuse and (out.is_relative_to(reuse) or reuse.is_relative_to(out)):
        raise ValueError("reuse must be a separate read-only model directory")
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((reuse / "model.json").read_text()) if reuse else {}
    manifest = json.loads((out / "model.json").read_text()) if (out / "model.json").exists() else {
        "schema": "hunyuan_video15.model.v1", "checkpoints": {}, "components": {}, "sources": {}}
    manifest["vision_profile"] = "google_siglip_so400m_14_384"
    components = {"vae": "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
                  "qwen": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
                  "byt5": "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors",
                  "tokenizer": "tokenizer.json", "vision": "google_siglip/vision_fp16.safetensors"}
    configs = {"vae": "vae/config.json", "qwen": "config.json", "byt5": "byt5/config.json"}
    files = [(COMFY, components[k], components[k], "component model terms") for k in ("vae", "qwen", "byt5")]
    files += [(QWEN, "tokenizer.json", "tokenizer.json", "apache-2.0"),
              (QWEN, "config.json", "config.json", "apache-2.0"),
              (BYT5, "config.json", "byt5/config.json", "apache-2.0"),
              (OFFICIAL, "vae/config.json", "vae/config.json", "tencent-hunyuan-community")]
    for profile in profiles:
        name = "split_files/diffusion_models/" + CHECKPOINTS[profile]
        manifest["checkpoints"][profile] = name
        configs[profile] = "transformer/" + FOLDERS[profile] + "/config.json"
        files += [(COMFY, name, name, "tencent-hunyuan-community"),
                  (OFFICIAL, configs[profile], configs[profile], "tencent-hunyuan-community")]
    files += [(VISION, name, "google_siglip/" + name, "apache-2.0")
              for name in ("config.json", "preprocessor_config.json")]
    # The exported vision checkpoint is reused only after its receipt is verified.
    exported = out / components["vision"]
    export_receipt = manifest.get("sources", {}).get(components["vision"], {})
    export_valid = (exported.is_file() and export_receipt.get("repo") == VISION[0]
                    and export_receipt.get("revision") == VISION[1]
                    and export_receipt.get("bytes") == exported.stat().st_size
                    and export_receipt.get("sha256") == digest(exported))
    if not export_valid:
        receipt = old.get("sources", {}).get(components["vision"])
        source = reuse / components["vision"] if reuse else None
        if (not exported.exists() and receipt and receipt.get("repo") == VISION[0]
                and receipt.get("revision") == VISION[1] and source.is_file()
                and source.stat().st_size == receipt["bytes"] and digest(source) == receipt["sha256"]):
            exported.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, exported)
            manifest["sources"][components["vision"]] = receipt
            export_valid = True
        else:
            files.append((VISION, "model.safetensors", "google_siglip/model.safetensors", "apache-2.0"))
    catalogs = {pin: {x.rfilename: x for x in HfApi().model_info(*pin[:1], revision=pin[1], files_metadata=True).siblings}
                for pin in dict.fromkeys(x[0] for x in files)}
    remaining = 2 << 30
    for pin, remote, name, _ in files:
        reusable = reuse and (reuse / name).is_file() and name in old.get("sources", {})
        if not (out / name).is_file() and not reusable:
            remaining += catalogs[pin][remote].size
    if shutil.disk_usage(out).free < remaining:
        raise RuntimeError(f"need {remaining / 2**30:.1f} GiB free disk")
    for pin, remote, name, license_name in files:
        item = catalogs[pin][remote]
        path = out / name
        receipt = old.get("sources", {}).get(name, {})
        if not path.exists() and reuse and receipt.get("revision") == pin[1] and receipt.get("file") == remote:
            source = reuse / name
            if source.is_file() and source.stat().st_size == item.size and digest(source) == receipt.get("sha256"):
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, path)
        if not path.exists() or path.with_suffix(path.suffix + ".aria2").exists() or path.stat().st_size != item.size:
            if path.exists() and path.stat().st_nlink > 1:
                path.unlink()  # Never resume a download through a reused hardlink.
            if transport == "aria2":
                path.parent.mkdir(parents=True, exist_ok=True)
                command = ["aria2c", "--continue=true", "--file-allocation=none", "--split=16",
                           "--max-connection-per-server=16", "--min-split-size=8M", "--summary-interval=30",
                           "--console-log-level=warn", "--download-result=hide", "--auto-file-renaming=false",
                           f"--dir={path.parent}", f"--out={path.name}"]
                if item.lfs: command.append(f"--checksum=sha-256={item.lfs.sha256}")
                subprocess.run(command + [hf_hub_url(pin[0], remote, revision=pin[1]) + "?download=true"], check=True)
            else:
                folder = out if name == remote else out / Path(name).parent
                downloaded = Path(hf_hub_download(pin[0], remote, revision=pin[1], local_dir=folder))
                if downloaded != path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    downloaded.rename(path)
        sha = digest(path)
        if path.with_suffix(path.suffix + ".aria2").exists() or path.stat().st_size != item.size or (item.lfs and sha != item.lfs.sha256):
            raise ValueError(f"invalid pinned asset: {name}")
        manifest["sources"][name] = {"repo": pin[0], "revision": pin[1], "file": remote,
                                     "sha256": sha, "bytes": item.size, "license": license_name}
        atomic_json(out / "model.json", manifest)
        print(f"VERIFIED {name}", flush=True)
    if not export_valid:
        import numpy as np
        from safetensors import safe_open
        from safetensors.numpy import save_file
        tensors = {}
        with safe_open(str(out / "google_siglip/model.safetensors"), framework="np") as source:
            for key in source.keys():
                if key.startswith("vision_model."):
                    tensors[key] = source.get_tensor(key).astype(np.float16)
        partial = exported.with_suffix(exported.suffix + ".partial")
        save_file(tensors, str(partial))
        partial.replace(exported)
        manifest["sources"][components["vision"]] = {"repo": VISION[0], "revision": VISION[1],
            "conversion": "vision_model tensors only, F32 to F16", "license": "apache-2.0",
            "sha256": digest(exported), "bytes": exported.stat().st_size}
    manifest["components"].update(components)
    manifest.setdefault("reference_configs", {}).update(configs)
    atomic_json(out / "model.json", manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reuse", type=Path)
    ap.add_argument("--profiles", nargs="+", choices=CHECKPOINTS, default=list(CHECKPOINTS))
    ap.add_argument("--transport", choices=("hub", "aria2"), default="hub")
    args = ap.parse_args()
    # Confine Hugging Face scratch/cache to this port's own staging directory.
    os.environ["HF_HOME"] = str(args.out.resolve() / ".hf-cache")
    stage(args.out, args.profiles, args.reuse, args.transport)


if __name__ == "__main__":
    main()
