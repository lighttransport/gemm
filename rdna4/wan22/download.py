"""Download only Q8_0 DiT and official ancillary weights at pinned revisions."""
import argparse
import json
from pathlib import Path
import concurrent.futures
import hashlib
import os
import threading
import time


def download_large(repo, revision, name, destination, workers):
    """Resume bounded HTTP ranges, then verify the Hub's content SHA-256."""
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    url = hf_hub_url(repo, name, revision=revision)
    metadata = get_hf_file_metadata(url)
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    progress = target.with_name(target.name + ".ranges.json")
    chunk = 64 << 20
    identity = {"size": metadata.size, "sha256": metadata.etag, "chunk": chunk}
    def digest(path):
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    if target.is_file() and target.stat().st_size == metadata.size and digest(target) == metadata.etag:
        return {"path": str(target), "bytes": metadata.size, "sha256": metadata.etag}
    completed = set()
    if partial.exists() and progress.exists():
        saved = json.loads(progress.read_text())
        if all(saved.get(key) == value for key, value in identity.items()):
            completed = set(saved["completed"])
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, metadata.size)
    lock = threading.Lock()
    total = (metadata.size + chunk - 1) // chunk
    def transfer(index):
        start = index * chunk
        end = min(metadata.size, start + chunk) - 1
        for attempt in range(5):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{metadata.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server did not honor requested byte range")
                    offset = start
                    for data in response.iter_content(1 << 20):
                        if offset + len(data) > end + 1:
                            raise RuntimeError("Oversized range response")
                        view = memoryview(data)
                        while view:
                            written = os.pwrite(fd, view, offset)
                            if written <= 0:
                                raise OSError("Short disk write")
                            offset += written
                            view = view[written:]
                    if offset != end + 1:
                        raise RuntimeError("Truncated range response")
                with lock:
                    os.fsync(fd)
                    completed.add(index)
                    stage = progress.with_suffix(".new")
                    stage.write_text(json.dumps({**identity, "completed": sorted(completed)}))
                    stage.replace(progress)
                    print(f"{name}: {len(completed)}/{total} ranges", flush=True)
                return
            except (requests.RequestException, RuntimeError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(transfer, [i for i in range(total) if i not in completed]))
    finally:
        os.close(fd)
    if digest(partial) != metadata.etag:
        progress.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch: {name}; retry to replace partial ranges")
    partial.replace(target)
    progress.unlink(missing_ok=True)
    print(f"Verified {name}: {metadata.etag}", flush=True)
    return {"path": str(target), "bytes": metadata.size, "sha256": metadata.etag}

GGUF_REPO = "QuantStack/Wan2.2-TI2V-5B-GGUF"
GGUF_REV = "57437632ddd08bdcbd1508c866aa22e126ed51d2"
PIPE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
PIPE_REV = "b8fff7315c768468a5333511427288870b2e9635"


def main():
    from huggingface_hub import snapshot_download, HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/mnt/disk01/models/wan22"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    files = [download_large(GGUF_REPO, GGUF_REV, "Wan2.2-TI2V-5B-Q8_0.gguf", args.model / "gguf", args.workers)]
    snapshot_download(PIPE_REPO, revision=PIPE_REV,
                      allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                                      "text_encoder/*.json", "transformer/config.json", "vae/*.json"],
                      local_dir=args.model / "pipeline")
    for item in HfApi().list_repo_tree(PIPE_REPO, revision=PIPE_REV, recursive=True):
        if item.path.endswith(".safetensors") and item.path.startswith(("text_encoder/", "vae/")):
            files.append(download_large(PIPE_REPO, PIPE_REV, item.path, args.model / "pipeline", args.workers))
    receipt = {"gguf": {"repo": GGUF_REPO, "revision": GGUF_REV},
               "pipeline": {"repo": PIPE_REPO, "revision": PIPE_REV}, "files": files}
    (args.model / "download.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
