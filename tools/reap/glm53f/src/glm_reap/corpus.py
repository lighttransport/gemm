from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import copy
import io
from contextlib import contextmanager

from .common import atomic_json, require_disk, GIB

SOURCES = {
    "instructions": ("theblackcat102/evol-codealpaca-v1", None),
    "code": ("bigcode/the-stack-smol-xs", None),
    "trajectories": ("nebius/SWE-rebench-openhands-trajectories", None),
    "general": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
}

STACK_REVISION = "1e3dd39b39787bddb20d7008e4d71c330d99f55b"


def stack_files(config):
    revision = config.get("code_revision", STACK_REVISION)
    # hf:// paths are filesystem names, not percent-encoded HTTP URLs.
    # datasets converts HTTP resolve URLs to hf:// without decoding c%2B%2B.
    return [f"hf://datasets/bigcode/the-stack-smol-xs@{revision}/data/{language}/data.json" for language in config["code_languages"]]


def load_source(repo, subset, kind, config):
    from datasets import load_dataset
    if kind == "code" and repo == "bigcode/the-stack-smol-xs":
        # This repository's loader is a legacy Python script. Its data files
        # are JSONL, so use the supported built-in JSON reader directly.
        files = stack_files(config)
        return load_dataset("json", data_files={"train": files}, split="train", streaming=True)
    if kind == "trajectories":
        return trajectory_rows(repo, subset)
    return load_dataset(repo, subset, split="train", streaming=True)


def trajectory_rows(repo, subset):
    """Read remote Parquet synchronously; no Arrow dataset scanner threads."""
    from datasets import load_dataset_builder
    import fsspec
    import pyarrow.parquet as parquet
    builder = load_dataset_builder(repo, subset)
    files = builder.config.data_files.get("train")
    if not files:
        raise ValueError(f"No train data files found in {repo}")
    for filename in files:
        if not str(filename).endswith(".parquet"):
            raise ValueError(f"Expected Parquet trajectory data: {filename}")
        with fsspec.open(str(filename), "rb", block_size=64*1024).open() as stream:
            reader = parquet.ParquetFile(stream, pre_buffer=False, buffer_size=64*1024)
            try:
                available = reader.schema_arrow.names
                columns = [name for name in ("trajectory", "tools", "trajectory_id", "instance_id", "repo") if name in available]
                if "trajectory" not in columns:
                    raise ValueError(f"Trajectory column missing: {filename}")
                for batch in reader.iter_batches(batch_size=8, columns=columns, use_threads=False):
                    yield from batch.to_pylist()
            finally:
                reader.close()


@contextmanager
def source_stream(repo, subset, kind, config):
    rows = iter(load_source(repo, subset, kind, config))
    try:
        yield rows
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            close()


def normalize(row, kind):
    if kind == "instructions":
        return [{"role": "user", "content": row["instruction"]}, {"role": "assistant", "content": row["output"]}]
    if kind == "code":
        return [{"role": "user", "content": "Continue this source file."}, {"role": "assistant", "content": row["content"]}]
    if kind == "general":
        return [{"role": "user", "content": "Continue this passage."}, {"role": "assistant", "content": row["text"]}]
    if kind == "vision":
        messages = [message for turn in row["texts"] for message in ({"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["assistant"]})]
        marker = "<|begin_of_image|><|image|><|end_of_image|>"
        if messages:
            if "<image>" in messages[0]["content"]:
                messages[0]["content"] = messages[0]["content"].replace("<image>", marker)
            elif "<|image|>" not in messages[0]["content"]:
                messages[0]["content"] = marker*len(row["images"])+messages[0]["content"]
        return messages
    messages = row["trajectory"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    if isinstance(messages, dict):
        messages = messages.get("messages", [])
    result = []
    for m in messages:
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            continue
        m = copy.deepcopy(m)
        if m.get("tool_calls") is None:
            m.pop("tool_calls", None)
        for call in m.get("tool_calls", []):
            args = call.get("function", {}).get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
        if m.get("content") is None:
            m["content"] = ""
        result.append(m)
    return result


def answer_tokens(tokenizer, messages, tools=None):
    """Mask assistant spans using offsets, without generation template tags."""
    options = {"tokenize": False, "add_generation_prompt": False, "reasoning_effort": "low"}
    if tools:
        options["tools"] = tools
    text = tokenizer.apply_chat_template(messages, **options)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = tokenizer.apply_chat_template(messages[:index], **{**options, "add_generation_prompt": True})
        after = tokenizer.apply_chat_template(messages[:index+1], **options)
        if not text.startswith(before) or not text.startswith(after):
            raise ValueError("Chat template prefixes changed; cannot safely construct answer mask")
        spans.append((len(before), len(after)))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = [any(a <= start and end <= b and end > start for a, b in spans) for start, end in encoded["offset_mapping"]]
    return encoded["input_ids"], mask


def corpus_manifest(config, dest, budget_reached=False):
    totals = {}
    for filename in dest.glob("*.jsonl"):
        with filename.open() as stream:
            totals[filename.stem] = sum(len(json.loads(line)["input_ids"]) for line in stream)
    atomic_json(dest / "manifest.json", {"seed": config["seed"], "tokens": totals, "sources": SOURCES, "code_revision": config.get("code_revision", STACK_REVISION), "code_languages": config["code_languages"], "vision": config["vision_subsets"], "budget_reached": budget_reached})
    if budget_reached:
        print("Corpus byte budget reached; saved records are ready to use. See manifest.json for achieved token counts.")


def download(config, destination):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file() and p.name != "manifest.json")
    byte_limit = int(config["corpus_limit_gib"]*GIB)
    if size >= byte_limit:
        corpus_manifest(config, dest, budget_reached=True)
        return
    require_disk(dest, byte_limit-size, config["reserve_disk_gib"]*GIB)
    tokenizer = AutoTokenizer.from_pretrained(config["source"], local_files_only=True)
    totals = {}
    # Each category has its own file and cursor. Restart skips existing records.
    for kind, fraction in config["corpus_mix"].items():
        target = int(config["corpus_tokens"]*fraction)
        filename = dest / f"{kind}.jsonl"
        records = [json.loads(line) for line in filename.open()] if filename.exists() else []
        seen = {r["id"] for r in records}
        tokens = sum(len(r["input_ids"]) for r in records)
        if tokens >= target:
            totals[kind] = tokens
            continue
        subsets = config["vision_subsets"] if kind == "vision" else [SOURCES[kind][1]]
        with filename.open("a") as stream:
            for subset in subsets:
                repo = "HuggingFaceM4/the_cauldron" if kind == "vision" else SOURCES[kind][0]
                quota = target // len(subsets)
                subset_tokens = sum(len(r["input_ids"]) for r in records if r["subset"] == subset)
                if subset_tokens >= quota:
                    continue
                with source_stream(repo, subset, kind, config) as data:
                    for row in data:
                        if kind == "code" and row.get("lang", row.get("language", "")).lower() not in config["code_languages"]:
                            continue
                        stable = {k: v for k, v in row.items() if k != "images"}
                        if kind == "vision":
                            stable["image_hashes"] = [hashlib.sha256(i.convert("RGB").tobytes()).hexdigest() for i in row["images"]]
                        ident = hashlib.sha256(json.dumps(stable, default=str, sort_keys=True).encode()).hexdigest()
                        if ident in seen:
                            continue
                        messages = normalize(row, kind)
                        tools = row.get("tools") if kind == "trajectories" else None
                        if isinstance(tools, str):
                            tools = json.loads(tools)
                        ids, mask = answer_tokens(tokenizer, messages, tools)
                        images = []
                        image_payloads = {}
                        if kind == "vision":
                            for image_index, image in enumerate(row["images"]):
                                image_dir = dest / "images"
                                image_dir.mkdir(exist_ok=True)
                                name = f"{ident}-{image_index}.png"
                                encoded_image = io.BytesIO()
                                image.convert("RGB").save(encoded_image, format="PNG")
                                image_payloads[image_dir / name] = encoded_image.getvalue()
                                images.append("images/"+name)
                        record = {"id": ident, "kind": kind, "subset": subset, "messages": messages, "tools": tools, "images": images, "input_ids": ids, "answer_mask": mask, "split": "heldout" if (int(ident[:8], 16)+config["seed"]) % 10 == 0 else "train"}
                        payload = json.dumps(record)+"\n"
                        added = len(payload.encode("utf-8")) + sum(len(data) for path, data in image_payloads.items() if not path.exists())
                        if size + added > byte_limit:
                            corpus_manifest(config, dest, budget_reached=True)
                            return
                        for path, data in image_payloads.items():
                            if not path.exists():
                                path.write_bytes(data)
                        stream.write(payload)
                        size += added
                        stream.flush()
                        seen.add(ident)
                        tokens += len(ids)
                        subset_tokens += len(ids)
                        if tokens >= target or subset_tokens >= quota:
                            break
                if tokens >= target:
                    break
        totals[kind] = tokens
    corpus_manifest(config, dest)


def windows(destination, length, split="train", limit=None, config=None):
    records = []
    counts = {}
    for filename in sorted(Path(destination).glob("*.jsonl")):
        with filename.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["split"] != split:
                    continue
                kind = row["kind"]
                if row["images"]:
                    if config is None:
                        raise ValueError("Vision windows require processor config")
                    if len(row["input_ids"]) <= length:
                        records.append((kind, row))
                        counts[kind] = counts.get(kind, 0) + 1
                    continue
                for start in range(0, len(row["input_ids"])-1, length):
                    ids = row["input_ids"][start:start+length]
                    mask = row["answer_mask"][start:start+length]
                    if len(ids) < 16 or not any(mask[1:]):
                        continue
                    records.append((kind, (ids, mask)))
                    counts[kind] = counts.get(kind, 0) + 1
    rng = random.Random(config.get("seed", 42) if config else 42)
    if config and config.get("corpus_mix"):
        import math
        # Weight windows by category, so short image examples do not dominate.
        records.sort(key=lambda item: -math.log(max(rng.random(), 1e-15)) * counts[item[0]] / config["corpus_mix"][item[0]])
    else:
        rng.shuffle(records)
    processor = None
    emitted = 0
    for kind, record in records:
        if limit is not None and emitted >= limit:
            break
        if isinstance(record, dict):
            if processor is None:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(config["source"], local_files_only=True)
            ids, mask, vision = vision_window(record, destination, config, processor=processor)
            if len(ids) > length or not any(mask[1:]):
                continue
            record = (ids, mask, vision)
        emitted += 1
        yield record


def vision_window(row, destination, config, processor=None):
    from PIL import Image
    from transformers import AutoProcessor
    if processor is None:
        processor = AutoProcessor.from_pretrained(config["source"], local_files_only=True)
    images = [Image.open(Path(destination)/name).convert("RGB") for name in row["images"]]
    text = processor.tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False, reasoning_effort="low")
    encoded = processor(text=text, images=images, return_tensors="pt", max_image_tokens=config["image_tokens"])
    ids = encoded["input_ids"][0].tolist()
    old, mask = row["input_ids"], row["answer_mask"]
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image|>")
    # Processor expands only placeholder token runs. Verify every other token.
    expanded, i, j = [], 0, 0
    while i < len(old) and j < len(ids):
        if old[i] == image_id:
            while j < len(ids) and ids[j] == image_id:
                expanded.append(False)
                j += 1
            i += 1
        else:
            if old[i] != ids[j]:
                raise ValueError("Processor changed a non-image token; answer mask would be invalid")
            expanded.append(mask[i])
            i, j = i+1, j+1
    if i != len(old) or j != len(ids):
        raise ValueError("Image expansion/token mask lengths disagree")
    return ids, expanded, {k: encoded[k] for k in ("pixel_values", "image_grid_thw")}
