from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
import sys

import numpy as np

from .checkpoint import expert_key, group_name, protected
from .common import atomic_json, require_disk, GIB, fingerprint
from .native import FORMATS


def exporter(config, checkpoint):
    import torch
    root = Path(config["llama_cpp"])
    sys.path[:0] = [str(root), str(root / "gguf-py")]
    import gguf
    from conversion import get_model_class
    cls = get_model_class("Glm5NextForConditionalGeneration")
    class Mapper(cls):
        model_arch = cls.model_arch
        no_mtp = True
        def index_tensors(self, remote_hf_model_id=None):
            return {}
    hp = copy.deepcopy(checkpoint.config)
    hp["text_config"]["n_routed_experts"] = config["keep_experts"]
    hp["text_config"]["num_nextn_predict_layers"] = 0
    mapper = Mapper(checkpoint.root, gguf.LlamaFileType.MOSTLY_Q2_K, Path(config["output"])/"glm53f-reap-gsq-rco.gguf", hparams=hp, eager=True)
    return mapper, gguf


def records(config, checkpoint, bank, selected, mapper):
    import torch
    result = []
    grouped = {}
    for source, candidates in sorted(bank.tensors.items()):
        chosen = candidates[bank.selection.get(group_name(source), 0)]
        key = expert_key(source)
        if key:
            grouped.setdefault((key[0], key[2]), {})[key[1]] = chosen
            continue
        name = source.replace("model.language_model.", "model.")
        layer_match = re.search(r"\.layers\.(\d+)\.", name)
        layer = int(layer_match[1]) if layer_match else None
        if protected(source, chosen.shape):
            value = bank.read(source)
            for canonical, converted in mapper.modify_tensors(torch.from_numpy(value), name, layer):
                data = converted.detach().cpu().numpy()
                kind = "F16" if "kv_b_proj" in source or "conv1d" in source else "F32"
                raw = np.ascontiguousarray(data, dtype=np.float16 if kind == "F16" else np.float32).view(np.uint8).reshape(-1)
                result.append((canonical, tuple(data.shape), kind, [raw]))
        else:
            # Mapping is one-to-one. Packed bytes bypass floating quantization.
            canonical = mapper.map_tensor_name(name)
            result.append((canonical, chosen.shape, chosen.qtype, [chosen.raw]))
    for (layer, projection), experts in sorted(grouped.items()):
        ids = selected[str(layer)]
        entries = [experts[i] for i in ids]
        if len({e.qtype for e in entries}) != 1:
            raise ValueError("All experts in a fused projection must share a native format")
        canonical = mapper.map_tensor_name(f"model.layers.{layer}.mlp.experts.{projection}.weight")
        result.append((canonical, (len(ids), *entries[0].shape), entries[0].qtype, [e.raw for e in entries]))
    if len({r[0] for r in result}) != len(result):
        raise ValueError("Duplicate canonical tensor name")
    return sorted(result)


def write(config, checkpoint, bank, selected):
    from .quality import selected_quality
    selected_quality(config, checkpoint, bank, selected)
    bank.retain_selected()
    mapper, gguf = exporter(config, checkpoint)
    tensors = records(config, checkpoint, bank, selected, mapper)
    writer = mapper.gguf_writer
    for name, shape, kind, chunks in tensors:
        writer.add_tensor_info(name, shape, np.dtype(np.float32), sum(c.nbytes for c in chunks), raw_dtype=gguf.GGMLQuantizationType(FORMATS[kind][0]))
    mapper.prepare_metadata(vocab_only=False)
    mapper.gguf_writer.add_string("compression.method", "REAP/native-format-GSQ/task-loss-RCO")
    payload = sum(c.nbytes for _, _, _, chunks in tensors for c in chunks)
    target = mapper.fname_out
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    temporary = target.with_suffix(".gguf.partial")
    from .resume import write_payload
    identity = fingerprint([checkpoint.identity(), config, selected, bank.original_selection])
    hashes = write_payload(writer, tensors, temporary, Path(config.get("work_dir", config["output"]))/"export-state.json", identity, config["reserve_disk_gib"]*GIB)
    if temporary.stat().st_size > config["weight_budget_bytes"]:
        raise RuntimeError(f"GGUF exceeds strict byte cap: {temporary.stat().st_size}")
    reader = gguf.GGUFReader(temporary)
    for tensor in reader.tensors:
        if hashlib.sha256(memoryview(np.ascontiguousarray(tensor.data).view(np.uint8))).hexdigest() != hashes[tensor.name]:
            raise RuntimeError(f"Serialized native candidate mismatch: {tensor.name}")
    del reader
    atomic_json(target.with_suffix(".manifest.json"), {"source": fingerprint(checkpoint.identity()), "config": fingerprint(config), "file_bytes": temporary.stat().st_size, "payload_bytes": payload, "sha256_tensors": hashes, "selection": bank.original_selection, "retained_experts": selected})
    temporary.replace(target)
    return str(target)
