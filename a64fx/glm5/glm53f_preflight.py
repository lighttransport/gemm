#!/usr/bin/env python3
"""Metadata-only GLM5.3 Flash preflight for a Fugaku checkout.

Never opens a safetensors payload. It validates the shard/index inventory,
prints the architecture fields needed by the A64FX graph, and checks that the
tokenizer and Jinja chat template have the expected GLM5.3 markers.
"""
import argparse
import json
import os
import struct
import sys


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def read_safetensors_header(path):
    """Read only the safetensors header; never maps or reads tensor payloads."""
    with open(path, "rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("short safetensors header: %s" % path)
        size = struct.unpack("<Q", raw)[0]
        if size > 64 * 1024 * 1024:
            raise ValueError("unreasonable safetensors header: %s" % path)
        header = stream.read(size)
        if len(header) != size:
            raise ValueError("truncated safetensors header: %s" % path)
    return json.loads(header)


def check_tensor_schema(root, shards, weight_map, text):
    """Validate the name-level contract needed by the future GLM5.3F loader.

    This intentionally checks headers only.  It catches accidentally supplied
    GLM5.2/GGUF layouts before an expensive multi-node launch, while leaving
    numerical graph validation to the A64FX runner tests.
    """
    tensors = {}
    for shard in shards:
        header = read_safetensors_header(os.path.join(root, shard))
        for name, desc in header.items():
            if name != "__metadata__":
                tensors[name] = desc

    prefix = "model.language_model."
    required = {
        prefix + "embed_tokens.weight",
        prefix + "norm.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        prefix + "layers.0.input_layernorm.weight",
        prefix + "layers.3.self_attn.indexer.k_norm.weight",
        prefix + "layers.3.mlp.gate.weight",
    }
    # The layer-0 linear-attention names and layer-3 sparse-attention names
    # distinguish GLM5.3F from the legacy all-MLA GLM5.2 implementation.
    required.discard("model.layers.0.input_layernorm.weight")
    missing = sorted(name for name in required if name not in tensors)
    layer_names = [name for name in tensors if name.startswith(prefix + "layers.")]
    layer_ids = set()
    for name in layer_names:
        rest = name[len(prefix + "layers."):]
        try:
            layer_ids.add(int(rest.split(".", 1)[0]))
        except (ValueError, IndexError):
            pass
    expected_layers = int(text.get("num_hidden_layers", 0))
    # GLM5.3F stores one auxiliary next-token/MTP layer directly after the
    # 45 decoder layers.  It is part of the checkpoint inventory but not part
    # of the main layer_types list.
    expected_ids = set(range(expected_layers + 1))
    errors = []
    if missing:
        errors.append("missing tensors: " + ", ".join(missing))
    if layer_ids != expected_ids:
        errors.append("layer ids %s, expected 0..%d (including MTP)" %
                      (sorted(layer_ids), expected_layers))
    sparse = [i for i in sorted(layer_ids) if (i < expected_layers and
                                                i >= 3 and (i - 3) % 4 == 0)]
    sparse_expert_gates = sum(1 for name in tensors
                              if ".mlp.experts." in name and name.endswith(".gate_proj.weight"))
    # Main layers 3..44 and the auxiliary MTP layer are MoE in this release;
    # the sparse-attention cadence applies only to the main decoder layers.
    moe_layers = max(0, expected_layers - int(text.get("first_k_dense_replace", 3))) + 1
    expected_expert_gates = moe_layers * int(text.get("n_routed_experts", 0))
    if sparse_expert_gates != expected_expert_gates:
        errors.append("expert gate tensors %d, expected %d" %
                      (sparse_expert_gates, expected_expert_gates))
    print("tensor_schema: names=%d layers=%d sparse_layers=%d expert_gate_weights=%d" %
          (len(tensors), len(layer_ids), len(sparse), sparse_expert_gates))
    if errors:
        for error in errors:
            print("tensor_schema: ERROR: " + error, file=sys.stderr)
        return False
    print("tensor_schema: GLM5.3F header contract ok")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default=os.path.expanduser("~/models/glm53f"))
    parser.add_argument("--check-tensors", action="store_true",
                        help="also inspect safetensors headers, never payloads")
    args = parser.parse_args(argv)
    root = os.path.abspath(os.path.expanduser(args.model_dir))
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json",
                "chat_template.jinja")
    missing = [name for name in required if not os.path.isfile(os.path.join(root, name))]
    if missing:
        print("missing: %s" % ", ".join(missing), file=sys.stderr)
        return 2

    config = read_json(os.path.join(root, "config.json"))
    text = config.get("text_config", config)
    index = read_json(os.path.join(root, "model.safetensors.index.json"))
    tokenizer = read_json(os.path.join(root, "tokenizer.json"))
    with open(os.path.join(root, "chat_template.jinja")) as stream:
        template = stream.read()

    weight_map = index.get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    present = [name for name in shards if os.path.isfile(os.path.join(root, name))]
    print("model_dir=%s" % root)
    print("model_type=%s" % config.get("model_type", text.get("model_type", "?")))
    print("shards=%d/%d" % (len(present), len(shards)))
    print("tensors=%d" % len(weight_map))
    print("architecture: layers=%s hidden=%s vocab=%s max_position=%s" %
          (text.get("num_hidden_layers"), text.get("hidden_size"),
           text.get("vocab_size"), text.get("max_position_embeddings")))
    print("architecture: attention=%s heads=%s kv_heads=%s head_dim=%s" %
          (text.get("layer_types"), text.get("num_attention_heads"),
           text.get("num_key_value_heads"), text.get("head_dim")))
    print("architecture: experts=%s active=%s shared=%s moe_inter=%s" %
          (text.get("n_routed_experts"), text.get("num_experts_per_tok"),
           text.get("n_shared_experts"), text.get("moe_intermediate_size")))
    print("architecture: q_lora=%s kv_lora=%s qk_head=%s v_head=%s" %
          (text.get("q_lora_rank"), text.get("kv_lora_rank"),
           text.get("qk_head_dim"), text.get("v_head_dim")))

    vocab = tokenizer.get("model", {}).get("vocab", {})
    merges = tokenizer.get("model", {}).get("merges", [])
    added = tokenizer.get("added_tokens", [])
    print("tokenizer: model=%s vocab=%d merges=%d added=%d" %
          (tokenizer.get("model", {}).get("type"), len(vocab), len(merges), len(added)))
    checks = ("[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>")
    missing_markers = [marker for marker in checks if marker not in template]
    print("chat_template: jinja_bytes=%d markers=%s" %
          (len(template.encode("utf-8")), "ok" if not missing_markers else "missing:" + ",".join(missing_markers)))
    schema_ok = True
    if args.check_tensors and len(present) == len(shards):
        try:
            schema_ok = check_tensor_schema(root, shards, weight_map, text)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print("tensor_schema: ERROR: %s" % error, file=sys.stderr)
            schema_ok = False
    if len(present) != len(shards) or missing_markers or not schema_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
