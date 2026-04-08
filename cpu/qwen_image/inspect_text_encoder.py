#!/usr/bin/env python3
"""Inspect Qwen-Image text encoder GGUF files."""
import struct, sys

def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")

def read_value(f, vtype):
    if vtype == 0: return struct.unpack("<B", f.read(1))[0]
    elif vtype == 1: return struct.unpack("<b", f.read(1))[0]
    elif vtype == 2: return struct.unpack("<H", f.read(2))[0]
    elif vtype == 3: return struct.unpack("<h", f.read(2))[0]
    elif vtype == 4: return struct.unpack("<I", f.read(4))[0]
    elif vtype == 5: return struct.unpack("<i", f.read(4))[0]
    elif vtype == 6: return struct.unpack("<f", f.read(4))[0]
    elif vtype == 7: return bool(struct.unpack("<B", f.read(1))[0])
    elif vtype == 8: return read_string(f)
    elif vtype == 9:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        if arr_len > 100:
            sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,8:0,10:8,11:8,12:8}
            if arr_type == 8:
                for _ in range(arr_len): read_string(f)
            else:
                f.read(arr_len * sizes.get(arr_type, 4))
            return f"[array of {arr_len}]"
        return [read_value(f, arr_type) for _ in range(arr_len)]
    elif vtype == 10: return struct.unpack("<Q", f.read(8))[0]
    elif vtype == 11: return struct.unpack("<q", f.read(8))[0]
    elif vtype == 12: return struct.unpack("<d", f.read(8))[0]
    else: raise ValueError(f"Unknown type {vtype}")

GGML_TYPE_NAME = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
    8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
    13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS',
    18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 26: 'BF16',
}

def inspect_gguf(path):
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print(f"Not GGUF: {magic}")
            return
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} KV pairs")

        print("\n--- Metadata ---")
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = read_value(f, vtype)
            val_str = str(value)
            if len(val_str) > 200: val_str = val_str[:200] + "..."
            print(f"  {key} = {val_str}")

        print(f"\n--- Tensors ({n_tensors}) ---")
        tensors = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append((name, dims, ttype))

        for name, dims, ttype in sorted(tensors, key=lambda x: x[0]):
            type_name = GGML_TYPE_NAME.get(ttype, f"type_{ttype}")
            print(f"  {name}: {dims} ({type_name})")

files = [
    "/mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
    "/mnt/disk01/models/qwen-image/text-encoder/mmproj-F16.gguf"
]
for f in files:
    inspect_gguf(f)
