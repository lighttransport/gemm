#!/usr/bin/env python3
"""Inspect Qwen-Image GGUF tensor names and shapes."""
import struct, sys, json

def read_gguf_header(path):
    """Read GGUF header and list all tensors."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print(f"Not a GGUF file: {magic}")
            return
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} KV pairs")

        # Read KV pairs
        GGUF_TYPE = {
            0: 'uint8', 1: 'int8', 2: 'uint16', 3: 'int16',
            4: 'uint32', 5: 'int32', 6: 'float32', 7: 'bool',
            8: 'string', 9: 'array', 10: 'uint64', 11: 'int64', 12: 'float64'
        }

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
                if arr_len > 1000:
                    # Skip large arrays
                    if arr_type == 8:
                        for _ in range(arr_len):
                            read_string(f)
                    else:
                        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                        f.read(arr_len * sizes.get(arr_type, 4))
                    return f"[array of {arr_len} {GGUF_TYPE.get(arr_type, '?')}]"
                return [read_value(f, arr_type) for _ in range(arr_len)]
            elif vtype == 10: return struct.unpack("<Q", f.read(8))[0]
            elif vtype == 11: return struct.unpack("<q", f.read(8))[0]
            elif vtype == 12: return struct.unpack("<d", f.read(8))[0]
            else:
                raise ValueError(f"Unknown type {vtype}")

        kv = {}
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = read_value(f, vtype)
            kv[key] = value

        print("\n=== KV Metadata ===")
        for k, v in sorted(kv.items()):
            val_str = str(v)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"  {k} = {val_str}")

        # Read tensor info
        GGML_TYPE_NAME = {
            0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
            8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
            13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS',
            18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 26: 'BF16',
            28: 'Q4_0_4_4', 29: 'Q4_0_4_8', 30: 'Q4_0_8_8'
        }

        print(f"\n=== Tensors ({n_tensors}) ===")
        tensors = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append((name, dims, ttype, offset))

        # Print sorted
        for name, dims, ttype, offset in sorted(tensors, key=lambda x: x[0]):
            type_name = GGML_TYPE_NAME.get(ttype, f"type_{ttype}")
            print(f"  {name}: {dims} ({type_name})")

        # Group by prefix
        print("\n=== Groups ===")
        groups = {}
        for name, dims, ttype, offset in tensors:
            parts = name.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                prefix = f"{parts[0]}.N"
            else:
                prefix = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((name, dims, ttype))

        for prefix in sorted(groups):
            group = groups[prefix]
            print(f"\n[{prefix}] ({len(group)} tensors)")
            # Show one representative
            shown = set()
            for name, dims, ttype in group:
                generic = name
                parts = name.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    generic = name.replace(f".{parts[1]}.", ".N.", 1)
                if generic not in shown:
                    shown.add(generic)
                    type_name = GGML_TYPE_NAME.get(ttype, f"type_{ttype}")
                    print(f"    {generic}: {dims} ({type_name})")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/disk01/models/qwen-image/diffusion-models/qwen-image-Q4_0.gguf"
    read_gguf_header(path)
