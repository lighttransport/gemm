#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Patch the FP16 cooperative-matrix shader emitted by glslc into a BF16 variant.
# glslang in this environment cannot spell SPV_KHR_bfloat16 yet, but RADV
# exposes VK_KHR_shader_bfloat16 and accepts the SPIR-V extension.

import struct
import sys


OP_CAPABILITY = 17
OP_EXTENSION = 10
OP_TYPE_FLOAT = 22

CAP_BFLOAT16_TYPE_KHR = 5116
CAP_BFLOAT16_COOPERATIVE_MATRIX_KHR = 5118
BFLOAT16_ENCODING_KHR = 0


def encode_extension(name: str) -> list[int]:
    data = name.encode("ascii") + b"\0"
    while len(data) % 4:
        data += b"\0"
    words = [struct.unpack("<I", data[i:i + 4])[0] for i in range(0, len(data), 4)]
    return [((1 + len(words)) << 16) | OP_EXTENSION] + words


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input.spv output.spv", file=sys.stderr)
        return 2

    with open(sys.argv[1], "rb") as f:
        raw = f.read()
    words = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    if len(words) < 5 or words[0] != 0x07230203:
        raise RuntimeError("not a SPIR-V binary")

    out = words[:5]
    inserted_caps = False
    inserted_ext = False
    patched_type = False
    i = 5
    while i < len(words):
        first = words[i]
        wc = first >> 16
        op = first & 0xffff
        inst = words[i:i + wc]

        if not inserted_caps and op != OP_CAPABILITY:
            out.extend([
                (2 << 16) | OP_CAPABILITY, CAP_BFLOAT16_TYPE_KHR,
                (2 << 16) | OP_CAPABILITY, CAP_BFLOAT16_COOPERATIVE_MATRIX_KHR,
            ])
            inserted_caps = True

        if not inserted_ext and op != OP_CAPABILITY and op != OP_EXTENSION:
            out.extend(encode_extension("SPV_KHR_bfloat16"))
            inserted_ext = True

        if op == OP_TYPE_FLOAT and wc == 3 and inst[2] == 16:
            inst = [(4 << 16) | OP_TYPE_FLOAT, inst[1], inst[2], BFLOAT16_ENCODING_KHR]
            patched_type = True

        out.extend(inst)
        i += wc

    if not patched_type:
        raise RuntimeError("did not find the 16-bit OpTypeFloat to patch")

    with open(sys.argv[2], "wb") as f:
        f.write(struct.pack("<%dI" % len(out), *out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
