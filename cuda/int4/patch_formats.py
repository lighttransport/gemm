#!/usr/bin/env python3
"""Patch only disassembled, byte-verified OMMA instructions in CUDA ELF .text.

Format-bit discovery: https://github.com/revollllt/sm120-e0m3-mma
This implementation resolves ELF sections and instruction addresses rather than
searching/replacing arbitrary occurrences of an opcode in the file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

MODES = {"e0m3_e2m1": 1, "e2m1_e0m3": 2, "e0m3_e0m3": 3}
FORMATS = ("E2M1.E2M1", "E0M3.E2M1", "E2M1.E0M3", "E0M3.E0M3")
OP = re.compile(
    r"/\*([0-9a-fA-F]+)\*/[^\n]*OMMA\.SF\.16864\.F32\."
    r"(E[02]M[13]\.E[02]M[13])\.UE4M3\.4X[^\n]*"
    r"/\* 0x([0-9a-fA-F]{16}) \*/\s*\n\s*/\* 0x([0-9a-fA-F]{16}) \*/"
)


def text_sections(data):
    if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", data, 18)[0] != 190:
        raise ValueError("expected a little-endian ELF64 CUDA CUBIN")
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, strings = struct.unpack_from("<HHH", data, 58)
    if stride != 64 or not count or strings >= count:
        raise ValueError("unsupported ELF section table")
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride) for i in range(count)]
    shstr = headers[strings]
    names = data[shstr[4]:shstr[4] + shstr[5]]
    result = {}
    for h in headers:
        end = names.index(0, h[0])
        name = names[h[0]:end].decode()
        if name.startswith(".text."):
            if h[4] + h[5] > len(data) or not (h[2] & 4):
                raise ValueError("invalid executable section")
            result[name[6:]] = (h[4], h[5])
    return result


def instructions(sass, prefixes=("gemm_", "probe_")):
    result = []
    for part in re.split(r"Function\s*:\s*", sass)[1:]:
        name = part.splitlines()[0].strip()
        if not name.startswith(prefixes):
            if "OMMA." in part:
                raise ValueError(f"unexpected OMMA outside designated kernels: {name}")
            continue
        matches = list(OP.finditer(part))
        if len(matches) != part.count("OMMA."):
            raise ValueError(f"unrecognized OMMA spelling/encoding in {name}")
        for m in matches:
            result.append((name, int(m[1], 16), m[2], int(m[3], 16), int(m[4], 16)))
    if not result:
        raise ValueError("no designated OMMA instructions found")
    return result


def patched_bytes(data, records, mode):
    sections = text_sections(data)
    out = bytearray(data)
    seen = set()
    for name, pc, fmt, low, high in records:
        if fmt != FORMATS[0] or (high >> 14) & 3:
            raise ValueError("source must contain unpatched E2M1 x E2M1 instructions")
        base, size = sections[name]
        if pc % 16 or pc + 16 > size or (name, pc) in seen:
            raise ValueError("invalid or duplicate instruction address")
        seen.add((name, pc))
        offset = base + pc
        if data[offset:offset + 16] != struct.pack("<QQ", low, high):
            raise ValueError(f"instruction bytes differ from disassembly: {name}+{pc:x}")
        struct.pack_into("<Q", out, offset + 8, high | mode << 14)
    return bytes(out)


def disassemble(tool, path):
    return subprocess.run([str(tool), "--dump-sass", str(path)], check=True,
                          capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump"))
    parser.add_argument("--scratch-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "tmp" / "int4")
    args = parser.parse_args()
    if not args.cuobjdump:
        parser.error("cuobjdump is required")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(args.scratch_dir.resolve())
    data = args.baseline.read_bytes()
    sass = disassemble(args.cuobjdump, args.baseline)
    records = instructions(sass)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e2m1_e2m1.sass").write_text(sass)
    manifest = {"source": str(args.baseline), "source_sha256": hashlib.sha256(data).hexdigest(),
                "cuobjdump": subprocess.check_output([args.cuobjdump, "--version"], text=True).strip(),
                "instruction_count": len(records), "variants": {}}
    for name, mode in MODES.items():
        path = args.output_dir / (name + ".cubin")
        if path.resolve() == args.baseline.resolve():
            raise ValueError("refusing to overwrite source")
        blob = patched_bytes(data, records, mode)
        pending = path.with_suffix(".pending.cubin")
        try:
            pending.write_bytes(blob)
            changed_sass = disassemble(args.cuobjdump, pending)
            changed = instructions(changed_sass)
            expected = [(n, pc, FORMATS[mode], lo, hi | mode << 14) for n, pc, _, lo, hi in records]
            if changed != expected:
                raise ValueError("post-patch disassembly did not match every expected instruction")
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
        (args.output_dir / (name + ".sass")).write_text(changed_sass)
        manifest["variants"][name] = {"sha256": hashlib.sha256(blob).hexdigest(), "format_bits": mode,
            "patches": [{"kernel": n, "pc": pc, "before": f"{hi:016x}", "after": f"{hi | mode << 14:016x}"}
                        for n, pc, _, _, hi in records]}
        print(f"{name}: verified {len(records)} OMMA instructions; sha256={hashlib.sha256(blob).hexdigest()}")
    (args.output_dir / "patch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
