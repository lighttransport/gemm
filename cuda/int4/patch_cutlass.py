#!/usr/bin/env python3
"""Patch and verify SM120 E0M3 instructions in a binary-only CUTLASS adapter.

The host ELF, fatbinary metadata and all non-format instruction bits remain
unchanged. PTX and compressed device ELFs are rejected to prevent an E2M1 JIT
fallback. All CUDA ELF entries are examined, including device-link stubs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import patch_formats as patch

PREFIXES = ("_ZN7cutlass13device_kernelI", "gemm_decode_")


def fatbin_section(data):
    if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", data, 18)[0] != 62:
        raise ValueError("expected a little-endian ELF64 x86-64 shared object")
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, strings = struct.unpack_from("<HHH", data, 58)
    if stride != 64 or not count or strings >= count or offset + count * stride > len(data):
        raise ValueError("invalid host ELF section table")
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride) for i in range(count)]
    shstr = headers[strings]
    names = data[shstr[4]:shstr[4] + shstr[5]]
    found = []
    for h in headers:
        name = names[h[0]:names.index(0, h[0])].decode()
        if name == ".nv_fatbin":
            if h[4] + h[5] > len(data):
                raise ValueError("fatbinary section outside host ELF")
            found.append((h[4], h[5]))
    if len(found) != 1:
        raise ValueError("expected exactly one .nv_fatbin section")
    return found[0]


def cuda_elves(data):
    base, length = fatbin_section(data)
    pos, end = base, base + length
    result = []
    while pos < end:
        if end - pos < 16:
            raise ValueError("truncated fatbinary header")
        magic, version, header, payload = struct.unpack_from("<IHHQ", data, pos)
        if magic != 0xBA55ED50 or version != 1 or header != 16 or pos + header + payload > end:
            raise ValueError("unsupported fatbinary header")
        entry, limit = pos + header, pos + header + payload
        while entry < limit:
            if limit - entry < 16:
                raise ValueError("truncated fatbinary entry")
            kind, version, header, size = struct.unpack_from("<HHIQ", data, entry)
            if kind != 2:
                raise ValueError("non-ELF entry found; compile with code=sm_120a and no PTX fallback")
            if version != 0x101 or header < 64 or size < 64 or entry + header + size > limit:
                raise ValueError("unsupported fatbinary entry")
            offset = entry + header
            blob = data[offset:offset + size]
            patch.text_sections(blob)  # Validates CUDA ELF machine and executable sections.
            if (struct.unpack_from("<I", blob, 48)[0] >> 8) & 255 != 120:
                raise ValueError("expected SM120 CUDA ELF")
            result.append((offset, blob))
            entry = offset + size
        pos = limit
    if not result:
        raise ValueError("no uncompressed CUDA ELFs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuobjdump", default="/usr/local/cuda-13.1/bin/cuobjdump")
    parser.add_argument("--scratch-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "tmp" / "int4")
    args = parser.parse_args()
    if args.baseline.resolve() == args.output.resolve():
        raise ValueError("refusing to overwrite baseline")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(args.scratch_dir.resolve())
    original = args.baseline.read_bytes()
    result = bytearray(original)
    modules, total, expected_changes = [], 0, set()
    with tempfile.TemporaryDirectory(dir=args.scratch_dir, prefix="patch_cutlass_") as work:
        for i, (offset, blob) in enumerate(cuda_elves(original)):
            path = Path(work) / f"module_{i}.cubin"
            path.write_bytes(blob)
            sass = patch.disassemble(args.cuobjdump, path)
            if "OMMA." not in sass:
                modules.append({"offset": offset, "bytes": len(blob), "instruction_count": 0})
                continue
            records = patch.instructions(sass, PREFIXES)
            changed = patch.patched_bytes(blob, records, 3)
            path.write_bytes(changed)
            changed_sass = patch.disassemble(args.cuobjdump, path)
            expected = [(n, pc, patch.FORMATS[3], lo, hi | 3 << 14) for n, pc, _, lo, hi in records]
            if patch.instructions(changed_sass, PREFIXES) != expected:
                raise ValueError("post-patch CUDA ELF disassembly mismatch")
            sections = patch.text_sections(blob)
            expected_changes.update(offset + sections[n][0] + pc + 9 for n, pc, *_ in records)
            result[offset:offset + len(blob)] = changed
            total += len(records)
            modules.append({"offset": offset, "bytes": len(blob), "instruction_count": len(records),
                            "kernels": sorted({n for n, *_ in records})})
        actual_changes = {i for i, (a, b) in enumerate(zip(original, result)) if a != b}
        if not total or actual_changes != expected_changes or any(original[i] ^ result[i] != 0xC0 for i in actual_changes):
            raise ValueError("unexpected host binary changes")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pending = args.output.with_suffix(".pending.so")
        try:
            pending.write_bytes(result)
            final_sass = patch.disassemble(args.cuobjdump, pending)
            final_records = patch.instructions(final_sass, PREFIXES)
            if len(final_records) != total or any(fmt != patch.FORMATS[3] for _, _, fmt, *_ in final_records):
                raise ValueError("final host shared-object disassembly mismatch")
            os.replace(pending, args.output)
        finally:
            pending.unlink(missing_ok=True)
    manifest = {"source": str(args.baseline), "source_sha256": hashlib.sha256(original).hexdigest(),
                "output_sha256": hashlib.sha256(result).hexdigest(), "format": "E0M3.E0M3",
                "cuobjdump": subprocess.check_output([args.cuobjdump, "--version"], text=True).strip(),
                "instruction_count": total, "changed_byte_count": len(actual_changes),
                "ptx_fallback": False, "modules": modules}
    args.output.with_suffix(".patch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.output.with_suffix(".sass").write_text(final_sass)
    print(f"{args.output}: verified {total} E0M3 x E0M3 instructions; only {len(actual_changes)} format bytes changed")


if __name__ == "__main__":
    main()
