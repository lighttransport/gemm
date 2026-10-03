#!/usr/bin/env python3
"""Rebuild compatible cached wheels without modifying the shared uv cache."""
import argparse
import csv
from email.parser import Parser
from pathlib import Path
import re
import zipfile

from packaging.requirements import Requirement
from packaging.tags import sys_tags, parse_tag
from packaging.utils import canonicalize_name

parser = argparse.ArgumentParser()
parser.add_argument("--cache", required=True)
parser.add_argument("--site-packages", action="append", default=[])
parser.add_argument("--output", required=True)
parser.add_argument("--skip", action="append", default=[])
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
supported = set(sys_tags())
roots = list((Path(args.cache)/"archive-v0").iterdir()) + [Path(p) for p in args.site_packages]
available = {}
for root in roots:
    for metadata in root.glob("*.dist-info/METADATA"):
        wheel = metadata.with_name("WHEEL")
        record = metadata.with_name("RECORD")
        if not wheel.exists() or not record.exists():
            continue
        fields = Parser().parsestr(metadata.read_text())
        tags = [line[5:] for line in wheel.read_text().splitlines() if line.startswith("Tag: ")]
        if not any(parse_tag(tag) & supported for tag in tags):
            continue
        name = canonicalize_name(fields["Name"])
        available.setdefault(name, []).append((root, metadata, fields, tags))
# Include alternatives for every dependency, allowing uv to enforce versions.
needed = {"numpy", "safetensors", "torch", "transformers", "datasets", "huggingface-hub", "pillow", "scipy", "psutil", "jinja2", "pytest", "setuptools"}
pending = list(needed)
while pending:
    name = pending.pop()
    for _, _, fields, _ in available.get(name, []):
        for raw in fields.get_all("Requires-Dist", []):
            requirement = Requirement(raw)
            # Include optional CUDA extras as well as ordinary dependencies.
            if requirement.marker and not any(requirement.marker.evaluate({"extra": extra}) for extra in ("", "cublas", "cudart", "cufft", "cufile", "curand", "cusolver", "cusparse", "cupti", "nvrtc", "nvjitlink", "nvtx")):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency not in needed:
                needed.add(dependency)
                pending.append(dependency)
for name in sorted(needed):
    if name in {canonicalize_name(value) for value in args.skip}:
        continue
    for root, metadata, fields, tags in available.get(name, []):
        # One compatible native tag suffices; pure wheels retain py3-none-any.
        tag = next(tag for tag in tags if parse_tag(tag) & supported)
        distribution = re.sub(r"[-.]", "_", fields["Name"])
        filename = output/f"{distribution}-{fields['Version']}-{tag}.whl"
        if filename.exists():
            continue
        temporary = filename.with_suffix(".whl.partial")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            with metadata.with_name("RECORD").open(newline="") as stream:
                for entry in csv.reader(stream):
                    relative = Path(entry[0])
                    if relative.is_absolute() or ".." in relative.parts:
                        continue
                    source = root/relative
                    if source.is_file() and not source.name.endswith(".pyc"):
                        archive.write(source, relative.as_posix())
        temporary.replace(filename)
        print(filename.name, flush=True)
print("Missing cached packages:", ", ".join(sorted(needed-available.keys())))
