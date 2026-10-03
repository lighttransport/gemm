#!/usr/bin/env python3
"""Materialize linked uv archives required by the installed PyTorch build."""
import argparse
from email.parser import Parser
from pathlib import Path
import shutil
import tomllib

from packaging.requirements import Requirement
from packaging.tags import parse_tag, sys_tags
from packaging.utils import canonicalize_name

parser = argparse.ArgumentParser()
parser.add_argument("--cache", required=True)
parser.add_argument("--torch-metadata", required=True)
args = parser.parse_args()
archives = Path(args.cache)/"archive-v0"
if not archives.exists():
    raise SystemExit(0)
available = {}
supported = set(sys_tags())
for root in archives.iterdir():
    for metadata in root.glob("*.dist-info/METADATA"):
        wheel = metadata.with_name("WHEEL")
        if not wheel.exists():
            continue
        tags = [line[5:] for line in wheel.read_text().splitlines() if line.startswith("Tag: ")]
        if not any(parse_tag(tag) & supported for tag in tags):
            continue
        fields = Parser().parsestr(metadata.read_text())
        available.setdefault(canonicalize_name(fields["Name"]), []).append((root, fields))
torch = Parser().parsestr(Path(args.torch_metadata).read_text())
pending = [(Requirement(raw), {""}) for raw in torch.get_all("Requires-Dist", [])]
project = Path(__file__).resolve().parents[1]/"pyproject.toml"
for raw in tomllib.loads(project.read_text())["project"]["dependencies"]:
    if canonicalize_name(Requirement(raw).name) != "torch":
        pending.append((Requirement(raw), {""}))
visited, selected = set(), set()
while pending:
    requirement, extras = pending.pop()
    if requirement.marker and not any(requirement.marker.evaluate({"extra": extra}) for extra in extras):
        continue
    identity = (str(requirement), tuple(sorted(extras)))
    if identity in visited:
        continue
    visited.add(identity)
    for root, fields in available.get(canonicalize_name(requirement.name), []):
        if fields["Version"] not in requirement.specifier:
            continue
        selected.add(root)
        for raw in fields.get_all("Requires-Dist", []):
            pending.append((Requirement(raw), {"", *requirement.extras}))
for root in sorted(selected):
    if not root.is_symlink():
        continue
    source = root.resolve(strict=True)
    temporary = root.with_name(root.name+".materializing")
    if temporary.exists():
        shutil.rmtree(temporary)
    print(f"Materializing {root.name}", flush=True)
    shutil.copytree(source, temporary)
    root.unlink()
    temporary.rename(root)
