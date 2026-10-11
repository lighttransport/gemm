"""Run an explicit local source/avatar matrix and collect numerical reports."""
from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
import signal
import threading

from .. import gpu, runtime
from ..service import EyeService
from . import motion, musclemimic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runtime.add_arguments(parser)
    musclemimic.add_arguments(parser)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam3d-body-model", required=True)
    args = parser.parse_args()
    runtime.configure_args(args)
    matrix = json.loads(args.matrix.read_text())
    heads = matrix["heads"]
    if not heads or not all(isinstance(h, str) for h in heads):
        parser.error("matrix.heads must list existing avatar IDs")
    service = EyeService(args.work)
    for head in heads:
        service.body_file(head, "avatar.glb")
    sources = [{"reference_motion": name} for name in matrix.get("references", [])]
    for path in matrix.get("videos", []):
        path = Path(path)
        if not path.is_absolute():
            path = args.matrix.resolve().parent / path
        with path.open("rb") as stream:
            upload = motion.upload(service, stream, path.stat().st_size,
                                   mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        if upload["kind"] != "video":
            parser.error("matrix.videos requires video files")
        sources.append({"upload_id": upload["id"], "fixed_camera": True})
    if not sources:
        parser.error("matrix needs references or videos")
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    output = {"version": 1, "research_only": True, "matrix": matrix,
              "perceptual_review": "pending", "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for head in heads:
        for source in sources:
            if cancel.is_set():
                return 130
            entry = {"head_id": head, "source": source}
            try:
                result = musclemimic.fit(service, {"head_id": head, **source},
                    lambda p, message: print(f"{head} {p:.0%} {message}", flush=True), cancel,
                    model_dir=args.sam3d_body_model, python=args.musclemimic_python,
                    assets=args.musclemimic_assets, device=args.musclemimic_device)
                entry["take"] = result
                evaluation = service.body_file(head, f"motions/{result['id']}/evaluation.json")
                entry["evaluation"] = json.loads(evaluation.read_text())
            except gpu.Cancelled:
                return 130
            except Exception as exc:
                entry["error"] = str(exc)
            output["results"].append(entry)
            output["accepted"] = sum(bool(r.get("take", {}).get("accepted")) for r in output["results"])
            partial = args.output.with_suffix(args.output.suffix + ".partial")
            partial.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
            partial.replace(args.output)
    return 0 if output["accepted"] == len(output["results"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
