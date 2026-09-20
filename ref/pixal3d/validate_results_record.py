"""Validate the consolidated Pixal3D result record and optional local artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--record", type=Path,
               default=ROOT / "ref/pixal3d/validation-results.json")
p.add_argument("--artifacts", action="store_true",
               help="also require and hash every extended-validation artifact")
a = p.parse_args()

record = json.loads(a.record.read_text())
extended = record["extended_validation"]
required = {
    "hardware", "checkpoint_provenance", "four_view_native_budget_runs",
    "four_view_pytorch_reference", "mask_provenance", "cuda_reliability_soak",
    "byte_identical_postprocess_replay", "real_queued_http_cuda", "evidence_policy",
}
assert required <= extended.keys(), required - extended.keys()

hardware = extended["hardware"]
assert hardware["gpu_name"] == "NVIDIA GeForce RTX 5060 Ti"
assert hardware["device_memory_mib"] >= 16 * 1024 - 100
assert hardware["nvidia_driver"] and hardware["cuda_toolkit"]

budgets = extended["four_view_native_budget_runs"]
runs = budgets["runs"]
assert [item["requested_budget_mib"] for item in runs] == [7168, 12288]
assert budgets["byte_identical"]
assert len({item["artifact"]["sha256"] for item in runs}) == 1
assert all(item["stats"]["vertices"] > 0 and item["stats"]["triangles"] > 0
           for item in runs)
assert all(item["peak_reserved_device_bytes"] <= item["effective_budget_bytes"]
           for item in runs)

reference = extended["four_view_pytorch_reference"]
assert reference["samples"] == 200000 and len(reference["renders"]) == 4
assert reference["geometry"]["symmetric_chamfer_rms"] < .01
for direction in ("native_to_reference", "reference_to_native"):
    assert reference["geometry"][direction]["normal_abs_cosine_mean"] > .95
for rendered in reference["renders"]:
    assert rendered["rgb_psnr"] > 20 and rendered["silhouette_iou"] > .98

masks = extended["mask_provenance"]
models = {item["name"]: item for item in masks["models"]}
assert {"briaai/RMBG-2.0", "ZhengPeng7/BiRefNet"} <= models.keys()
assert models["briaai/RMBG-2.0"]["normal_semantics"]["source_alpha_iou"] > .99
assert models["ZhengPeng7/BiRefNet"]["normal_semantics"]["source_alpha_iou"] > .998
assert models["ZhengPeng7/BiRefNet"]["asset_tuned_best"]["iou"] > .999
assert "diagnostic only" in masks["interpretation"]

reliability = extended["cuda_reliability_soak"]
assert reliability["repeats"] >= 2 and len(reliability["checks"]) >= 7
assert {item["name"] for item in reliability["checks"]} >= {
    "resident-1", "resident-2", "structure-mixed", "structure-fp32",
    "native-invalid-input-recovery", "web-queue-cancellation",
    "multiview-budget-artifacts",
}
assert max(reliability["matched_structure_nrmse"].values()) < .001

postprocess = extended["byte_identical_postprocess_replay"]
assert postprocess["texture_size"] == 4096 and postprocess["triangle_target"] == 1000000
assert postprocess["byte_identical_to_baseline"]
assert postprocess["artifact"]["sha256"] == (
    "6d8c267b006df8cf60e3c5ea71d470e89c3cb734f61b91b5e1c622e1f711a8b7")

web = extended["real_queued_http_cuda"]
assert web["automatic_rmbg"]["mask_source"] == "rmbg-2.0"
assert web["automatic_rmbg"]["cancel_then_recover"]
assert web["explicit_mask_paired_reference"]["mask_source"] == "mask"
assert web["explicit_mask_paired_reference"]["surface"]["available"]


def artifacts(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        for item in value.values():
            yield from artifacts(item)
    elif isinstance(value, list):
        for item in value:
            yield from artifacts(item)


checked = 0
if a.artifacts:
    for item in artifacts(extended):
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == item["sha256"], path
        checked += 1

print(json.dumps({"record": str(a.record), "required_sections": len(required),
                  "artifacts_checked": checked}))
print("Pixal3D consolidated validation record: PASS")
