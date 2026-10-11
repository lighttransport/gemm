"""Prepared-observation integration; capability gates prevent partial inference.

The native primitive library is usable independently. Reconstruction is kept
unavailable until the conditioning/decoder/loss/sampler parity milestones land.
An unguided SAM3D result must never be published as a completed GenIA result.
"""
from .. import gpu
from ..service import ROOT, ServiceError
from . import RELEASE
from .observations import load
from .upload import resolve

CONVERTED = ("sam3d_dinov2.safetensors", "sam3d_cond_fuser.safetensors",
             "sam3d_point_patch_embed.safetensors", "sam3d_ss_dit.safetensors",
             "sam3d_ss_decoder.safetensors", "sam3d_slat_dit.safetensors",
             "sam3d_slat_gs_decoder.safetensors", "sam3d_slat_cond_fuser.safetensors")
PENDING = ("four_stream_conditioning", "fixed_shape_pose_sampling", "entropy_shape_fusion",
           "visibility_attention_hooks", "gaussian_decoder_backward", "native_lpips_losses",
           "guided_slat_sampling", "dora_refinement", "end_to_end_parity")


def availability():
    model = gpu.model_path("sam3d")
    missing = [str(model / "safetensors" / name) for name in CONVERTED
               if not (model / "safetensors" / name).is_file()]
    selected = gpu.backend()
    library = ROOT / "cuda/genia/libgenia_cuda.so"
    return {"available": False, "revision": RELEASE, "model_dir": str(model),
            "backend": selected, "device": gpu.device_index(), "missing": missing,
            "capabilities": {"prepared_observations": True,
                             "cuda_primitives_built": library.is_file(),
                             "guided_reconstruction": False, "test_time_refinement": False},
            "pending": list(PENDING),
            "reason": "Native GenIA reconstruction is not implemented yet; prepared observations and CUDA primitives are available"}


def check(path):
    observations = load(path)
    return {"observations": observations.provenance(), "genia": availability()}


def validate_job(service, request, progress, cancel):
    if any(key in request for key in ("observations", "model", "runner", "models_root", "path")):
        raise ServiceError("GenIA local paths are available only through the CLI")
    if cancel.is_set():
        raise gpu.Cancelled("cancelled")
    progress(.1, "Validating prepared GenIA observations")
    result = check(resolve(service, request.get("upload_id")))
    progress(1, "Prepared observations validated")
    return result


def reconstruct_job(service, request, progress, cancel):
    validate_job(service, request, progress, cancel)
    raise ServiceError(availability()["reason"])
