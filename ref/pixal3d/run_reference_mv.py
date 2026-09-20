"""Run pinned multiview inference without loading RMBG for RGBA-only inputs.

The upstream entry point checks every view's alpha before calling its matting
model, but constructs RMBG eagerly. This launcher replaces only that eager
constructor. An opaque/RGB view still fails instead of silently changing the
reference preprocessing contract.
"""
from pathlib import Path
import functools
import math
import runpy
import sys
import types
import torch

UPSTREAM = Path(__file__).with_name("upstream")
sys.path.insert(0, str(UPSTREAM))

# The official Torch 2.7/cu128 NATTEN wheel predates sm_120 and its CUTLASS
# binary has no Blackwell kernel image. FlexAttention implements the same
# neighborhood operation through PyTorch/Triton and runs on the 5060 Ti.
import natten.functional
import natten

_natten_na2d = natten.functional.na2d


def _chunked_na2d(q, k, v, kernel_size, dilation, scale):
    """Exact inference-only NA for NAF's unequal QK/V head widths."""
    kh, kw = ((kernel_size, kernel_size) if isinstance(kernel_size, int)
              else kernel_size)
    dh, dw = ((dilation, dilation) if isinstance(dilation, int) else dilation)
    height, width = q.shape[1:3]
    row_base = torch.arange(height, device=q.device)
    col_base = torch.arange(width, device=q.device)
    row_start = (row_base - (kh // 2) * dh).clamp(0, height - 1 - (kh - 1) * dh)
    col_start = (col_base - (kw // 2) * dw).clamp(0, width - 1 - (kw - 1) * dw)
    outputs = []
    for first in range(0, height, 8):
        last = min(first + 8, height)
        qc = q[:, first:last]
        logits = []
        for iy in range(kh):
            rows = row_start[first:last] + iy * dh
            for ix in range(kw):
                cols = col_start + ix * dw
                logits.append((qc * k[:, rows[:, None], cols[None, :]]).sum(-1) * scale)
        weights = torch.softmax(torch.stack(logits, dim=-1), dim=-1)
        out = torch.zeros((*qc.shape[:-1], v.shape[-1]), device=v.device, dtype=v.dtype)
        offset = 0
        for iy in range(kh):
            rows = row_start[first:last] + iy * dh
            for ix in range(kw):
                cols = col_start + ix * dw
                out.add_(v[:, rows[:, None], cols[None, :]] * weights[..., offset, None])
                offset += 1
        outputs.append(out)
    return torch.cat(outputs, dim=1)


@functools.wraps(_natten_na2d)
def _na2d_blackwell(*args, **kwargs):
    if kwargs.get("backend") == "cutlass-fna":
        if len(args) >= 3 and args[0].shape[-1] != args[2].shape[-1]:
            q, k, v = args[:3]
            return _chunked_na2d(q, k, v, kwargs["kernel_size"],
                                 kwargs.get("dilation", 1),
                                 kwargs.get("scale", 1.0 / math.sqrt(q.shape[-1])))
        kwargs["backend"] = "flex-fna"
    return _natten_na2d(*args, **kwargs)


natten.functional.na2d = _na2d_blackwell
natten.na2d = _na2d_blackwell

from pixal3d.pipelines import rembg
from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as _proj


class _DeferredNaf:
    def __init__(self, model, image, features, output_size):
        encoded = model.image_encoder(image, output_size=output_size)
        self.query = model.query_encoder(encoded)
        self.key = model.key_encoder(encoded, features)
        self.value = features
        self.heads = model.upsampler.num_heads
        self.kernel_size = model.upsampler.kernel_size
        self.scale = model.upsampler.scale
        self.height, self.width = self.query.shape[-2:]


def _deferred_naf_forward(model, image, features, output_size,
                          return_weights=False, *args, **kwargs):
    if return_weights:
        raise ValueError("Blackwell deferred NAF does not support attention weights")
    return _DeferredNaf(model, image, features, output_size)


def _gather_bchw(tensor, rows, cols, heads):
    batch = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
    gathered = tensor.permute(0, 2, 3, 1)[batch, rows, cols]
    return gathered.reshape(tensor.shape[0], rows.shape[1], heads, -1)


def _deferred_pixels(deferred, rows, cols, chunk_size=4096):
    """Evaluate NAF only at integer output pixels, preserving shifted windows."""
    kh, kw = deferred.kernel_size
    dh = deferred.height // deferred.key.shape[-2]
    dw = deferred.width // deferred.key.shape[-1]
    outputs = []
    for first in range(0, rows.shape[1], chunk_size):
        r = rows[:, first:first + chunk_size]
        c = cols[:, first:first + chunk_size]
        q = _gather_bchw(deferred.query, r, c, deferred.heads)
        r0 = (r - (kh // 2) * dh).clamp(0, deferred.height - 1 - (kh - 1) * dh)
        c0 = (c - (kw // 2) * dw).clamp(0, deferred.width - 1 - (kw - 1) * dw)
        logits = []
        key_indices = []
        for iy in range(kh):
            for ix in range(kw):
                tr = r0 + iy * dh
                tc = c0 + ix * dw
                sr = torch.floor((tr + 0.5) * deferred.key.shape[-2] / deferred.height).long()
                sc = torch.floor((tc + 0.5) * deferred.key.shape[-1] / deferred.width).long()
                sr.clamp_(0, deferred.key.shape[-2] - 1)
                sc.clamp_(0, deferred.key.shape[-1] - 1)
                key_indices.append((sr, sc))
                key = _gather_bchw(deferred.key, sr, sc, deferred.heads)
                logits.append((q * key).sum(-1) * deferred.scale)
        weights = torch.softmax(torch.stack(logits, dim=-1), dim=-1)
        value_width = deferred.value.shape[1] // deferred.heads
        out = torch.zeros((*q.shape[:-1], value_width), device=q.device,
                          dtype=deferred.value.dtype)
        for offset, (sr, sc) in enumerate(key_indices):
            value = _gather_bchw(deferred.value, sr, sc, deferred.heads)
            out.add_(value * weights[..., offset, None])
        outputs.append(out.flatten(2))
    return torch.cat(outputs, dim=1)


def _project_deferred(module, deferred, camera_angle_x, distance, mesh_scale,
                      transform_matrix):
    batch = deferred.query.shape[0]
    grid_points = module.grid_points.expand(batch, -1, -1)
    grid_points = grid_points / mesh_scale[:, None, None] / 2
    if transform_matrix is None:
        transform_matrix = module.front_view_transform_matrix.expand(batch, -1, -1).clone()
        transform_matrix[:, 1, 3] = -distance
    image_points, _, _ = _proj.project_points_to_image_batch(
        grid_points, transform_matrix, camera_angle_x, module.image_resolution)
    x = image_points[..., 0].clamp(0, deferred.width - 1)
    y = image_points[..., 1].clamp(0, deferred.height - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1 = (x0 + 1).clamp(max=deferred.width - 1)
    y1 = (y0 + 1).clamp(max=deferred.height - 1)
    wx, wy = (x - x0).unsqueeze(-1), (y - y0).unsqueeze(-1)
    output = _deferred_pixels(deferred, y0, x0)
    output.mul_((1 - wx) * (1 - wy))
    corner = _deferred_pixels(deferred, y0, x1)
    corner.mul_(wx * (1 - wy))
    output.add_(corner)
    del corner
    corner = _deferred_pixels(deferred, y1, x0)
    corner.mul_((1 - wx) * wy)
    output.add_(corner)
    del corner
    corner = _deferred_pixels(deferred, y1, x1)
    corner.mul_(wx * wy)
    output.add_(corner)
    return output


_original_load_naf = _proj.DinoV3ProjFeatureExtractor._load_naf
_original_proj_forward = _proj.ProjGrid.forward
_original_proj_mv_forward = _proj.ProjGridMV.forward


def _load_deferred_naf(extractor):
    _original_load_naf(extractor)
    if not getattr(extractor.naf_model, "_pixal_deferred", False):
        extractor.naf_model.forward = types.MethodType(_deferred_naf_forward,
                                                       extractor.naf_model)
        extractor.naf_model._pixal_deferred = True


def _proj_forward(module, features_map, camera_angle_x, distance, mesh_scale,
                  transform_matrix=None, BHWC=True):
    if isinstance(features_map, _DeferredNaf):
        return _project_deferred(module, features_map, camera_angle_x, distance,
                                 mesh_scale, transform_matrix)
    return _original_proj_forward(module, features_map, camera_angle_x, distance,
                                  mesh_scale, transform_matrix, BHWC)


def _proj_mv_forward(module, features_map, camera_angle_x, distance, mesh_scale,
                     transform_matrix=None, BHWC=True):
    if isinstance(features_map, _DeferredNaf):
        return _project_deferred(module, features_map, camera_angle_x, distance,
                                 mesh_scale, transform_matrix)
    return _original_proj_mv_forward(module, features_map, camera_angle_x, distance,
                                     mesh_scale, transform_matrix, BHWC)


_proj.DinoV3ProjFeatureExtractor._load_naf = _load_deferred_naf
_proj.ProjGrid.forward = _proj_forward
_proj.ProjGridMV.forward = _proj_mv_forward


class AlphaOnlyRmbg:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image):
        raise RuntimeError(
            "reference view lacks useful alpha; install the authorized RMBG-2.0 weights")

    def to(self, device):
        return self

    def cpu(self):
        return self


rembg.BiRefNet = AlphaOnlyRmbg


def main():
    runpy.run_path(str(UPSTREAM / "inference_mv.py"), run_name="__main__")


if __name__ == "__main__":
    main()
