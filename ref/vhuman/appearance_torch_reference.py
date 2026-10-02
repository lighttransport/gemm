"""Optional legacy Torch/gsplat oracle; native CPU fitting is the default.

Fixed-light appearance fitting to cleared posed reference frames.

No image-based mesh renderer is used as a photographic training target.
Topology and poses must already be registered to the supplied real references.
"""
import json
from pathlib import Path
import numpy as np
from server.vhuman.realtime.src.avatar.bundle import GaussianAvatar, bind
from server.vhuman.realtime.src.avatar.provenance import verify_files
from server.vhuman.realtime.src.avatar.geometry import deform_torch


def fit(manifest, output, count=50000, steps=1000, device="cuda:0", seed=7):
    import torch
    from gsplat import rasterization
    manifest = Path(manifest)
    spec = json.loads(manifest.read_text())
    if spec.get("format") != "vhuman.appearance_corpus.v1":
        raise ValueError("unsupported appearance corpus")
    verify_files(spec["provenance"], manifest.parent)
    if spec["data"] not in {r["path"] for r in spec["provenance"]}:
        raise ValueError("appearance data lacks a checksum receipt")
    path = (manifest.parent / spec["data"]).resolve()
    if not path.is_relative_to(manifest.parent.resolve()): raise ValueError("corpus path escapes root")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    pos, tri, images = data["vertices"], data["triangles"], data["images"]
    names, controls = spec["control_names"], data["controls"]
    frames = len(pos)
    if pos.ndim != 3 or pos.shape[2] != 3 or images.ndim != 4 or images.shape[0] != frames or images.shape[-1] != 3:
        raise ValueError("invalid appearance frame shapes")
    if controls.shape != (frames, len(names)) or data["view"].shape != (frames, 4, 4) or data["intrinsics"].shape != (frames, 3, 3):
        raise ValueError("appearance camera/control mismatch")
    if any(not np.isfinite(x).all() for x in data.values()) or (images < 0).any() or (images > 1).any():
        raise ValueError("nonfinite corpus or images outside linear RGB 0..1")
    masks = data.get("masks", (images.max(-1) > 1e-5).astype(np.float32))
    if masks.shape != images.shape[:3] or (masks < 0).any() or (masks > 1).any():
        raise ValueError("invalid appearance masks")
    if not 1 <= steps: raise ValueError("steps must be positive")
    avatar = bind(pos[0], tri, names, count, seed, purpose=spec.get("purpose", "production"), provenance=spec["provenance"])
    torch.manual_seed(seed)
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    ids = torch.as_tensor(avatar.arrays["triangle"], device=device, dtype=torch.long)
    triangles = torch.as_tensor(tri, device=device, dtype=torch.long)
    bary = tensor(avatar.arrays["barycentric"])
    # Project the cleared first reference onto attached points for an informative
    # initialization. This is a fitting initializer, not a photographic quality claim.
    initial = tensor(pos[0])[triangles[ids]]
    initial = (initial * bary[..., None]).sum(1)
    camera = initial @ tensor(data["view"][0][:3, :3]).T + tensor(data["view"][0][:3, 3])
    screen = camera @ tensor(data["intrinsics"][0]).T
    uv = screen[:, :2] / screen[:, 2:3].clamp_min(1e-6)
    grid = uv / tensor([images.shape[2]-1, images.shape[1]-1]) * 2 - 1
    initial_rgb = torch.nn.functional.grid_sample(tensor(images[0]).permute(2, 0, 1)[None],
        grid[None, None], mode="bilinear", align_corners=True)[0, :, 0].T.clamp(.01, .99)
    rgb_raw = torch.nn.Parameter(torch.logit(initial_rgb))
    opacity_raw = torch.nn.Parameter(torch.full((count,), 1., device=device))
    log_scales = torch.nn.Parameter(tensor(np.tile(np.log([.15, .15, .00015]), (count, 1))))
    offset_raw = torch.nn.Parameter(torch.zeros(count, device=device))
    color_basis = torch.nn.Parameter(torch.zeros(count, 8, 3, device=device))
    expression = torch.nn.Parameter(torch.randn(len(names), 8, device=device) * .05)
    params = [rgb_raw, opacity_raw, log_scales, offset_raw, color_basis, expression]
    limits = tensor([1., 1., .002])
    optimizer = torch.optim.Adam(params, lr=.01)
    rng = np.random.default_rng(seed)
    loss_value = None
    for step in range(steps):
        f = int(rng.integers(frames))
        scales = torch.minimum(log_scales.exp().clamp_min(1e-5), limits)
        current = dict(triangle=ids, barycentric=bary, normal_offset=.005 * offset_raw.tanh(),
            covariance_local=torch.diag_embed(scales.square()), opacity=opacity_raw.sigmoid(),
            rgb=rgb_raw.sigmoid(), color_basis=color_basis, expression_matrix=expression)
        means, cov, opacity, colors = deform_torch(tensor(pos[f]), triangles, current,
                                                  tensor(controls[f]), "trace-v1")
        image, alpha, _ = rasterization(means=means, quats=None, scales=None, covars=cov,
                                    opacities=opacity, colors=colors,
                                    viewmats=tensor(data["view"][f])[None], Ks=tensor(data["intrinsics"][f])[None],
                                    width=images.shape[2], height=images.shape[1], packed=True)
        image_l1 = (image[0] - tensor(images[f])).abs().mean()
        loss = image_l1 + .05 * (alpha[0, ..., 0] - tensor(masks[f])).abs().mean() + 1e-4 * color_basis.square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if not torch.isfinite(loss): raise RuntimeError("appearance loss became nonfinite")
        loss_value = float(image_l1.detach().cpu())
    cpu = lambda x: x.detach().cpu().numpy().astype(np.float32)
    avatar.arrays.update(rgb=cpu(rgb_raw.sigmoid()), opacity=cpu(opacity_raw.sigmoid()),
                         normal_offset=cpu(.005 * offset_raw.tanh()), color_basis=cpu(color_basis),
                         expression_matrix=cpu(expression),
                         covariance_local=cpu(torch.diag_embed(torch.minimum(log_scales.exp().clamp_min(1e-5), limits).square())))
    avatar.metadata.update(trained=True, covariance_policy="trace-v1", training_steps=steps, training_l1=loss_value,
                           reference_camera=dict(view=data["view"][0].tolist(), intrinsics=data["intrinsics"][0].tolist(),
                                                 size=[images.shape[2], images.shape[1]]),
                           corpus_sha256=next(r["sha256"] for r in spec["provenance"] if r["path"] == spec["data"]))
    avatar.save(output)
    return {"steps": steps, "loss_l1": loss_value, "gaussians": count}
