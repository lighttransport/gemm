"""Fixed-light appearance fitting to cleared posed reference frames.

No image-based mesh renderer is used as a photographic training target.
Topology and poses must already be registered to the supplied real references.
"""
import json
from pathlib import Path
import numpy as np
from .bundle import GaussianAvatar, bind
from .provenance import verify_files
from .native_training import AppearanceTrainer, initialize_rgb


def fit(manifest, output, count=50000, steps=1000, device="cpu", seed=7, threads=4, memory_mb=512):
    from ....native_gpu_training import device_index
    gpu=device_index(device) is not None
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
    if (pos.ndim != 3 or pos.shape[2] != 3 or not frames or images.ndim != 4 or images.shape[0] != frames or
            images.shape[-1] != 3 or not all(1<=size<=4096 for size in images.shape[1:3])):
        raise ValueError("invalid appearance frame shapes")
    if controls.shape != (frames, len(names)) or data["view"].shape != (frames, 4, 4) or data["intrinsics"].shape != (frames, 3, 3):
        raise ValueError("appearance camera/control mismatch")
    if any(not np.isfinite(x).all() for x in data.values()) or (images < 0).any() or (images > 1).any():
        raise ValueError("nonfinite corpus or images outside linear RGB 0..1")
    masks = data.get("masks", (images.max(-1) > 1e-5).astype(np.float32))
    if masks.shape != images.shape[:3] or (masks < 0).any() or (masks > 1).any():
        raise ValueError("invalid appearance masks")
    if type(steps) is not int or not 1<=steps<=1000000: raise ValueError("steps must be 1..1000000")
    if not isinstance(names,list) or not 1<=len(names)<=512 or any(not isinstance(name,str) or not name for name in names):
        raise ValueError('invalid appearance control names')
    avatar = bind(pos[0], tri, names, count, seed, purpose=spec.get("purpose", "production"), provenance=spec["provenance"])
    points=(pos[0][tri[avatar.arrays['triangle']]]*avatar.arrays['barycentric'][...,None]).sum(1)
    initial_rgb=initialize_rgb(points,images[0],data['view'][0],data['intrinsics'][0])
    model=AppearanceTrainer(avatar,tri,initial_rgb,seed,threads,device=device,resident=gpu,memory_mb=memory_mb)
    rng=np.random.default_rng(seed);loss_value=None
    try:
        for step in range(steps):
            frame=int(rng.integers(frames))
            _,loss,_=model.compute(pos[frame],controls[frame],data['view'][frame],data['intrinsics'][frame],
                (images.shape[2],images.shape[1]),images[frame],masks[frame],update=True,
                return_gradient=False,return_output=False)
            loss_value=float(loss[1])
        model.export(avatar)
        peak_bytes=model._gpu.peak_bytes if gpu else 0
    finally:model.close()
    backend='repository_cuda_gemm' if gpu else 'repository_cpu_gemm'
    avatar.metadata.update(trained=True, covariance_policy="trace-v1", training_steps=steps, training_l1=loss_value,
                           training_backend=backend, training_device=device, thread_budget=threads,
                           cuda_peak_bytes=peak_bytes,cuda_memory_budget_mb=memory_mb if gpu else None,
                           reference_camera=dict(view=data["view"][0].tolist(), intrinsics=data["intrinsics"][0].tolist(),
                                                 size=[images.shape[2], images.shape[1]]),
                           corpus_sha256=next(r["sha256"] for r in spec["provenance"] if r["path"] == spec["data"]))
    avatar.save(output)
    return {"steps": steps, "loss_l1": loss_value, "gaussians": count, "backend": backend, "device": device,"cuda_peak_bytes":peak_bytes}
