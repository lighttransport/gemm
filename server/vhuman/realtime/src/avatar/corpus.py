"""Initial neutral-only registered corpus from a clean vhuman reconstruction.

This proves fitting and rendering. Expression registration and photographic
segmentation must precede production use; a projected convex mask is diagnostic.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from .rig import RigAvatar
from .provenance import verify_files, sha256
from ....reconstruction.reference import Camera, srgb_to_linear


def export_neutral(head, identity, output, build_dir):
    head, identity, output = Path(head), Path(identity), Path(output)
    spec = json.loads((identity / "manifest.json").read_text())
    verify_files(spec["provenance"], identity)
    if sha256(head / "portrait.png") != sha256(identity / "neutral.png"):
        raise ValueError("rig portrait differs from clean identity")
    reports = sorted((head / "reconstruction").glob("*/manifest.json"))
    if len(reports) != 1: raise ValueError("neutral exporter requires an unambiguous reconstruction")
    report = json.loads(reports[0].read_text())
    camera = Camera.from_dict(report["geometry"]["fitted_cameras"][0])
    rig_dir = reports[0].parent / "rig"
    rig = RigAvatar(rig_dir, build_dir)
    try:
        vertices = rig.deform(np.zeros(len(rig.names), np.float32))
        image = np.asarray(Image.open(identity / "neutral.png").convert("RGB"), np.float32) / 255
        xy, depth = camera.project(vertices)
        valid = (depth > .02) & np.isfinite(xy).all(1)
        if valid.sum() < 3: raise ValueError("no valid projected neutral rig")
        outline = xy[valid][ConvexHull(xy[valid]).vertices]
        mask_image = Image.new("L", (image.shape[1], image.shape[0]))
        ImageDraw.Draw(mask_image).polygon([tuple(p) for p in outline], fill=255)
        mask = np.asarray(mask_image, np.float32) / 255
        target = srgb_to_linear(image) * mask[..., None]
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = np.diag([1, -1, -1]) @ camera.rotation
        view[:3, 3] = -view[:3, :3] @ camera.origin
        K = np.array([[camera.focal, camera.skew, camera.cx],
                      [0, camera.focal_y or camera.focal, camera.cy], [0, 0, 1]], np.float32)
        # Exact camera convention parity before allowing a corpus to be written.
        p = vertices @ view[:3, :3].T + view[:3, 3]
        projected = p @ K.T; projected = projected[:, :2] / projected[:, 2:3]
        error = float(np.max(abs(projected[valid] - xy[valid])))
        if error > .01: raise ValueError("camera conversion exceeds .01 pixel")
        output.mkdir(parents=True, exist_ok=True)
        data = output / "frames.npz"
        np.savez_compressed(data, vertices=vertices[None], triangles=rig.triangles, images=target[None], masks=mask[None],
                             controls=np.zeros((1, len(rig.names)), np.float32), view=view[None], intrinsics=K[None])
        Image.fromarray(np.uint8(mask * 255)).save(output / "mask.png")
        Image.open(identity / "neutral.png").save(output / "neutral.png")
        origin = next(r for r in spec["provenance"] if r["path"] == "neutral.png")
        receipts = [dict(origin, path="neutral.png", sha256=sha256(output / "neutral.png")),
                    dict(path="frames.npz", source="original neutral corpus from clean FLUX portrait and GNM rig",
                         revision="neutral-v1", license="Apache-2.0", roles=["appearance-training"], sha256=sha256(data),
                         identity_sha256=origin["sha256"], rig_sha256=sha256(rig_dir / "rig_deformer.safetensors"))]
        manifest = output / "manifest.json"
        manifest.write_text(json.dumps(dict(format="vhuman.appearance_corpus.v1", purpose="diagnostic", data=data.name,
             control_names=list(rig.names), provenance=receipts, camera_projection_max_error_px=error,
             registration="single neutral fit; projected convex mask; expression and mouth quality unvalidated"), indent=2))
        return dict(manifest=str(manifest), camera_projection_max_error_px=error)
    finally: rig.close()
