"""Render a registered corpus and report reconstruction metrics, never held-out by default."""
import json
from pathlib import Path
import numpy as np
from ..avatar.bundle import GaussianAvatar
from ..avatar.provenance import verify_files, sha256


def evaluate(manifest, avatar_path, output):
    from contextlib import closing
    from ..renderer.gaussian import GaussianRenderer
    from PIL import Image
    manifest = Path(manifest)
    spec = json.loads(manifest.read_text())
    if spec.get("format") != "vhuman.appearance_corpus.v1": raise ValueError("unsupported corpus")
    verify_files(spec["provenance"], manifest.parent)
    data_path = (manifest.parent / spec["data"]).resolve()
    if not data_path.is_relative_to(manifest.parent.resolve()): raise ValueError("corpus escapes root")
    if spec["data"] not in {r["path"] for r in spec["provenance"]}: raise ValueError("missing data receipt")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as data:
        avatar = GaussianAvatar.load(avatar_path, data["triangles"])
        if avatar.metadata["control_names"] != spec["control_names"]: raise ValueError("control order mismatch")
        results = []
        with closing(GaussianRenderer(avatar, data["triangles"])) as renderer:
            for i, target in enumerate(data["images"]):
                if not np.isfinite(target).all(): raise ValueError("nonfinite target")
                with renderer.render(data["vertices"][i], data["view"][i], data["intrinsics"][i],
                        (target.shape[1], target.shape[0]), data["controls"][i]) as handle:
                    rgba = handle.rgba.numpy()
                error = rgba[..., :3] - target
                mask = target.max(-1) > 1e-5
                foreground = error[mask] if mask.any() else error.reshape(-1, 3)
                mse = float(np.square(error).mean())
                results.append(dict(frame=i, linear_l1=float(abs(error).mean()),
                    linear_psnr_db=float(-10*np.log10(max(mse, 1e-12))),
                    foreground_linear_l1=float(abs(foreground).mean()),
                    foreground_alpha_mean=float(rgba[..., 3][mask].mean()) if mask.any() else None))
                # Show the actual renderer RGB over black, not unpremultiplied radiance.
                linear = np.clip(rgba[..., :3], 0, 1)
                rgb = np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)
                Image.fromarray(np.uint8(np.clip(rgb*255+.5, 0, 255))).save(output / f"frame-{i:04d}.png")
    digest = sha256(data_path)
    result = dict(format="vhuman.appearance_evaluation.v1", avatar_sha256=sha256(avatar_path),
        corpus_sha256=digest, evaluation="training-reconstruction" if digest == avatar.metadata.get("corpus_sha256") else "separate-corpus",
        purpose=avatar.metadata["purpose"], backend="native_cuda", frames=results)
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return result
