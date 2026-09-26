"""The Image-to-3D preprocessing API.

    preprocess_object     photo -> clean RGBA object (Qwen extraction or RMBG-2.0)
    edit_object           object-preserving image-to-image edit (optional mask)
    complete_occlusion    mask/rect/circle region completion
    texture_preprocess    lighting / shadow / specular cleanup for texturing
    generate_view(s)      one or many requested views of a reference object
    generate_multiview    views + metadata.json + transforms.json + validation.json
    generate_turntable    evenly spaced views at one elevation (flat layout)
    generate_image_to_3d_dataset   the whole chain from one photo

Every function takes a Backend (backends.select_backend) so the model is
loaded once and reused across calls; images go to disk as they are produced
and are never held in memory as a set.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import imageops, views as viewlib
from .backends import Backend, BackendError, GenRequest
from .dataset import DatasetWriter
from .validate import validate_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]

PRESERVE_CLAUSE = (
    "Keep the main object's identity, geometry, silhouette, proportions, position and scale unchanged, and "
    "preserve its colors, materials, fine surface details, text and logos, except where the instruction "
    "explicitly asks for a change. Do not add new objects and do not remove parts of the object.")

FIDELITY_CLAUSE = (
    "This is preparation for 3D reconstruction, so fidelity matters more than looks: do not beautify, "
    "stylize, sharpen, smooth or add detail, and keep the exact colors, materials, surface textures, text "
    "and logos.")

EXTRACT_PROMPT = (
    "Extract the main object from the image onto a transparent background. Keep the object exactly as it "
    "is: the same shape, silhouette, proportions, position and scale, colors, materials, text and logos. "
    "Remove everything else, including the background, supporting surfaces and cast shadows. "
    + viewlib.BACKGROUND_CLAUSES["transparent"])

TEXTURE_OPERATIONS = {
    "neutralize-lighting": "Relight the object with soft, even, neutral studio lighting.",
    "reduce-shadows": "Remove cast shadows and soften strong self-shadowing.",
    "reduce-specular": "Reduce strong specular highlights and glare so the underlying base color is visible.",
    "remove-reflections": "Remove reflections of the surroundings from the object's surfaces.",
    "repair-defects": "Repair small scratches, dust, noise and compression artifacts.",
    "remove-background": viewlib.BACKGROUND_CLAUSES["transparent"],
}
DEFAULT_TEXTURE_OPERATIONS = ("neutralize-lighting", "reduce-shadows", "reduce-specular")

OCCLUSION_INSTRUCTION = (
    "Reconstruct the hidden or missing part of the object inside the marked region so that it continues the "
    "surrounding geometry, materials and colors seamlessly.")


@dataclass
class ViewParams:
    steps: int = 20
    seed: int = 0
    seed_mode: str = "shared"
    background: str = "transparent"
    template: str | None = None
    instruction: str | None = None
    negative_prompt: str | None = None
    true_cfg_scale: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


def _output_size(width: int, height: int, max_side: int = 1024) -> tuple:
    """The source aspect ratio at no more than 1024^2, multiples of 32."""
    area = min(width * height, max_side * max_side)
    return imageops.snap_size(width, height, area)


def _as_rgba_file(image, workdir: Path, name: str) -> Path:
    """A path to an RGBA PNG of `image` (path or array), in one colour form
    for every consumer."""
    path = workdir / f"{name}.png"
    rgba = imageops.load_rgba(image) if not isinstance(image, np.ndarray) else image
    imageops.save_png(rgba, path)
    return path


# ---- single-image operations -------------------------------------------

def preprocess_object(image, out, backend: Backend | None = None, *, method: str = "qwen",
                      pixels: str = "original", size: tuple | None = None, fill: float | None = 0.85,
                      pad: int = 0, center: bool = True, normalize_scale: bool = True, crop: bool = False,
                      steps: int = 20, seed: int = 0, align_threshold: float = 0.85) -> dict:
    """Object extraction to a clean RGBA PNG.

    method: "qwen" (the model's native subject extraction), "rmbg" (RMBG-2.0
    matting) or "alpha" (the input already has a usable alpha channel).
    pixels: "original" keeps the source RGB and takes only the alpha from the
    extraction, so no pixel of the object is regenerated; this needs the
    extraction to stay aligned with the source and falls back to the generated
    pixels (with a warning) when it measurably is not. "generated" keeps the
    model's RGBA as is.
    Then frames the object: fill/pad/center/crop (see imageops.normalize_object)."""
    if method not in ("qwen", "rmbg", "alpha"):
        raise ValueError(f"method must be qwen, rmbg or alpha, got {method!r}")
    if pixels not in ("original", "generated"):
        raise ValueError(f"pixels must be original or generated, got {pixels!r}")
    started = time.perf_counter()
    source = imageops.load_rgba(image)
    info = {"method": method, "pixels": pixels, "source_size": [source.shape[1], source.shape[0]],
            "warnings": []}
    with tempfile.TemporaryDirectory(prefix="qimg21-i23d-") as td:
        work = Path(td)
        if method == "alpha":
            if source[..., 3].min() == 255:
                raise ValueError("method alpha needs an input with transparency; its alpha channel is opaque")
            extracted = source
        elif method == "rmbg":
            from .rmbg import remove_background
            alpha = remove_background(source)
            extracted = source.copy()
            extracted[..., 3] = alpha
        else:
            if backend is None:
                raise BackendError("method qwen needs a backend")
            width, height = _output_size(source.shape[1], source.shape[0])
            source_file = _as_rgba_file(source, work, "source")
            result = backend.generate(GenRequest(prompt=EXTRACT_PROMPT, out=work / "extracted.png", width=width,
                                                 height=height, steps=steps, seed=seed,
                                                 references=(source_file,)))
            info["generation"] = {"seconds": round(result.seconds, 3), "backend": result.backend,
                                  "width": width, "height": height, "steps": steps, "seed": seed}
            generated = imageops.load_rgba(result.path)
            generated = imageops.resize_rgba(generated, source.shape[1], source.shape[0])
            extracted = generated
            if pixels == "original":
                score = alignment(source, generated)
                info["alignment"] = round(score, 4)
                if score >= align_threshold:
                    extracted = source.copy()
                    extracted[..., 3] = generated[..., 3]
                else:
                    info["warnings"].append(
                        f"the extraction moved or reshaped the object (alignment {score:.3f} < "
                        f"{align_threshold}); kept the generated pixels instead of the originals")
        if normalize_scale or center or crop or size:
            framed, transform = imageops.normalize_object(extracted, size=size, fill=fill if normalize_scale else None,
                                                          pad=pad, center=center, crop=crop)
            info["transform"] = transform.to_dict()
        else:
            framed = extracted
        imageops.save_png(framed, out)
    info.update(output=str(out), seconds=round(time.perf_counter() - started, 3),
                output_size=[framed.shape[1], framed.shape[0]])
    return info


def alignment(source: np.ndarray, generated: np.ndarray) -> float:
    """How well a generated extraction lines up with the source photo: the
    correlation of their luminance inside the generated foreground (1.0 is a
    perfect match). A shifted or re-drawn object scores low."""
    mask = generated[..., 3] > 127
    if mask.sum() < 64:
        return 0.0
    lum = lambda rgba: rgba[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    a, b = lum(source)[mask], lum(generated)[mask]
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def edit_object(image, instruction: str, out, backend: Backend, *, strength: float = 1.0, mask=None,
                rect=None, circle=None, mask_feather: int = 0, keep_background: bool = False,
                transparent: bool = False, size: tuple | None = None, steps: int = 20, seed: int = 0,
                preserve: bool = True, mask_as_reference: bool = True) -> dict:
    """Object-preserving image-to-image edit.

    strength 1 edits through the model's image conditioning alone; below 1 the
    edit starts from the source image itself (SDEdit), which holds geometry
    tighter. A mask/rect/circle limits the change: it is enforced in the
    denoise loop and the original pixels are pasted back outside it."""
    if not instruction or not instruction.strip():
        raise ValueError("an edit needs an instruction")
    started = time.perf_counter()
    source = imageops.load_rgba(image)
    width, height = size or _output_size(source.shape[1], source.shape[0])
    prompt = instruction.strip().rstrip(".") + "."
    if preserve:
        prompt += " " + PRESERVE_CLAUSE
    if transparent:
        prompt += " " + viewlib.BACKGROUND_CLAUSES["transparent"]
    with tempfile.TemporaryDirectory(prefix="qimg21-i23d-") as td:
        work = Path(td)
        resized = source if source.shape[:2] == (height, width) else imageops.resize_rgba(source, width, height)
        source_file = _as_rgba_file(resized, work, "source")
        mask_array = None
        if any(v is not None for v in (mask, rect, circle)):
            scale_x, scale_y = width / source.shape[1], height / source.shape[0]
            rect_s = None if rect is None else (rect[0] * scale_x, rect[1] * scale_y, rect[2] * scale_x, rect[3] * scale_y)
            circle_s = None if circle is None else (circle[0] * scale_x, circle[1] * scale_y,
                                                     circle[2] * (scale_x + scale_y) / 2)
            mask_array = imageops.make_mask((width, height), image=mask, rect=rect_s, circle=circle_s,
                                            resize=True, feather=mask_feather)
            mask_file = work / "mask.png"
            imageops.save_png(mask_array, mask_file, mode="L")
        request = GenRequest(prompt=prompt, out=work / "edited.png", width=width, height=height, steps=steps,
                             seed=seed, references=(source_file,),
                             init_image=source_file if (strength < 1.0 or mask_array is not None) else None,
                             strength=strength, mask=mask_file if mask_array is not None else None,
                             mask_as_reference=mask_as_reference)
        result = backend.generate(request)
        edited = imageops.load_rgba(result.path)
        if mask_array is not None:
            edited = imageops.paste_outside(resized, edited, mask_array)
        if keep_background and not transparent:
            # Keep the source's alpha: the edit changes pixels, not coverage.
            edited[..., 3] = resized[..., 3]
        imageops.save_png(edited, out)
    return {"output": str(out), "prompt": prompt, "strength": strength, "masked": mask_array is not None,
            "width": width, "height": height, "steps": steps, "seed": seed, "backend": result.backend,
            "seconds": round(time.perf_counter() - started, 3)}


def complete_occlusion(image, out, backend: Backend, *, mask=None, rect=None, circle=None,
                       instruction: str | None = None, **options) -> dict:
    """Fill an occluded or missing region (mask, rect or circle) so it
    continues the visible object; nothing outside the region changes."""
    if all(v is None for v in (mask, rect, circle)):
        raise ValueError("occlusion completion needs the region: a mask image, a rect or a circle")
    text = instruction or OCCLUSION_INSTRUCTION
    return edit_object(image, text, out, backend, mask=mask, rect=rect, circle=circle, **options)


def texture_preprocess(image, out, backend: Backend, *, operations=DEFAULT_TEXTURE_OPERATIONS,
                       strength: float = 1.0, **options) -> dict:
    """Texture-friendly cleanup: flatter lighting, fewer shadows and highlights,
    with colors, materials, text and logos kept -- never a beautification."""
    unknown = [op for op in operations if op not in TEXTURE_OPERATIONS]
    if unknown:
        raise ValueError(f"unknown texture operations {unknown}; choose from {', '.join(TEXTURE_OPERATIONS)}")
    if not operations:
        raise ValueError("choose at least one texture operation")
    instruction = " ".join(TEXTURE_OPERATIONS[op] for op in operations if op != "remove-background")
    transparent = "remove-background" in operations
    instruction = (instruction + " " + FIDELITY_CLAUSE).strip()
    info = edit_object(image, instruction, out, backend, strength=strength, transparent=transparent, **options)
    info["operations"] = list(operations)
    return info


# ---- views ---------------------------------------------------------------

def generate_view(references, spec: viewlib.ViewSpec, out, backend: Backend,
                  params: ViewParams = ViewParams()) -> dict:
    """One requested view of the reference object. Reproducible from the
    references, params.seed/seed_mode, the spec and the generation params."""
    spec = spec.validated()
    refs = tuple(Path(r) for r in references)
    if not refs:
        raise ValueError("a view needs at least one reference image")
    seed = viewlib.derive_seed(params.seed, spec, params.seed_mode)
    prompt = viewlib.view_prompt(spec, background=params.background, template=params.template,
                                 instruction=params.instruction)
    result = backend.generate(GenRequest(prompt=prompt, out=Path(out), width=spec.width, height=spec.height,
                                         steps=params.steps, seed=seed, references=refs,
                                         negative_prompt=params.negative_prompt,
                                         true_cfg_scale=params.true_cfg_scale, tags={"view": spec.to_dict()}))
    return {"file": str(out), "seed": seed, "prompt": prompt, "seconds": round(result.seconds, 3),
            "backend": result.backend, "spec": spec.to_dict(), "details": result.details}


def generate_views(references, specs, outs, backend: Backend, params: ViewParams = ViewParams(), *,
                   batch_size: int = 1, on_view=None):
    """Views in the given order, written as they finish. batch_size groups
    requests for backends that can batch (TorchBackend); it never changes a
    view's prompt, seed or conditioning, only how many run together. Yields one
    record per view; nothing is kept in memory."""
    specs, outs = list(specs), list(outs)
    if len(specs) != len(outs):
        raise ValueError("one output path per view")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    batcher = getattr(backend, "generate_batch", None)
    for start in range(0, len(specs), batch_size):
        chunk = list(zip(specs[start:start + batch_size], outs[start:start + batch_size]))
        if batcher and len(chunk) > 1:
            refs = tuple(Path(r) for r in references)
            requests, meta = [], []
            for spec, out in chunk:
                spec = spec.validated()
                seed = viewlib.derive_seed(params.seed, spec, params.seed_mode)
                prompt = viewlib.view_prompt(spec, background=params.background, template=params.template,
                                             instruction=params.instruction)
                requests.append(GenRequest(prompt=prompt, out=Path(out), width=spec.width, height=spec.height,
                                           steps=params.steps, seed=seed, references=refs,
                                           negative_prompt=params.negative_prompt,
                                           true_cfg_scale=params.true_cfg_scale))
                meta.append((spec, seed, prompt, out))
            for (spec, seed, prompt, out), result in zip(meta, batcher(requests)):
                record = {"file": str(out), "seed": seed, "prompt": prompt, "seconds": round(result.seconds, 3),
                          "backend": result.backend, "spec": spec.to_dict(), "details": result.details}
                if on_view:
                    on_view(record)
                yield record
        else:
            for spec, out in chunk:
                record = generate_view(references, spec, out, backend, params)
                if on_view:
                    on_view(record)
                yield record


def generate_multiview(references, specs, root, backend: Backend, params: ViewParams = ViewParams(), *,
                       layout: str | None = None, batch_size: int = 1, include_reference: bool = True,
                       reference_spec: viewlib.ViewSpec | None = None, preprocessing: dict | None = None,
                       model: str = "", validate: bool = True, on_view=None) -> dict:
    """Generate `specs` from the shared references into a dataset at `root`.

    The references are copied into root/reference/ exactly as they are fed to
    the model, so the dataset records what every view was conditioned on.
    Generated views are never fed back as references."""
    specs = [s.validated() for s in specs]
    if not specs:
        raise ValueError("no views requested")
    sizes = {(s.width, s.height) for s in specs}
    if len(sizes) != 1:
        raise ValueError(f"all views must share one output size, got {sorted(sizes)}")
    if layout is None:
        layout = "rings" if len({s.elevation_deg for s in specs}) > 1 else "flat"
    writer = DatasetWriter(root, layout)
    names = [writer.relpath(i, s) for i, s in enumerate(specs)]
    if len(set(names)) != len(names):
        raise ValueError("two views map to the same file; views must differ in azimuth or elevation")
    ref_dir = Path(root) / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_files, ref_meta = [], []
    for index, ref in enumerate(references):
        copied = ref_dir / f"ref_{index:02d}.png"
        imageops.save_png(imageops.load_rgba(ref), copied)
        ref_files.append(copied)
        ref_meta.append({"file": f"reference/{copied.name}", "source": str(ref), "sha256": imageops.sha256_file(copied)})
    records = []
    for record in generate_views(ref_files, specs, [writer.path(i, s) for i, s in enumerate(specs)], backend,
                                 params, batch_size=batch_size, on_view=on_view):
        index = len(records)
        record["file"] = names[index]
        record["index"] = index
        records.append(record)
    views = [{"index": r["index"], "file": r["file"], "azimuth_deg": r["spec"]["azimuth_deg"],
              "elevation_deg": r["spec"]["elevation_deg"], "roll_deg": r["spec"]["roll_deg"],
              "distance": r["spec"]["distance"], "fov_deg": r["spec"]["fov_deg"],
              "projection": r["spec"]["projection"], "width": r["spec"]["width"], "height": r["spec"]["height"],
              "seed": r["seed"], "prompt": r["prompt"], "seconds": r["seconds"], "backend": r["backend"]}
             for r in records]
    backend_name = records[0]["backend"] if records else getattr(backend, "name", "?")
    writer.write_metadata(model=model, backend=backend_name, seed=params.seed, seed_mode=params.seed_mode,
                          params={**params.to_dict(), "batch_size": batch_size}, references=ref_meta,
                          preprocessing=preprocessing, views=views, repo_root=REPO_ROOT)
    reference_entry = None
    if include_reference and ref_files:
        spec0 = reference_spec or viewlib.ViewSpec(0.0, 0.0, width=specs[0].width, height=specs[0].height)
        reference_entry = (ref_meta[0]["file"], spec0.validated())
    writer.write_transforms([(names[i], s) for i, s in enumerate(specs)], reference=reference_entry)
    summary = {"root": str(root), "views": len(records), "layout": layout,
               "seconds": round(sum(r["seconds"] for r in records), 3)}
    if validate:
        report = validate_dataset(root, names, width=specs[0].width, height=specs[0].height,
                                  expect_alpha=params.background == "transparent")
        summary["validation"] = report["summary"]
    return summary


def generate_turntable(references, root, backend: Backend, *, views: int = 24, elevation_deg: float = 0.0,
                       width: int = 512, height: int = 512, params: ViewParams = ViewParams(), **options) -> dict:
    specs = viewlib.turntable(views, elevation_deg, width=width, height=height)
    return generate_multiview(references, specs, root, backend, params, layout="flat", **options)


def generate_image_to_3d_dataset(image, root, backend: Backend, *, azimuth_views: int = 24,
                                 elevations=(0.0,), width: int = 512, height: int = 512,
                                 extract: str | None = "qwen", pixels: str = "original", fill: float = 0.85,
                                 cleanup=(), params: ViewParams = ViewParams(), top_bottom: bool = False,
                                 projection: str = "perspective", batch_size: int = 1, extra_references=(),
                                 model: str = "", on_view=None) -> dict:
    """Photo -> object extraction -> RGBA normalization -> optional cleanup ->
    views -> metadata.json / transforms.json / validation.json."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prep_dir = root / "preprocess"
    prep_dir.mkdir(exist_ok=True)
    preprocessing = {}
    prepared = prep_dir / "object_rgba.png"
    if extract:
        preprocessing["extract"] = preprocess_object(image, prepared, backend, method=extract, pixels=pixels,
                                                     size=(width, height), fill=fill, steps=params.steps,
                                                     seed=params.seed)
    else:
        framed, transform = imageops.normalize_object(imageops.load_rgba(image), size=(width, height), fill=fill)
        imageops.save_png(framed, prepared)
        preprocessing["extract"] = {"method": "none", "transform": transform.to_dict()}
    if cleanup:
        cleaned = prep_dir / "object_clean.png"
        preprocessing["cleanup"] = texture_preprocess(prepared, cleaned, backend, operations=tuple(cleanup),
                                                      size=(width, height), steps=params.steps, seed=params.seed,
                                                      keep_background=True)
        prepared = cleaned
    common = {"width": width, "height": height, "projection": projection}
    specs = viewlib.rings(azimuth_views, list(elevations), **common) if len(elevations) > 1 else \
        viewlib.ring(azimuth_views, elevations[0], **common)
    if top_bottom:
        specs += viewlib.top_bottom(**common)
    layout = "rings" if len(elevations) > 1 or top_bottom else "flat"
    summary = generate_multiview([prepared, *extra_references], specs, root, backend, params, layout=layout,
                                 batch_size=batch_size, preprocessing=preprocessing, model=model, on_view=on_view)
    summary["preprocessing"] = preprocessing
    (root / "pipeline.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary
