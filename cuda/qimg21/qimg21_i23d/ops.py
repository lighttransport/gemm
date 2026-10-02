"""The Image-to-3D preprocessing API.

    preprocess_object     photo -> clean RGBA object (Qwen extraction or RMBG-2.0)
    edit_object           object-preserving image-to-image edit (optional mask)
    complete_occlusion    mask/rect/circle region completion
    texture_preprocess    lighting / shadow / specular cleanup for texturing
    generate_view(s)      one or many requested views of a reference object
    generate_multiview    views + metadata.json + transforms.json + validation.json
    generate_turntable    evenly spaced views at one elevation (flat layout)
    generate_image_to_3d_dataset   the whole chain from one photo, optionally
                          edited, optionally reconstructed with Pixal3D
    reconstruct_3d        an RGBA object or a view dataset -> GLB (Pixal3D native
                          runner and/or PyTorch reference)

Every function takes a Backend (backends.select_backend) so the model is
loaded once and reused across calls; images go to disk as they are produced
and are never held in memory as a set.
"""
from __future__ import annotations

import json
import math
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


def _scratch() -> Path:
    """Scratch space inside the repository (AGENTS.md: tmp/, not /tmp)."""
    path = REPO_ROOT / "tmp"
    path.mkdir(exist_ok=True)
    return path


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
    with tempfile.TemporaryDirectory(prefix="qimg21-i23d-", dir=_scratch()) as td:
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
    with tempfile.TemporaryDirectory(prefix="qimg21-i23d-", dir=_scratch()) as td:
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

def _view_request(refs: tuple, spec: viewlib.ViewSpec, out, params: ViewParams):
    spec = spec.validated()
    seed = viewlib.derive_seed(params.seed, spec, params.seed_mode)
    prompt = viewlib.view_prompt(spec, background=params.background, template=params.template,
                                 instruction=params.instruction)
    request = GenRequest(prompt=prompt, out=Path(out), width=spec.width, height=spec.height, steps=params.steps,
                         seed=seed, references=refs, negative_prompt=params.negative_prompt,
                         true_cfg_scale=params.true_cfg_scale, tags={"view": spec.to_dict()})
    return request, spec


def _view_record(request: GenRequest, spec: viewlib.ViewSpec, result) -> dict:
    return {"file": str(request.out), "seed": request.seed, "prompt": request.prompt,
            "seconds": round(result.seconds, 3), "backend": result.backend, "spec": spec.to_dict(),
            "details": result.details}


def _references(references) -> tuple:
    refs = tuple(Path(r) for r in references)
    if not refs:
        raise ValueError("a view needs at least one reference image")
    return refs


def generate_view(references, spec: viewlib.ViewSpec, out, backend: Backend,
                  params: ViewParams = ViewParams()) -> dict:
    """One requested view of the reference object. Reproducible from the
    references, params.seed/seed_mode, the spec and the generation params."""
    request, spec = _view_request(_references(references), spec, out, params)
    return _view_record(request, spec, backend.generate(request))


def generate_views(references, specs, outs, backend: Backend, params: ViewParams = ViewParams(), *,
                   batch_size: int = 1, on_view=None):
    """Views in the given order, written as they finish. batch_size groups
    requests for backends that can batch (TorchBackend); it never changes a
    view's prompt, seed or conditioning, only how many run together. Yields one
    record per view; nothing is kept in memory.

    A backend with prepare() sees every request first (NativeBackend encodes
    all the view prompts in one text-encoder pass); that changes when work
    happens, never a view's result."""
    specs, outs = list(specs), list(outs)
    if len(specs) != len(outs):
        raise ValueError("one output path per view")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    refs = _references(references)
    planned = [_view_request(refs, spec, out, params) for spec, out in zip(specs, outs)]
    prepare = getattr(backend, "prepare", None)
    if prepare:
        prepare([request for request, _ in planned])
    batcher = getattr(backend, "generate_batch", None)
    for start in range(0, len(planned), batch_size):
        chunk = planned[start:start + batch_size]
        if batcher and len(chunk) > 1:
            results = batcher([request for request, _ in chunk])
        else:
            results = (backend.generate(request) for request, _ in chunk)   # lazily, one at a time
        for (request, spec), result in zip(chunk, results):
            record = _view_record(request, spec, result)
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
              "seed": r["seed"], "prompt": r["prompt"], "seconds": r["seconds"], "backend": r["backend"],
              "timings": (r.get("details") or {}).get("timings")}
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


OBJECT_CLAUSE = ("A single complete object, centered, fully inside the frame, on its own, with nothing else in "
                 "the scene. Neutral studio lighting.")


def generate_object(prompt: str, out, backend: Backend, *, width: int = 1024, height: int = 1024,
                    size=(512, 512), fill: float = 0.85, transparent: bool = True, steps: int = 20, seed: int = 0,
                    negative_prompt: str | None = None) -> dict:
    """Text -> one framed RGBA object (the studio's first step).

    Qwen-Image 2.1 draws the object alone, on a native transparent
    background by default; the result is framed like preprocess_object
    (centred, longest side `fill` of a `size` canvas; size None keeps the
    generated canvas). The unframed image is kept as <out>_raw.png."""
    if not prompt or not prompt.strip():
        raise ValueError("an object needs a prompt")
    out = Path(out)
    raw = out.with_name(out.stem + "_raw.png")
    full = " ".join([prompt.strip(), OBJECT_CLAUSE,
                     viewlib.BACKGROUND_CLAUSES["transparent" if transparent else "white"]])
    result = backend.generate(GenRequest(prompt=full, out=raw, width=width, height=height, steps=steps, seed=seed,
                                         negative_prompt=negative_prompt))
    method = "alpha" if (imageops.load_rgba(raw)[..., 3] < 250).any() else "rmbg"
    info = preprocess_object(raw, out, backend, method=method, size=size, fill=fill, steps=steps, seed=seed)
    return {"output": str(out), "raw": str(raw), "prompt": full, "seconds": round(result.seconds, 3),
            "backend": result.backend, "extraction": info["method"], "transform": info.get("transform"),
            "timings": result.details.get("timings")}


TURNAROUND_VIEWS = {
    4: ((0.0, "front view facing the viewer"), (90.0, "side view with the character facing left"),
        (180.0, "back view"), (270.0, "side view with the character facing right")),
    3: ((0.0, "front view facing the viewer"), (90.0, "side view with the character facing left"),
        (180.0, "back view")),
}
TURNAROUND_TEMPLATE = (
    "Character turnaround reference sheet of {subject}. {count} views of the same character side by side in one "
    "row, evenly spaced, same size, same height and same ground line, orthographic, no overlap: from left to "
    "right, {order}. Consistent proportions and details in every view. No text, no labels. {background}")


def turnaround_prompt(subject: str, views: int = 4, background: str = "transparent") -> str:
    order = ", ".join(f"({i + 1}) {words}" for i, (_, words) in enumerate(TURNAROUND_VIEWS[views]))
    return " ".join(TURNAROUND_TEMPLATE.format(subject=subject, count=views, order=order,
                                               background=viewlib.BACKGROUND_CLAUSES[background]).split())


def generate_turnaround(root, backend: Backend, *, prompt: str | None = None, reference=None, views: int = 4,
                        size: int = 512, fill: float = 0.85, steps: int = 20, seed: int = 0,
                        negative_prompt: str | None = None, sheet=None) -> dict:
    """Consistent posed views of a character from ONE turnaround sheet.

    Views generated one by one (generate_views) drift: each is a separate
    sample. A turnaround sheet -- front, left profile, back (and right
    profile) side by side in one image -- is one sample, so the character,
    its proportions and details agree across views far better. The sheet is
    drawn at size x (views * size) on a transparent background, from `prompt`
    (text) and/or a `reference` image of the character, split into views
    with one shared scale and ground line (imageops.split_sheet) and written
    as a dataset: views/view_aNNN.png, sheet.png, metadata.json,
    transforms.json (front view first; cameras requested, not calibrated),
    validation.json.

    sheet: an existing turnaround sheet (RGBA, one row, views in the order
    above -- e.g. drawn by hand or kept from an earlier run) to split instead
    of generating one."""
    if views not in TURNAROUND_VIEWS:
        raise ValueError(f"views must be one of {sorted(TURNAROUND_VIEWS)}")
    if not prompt and reference is None and sheet is None:
        raise ValueError("a turnaround needs a prompt, a reference image, or an existing sheet")
    root = Path(root)
    (root / "views").mkdir(parents=True, exist_ok=True)
    subject = prompt or "the character in the reference image"
    if reference is not None:
        subject += (", exactly as in the reference image: the same identity, shapes, colors, materials and "
                    "details")
    text = turnaround_prompt(subject, views)
    started = time.perf_counter()
    if sheet is not None:
        imageops.save_png(imageops.load_rgba(sheet), root / "sheet.png")
        text, backend_name = None, "given sheet"
    else:
        result = backend.generate(GenRequest(prompt=text, out=root / "sheet.png", width=size * views, height=size,
                                             steps=steps, seed=seed,
                                             references=(Path(reference),) if reference else (),
                                             negative_prompt=negative_prompt))
        backend_name = result.backend
    # Front/back (and left/right profile) are 180 degrees apart: register
    # each opposite view on its partner's mirror image.
    registration: list = []
    panels, boxes = imageops.split_sheet(imageops.load_rgba(root / "sheet.png"), views, size=size, fill=fill,
                                         register=((0, 2), (1, 3)) if views == 4 else ((0, 2),),
                                         report=registration)
    frames, files, records = [], [], []
    for (azimuth, words), panel, box in zip(TURNAROUND_VIEWS[views], panels, boxes):
        spec = viewlib.ViewSpec(azimuth_deg=azimuth, width=size, height=size, tags={"turnaround": True}).validated()
        name = f"views/view_a{int(azimuth):03d}.png"
        imageops.save_png(panel, root / name)
        frames.append((name, spec))
        files.append(name)
        records.append({"file": name, "azimuth_deg": azimuth, "elevation_deg": 0.0, "sheet_box": list(box),
                        "view": words})
    writer = DatasetWriter(root, layout="flat")
    writer.write_transforms(frames)
    report = validate_dataset(root, files, width=size, height=size, expect_alpha=True)
    summary = {"root": str(root), "views": records, "prompt": text, "seed": seed, "steps": steps,
               "reference": str(reference) if reference else None, "sheet": "sheet.png",
               "seconds": round(time.perf_counter() - started, 3), "backend": backend_name,
               "validation": report["summary"], "registration": registration,
               "camera_parameters": "requested view metadata, not calibration: panels of one generated "
                                    "turnaround sheet, framed with one shared scale and ground line"}
    (root / "metadata.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def generate_image_to_3d_dataset(image, root, backend: Backend, *, azimuth_views: int = 24,
                                 elevations=(0.0,), width: int = 512, height: int = 512,
                                 extract: str | None = "qwen", pixels: str = "original", fill: float = 0.85,
                                 cleanup=(), params: ViewParams = ViewParams(), top_bottom: bool = False,
                                 projection: str = "perspective", batch_size: int = 1, extra_references=(),
                                 model: str = "", on_view=None, edit: dict | None = None,
                                 reconstructors=(), recon_mode: str = "single", recon_fov_deg=None,
                                 recon_elevations=(0.0,), recon_max_frames: int = 16) -> dict:
    """Photo -> object extraction -> RGBA normalization -> optional cleanup ->
    optional edit -> views -> metadata.json / transforms.json /
    validation.json -> optional Pixal3D reconstruction (reconstruct_3d).

    edit: edit_object keyword arguments plus "instruction" (e.g. {"instruction":
    "make the base blue", "rect": (x, y, w, h)}); the edited object is what the
    views and the reconstruction see. azimuth_views=0 (and no top/bottom)
    generates no views.

    recon_mode "single" (the default) reconstructs from the prepared object
    alone; "multiview" poses the generated views with their requested
    cameras. Measured against Pixal3D on real renders of the same object,
    single view was clearly better (Chamfer RMS 0.024 vs 0.063): the model
    keeps a view's outline close to the reference's width, so a deep object's
    generated side views are too narrow to be used as calibrated cameras. The
    image backend is closed (resident processes and device memory released)
    before Pixal3D runs."""
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
    if edit:
        options = dict(edit)
        instruction = options.pop("instruction")
        edited = prep_dir / "object_edited.png"
        preprocessing["edit"] = edit_object(prepared, instruction, edited, backend, transparent=True,
                                            size=(width, height), steps=params.steps, seed=params.seed, **options)
        prepared = edited
    if recon_mode not in ("single", "multiview"):
        raise ValueError("recon_mode must be single or multiview")
    if azimuth_views < 0:
        raise ValueError("azimuth_views must be >= 0")
    if reconstructors and recon_mode == "multiview" and not azimuth_views:
        raise ValueError("multiview reconstruction needs generated views (azimuth_views > 0)")
    common = {"width": width, "height": height, "projection": projection}
    specs = []
    if azimuth_views:
        specs = viewlib.rings(azimuth_views, list(elevations), **common) if len(elevations) > 1 else \
            viewlib.ring(azimuth_views, elevations[0], **common)
    if top_bottom:
        specs += viewlib.top_bottom(**common)
    if specs:
        layout = "rings" if len(elevations) > 1 or top_bottom else "flat"
        summary = generate_multiview([prepared, *extra_references], specs, root, backend, params, layout=layout,
                                     batch_size=batch_size, preprocessing=preprocessing, model=model,
                                     on_view=on_view)
    else:
        summary = {"root": str(root), "views": 0}
    summary["prepared"] = str(prepared)
    summary["preprocessing"] = preprocessing
    if reconstructors:
        backend.close()
        if recon_mode == "single":
            summary["reconstruction"] = reconstruct_3d(prepared, root / "reconstruction", reconstructors,
                                                       mode="single", fov_deg=recon_fov_deg)
        else:
            summary["reconstruction"] = reconstruct_3d(root, root / "reconstruction", reconstructors,
                                                       mode="multiview", elevations=recon_elevations,
                                                       max_frames=recon_max_frames)
    (root / "pipeline.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def reconstruct_3d(source, out_dir, reconstructors, *, mode: str = "multiview", fov_deg=None,
                   mesh_scale: float = 1.0, elevations=(0.0,), max_frames: int = 16,
                   compare: bool = True) -> dict:
    """Reconstruct a GLB per Pixal3D runner (reconstruct.Pixal3DNative /
    Pixal3DReference) from:
    - mode "multiview": a view dataset (transforms.json + RGBA frames); the
      reference and the generated views at `elevations` (None: all) are
      staged, at most max_frames (Pixal3D's limit is 16);
    - mode "single": an RGBA object image; its camera FOV is fov_deg, or
      estimated with MoGe-2 when None.
    With two runners the meshes are compared (symmetric Chamfer). Writes
    out_dir/<runner>.glb and out_dir/reconstruction.json."""
    from . import reconstruct as recon
    if not reconstructors:
        raise ValueError("no reconstructor given")
    source = Path(source)
    if not source.exists():
        raise ValueError(f"{source} does not exist")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"mode": mode, "camera_parameters": None, "runs": []}
    if mode == "multiview":
        if not (source / "transforms.json").is_file():
            raise ValueError(f"{source} has no transforms.json; multiview reconstruction needs a view dataset")
        views, files = recon.stage_views(source, out_dir, max_frames, elevations)
        if len(files) < 2:
            raise ValueError(f"multiview reconstruction found only {len(files)} frame(s) in {source} at elevations "
                             f"{list(elevations) if elevations is not None else 'any'}; pass the elevations the "
                             "views were made at, or use single view")
        transforms = json.loads((views / "transforms.json").read_text())
        record.update(views_dir=str(views), frames=files, camera_parameters=transforms.get("camera_parameters"))
        for runner in reconstructors:
            record["runs"].append(runner.multiview(views, out_dir / f"{runner.name}.glb", out_dir / runner.name))
    elif mode == "single":
        if fov_deg is None:
            camera = recon.estimate_camera(source, out_dir, getattr(reconstructors[0], "backend", "cuda"),
                                           mesh_scale, cancel=getattr(getattr(reconstructors[0], 'settings', None), 'cancel', None))
            fov = float(camera["fov"])
            record["camera"] = {"fov_rad": fov, "source": camera.get("camera_source", "moge-2")}
        else:
            fov = math.radians(float(fov_deg))
            record["camera"] = {"fov_rad": fov, "source": "given"}
        record["input"] = str(source)
        for runner in reconstructors:
            record["runs"].append(runner.single(source, out_dir / f"{runner.name}.glb", out_dir / runner.name,
                                                fov_rad=fov, mesh_scale=mesh_scale))
    else:
        raise ValueError("mode must be single or multiview")
    if compare and len(record["runs"]) == 2:
        record["comparison"] = recon.compare_meshes(record["runs"][0]["output"], record["runs"][1]["output"],
                                                    out_dir)
    (out_dir / "reconstruction.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record
