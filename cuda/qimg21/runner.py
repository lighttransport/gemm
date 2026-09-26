#!/usr/bin/env python3
"""Qwen-Image 2.1 runner: Image-to-3D preprocessing commands.

    runner.py object-preprocess   photo -> RGBA object (background removal, framing)
    runner.py edit                object-preserving image-to-image edit (optional mask)
    runner.py texture-preprocess  lighting / shadow / specular cleanup for texturing
    runner.py multiview           views of a reference object (rings, top/bottom)
    runner.py turntable           evenly spaced views at one elevation
    runner.py image-to-3d         the whole chain into a dataset
    runner.py validate            re-run the 2D checks on a dataset

Text-to-image generation stays in native_generate.py. Camera values in the
outputs are requested view metadata, not calibration; see IMAGE_TO_3D.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qimg21_i23d import backends, ops, views as viewlib  # noqa: E402
from qimg21_i23d.validate import validate_dataset  # noqa: E402

DEFAULT_MODEL = "/mnt/nvme01/models/qimg-21"


def add_backend_options(parser):
    group = parser.add_argument_group("backend")
    group.add_argument("--backend", default="auto", choices=("auto", "native", "torch", "mock"),
                       help="native: the CUDA runner (one reference image); torch: the PyTorch pipeline "
                            "(up to 10 references); auto picks native when it can (default)")
    group.add_argument("--model", default=DEFAULT_MODEL, help="Qwen-Image 2.1 model directory")
    group.add_argument("--preset", default="fast12", help="native fast-runner preset (low8, low8-fp4, fast12, "
                                                          "accurate); '' for the parity harness")
    group.add_argument("--attention", default=None, choices=(None, "sage", "flash", "exact"),
                       help="native attention kernel; exact is closest to PyTorch")
    group.add_argument("--device", default="cuda", help="torch backend device")
    group.add_argument("--condition-resolution", type=int, default=1024,
                       help="reference images are shown to the model at about this many pixels squared "
                            "(256-1024, default 1024 as the pipeline does); lower is faster")
    group.add_argument("--exact-prompts", action="store_true",
                       help="native: encode each image prompt unsplit, bitwise as PyTorch does, instead of "
                            "sharing the image prefix between prompts (faster; 1 - cos ~ 4e-5)")
    group.add_argument("--no-resident", action="store_true",
                       help="native: run every image one-shot instead of keeping the denoiser and VAE "
                            "decoder loaded between images")
    group.add_argument("--steps", type=int, default=20, help="denoising steps (default 20)")
    group.add_argument("--seed", type=int, default=0, help="base seed (default 0); every output is reproducible "
                                                            "from it")


def add_mask_options(parser):
    group = parser.add_argument_group("region (255 = may change)")
    group.add_argument("--mask", help="mask image: white = edit; RGBA/LA masks use their alpha")
    group.add_argument("--mask-rect", help="x,y,w,h in source pixels")
    group.add_argument("--mask-circle", help="cx,cy,r in source pixels")
    group.add_argument("--mask-feather", type=int, default=0, help="Gaussian feather radius in pixels")
    group.add_argument("--no-mask-reference", action="store_true",
                       help="do not show the mask to the model as a reference image (the region is still "
                            "enforced)")


def add_view_options(parser):
    group = parser.add_argument_group("views")
    group.add_argument("--width", type=int, default=512)
    group.add_argument("--height", type=int, default=512)
    group.add_argument("--seed-mode", default="shared", choices=viewlib.SEED_MODES,
                       help="shared: every view starts from the same noise (default, more consistent); "
                            "per_view: a stable seed per camera")
    group.add_argument("--background", default="transparent", choices=tuple(viewlib.BACKGROUND_CLAUSES))
    group.add_argument("--instruction", help="extra text appended to every view prompt")
    group.add_argument("--template-file", help="a view prompt template (see views.DEFAULT_VIEW_TEMPLATE)")
    group.add_argument("--projection", default="perspective", choices=viewlib.PROJECTIONS,
                       help="orthographic asks for clean orthographic-like views")
    group.add_argument("--fov", type=float, default=viewlib.DEFAULT_FOV_DEG,
                       help="requested horizontal FOV in degrees (metadata; default Pixal3D's 20)")
    group.add_argument("--distance", type=float, default=viewlib.DEFAULT_DISTANCE,
                       help="requested camera distance (metadata)")
    group.add_argument("--batch-size", type=int, default=1, help="views per batch (torch backend batches; "
                                                                   "outputs do not depend on it)")
    group.add_argument("--no-reference-frame", action="store_true",
                       help="leave the reference image out of transforms.json (frame 0 by default)")


def parse_ints(text, count, name):
    if text is None:
        return None
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        raise SystemExit(f"{name} must be {count} comma-separated numbers, got {text!r}")
    if len(values) != count:
        raise SystemExit(f"{name} must be {count} comma-separated numbers, got {text!r}")
    return values


def make_backend(args, references: int = 1):
    options = {"model": args.model, "preset": args.preset or None, "attention": args.attention,
               "device": args.device, "condition_resolution": args.condition_resolution,
               "resident": not args.no_resident, "exact_prompts": args.exact_prompts}
    backend = backends.select_backend(args.backend, references=references, **options)
    args.backends.append(backend)   # closed by main(): stops resident processes
    return backend


def view_params(args) -> ops.ViewParams:
    template = Path(args.template_file).read_text() if getattr(args, "template_file", None) else None
    return ops.ViewParams(steps=args.steps, seed=args.seed, seed_mode=args.seed_mode, background=args.background,
                          template=template, instruction=args.instruction)


def view_common(args) -> dict:
    return {"width": args.width, "height": args.height, "projection": args.projection, "fov_deg": args.fov,
            "distance": args.distance}


def cmd_object_preprocess(args):
    backend = make_backend(args) if args.method == "qwen" else None
    size = (args.size, args.size) if args.size else None
    info = ops.preprocess_object(args.input, args.output, backend, method=args.method, pixels=args.pixels,
                                 size=size, fill=args.fill, pad=args.pad, center=not args.no_center,
                                 normalize_scale=not args.keep_scale, crop=args.crop, steps=args.steps,
                                 seed=args.seed)
    return info


def cmd_edit(args):
    backend = make_backend(args, 2)
    return ops.edit_object(args.input, args.instruction, args.output, backend, strength=args.strength,
                           mask=args.mask, rect=parse_ints(args.mask_rect, 4, "--mask-rect"),
                           circle=parse_ints(args.mask_circle, 3, "--mask-circle"), mask_feather=args.mask_feather,
                           mask_as_reference=not args.no_mask_reference, keep_background=args.keep_alpha,
                           transparent=args.transparent, steps=args.steps, seed=args.seed)


def cmd_texture(args):
    backend = make_backend(args)
    return ops.texture_preprocess(args.input, args.output, backend, operations=tuple(args.ops.split(",")),
                                  strength=args.strength, steps=args.steps, seed=args.seed,
                                  keep_background=not args.transparent)


def references_of(args):
    refs = list(args.reference or [])
    if args.input:
        refs.insert(0, args.input)
    if not refs:
        raise SystemExit("give --input or at least one --reference")
    return refs


def cmd_multiview(args):
    refs = references_of(args)
    common = view_common(args)
    if args.azimuth_views:
        specs = viewlib.rings(args.azimuth_views, viewlib.parse_elevations(args.elevations or "0"), **common)
    else:
        specs = viewlib.ring(args.views, args.elevation, **common)
    if args.top_bottom:
        specs += viewlib.top_bottom(**common)
    layout = "rings" if args.azimuth_views or args.top_bottom else "flat"
    backend = make_backend(args, len(refs))
    return ops.generate_multiview(refs, specs, args.output, backend, view_params(args), layout=layout,
                                  batch_size=args.batch_size, include_reference=not args.no_reference_frame,
                                  model=args.model, on_view=progress)


def cmd_turntable(args):
    refs = references_of(args)
    backend = make_backend(args, len(refs))
    specs = viewlib.turntable(args.views, args.elevation, **view_common(args))
    return ops.generate_multiview(refs, specs, args.output, backend, view_params(args), layout="flat",
                                  batch_size=args.batch_size, include_reference=not args.no_reference_frame,
                                  model=args.model, on_view=progress)


def make_reconstructors(args):
    from qimg21_i23d import reconstruct
    settings = reconstruct.ReconSettings(seed=args.recon_seed, texture_size=args.texture_size,
                                         triangle_target=args.triangle_target, vram_budget_mib=args.vram_budget_mib)
    runners = reconstruct.make_reconstructors(args.reconstruct, args.pixal3d_backend, settings)
    for runner in runners:
        ok, missing = runner.available()
        if not ok:   # fail before any image work is spent
            raise backends.BackendError(f"Pixal3D {runner.name} ({runner.backend}) is not ready: "
                                        f"missing {', '.join(missing)}")
    return runners


def recon_elevations(args):
    if args.recon_elevations == "all":
        return None
    return viewlib.parse_elevations(args.recon_elevations)


def cmd_image_to_3d(args):
    extra = list(args.reference or [])
    runners = make_reconstructors(args)
    edit = None
    if args.edit:
        edit = {"instruction": args.edit, "strength": args.edit_strength, "mask": args.mask,
                "rect": parse_ints(args.mask_rect, 4, "--mask-rect"),
                "circle": parse_ints(args.mask_circle, 3, "--mask-circle"), "mask_feather": args.mask_feather,
                "mask_as_reference": not args.no_mask_reference}
    elif args.mask or args.mask_rect or args.mask_circle:
        raise SystemExit("a mask needs --edit")
    backend = make_backend(args, 1 + len(extra))
    elevations = viewlib.parse_elevations(args.elevations) if args.elevations else [args.elevation]
    return ops.generate_image_to_3d_dataset(
        args.input, args.output, backend, azimuth_views=args.views, elevations=elevations, width=args.width,
        height=args.height, extract=None if args.keep_background else args.method, pixels=args.pixels,
        fill=args.fill, cleanup=tuple(args.cleanup.split(",")) if args.cleanup else (), params=view_params(args),
        top_bottom=args.top_bottom, projection=args.projection, batch_size=args.batch_size,
        extra_references=extra, model=args.model, on_view=progress, edit=edit, reconstructors=runners,
        recon_mode=args.recon_mode, recon_fov_deg=args.recon_fov, recon_elevations=recon_elevations(args),
        recon_max_frames=args.recon_max_frames)


def cmd_turnaround(args):
    runners = make_reconstructors(args)
    if not args.prompt and not args.input:
        raise SystemExit("give --prompt, --input (a reference image of the character), or both")
    backend = make_backend(args, 1 if args.input else 0)
    summary = ops.generate_turnaround(args.output, backend, prompt=args.prompt, reference=args.input,
                                      views=args.views, size=args.size, steps=args.steps, seed=args.seed)
    if runners:
        backend.close()
        mode = args.recon_mode if args.recon_mode_given else "multiview"
        source = Path(args.output) if mode == "multiview" else Path(args.output) / summary["views"][0]["file"]
        summary["reconstruction"] = ops.reconstruct_3d(source, Path(args.output) / "reconstruction", runners,
                                                       mode=mode, fov_deg=args.recon_fov if mode == "single"
                                                       else None)
    return summary


def cmd_reconstruct(args):
    runners = make_reconstructors(args)
    if not runners:
        raise SystemExit("--reconstruct none: nothing to do")
    source = Path(args.input)
    mode = args.recon_mode if args.recon_mode_given else ("multiview" if source.is_dir() else "single")
    return ops.reconstruct_3d(source, args.output, runners, mode=mode, fov_deg=args.recon_fov,
                              elevations=recon_elevations(args), max_frames=args.recon_max_frames)


def cmd_validate(args):
    root = Path(args.dataset)
    meta = json.loads((root / "metadata.json").read_text())
    files = [view["file"] for view in meta["views"]]
    size = (meta["views"][0]["width"], meta["views"][0]["height"]) if meta["views"] else (None, None)
    report = validate_dataset(root, files, width=size[0], height=size[1],
                              expect_alpha=meta["generation"].get("background") == "transparent")
    return report["summary"]


def progress(record):
    print(f"  view {record['index'] if 'index' in record else ''} {record['file']}: {record['seconds']:.1f} s",
          file=sys.stderr, flush=True)


def add_reconstruct_options(parser, default: str):
    group = parser.add_argument_group("Pixal3D reconstruction")
    group.add_argument("--reconstruct", default=default, choices=("none", "native", "reference", "both"),
                       help="native: cpu/pixal3d/pixal3d; reference: the pinned PyTorch pipeline; both: run "
                            f"both and compare the meshes (default {default})")
    group.add_argument("--pixal3d-backend", default="cuda", choices=("cuda", "rocm", "cpu"),
                       help="device for Pixal3D (the reference runs on cuda or rocm)")
    group.add_argument("--recon-mode", default="single", choices=("single", "multiview"),
                       help="single (default for image-to-3d; for reconstruct: an image input): the prepared "
                            "object alone -- measurably better than posing generated views; multiview: the "
                            "reference plus generated views, posed by their requested cameras")
    group.add_argument("--recon-fov", type=float,
                       help="single-view camera FOV in degrees (default: MoGe-2 estimate)")
    group.add_argument("--recon-elevations", default="0",
                       help="elevation rings whose views go to Pixal3D, comma-separated, or 'all' (default 0)")
    group.add_argument("--recon-max-frames", type=int, default=16, help="frames given to Pixal3D (at most 16)")
    group.add_argument("--recon-seed", type=int, default=42)
    group.add_argument("--texture-size", type=int, default=4096, choices=(1024, 2048, 4096))
    group.add_argument("--triangle-target", type=int, default=1_000_000)
    group.add_argument("--vram-budget-mib", type=int, help="native Pixal3D device-memory budget")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="runner.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("object-preprocess", aliases=["image-to-3d-preprocess"],
                       help="background removal and framing into an RGBA PNG")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="RGBA PNG")
    p.add_argument("--remove-background", action="store_true", help="accepted for clarity; extraction always "
                                                                     "runs unless --method alpha")
    p.add_argument("--method", default="qwen", choices=("qwen", "rmbg", "alpha"),
                   help="qwen: the model's native extraction (default); rmbg: RMBG-2.0 matting; alpha: use "
                        "the input's alpha")
    p.add_argument("--pixels", default="original", choices=("original", "generated"),
                   help="original: keep the photo's pixels, take only the alpha (default)")
    p.add_argument("--size", type=int, help="square output canvas (default: the source size)")
    p.add_argument("--fill", type=float, default=0.85, help="object's longest side / canvas (default 0.85)")
    p.add_argument("--pad", type=int, default=0, help="padding in pixels")
    p.add_argument("--keep-scale", action="store_true", help="do not rescale the object")
    p.add_argument("--no-center", action="store_true", help="do not center the object")
    p.add_argument("--crop", action="store_true", help="crop to the object (plus --pad)")
    add_backend_options(p)
    p.set_defaults(run=cmd_object_preprocess)

    p = sub.add_parser("edit", help="object-preserving image-to-image edit")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--instruction", "--edit", required=True, help='e.g. "remove the person\'s hand"')
    p.add_argument("--strength", type=float, default=1.0,
                   help="1: edit through image conditioning (default); below 1 also starts from the source "
                        "(tighter geometry)")
    p.add_argument("--keep-alpha", action="store_true", help="keep the input's alpha channel")
    p.add_argument("--transparent", action="store_true", help="ask for a transparent background")
    add_mask_options(p)
    add_backend_options(p)
    p.set_defaults(run=cmd_edit)

    p = sub.add_parser("texture-preprocess", help="texture-friendly cleanup")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ops", default=",".join(ops.DEFAULT_TEXTURE_OPERATIONS),
                   help=f"comma-separated: {', '.join(ops.TEXTURE_OPERATIONS)}")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--transparent", action="store_true", help="output a transparent background")
    add_backend_options(p)
    p.set_defaults(run=cmd_texture)

    for name, runner, help_text in (("multiview", cmd_multiview, "views of a reference object"),
                                    ("turntable", cmd_turntable, "evenly spaced views at one elevation")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--input", help="the reference image (an RGBA object works best)")
        p.add_argument("--reference", action="append",
                       help="another reference image (repeatable; more than one needs --backend torch)")
        p.add_argument("--output", required=True, help="dataset directory")
        p.add_argument("--views", type=int, default=12 if name == "multiview" else 24,
                       help="views around the object")
        p.add_argument("--elevation", type=float, default=0.0, help="elevation in degrees")
        if name == "multiview":
            p.add_argument("--azimuth-views", type=int, help="views per elevation ring (with --elevations)")
            p.add_argument("--elevations", help="comma-separated elevation rings, e.g. -20,0,20,45")
            p.add_argument("--top-bottom", action="store_true", help="add top and bottom views")
        add_view_options(p)
        add_backend_options(p)
        p.set_defaults(run=runner)

    p = sub.add_parser("image-to-3d", help="photo -> RGBA object -> views -> dataset")
    p.add_argument("--input", required=True)
    p.add_argument("--reference", action="append", help="extra real reference photos (needs --backend torch)")
    p.add_argument("--output", required=True)
    p.add_argument("--views", type=int, default=24, help="views per elevation (0: none)")
    p.add_argument("--elevation", type=float, default=0.0)
    p.add_argument("--elevations", help="comma-separated elevation rings")
    p.add_argument("--top-bottom", action="store_true")
    p.add_argument("--remove-background", action="store_true", help="accepted for clarity (the default)")
    p.add_argument("--keep-background", action="store_true", help="skip extraction; the input is already clean")
    p.add_argument("--method", default="qwen", choices=("qwen", "rmbg", "alpha"))
    p.add_argument("--pixels", default="original", choices=("original", "generated"))
    p.add_argument("--fill", type=float, default=0.85)
    p.add_argument("--cleanup", help=f"texture cleanup ops before generating views: {', '.join(ops.TEXTURE_OPERATIONS)}")
    p.add_argument("--edit", help="an object-preserving edit applied to the extracted object before views and "
                                  "reconstruction (e.g. 'make the base dark blue')")
    p.add_argument("--edit-strength", type=float, default=1.0, help="1: edit by conditioning; below 1: SDEdit")
    add_mask_options(p)
    add_view_options(p)
    add_backend_options(p)
    add_reconstruct_options(p, "none")
    p.set_defaults(run=cmd_image_to_3d)

    p = sub.add_parser("turnaround", help="a character's consistent front/side/back views from one "
                                          "turnaround sheet -> posed dataset (-> GLB)")
    p.add_argument("--prompt", help="the character, e.g. 'a cute chibi bunny with a pink scarf'")
    p.add_argument("--input", help="a reference image of the character (with or without --prompt)")
    p.add_argument("--output", required=True, help="dataset directory")
    p.add_argument("--views", type=int, default=4, choices=(3, 4),
                   help="4: front, left, back, right; 3: front, left, back")
    p.add_argument("--size", type=int, default=512, help="view size in pixels (the sheet is size x views*size)")
    add_backend_options(p)
    add_reconstruct_options(p, "none")
    p.set_defaults(run=cmd_turnaround)

    p = sub.add_parser("reconstruct", help="an RGBA object or a view dataset -> GLB with Pixal3D")
    p.add_argument("--input", required=True, help="a dataset directory (multiview) or an RGBA PNG (single)")
    p.add_argument("--output", required=True, help="output directory (<runner>.glb, reconstruction.json)")
    add_reconstruct_options(p, "native")
    p.set_defaults(run=cmd_reconstruct)

    p = sub.add_parser("validate", help="re-run the 2D checks on a dataset")
    p.add_argument("dataset")
    p.set_defaults(run=cmd_validate)
    return ap


# Values that may start with "-" (e.g. --elevations -20,0,20): argparse would
# read them as options, so they are joined to their flag first.
SIGNED_VALUES = ("--elevations", "--elevation", "--mask-rect", "--mask-circle", "--recon-elevations")


def join_signed_values(argv: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(argv):
        if argv[i] in SIGNED_VALUES and i + 1 < len(argv):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv=None) -> int:
    raw = join_signed_values(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(raw)
    args.recon_mode_given = any(a == "--recon-mode" or a.startswith("--recon-mode=") for a in raw)
    args.backends = []
    try:
        result = args.run(args)
    except (ValueError, backends.BackendError) as exc:
        print(f"runner: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:   # reconstruct.ReconstructionError: a runner failed
        print(f"runner: {exc}", file=sys.stderr)
        return 1
    finally:
        for backend in args.backends:
            close = getattr(backend, "close", None)
            if close:
                close()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
