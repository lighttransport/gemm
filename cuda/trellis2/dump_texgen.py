"""TRELLIS.2 texgen ground-truth dumper.

Runs the full image -> mesh + tex_voxels pipeline (CUDA reference, pipeline
type '512'), then runs a *patched* copy of `o_voxel.postprocess.to_glb`
which np.save's the artefacts listed in
`rdna4/trellis2/docs/verify_texgen.md` so the rdna4 cumesh_xatlas shim
can be validated against CUDA without UV-unwrap nondeterminism.

The patch is applied by rewriting `to_glb`'s source (via inspect) and
exec'ing it in the postprocess module's namespace — keeps us in lockstep
with upstream without maintaining a 300-line copy.

Outputs (in --output-dir, default cuda/trellis2/verify-dumps):

  Required (per docs/verify_texgen.md):
    uv_vertices.npy, uv_faces.npy, uv_uvs.npy, uv_vmaps.npy, uv_normals.npy
    bvh_vertices.npy, bvh_faces.npy
    attr_volume.npy, attr_coords.npy, attr_grid_size.npy
    texgen_consts.json
    textured_cuda.glb

  Optional (dumped by default; cheap):
    rast.npy, mask.npy, valid_pos.npy, bvh_face_id.npy, bvh_uvw.npy,
    attrs_pre_inpaint.npy
"""
import argparse
import inspect
import json
import os
import sys
import textwrap

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'cpu', 'trellis2', 'trellis2_repo'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rdna4', 'trellis2'))

import numpy as np
import torch
from PIL import Image

from gen_image_to_3d import _patch_dinov3_extractor, _patch_birefnet_noop


def _save(out_dir: str, name: str, t) -> None:
    if isinstance(t, torch.Tensor):
        a = t.detach().cpu().numpy()
    else:
        a = np.asarray(t)
    np.save(os.path.join(out_dir, f'{name}.npy'), a)
    print(f'[dump-texgen] {name:24s} shape={a.shape} dtype={a.dtype}', flush=True)


def _build_patched_to_glb(out_dir: str):
    """Rewrite o_voxel.postprocess.to_glb to inject np.save at known points.

    Insertion points are matched by exact substrings; if upstream renames
    these locals the script will raise (preferred over silently dumping
    nothing).
    """
    from o_voxel import postprocess as _pp
    # NOTE: do NOT textwrap.dedent — it normalizes whitespace-only lines and
    # would break our multi-line anchors below.
    src = inspect.getsource(_pp.to_glb)

    I = "    "  # function body indent
    repls = [
        # Post-UV-unwrap mesh state.
        (
            "out_normals = mesh.read_vertex_normals()[out_vmaps]",
            "out_normals = mesh.read_vertex_normals()[out_vmaps]\n"
            f"{I}_dump('uv_vertices', out_vertices)\n"
            f"{I}_dump('uv_faces',    out_faces)\n"
            f"{I}_dump('uv_uvs',      out_uvs)\n"
            f"{I}_dump('uv_vmaps',    out_vmaps)\n"
            f"{I}_dump('uv_normals',  out_normals)",
        ),
        # Composed rast + mask after the chunk loop.
        (
            "mask = rast[0, ..., 3] > 0",
            "mask = rast[0, ..., 3] > 0\n"
            f"{I}_dump('rast', rast)\n"
            f"{I}_dump('mask', mask)",
        ),
        # BVH lookup + valid_pos.
        (
            "_, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)",
            f"_dump('valid_pos', valid_pos)\n"
            f"{I}_, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)\n"
            f"{I}_dump('bvh_face_id', face_id)\n"
            f"{I}_dump('bvh_uvw',     uvw)",
        ),
        # attrs after grid_sample_3d, before inpaint.
        (
            "if use_tqdm:\n        pbar.update(1)\n    if verbose:\n        print(\"Done\")\n    \n    # --- Texture Post-Processing",
            "_dump('attrs_pre_inpaint', attrs)\n"
            f"{I}if use_tqdm:\n        pbar.update(1)\n    if verbose:\n        print(\"Done\")\n    \n    # --- Texture Post-Processing",
        ),
        ("def to_glb(", "def to_glb_dump("),
    ]
    for old, new in repls:
        if old not in src:
            raise RuntimeError(
                f'patched to_glb: anchor not found:\n{old[:80]!r}\n'
                'Upstream postprocess.to_glb has been refactored — update dump_texgen.py.'
            )
        src = src.replace(old, new, 1)

    ns = dict(_pp.__dict__)
    ns['_dump'] = lambda name, t: _save(out_dir, name, t)
    exec(src, ns)
    return ns['to_glb_dump']


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--model-root', required=True)
    ap.add_argument('--dinov3', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--texture-size', type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _patch_dinov3_extractor(args.dinov3)
    _patch_birefnet_noop()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel  # noqa: F401  — needed for patched to_glb dependencies

    print(f'[load] {args.model_root}', flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_root)
    pipeline.cuda()

    img = Image.open(args.image)
    if img.mode != 'RGBA':
        rgb = np.array(img.convert('RGB'))
        alpha = ((rgb.sum(-1) > 15) * 255).astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb, alpha]), 'RGBA')

    torch.manual_seed(args.seed)
    print(f'[run] pipeline_type=512 seed={args.seed}', flush=True)
    out = pipeline.run(img, seed=args.seed, pipeline_type='512')
    mesh = out[0]
    mesh.simplify(16777216)  # match gen_image_to_3d.py — no-op when below budget

    # ---- Inputs to to_glb (BVH source + attribute volume) ----
    _save(args.output_dir, 'bvh_vertices', mesh.vertices)
    _save(args.output_dir, 'bvh_faces',    mesh.faces)
    _save(args.output_dir, 'attr_volume',  mesh.attrs)
    _save(args.output_dir, 'attr_coords',  mesh.coords)

    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        dtype=torch.float32, device=mesh.coords.device)
    grid_size = ((aabb[1] - aabb[0]) / mesh.voxel_size).round().int()
    _save(args.output_dir, 'attr_grid_size', grid_size)

    # attr_layout serializable form (dict[str, list|int]).
    layout_json = {}
    for k, v in mesh.layout.items():
        if isinstance(v, slice):
            r = list(range(v.start or 0, v.stop or 0))
            layout_json[k] = r if len(r) != 1 else r[0]
        else:
            layout_json[k] = v
    consts = {
        'attr_layout':  layout_json,
        'voxel_size':   float(mesh.voxel_size) if hasattr(mesh.voxel_size, '__float__')
                        else (mesh.voxel_size.tolist() if torch.is_tensor(mesh.voxel_size)
                              else mesh.voxel_size),
        'aabb':         [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        'texture_size': args.texture_size,
        'remesh':       True,
        'remesh_band':  1,
        'remesh_project': 0,
        'decimation_target': 1000000,
    }
    with open(os.path.join(args.output_dir, 'texgen_consts.json'), 'w') as f:
        json.dump(consts, f, indent=2)
    print(f'[dump-texgen] texgen_consts.json -> {consts}', flush=True)

    # ---- Patched to_glb: writes uv_*, rast/mask, valid_pos, bvh_face_id/uvw, attrs ----
    to_glb_dump = _build_patched_to_glb(args.output_dir)
    glb = to_glb_dump(
        vertices          = mesh.vertices,
        faces             = mesh.faces,
        attr_volume       = mesh.attrs,
        coords            = mesh.coords,
        attr_layout       = mesh.layout,
        voxel_size        = mesh.voxel_size,
        aabb              = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target = 1000000,
        texture_size      = args.texture_size,
        remesh            = True,
        remesh_band       = 1,
        remesh_project    = 0,
        verbose           = True,
    )
    glb_path = os.path.join(args.output_dir, 'textured_cuda.glb')
    glb.export(glb_path, extension_webp=False)
    print(f'[done] {glb_path}', flush=True)
    print(f'[vram peak] {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
