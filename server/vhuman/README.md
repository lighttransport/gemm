# Independent procedural virtual humans

A local eye/head viewer with synthetic procedural textures, analytic refraction,
portable glTF export, portrait-based eye fitting and optional Qwen/Pixal3D jobs.
The runtime requires no Unreal Engine installation, source, content or measured
character data. Engine-specific import/export adapters are not included.

## Model and measurement inputs

The eye is a smooth union of a sclera sphere and an offset corneal sphere, with
an opaque iris plane behind it. The limbus determines the spheres' intersection.
Defaults are **synthetic demonstration choices**, not a clinical specification
or measurements of a licensed character:

| Input | Default | Unit |
| --- | --- | --- |
| Sclera radius | 0.012 | metres |
| Limbus radius | 0.006 | metres |
| Cornea curvature radius | 0.008 | metres |
| Junction blend width | 0.0005 | metres |
| Chamber depth | 0.0035 | metres |
| Refractive index | 1.336 | dimensionless |

Users or an LLM may choose different inputs through the same validated JSON
schema. Non-finite inputs and impossible radius combinations are rejected.
For example, save this as `tmp/my-eye.json`:

```json
{
  "optics": {
    "sclera_radius": 0.014,
    "limbus_radius": 0.0065,
    "cornea_radius": 0.009,
    "limbus_blend": 0.0003,
    "chamber_depth": 0.0038,
    "ior": 1.34
  }
}
```

```sh
python3 -m server.vhuman.cli eye --params @tmp/my-eye.json --res 1024 --out tmp/my-eye
```

Physical inputs control standalone eye meshes, CPU rendering and browser
uniforms. Automatic portrait head fitting currently uses the synthetic default
profile for scale, carving and lids. Custom standalone eye measurements are not
silently applied to head fitting. Skin controls use the separate `--skin` JSON.
Keep the provenance of any measurements you supply; the application cannot
establish redistribution rights for arbitrary user-provided values or images.

## Independent implementation

- Texture coordinates use two linear angular intervals: the limbus maps to UV
  radius 0.2 and the back pole to 0.7. These are atlas packing choices. No
  measured UV table is imported, embedded or reconstructed.
- Geometry uses sphere intersections and a smooth union. Refraction uses
  Snell's law; reflection uses Schlick's approximation with F0 derived from
  the selected refractive index. Iris diffuse lighting is Lambertian.
- Iris appearance uses original spectral-noise fibres, crypts, spots and
  furrows, two-colour blending, neutral occlusion, a soft pupil and an outer
  ring. There is no fitted caustic term or external material-graph replica.
- Sclera vessels are original branching random walks. Synthetic optical
  density produces their colour. Lid occlusion is a Gaussian falloff from
  the portrait's eye opening; wet margins are original ribbon meshes.
- The CPU and GLSL implement the same mapping, pupil warp and albedo equations.
  The portable glTF eye bakes albedo and uses standard transmission materials.
- Eye-aligned skin masks protect lips and dark brows. Pores/freckles are
  deterministic object-space fields. Optional `delight_strength` reduces
  broad portrait shading; it does not recover ground-truth albedo.

## Running locally

```sh
sh server/vhuman/run.sh
# http://127.0.0.1:8790/ and /head
python3 -m server.vhuman.cli --mock head --subject "test portrait"
python3 -m server.vhuman.cli head-fit --portrait portrait.png --glb pixal3d.glb
python3 -m server.vhuman.cli render --preset blue --out tmp/eye-preview.png
```

The default work directory is `tmp/vhuman-independent/`. Previous development
exports under `tmp/vhuman/` are legacy local artifacts and are not reused or
redistributed. Regenerate meshes and textures rather than relabel old exports.
All generated work remains outside Git. Qwen and Pixal3D jobs use the existing
local GPU backends and lock; CPU eye/head operations do not need those models.
The server binds to loopback by default and is a local development tool.

Exports include `eye.glb`, a neutral texture ZIP, `params.json` and
`measurements.json`. Texture normals use the OpenGL +Y convention. API export
formats are `glb` and `textures`; the result keys are `glb` and `textures_zip`.
There is no engine-specific parameter schema or asset-measurement command.

For generated portrait heads the pipeline detects eye openings, fits procedural
eyes, carves/drapes the lids, adds lining/wet margins/occlusion, and bakes skin.
The head viewer supports analytic and portable eye rendering. Source portraits,
raw generated heads, fit metadata and maps remain beside each exported GLB.

## Tests and limitations

```sh
python3 -m server.vhuman.test_all
```

Optional reconstruction tests need `server/vhuman/requirements-remesh.txt`.
Browser tests use Chrome/SwiftShader and report skips if dependencies are
unavailable. Tests cover finite geometry, UV round trips, user measurements,
cache invalidation, materials, API behavior and both viewers.

Single-view reconstruction can retain folded lids, uneven canthi and socket
layer gaps. These are accepted fitting limitations, not guarantees of anatomical
accuracy. Hair/beard segmentation is incomplete. Relighting cannot uniquely
separate complexion from cast shadows. See [QUALITY_PLAN.md](QUALITY_PLAN.md)
for the current references and validation.

## Licensing and provenance

Repository-authored code is under the root MIT license. NumPy, Pillow and
Three.js are external dependencies with their own permissive licenses; the
optional mapbox-earcut bindings declare ISC and earcut has its own notices.
No GPL code, Unreal source, Unreal assets or measurements extracted from those
assets are included in this implementation. Standard mathematical operations
and original procedural algorithms are used instead of engine material graphs.
Khronos PBR Neutral tone mapping is used for the preview; third-party libraries
retain their own licenses. The tone-mapping adaptation notice is preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Qwen-generated imagery and model weights have separate terms (Qwen Research
License for the configured Qwen-Image 2.1 model). Their outputs are tagged
`qwen-research` and remain in the local work directory. The MIT code license
does not relicense third-party models or generated assets. No user/LLM-supplied
measurement is treated as independently verified anatomy.
