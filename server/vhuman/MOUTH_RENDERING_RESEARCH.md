# Research: occlusion and rendering quality around lips, mouth, teeth and tongue

Research date: 2026-10-09. Scope: the vhuman candidate's offline Blender/USD
path and the WebGL2/mobile runtime. Measured numbers below refer to
`tmp/vhuman-mouth-occlusion/` and the refined-ear candidate (`a2f43542…`).

## Problem

On mobile/browser, teeth, gums, tongue and the oral cavity were flat-colour
PBR materials without any occlusion or shadowing. Teeth deep in the mouth
received the same ambient and key light as the face, so they glowed and
the mouth read as a pasted-on texture. The offline Cycles path traces true
occlusion, but its oral materials are uniform Principled shaders (light SSS on
gums and tongue only), so teeth look plasticky and the tongue lacks texture.

The ray-traced reference (below) shows how dark the interior really is: mean
cosine-weighted escape visibility of oral vertices is only a few percent; only
labial incisor surfaces, the tongue tip and gums near the lips see more.

## Occlusion techniques considered

| Technique | What it captures | Runtime cost | Fit for this avatar |
| --- | --- | --- | --- |
| Static baked AO (vertex or texture) | crevices; one mouth pose | none | Baseline only: wrong as soon as the lips move |
| Pose-space regression AO ([Kontkanen & Aila 2006]; [Kirk & Arikan 2007]; EA SEED bi-level regression [Le et al. 2019]) | AO as a learned function of pose | small (per-vertex weights) | Good when trained on broad poses; needs ground truth and can extrapolate badly |
| Analytic lip-aperture form factor (tested, rejected as predictor) | point-to-polygon form factor of the lip rim (Lambert contour integral) | O(vertices × rim points) | Uncorrelated with ray-traced visibility here: the rim lies inside the lips and teeth protrude past it |
| **Lip-shape "morphable" AO (implemented)** | per-vertex visibility affine in lip-opening area and height, fitted to ray-traced visibility | 3 weights/vertex, ~5 shader ops | Best held-out accuracy; follows the lip shape every frame |
| **Analytic aperture shadow (implemented)** | soft direct-light shadow of the lips on teeth/tongue: light ray vs. rim polygon with distance-scaled penumbra | same loop | Cheap contact/lip shadow for each directional light |
| Analytic proxy occluders (spheres/capsules, e.g. capsule shadows, Quilez sphere occlusion) | occlusion by tongue/teeth proxies | cheap in shader | Possible add-on for tongue self-shadowing |
| Screen-space AO: GTAO ([Jimenez et al. 2016], three.js `GTAOPass`), N8AO | generic contact occlusion incl. lips/nose | full-screen pass, normal/depth buffers | Desktop option; costly and view-dependent on mobile; misses off-screen occluders |
| Shadow maps (PCF/PCSS) | exact direct shadows | extra depth pass of the head | Desktop option; lip→teeth contact needs small near-plane/bias tuning |
| Bent normals / specular occlusion ([Lagarde & de Rousiers 2014]) | reduces glossy leaks in cavities | per-pixel | Implemented as indirect-specular × visibility² |

Production digital-human shaders (e.g. Reallusion Character Creator) expose
"AO masking" to hide over-lit inner teeth and tongue; MetaHuman exposes a jaw-
open preview but the published docs do not describe a dedicated mouth AO input.
Our analytic aperture term is a geometric version of that masking that follows
the actual lip shape every frame.

## Recommended plan (priority order)

1. **Dynamic mouth occlusion on mobile (done in this iteration).** Per-vertex
   indirect visibility `w0 + w1·area + w2·height` of the lip-rim opening, a
   GTAO multi-bounce lift, and an aperture shadow per directional light gated
   by `min(1, visibility/0.5)`; applied in the oral materials' vertex/fragment
   shaders (CPU work per pose: uniform upload, ~0.1–0.3 ms). Toggle in the
   player.
2. **Static local AO for crevices.** Bake interproximal/gingival-margin AO
   (short-range rays) into the oral vertex weights or a small texture; it
   multiplies the aperture term and costs nothing at runtime.
3. **Inner-lip and commissure occlusion on skin.** Apply the same aperture
   visibility to skin vertices of the inner vermilion/mucosa and mouth corners
   (currently fully lit), with a short feather to the outer lip.
4. **Teeth appearance.** Per-tooth albedo variation (cervical yellower, incisal
   edge greyer/bluer translucency), enamel specular (roughness ≈0.15–0.25,
   F0≈0.04), wrap/transmission lighting on mobile to fake enamel
   translucency; offline: layered enamel/dentin SSS (cf. [Shetty & Bailey],
   [Velinov et al. 2018], "Physically Based Real-Time Rendering of Teeth and Partial Restorations", Eurographics 2020). Separate
   crowns from gums in the asset (currently one tooth material incl. roots).
5. **Gums and tongue.** Wet specular layer (low roughness clearcoat), SSS,
   gingival-margin darkening; tongue papillae normal/roughness detail and
   dorsum colour variation (an optional micro-structure prior exists in earlier
   iterations); keep the aperture occlusion on top.
6. **Lips.** Wet glossy band along the inner vermilion (roughness gradient to
   the contact line), defined vermilion border in albedo, inner mucosa colour
   transition; drive gloss with lip stretch from the existing wrinkle drivers.
7. **Desktop-only extras.** Half-resolution GTAO pass and a head-only shadow map
   (lip→teeth contact) behind a quality switch; measure on the device test page.
8. **Validation.** Keep the ray-traced reference as the gate: report held-out
   MAE/p95 per part and matched Blender-vs-browser mouth crops for open,
   speaking and closed poses.

## Measured results

Reference: cosine-weighted escape visibility of all 4,611 oral vertices, GPU
ray traced (64 rays/vertex) against skin and oral geometry within 90 mm.
Training: 200 expressions sampled from the motion tracks' coefficient
distribution (±1.5σ PCA plus extrapolated interpolations), evaluated with the
native runtime. Test: the default (81 samples) and authored oral-stress (43)
tracks, fully held out. Normals: raw GNM oral normals face the mouth air
(checked on incisal edges, labial surfaces, tongue dorsum and cavity roof);
an early signed-volume/free-path orientation heuristic was wrong and its
results were discarded.

Indirect visibility, mean absolute error (p95):

| Model | default | stress |
| --- | --- | --- |
| no occlusion (1.0) | 0.94 | 0.93 |
| static baked per-vertex AO | 0.028 (0.128) | 0.047 (0.234) |
| analytic rim form factor | 0.135 (0.665) | 0.120 (0.486) |
| per-vertex affine in form factor | 0.021 (0.100) | 0.033 (0.150) |
| **per-vertex affine in area + height (shipped)** | **0.012 (0.064)** | **0.018 (0.071)** |

Adding the form factor to area + height changes nothing (0.012/0.018). Mesh
or per-tooth smoothing of the weights raises held-out error (e.g. 50 % tooth
averaging: teeth MAE 0.019→0.038 on stress), so the within-tooth lip-shadow
pattern is real; visible blotches on the coarse tooth mesh are a resolution
limit (proposal item 4).

Direct light (browser key/fill lights, 0.12 rad soft cone), held-out MAE:

| Light / track | no shadow | aperture shadow | aperture × min(1, vis/0.5) (shipped) |
| --- | --- | --- | --- |
| key / default | 0.968 | 0.147 | 0.036 |
| key / stress | 0.918 | 0.222 | 0.062 |
| fill / default | 0.976 | 0.119 | 0.030 |
| fill / stress | 0.928 | 0.230 | 0.059 |

The gate constant 0.5 was selected on the default track only. Upper front teeth
in the rest smile are ~92 % occluded under the browser's key-from-above rig
(reference 0.081 vs predicted 0.078), so they render darker than in the
flash-lit portrait; that is lighting, not an occlusion error.

Browser: 75,620 triangles unchanged; hardware verification 30 FPS on RTX 5060
Ti, exact WASM/native parity, binding p95 3.3e-6 mm. Not measured on a phone.
The iOS/Filament player does not yet implement the oral shader terms.

## References

- J. Kontkanen, T. Aila, "Ambient Occlusion for Animated Characters", EGSR 2006.
- A. G. Kirk, O. Arikan, "Real-time ambient occlusion for dynamic character skins", I3D 2007 ([SIGGRAPH archive entry](https://history.siggraph.org/learning/precomputed-ambient-occlusion-for-character-skins-by-kirk-and-arikan)).
- B. H. Le et al., "High-Quality Object-Space Dynamic Ambient Occlusion for Characters Using Bi-level Regression", I3D 2019 ([EA SEED](https://www.ea.com/seed/news/i3d2019-dynamic-ao)).
- J. Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion" (GTAO), SIGGRAPH 2016 course; [three.js GTAOPass](https://threejs.org/docs/pages/GTAOPass.html); [N8AO](https://github.com/N8python/n8ao).
- S. Lagarde, C. de Rousiers, "Moving Frostbite to Physically Based Rendering", SIGGRAPH 2014 course (specular occlusion).
- J. H. Lambert, point-to-polygon form factor (contour integral), as used for polygonal area lights.
- [Shetty & Bailey, "A physical rendering model for human teeth"](https://history.siggraph.org/?p=135987); [Velinov et al., "Appearance Capture and Modeling of Human Teeth", SIGGRAPH Asia 2018](https://www.jannovak.info/publications/TeethAppearance/index.html); "Physically Based Real-Time Rendering of Teeth and Partial Restorations" (Eurographics 2020; Henyey–Greenstein scattering with fitted dentin cores).
- [Reallusion digital human shader (teeth/tongue AO masking)](https://reallusion.com/character-creator/digital-human-shader.html); [MetaHuman teeth controls](https://dev.epicgames.com/documentation/metahuman/teeth-and-eyelash-controls).
