# Skin subsurface scattering: survey and UE MetaHuman vs Cycles mapping (2026-11-04)

## 1. Models in use

| Model | Idea | Where |
|---|---|---|
| Dipole diffusion (Jensen et al. 2001) | semi-infinite slab, point source + mirrored sink; R(r) from sigma_a, sigma_s' | early offline skin, Pixar/RenderMan legacy |
| Multipole / quantized diffusion (Donner & Jensen 2005; d'Eon & Irving 2011) | layered slabs, sum of Gaussians fit | multi-layer skin (epidermis/dermis) |
| Sum-of-Gaussians texture-space diffusion (d'Eon & Luebke 2007) | blur irradiance in UV with fitted Gaussians | GPU Gems 3 "Human Head" |
| Separable screen-space SSS (Jimenez et al. 2009/2015) | two 1D screen-space blurs of diffuse lighting with a kernel from the profile | UE "Separable" profile, many games |
| Burley normalized diffusion (Burley 2015; Christensen & Burley 2015) | R(r) = A s (e^{-sr} + e^{-sr/3}) / (8 pi r); s from surface albedo A by fitted curves (perpendicular, diffuse-surface, search-light configurations); parameter d (diffuse mean free path) | Disney BSDF, UE5 Burley (screen-space importance sampled), Cycles "Christensen-Burley" |
| Volumetric random walk (Chiang et al. 2016; Wrenninge et al. 2017) | path-trace inside the mesh with sigma_t = 1/mfp, single-scatter albedo inverted from the surface colour | Cycles Random Walk / Random Walk (Skin), Arnold randomwalk, RenderMan PxrSurface, UE path tracer |

The diffusion models assume a flat semi-infinite medium, so they over-blur thin parts (ears, nostrils, eyelids) and miss
the back-lit glow. Random walk handles the real geometry, which is why it is the offline reference. UE's raster path
blurs the *diffuse lighting* in screen space with a Burley kernel (`PostProcessSubsurface.usf`, AFIS), and adds a
separate transmission term for thin parts (profile `Transmission` group, `ExtinctionScale`, `NormalScale`).

## 2. UE MetaHuman skin (dumped from UE 5.8.4, `ue_scripts/dump_skin_material.py`, `v1/presets/skin_material.json`)

- Material chain of the reference renders: `MID_MI_Skin_Head_UI_LOD0_Editable` -> `MI_Skin_Head_UI_LOD0` ->
  `MI_skin_unified_MH_preset` -> `MI_skin_unified` -> `M_skin_unified_UI`; shading model from the material expression
  (Subsurface Profile); profile `SSP_skin_unified` on every level. `r.Substrate 0`, `r.SubsurfaceScattering 1`,
  `r.SSS.Burley.Quality 1`, `r.SSS.Scale 1`, `r.SSS.Checkerboard 2`. No ray tracing, no Lumen.
- `SSP_skin_unified` (Burley, mean-free-path mode): mean free path colour (.350, .136, .0875) x distance 0.3 cm =
  **1.05 / .41 / .26 mm** (R/G/B); world unit scale 0.1 (with the shader's mm<->cm handling this makes the distance
  a plain cm value on a cm-scale head); surface albedo (1, 1, 1) (only used for the mfp <-> dmfp conversion: the
  shader takes the scattered colour from the GBuffer base colour); tint 1; boundary bleed 1.
  Dual specular: **lobe roughness 0.8 r and 1.1 r, mix .5** (`SubsurfaceProfileCommon.ush`
  `GetSubsurfaceProfileDualSpecular`, `ShadingModels.ush` `DualSpecularGGX`; faded to one lobe when Opacity < .1).
  Transmission: extinction scale 18.6, normal scale .3, scattering distribution .5, IOR 1.55,
  transmission tint (.10, .033, 0).
- Material parameters (mh_000000; skin synthesis does not change these): Opacity scales the scatter radius
  (`PostProcessSubsurface.usf`: "opacity allows to scale the radius"); `Scatter Distance Base .5`,
  `Scatter Distance Global Multiply .5`, all regions 1 -> radius scale ~.25-.5 of the profile's mfp (the material graph
  that combines them is not visible from Python). `Fake SSS Strength .8` (a texture-mip based diffuse soften).
  Roughness per region (x `Roughness Adjust .95` x `Roughness Global Multiply 1.05` ~= x1.0): forehead .58,
  cheeks .64, chin .65, temples .63, nose .46-.55, upper/lower lip .62/.60, eyelids .55, ear .70 (inner .35),
  `Roughness Concavity Multiply 1.1`, `Convexity .85`, `Micro Cavity 1.35`, `Micro Skin Roughness 1.05`.
  Specular per region .75-.9 (nasal creases .45), x `Spec Adjust .9`, `Concavity .6`, `Convexity 1.2`; UE F0 =
  .08 x Specular -> **~.055-.065**. Micro normal tiling 44, strength .7.

## 3. Cycles (Blender 5.2, `cycles_scene.full_skin_graph`)

- Principled BSDF v2 (since 4.0): `Subsurface Weight` mixes diffuse and subsurface (it is not a radius multiplier);
  `Subsurface Radius` x `Subsurface Scale` = per-channel mean free path in scene units (metres here);
  `RANDOM_WALK_SKIN` = random walk with the albedo inverted from Base Color and an entry lobe mixing diffuse and
  specular transmission with `Subsurface IOR`; `Subsurface Anisotropy` (skin ~.8 in the literature, Cycles default 0).
- `Specular IOR Level` .5 = F0 from IOR; F0 is scaled by 2 x level (clamped by the graph at level 1).
- vhuman now: SSS weight .6, radius (3.67, 1.37, .68) mm x scale .5 = **1.84 / .69 / .34 mm**, IOR 1.4; F0 .028
  (level = template .42-.60 x (1 - .7 cavity) -> F0 ~.024-.034); roughness template .36-.52 x (1 + .6 cavity), lips
  -.12; coat .10 (1 - cavity) at roughness .30 (+.35 on lips).

## 4. Mapping and the gaps it shows

| Quantity | UE MetaHuman | vhuman Cycles | Ratio / note |
|---|---|---|---|
| Scatter mfp R/G/B | 1.05/.41/.26 mm x (~.25-.5 Opacity) | 1.84/.69/.34 mm | Cycles 2-7x wider; colour ratios similar (R:G:B 1:.39:.25 vs 1:.37:.19) |
| SSS amount | all diffuse through the profile | 60% SSS, 40% Lambert | |
| Roughness | ~.55-.65 per region, lobes .8r/1.1r | ~.42-.52 | UE ~1.25x (the sweep's best global scale was x1.25) |
| Second lobe | none (dual lobes both near r) | coat .10 at .30 | coat 0 won on 12/12 heads |
| F0 | ~.06 | ~.028 | UE ~2x |
| Thin-part transmission | separate transmission profile | random walk through geometry | not matched; ears differ |

The two earlier empirical wins (roughness x1.25, coat 0) are what the UE parameters predict. The remaining
untested gaps are the SSS radius / weight and the specular level.

## 5. Experiment (`compare_render.UE_SET`, `--ue-match`; 12 heads x front/+-40, fast profile, vs `normal`)

`ue_sss25|50|100` = weight 1 with the UE mfp x .25/.5/1; `ue_spec` = specular x1.8 (F0 ~.05); `ue_match` = roughness
x1.25 + coat 0 + ue_sss50; `ue_match_spec` = ue_match + specular x1.8. Results pending (`compare_v1/ue_match_vhstudio__fast.json`).

## References
- H. W. Jensen, S. Marschner, M. Levoy, P. Hanrahan, "A Practical Model for Subsurface Light Transport", SIGGRAPH 2001.
- C. Donner, H. W. Jensen, "Light Diffusion in Multi-Layered Translucent Materials", SIGGRAPH 2005.
- E. d'Eon, D. Luebke, "Advanced Techniques for Realistic Real-Time Skin Rendering", GPU Gems 3, 2007.
- E. d'Eon, G. Irving, "A Quantized-Diffusion Model for Rendering Translucent Materials", SIGGRAPH 2011.
- J. Jimenez et al., "Separable Subsurface Scattering", CGF 2015.
- B. Burley, "Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering", SIGGRAPH 2015 course.
- P. Christensen, B. Burley, "Approximate Reflectance Profiles for Efficient Subsurface Scattering", Pixar TM 2015.
- M. Chiang, P. Kutz, B. Burley, "Practical and Controllable Subsurface Scattering for Production Path Tracing", SIGGRAPH 2016 talk.
- UE 5.8 source shipped with the build: `Engine/Source/Runtime/Engine/Classes/Engine/SubsurfaceProfile.h`,
  `Engine/Shaders/Private/{SubsurfaceProfileCommon.ush, BurleyNormalizedSSSCommon.ush, PostProcessSubsurface.usf, ShadingModels.ush}`.
- Blender manual, Principled BSDF (subsurface weight/radius/scale, Random Walk (Skin)); Cycles PR #110989 (Principled v2 subsurface).
