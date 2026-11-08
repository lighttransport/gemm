"""Physically layered procedural eyeball for Cycles (fresh path, 2026-11).

Separate meshes in an eye-local frame (metres, origin = eyeball centre, +Z = optical axis, +Y = up):
an SSS sclera shell open at the limbus, a closed refractive cornea/aqueous volume, a displaced iris
annulus inside that volume and a dark pupil cup. Textures are polar (iris) and spherical (sclera) maps
that match the mesh UVs, so no remapping is needed. See ``bake.build_eye``.
"""
