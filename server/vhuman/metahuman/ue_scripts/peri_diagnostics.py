"""UE-only periocular SSS / eye-occlusion ablation captures, separate from references.

Run through launch.py with a VHStudio winjob batch. Never skip existing reference
markers. Restore cvars/sections and write per-head provenance for each new run.
"""
import json
import os
import time

import unreal
import gen_batch as gb
import mh_common as mh
import render_batch as rb


def steps(batch, results):
    world = rb.load_environment('Studio')
    original_sss = unreal.SystemLibrary.get_console_variable_int_value('r.SubsurfaceScattering')
    unreal.SystemLibrary.execute_console_command(world, 'r.Streaming.FullyLoadUsedTextures 1')
    cap, comp, rt = rb.make_capture_actor(world, int(batch.get('size', 1024)))
    lights = rb.spawn_replica(batch['lights'][batch['ids'][0]])
    rb.hide_environment(world, keep={cap.get_actor_label()} | {c.get_owner().get_actor_label() for c, _ in lights.values()})
    sub = mh.subsystem()
    try:
        yield from rb.settle(world)
        for sid in batch['ids']:
            root = os.path.join(rb.V1, 'samples', sid)
            params = json.load(open(os.path.join(root, 'params.json')))
            cams = json.load(open(os.path.join(root, 'cameras.json')))['cameras']
            out = os.path.join(mh.OUT_ROOT, 'peri_diagnostics', batch.get('tag', 'default'), sid)
            os.makedirs(out, exist_ok=True)
            character = mh.create_character('/Game/MHGen/Render', sid + '_peri_diag')
            actor = None
            try:
                character.preview_material_type = unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE
                with mh.Editing(character):
                    sub.set_face_model_coefficients(character=character, coefficients=[float(x) for x in params['coefficients']])
                    sub.commit_face_state(character=character)
                    sub.commit_skin_settings(character=character, skin_settings=gb.skin_settings_from_job(params['skin']))
                    gb.apply_eyes(character, params['eyes'])
                    actor = sub.spawn_meta_human_actor(character=character)
                    rb.place_replica(lights, batch['lights'][sid])
                    faces = [c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_name() == 'Face']
                    if len(faces) != 1:
                        raise RuntimeError('expected one Face component')
                    face = faces[0]; face.set_forced_lod(1)
                    slots = [str(x) for x in face.get_material_slot_names()]
                    occlusion = [i for i, slot in enumerate(slots) if slot.lower().startswith('eyeshell_')]
                    if len(occlusion) != 1:
                        raise RuntimeError(f'expected eye occlusion slot: {slots}')
                    comp.clear_show_only_components(); comp.show_only_actor_components(actor, True)
                    yield from rb.settle(world)
                    files = []
                    modes = batch.get('modes', [dict(name=f'sss{s}_occlusion{o}', sss=s, occlusion=o)
                                               for s, o in ((1,1), (0,1), (1,0), (0,0))])
                    shell_material = face.get_material(occlusion[0])
                    scalar_defaults = {}
                    if any('blur_size' in mode or 'shadow_multiply' in mode for mode in modes):
                        getter = getattr(shell_material, 'k2_get_scalar_parameter_value', None)
                        if getter is None:
                            getter = getattr(shell_material, 'get_scalar_parameter_value')
                        scalar_defaults = {name: float(getter(name)) for name in ('Blur Size', 'ShadowMultiply')}
                    edge_sections = [i for i, slot in enumerate(slots) if slot.lower().startswith('eyeedge_')]
                    if any('lacrimal' in mode for mode in modes) and not edge_sections:
                        raise RuntimeError('lacrimal section missing')
                    for mode in modes:
                        sss, shell = int(mode.get('sss', 1)), bool(mode.get('occlusion', 1))
                        unreal.SystemLibrary.execute_console_command(world, f'r.SubsurfaceScattering {sss}')
                        for i in occlusion:
                            face.show_material_section(i, i, shell, 0)
                        for i in edge_sections:
                            face.show_material_section(i, i, bool(mode.get('lacrimal', 1)), 0)
                        for key, name in (('blur_size', 'Blur Size'), ('shadow_multiply', 'ShadowMultiply')):
                            if name in scalar_defaults:
                                shell_material.set_scalar_parameter_value(name, float(mode.get(key, scalar_defaults[name])))
                        yield from rb.settle(world)
                        for name in batch.get('cameras', ['eye_left', 'eye_right', 'front']):
                            file = f"{name}__{mode['name']}"
                            rb.shoot(world, comp, rt, cams[name], unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR, False, out, file)
                            files.append(file + '.exr')
                            yield
                    for name, value in scalar_defaults.items():
                        shell_material.set_scalar_parameter_value(name, value)
                    if batch.get('capture_normals'):
                        for name in batch.get('cameras', ['eye_left', 'eye_right', 'front']):
                            rb.shoot(world, comp, rt, cams[name], unreal.SceneCaptureSource.SCS_NORMAL, True, out, name + '__normal')
                            files.append(name + '__normal.exr')
                            yield
                    for name in batch.get('cameras', ['eye_left', 'eye_right', 'front']):
                        rb.shoot(world, comp, rt, cams[name], unreal.SceneCaptureSource.SCS_BASE_COLOR, True, out, name + '__basecolor')
                        files.append(name + '__basecolor.exr')
                        yield
                    rec = dict(ok=True, id=sid, slots=slots, occlusion_sections=occlusion,
                               finished=time.time(), files=files, modes=modes, scalar_defaults=scalar_defaults,
                               size=int(batch.get('size', 1024)),
                               normal_encoding='signed UE world xyz; SCS_Normal' if batch.get('capture_normals') else None,
                               sha256={f: gb.sha256(os.path.join(out, f)) for f in files})
                    mh.write_json(os.path.join(out, 'done.json'), rec)
                    results[sid] = rec
            finally:
                comp.clear_show_only_components()
                if actor:
                    actor.destroy_actor()
                mh.delete_asset(character)
    finally:
        unreal.SystemLibrary.execute_console_command(world, f'r.SubsurfaceScattering {original_sss}')
        cap.destroy_actor()
        for component, _ in lights.values():
            component.get_owner().destroy_actor()


batch = json.load(open(mh.script_args()[0]))
rb.batch_steps = steps
rb.run_ticked(batch, os.path.join(mh.OUT_ROOT, 'peri_diagnostics', batch.get('tag', 'default'), 'result.json'))
