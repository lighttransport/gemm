"""Capture only MetaHuman eye material sections under VHStudio, without the face mesh.

Runs in Unreal via launch.py: eye_isolate.py <winjob batch.json>. Outputs go to a separate
directory per identity under the dataset root, never into the reference render set. By default
only the eyeballs are included; the optional batch flag include_eye_occlusion also includes the
separate translucent eyelid-occlusion shell. Rerunning an identity replaces earlier captures.
"""
import json
import math
import os
import time

import unreal

import gen_batch as gb
import mh_common as mh
import render_batch as rb


def steps(batch, results):
    world = rb.load_environment('Studio')
    unreal.SystemLibrary.execute_console_command(world, 'r.Streaming.FullyLoadUsedTextures 1')
    cap, comp, rt = rb.make_capture_actor(world, 1024)
    lights = rb.spawn_replica(batch['lights'][batch['ids'][0]])
    rb.hide_environment(world, keep={cap.get_actor_label()} | {c.get_owner().get_actor_label() for c, _ in lights.values()})
    yield from rb.settle(world)
    sub = mh.subsystem()
    for sid in batch['ids']:
        root = os.path.join(rb.V1, 'samples', sid)
        params = json.load(open(os.path.join(root, 'params.json')))
        cams = json.load(open(os.path.join(root, 'cameras.json')))['cameras']
        include_occlusion = bool(batch.get('include_eye_occlusion', False))
        out = os.path.join(mh.OUT_ROOT, 'eye_isolate_occlusion' if include_occlusion else 'eye_isolate_eyeball', sid)
        os.makedirs(out, exist_ok=True)
        character = mh.create_character('/Game/MHGen/Render', sid + '_eye_isolate')
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
                yield from rb.settle(world)
                faces = [c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_name() == 'Face']
                if len(faces) != 1:
                    raise RuntimeError(f'expected one Face component, found {len(faces)}')
                face = faces[0]
                slots = [str(x) for x in face.get_material_slot_names()]
                prefixes = ('eyeleft_', 'eyeright_', 'eyeshell_') if include_occlusion else ('eyeleft_', 'eyeright_')
                keep = [i for i, name in enumerate(slots) if name.lower().startswith(prefixes)]
                if len(keep) != (3 if include_occlusion else 2):
                    raise RuntimeError(f'expected eye material slots: {slots}')
                face.set_forced_lod(1)
                comp.clear_show_only_components()
                comp.show_only_component(face)
                yield from rb.settle(world)
                for i in range(len(slots)):
                    if i not in keep:
                        face.show_material_section(i, i, False, 0)
                yield from rb.settle(world)
                files = []
                for name in ('eye_left', 'eye_right'):
                    cam = dict(cams[name]); cam['ue'] = dict(cam['ue'])
                    fov = math.radians(cam['ue']['fov_deg'])
                    cam['ue']['fov_deg'] = math.degrees(2 * math.atan(math.tan(fov / 2) / 2.5))
                    rb.shoot(world, comp, rt, cam, unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR, True, out, name)
                    files.append(name + '.exr')
                    yield
                    rb.shoot(world, comp, rt, cam, unreal.SceneCaptureSource.SCS_BASE_COLOR, True, out, name + '_basecolor')
                    files.append(name + '_basecolor.exr')
                    yield
                record = dict(ok=True, id=sid, slots=slots, kept=keep, include_occlusion=include_occlusion, files=files,
                              sha256={f: gb.sha256(os.path.join(out, f)) for f in files})
                mh.write_json(os.path.join(out, 'done.json'), record)
                results[sid] = record
                comp.clear_show_only_components()
                actor.destroy_actor(); actor = None
        finally:
            if actor:
                actor.destroy_actor()
            mh.delete_asset(character)
    cap.destroy_actor()


batch = json.load(open(mh.script_args()[0]))
rb.batch_steps = steps
rb.run_ticked(batch, os.path.join(mh.OUT_ROOT, 'eye_isolate', f'result-{int(time.time())}.json'))
