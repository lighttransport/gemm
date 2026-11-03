# MetaHuman reference renders (runs inside UE): rebuilds each listed sample's character from its params.json (face
# coefficients, skin, eyes), spawns the preview actor in the MetaHuman lighting environment and captures every camera
# in samples/<id>/cameras.json to linear EXR: lit HDR (scene), base colour, and a show-only-actor base colour pass used
# as the coverage mask. usage (via launch.py): render_batch.py <batch.json>  with {"ids": [...], "env": "Studio"}.
import json
import os
import time
import traceback

import unreal

import mh_common as mh
import gen_batch as gb

V1 = os.path.join(mh.OUT_ROOT, 'v1')
ENV_ROOT = '/MetaHumanCharacter/LightingEnvironments'
BASE_LEVEL = f'{ENV_ROOT}/L_BaseEnvironment'


def load_environment(env):
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not les.load_level(BASE_LEVEL):
        raise RuntimeError('cannot load ' + BASE_LEVEL)
    world = mh.editor_world()
    if env:
        unreal.EditorLevelUtils.add_level_to_world(world, f'{ENV_ROOT}/{env}', unreal.LevelStreamingAlwaysLoaded)
    return mh.editor_world()


def make_capture_actor(world, size):
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    comp = cap.capture_component2d
    comp.capture_every_frame = False
    comp.always_persist_rendering_state = True
    pps = comp.post_process_settings
    pps.override_auto_exposure_method = True; pps.auto_exposure_method = unreal.AutoExposureMethod.AEM_MANUAL
    pps.override_auto_exposure_bias = True; pps.auto_exposure_bias = 0.0
    pps.override_bloom_intensity = True; pps.bloom_intensity = 0.0
    pps.override_vignette_intensity = True; pps.vignette_intensity = 0.0
    comp.post_process_settings = pps
    rt = unreal.RenderingLibrary.create_render_target2d(world, size, size, unreal.TextureRenderTargetFormat.RTF_RGBA16F, unreal.LinearColor(0, 0, 0, 0), False)
    comp.texture_target = rt
    return cap, comp, rt


def shoot(world, comp, rt, cam, source, show_only, path, name):
    ue = cam['ue']
    loc = unreal.Vector(*ue['location_cm'])
    rot = unreal.MathLibrary.make_rot_from_xz(unreal.Vector(*ue['forward']), unreal.Vector(*ue['up']))
    comp.get_owner().set_actor_location_and_rotation(loc, rot, False, True)
    comp.fov_angle = ue['fov_deg']
    comp.capture_source = source
    comp.primitive_render_mode = (unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST if show_only
                                  else unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES)
    comp.capture_scene()
    unreal.RenderingLibrary.export_render_target(world, rt, path, f'{name}.exr')
    return f'renders/{name}.exr'


def render_sample(sid, world, comp, rt, env):
    d = os.path.join(V1, 'samples', sid)
    params = json.load(open(os.path.join(d, 'params.json')))
    cams = json.load(open(os.path.join(d, 'cameras.json')))
    out_dir = os.path.join(d, 'renders'); os.makedirs(out_dir, exist_ok=True)
    sub = mh.subsystem()
    character = mh.create_character('/Game/MHGen/Render', sid)
    files = []
    try:
        with mh.Editing(character):
            sub.set_face_model_coefficients(character=character, coefficients=[float(x) for x in params['coefficients']])
            sub.commit_face_state(character=character)
            character.preview_material_type = unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE
            sub.commit_skin_settings(character=character, skin_settings=gb.skin_settings_from_job(params['skin']))
            gb.apply_eyes(character, params['eyes'])
            actor = sub.spawn_meta_human_actor(character=character)
            comp.show_only_actors = [actor]
            for name, cam in cams['cameras'].items():
                files.append(shoot(world, comp, rt, cam, unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR, False, out_dir, f'{name}__{env.lower()}'))
                files.append(shoot(world, comp, rt, cam, unreal.SceneCaptureSource.SCS_BASE_COLOR, True, out_dir, f'{name}__basecolor'))
            comp.show_only_actors = []
            actor.destroy_actor()
    finally:
        mh.delete_asset(character)
    rec = dict(ok=True, id=sid, env=env, files=files, size=cams['size'], finished=time.time(),
               sha256={f: gb.sha256(os.path.join(d, f)) for f in files if os.path.isfile(os.path.join(d, f))})
    mh.write_json(os.path.join(out_dir, 'renders_done.json'), rec)
    return rec


def main():
    batch = json.load(open(mh.script_args()[0]))
    env = batch.get('env', 'Studio')
    world = load_environment(env)
    size = int(batch.get('size', 1024))
    cap, comp, rt = make_capture_actor(world, size)
    results = {}
    for sid in batch['ids']:
        done = os.path.join(V1, 'samples', sid, 'renders', 'renders_done.json')
        if os.path.isfile(done) and json.load(open(done)).get('ok'):
            results[sid] = 'skip'; continue
        try:
            render_sample(sid, world, comp, rt, env); results[sid] = 'ok'
        except Exception:      # noqa: BLE001
            tb = traceback.format_exc(); mh.log(tb)
            os.makedirs(os.path.dirname(done), exist_ok=True)
            mh.write_json(done, dict(ok=False, id=sid, error=tb)); results[sid] = 'fail'
        mh.log(f'{sid}: {results[sid]}')
    cap.destroy_actor()
    return dict(results=results)


mh.guarded(main, os.path.join(V1, 'logs', f'render-result-{int(time.time())}.json'))
