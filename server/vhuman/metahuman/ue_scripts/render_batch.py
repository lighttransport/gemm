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


def spawn_replica(replica):
    """vhuman studio lamps as equal-area RectLights (light_calibration.py plan/calibration records)."""
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem); lights = {}
    for r in replica:
        rot = unreal.MathLibrary.make_rot_from_x(unreal.Vector(*r['direction']))
        a = eas.spawn_actor_from_class(unreal.RectLight, unreal.Vector(*r['location_cm']), rot)
        c = a.rect_light_component
        c.set_mobility(unreal.ComponentMobility.MOVABLE)
        for k, v in (('source_width', r['width_cm']), ('source_height', r['width_cm']), ('barn_door_angle', 0.),
                     ('barn_door_length', 0.), ('attenuation_radius', 2000.), ('use_temperature', False),
                     ('intensity_units', unreal.LightUnits.CANDELAS), ('cast_shadows', True)):
            c.set_editor_property(k, v)
        c.set_light_color(unreal.LinearColor(1, 1, 1, 1))
        a.set_actor_label(f'vh_{r["name"]}')
        lights[r['name']] = (c, float(r['intensity_cd']))
    return lights


def hide_environment(world, keep):
    """Hide every light, sky light and reflection capture of the loaded environment (keep: our own actors)."""
    hidden = []
    for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
        if a.get_actor_label() in keep:
            continue
        for cls in (unreal.LightComponentBase, unreal.ReflectionCaptureComponent):
            for c in a.get_components_by_class(cls):
                if c.is_visible():
                    c.set_visibility(False); hidden.append(f'{a.get_actor_label()}:{c.get_class().get_name()}')
    return hidden


def place_replica(lights, records):
    for r in records:
        c, _ = lights[r['name']]
        c.get_owner().set_actor_location_and_rotation(unreal.Vector(*r['location_cm']), unreal.MathLibrary.make_rot_from_x(unreal.Vector(*r['direction'])), False, True)
        c.set_intensity(float(r['intensity_cd']))


SETTLE_S = float(os.environ.get('MH_RENDER_SETTLE_S', '8'))      # after compilation finished: streaming / GPU texture synthesis
SETTLE_TICKS = 30
FINISH_ALL = ('Editor.AsyncAssetCompilationFinishAll', 'Editor.AsyncTextureCompilationFinishAll',
              'Editor.AsyncStaticMeshCompilationFinishAll', 'Editor.AsyncSkinnedAssetCompilationFinishAll')


def settle(world):
    """Generator: let the editor tick until shaders/textures are compiled and streamed (the whole batch used to run inside
    one frame, so every capture showed placeholder materials: checkerboard skin, no eyes)."""
    for cmd in FINISH_ALL:
        unreal.SystemLibrary.execute_console_command(world, cmd)
    t0, n = time.time(), 0
    while n < SETTLE_TICKS or time.time() - t0 < SETTLE_S:
        n += 1
        if n % 10 == 0:
            for cmd in FINISH_ALL:
                unreal.SystemLibrary.execute_console_command(world, cmd)
        yield


def render_sample(sid, world, comp, rt, env, rec_out, replica=None):
    """Generator (one step per editor tick). replica: (lights, per-sample records) for the VHStudio env."""
    d = os.path.join(V1, 'samples', sid)
    params = json.load(open(os.path.join(d, 'params.json')))
    cams = json.load(open(os.path.join(d, 'cameras.json')))
    out_dir = os.path.join(d, 'renders'); os.makedirs(out_dir, exist_ok=True)
    sub = mh.subsystem()
    character = mh.create_character('/Game/MHGen/Render', sid)
    files = []
    # set before the edit session starts: the editing data picks its preview material at try_add_object_to_edit, and a later
    # property write does not reach the spawned actor (it rendered the 'Topology' region/wire material)
    character.preview_material_type = unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE
    try:
        with mh.Editing(character):
            sub.set_face_model_coefficients(character=character, coefficients=[float(x) for x in params['coefficients']])
            sub.commit_face_state(character=character)
            character.preview_material_type = unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE
            sub.commit_skin_settings(character=character, skin_settings=gb.skin_settings_from_job(params['skin']))
            gb.apply_eyes(character, params['eyes'])
            actor = sub.spawn_meta_human_actor(character=character)
            if replica:
                place_replica(*replica)
            yield from settle(world)
            comp.clear_show_only_components(); comp.show_only_actor_components(actor, True)   # list property is template-locked
            for name, cam in cams['cameras'].items():
                files.append(shoot(world, comp, rt, cam, unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR, False, out_dir, f'{name}__{env.lower()}'))
                if not replica:      # base colour / coverage do not depend on the lights
                    files.append(shoot(world, comp, rt, cam, unreal.SceneCaptureSource.SCS_BASE_COLOR, True, out_dir, f'{name}__basecolor'))
                yield
            comp.clear_show_only_components()
            actor.destroy_actor()
    finally:
        mh.delete_asset(character)
    rec = dict(ok=True, id=sid, env=env, files=files, size=cams['size'], finished=time.time(), settle_s=SETTLE_S,
               sha256={f: gb.sha256(os.path.join(d, f)) for f in files if os.path.isfile(os.path.join(d, f))})
    mh.write_json(os.path.join(out_dir, done_name(env)), rec)
    rec_out.append(rec)


def done_name(env):
    return 'renders_done.json' if env == 'Studio' else f'renders_done_{env.lower()}.json'


def batch_steps(batch, results):
    """env 'Studio' (MetaHuman lighting environment) or 'VHStudio': the Studio level with its lights hidden and the
    calibrated vhuman lamp rig (batch['lights'][id], from light_calibration.replica_batch) placed around each head."""
    env = batch.get('env', 'Studio')
    world = load_environment('Studio' if env == 'VHStudio' else env)
    size = int(batch.get('size', 1024))
    unreal.SystemLibrary.execute_console_command(world, 'r.Streaming.FullyLoadUsedTextures 1')
    cap, comp, rt = make_capture_actor(world, size)
    lights = None
    if env == 'VHStudio':
        lights = spawn_replica(batch['lights'][batch['ids'][0]])
        hidden = hide_environment(world, keep={a.get_actor_label() for a in [cap] + [c.get_owner() for c, _ in lights.values()]})
        mh.log(f'VHStudio: hidden {hidden}')
    yield from settle(world)          # environment level materials
    for sid in batch['ids']:
        done = os.path.join(V1, 'samples', sid, 'renders', done_name(env))
        if os.path.isfile(done) and json.load(open(done)).get('ok'):
            results[sid] = 'skip'; continue
        try:
            yield from render_sample(sid, world, comp, rt, env, [], replica=(lights, batch['lights'][sid]) if lights else None)
            results[sid] = 'ok'
        except Exception:      # noqa: BLE001
            tb = traceback.format_exc(); mh.log(tb)
            os.makedirs(os.path.dirname(done), exist_ok=True)
            mh.write_json(done, dict(ok=False, id=sid, error=tb)); results[sid] = 'fail'
        mh.log(f'{sid}: {results[sid]}')
    cap.destroy_actor()


def run_ticked(batch, result_path):
    """Drive batch_steps from Slate's post-tick callback so the engine renders/compiles between steps; quit at the end."""
    results = {}
    steps = batch_steps(batch, results)
    state = {}

    def tick(_dt):
        if state.get('busy') or state.get('done'):
            return                     # level loads / exports pump Slate ticks re-entrantly
        state['busy'] = True
        try:
            next(steps)
        except StopIteration:
            state['done'] = True
            unreal.unregister_slate_post_tick_callback(state['handle'])
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)
            mh.finish(result_path, dict(results=results), ok=True)
        except Exception:      # noqa: BLE001
            state['done'] = True
            unreal.unregister_slate_post_tick_callback(state['handle'])
            tb = traceback.format_exc(); mh.log(tb)
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)
            mh.finish(result_path, dict(results=results, error=tb), ok=False)
        finally:
            state['busy'] = False

    # -ExecutePythonScript quits the editor once the script returns unless the script asks to stay alive
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    state['handle'] = unreal.register_slate_post_tick_callback(tick)


if __name__ != 'render_batch':     # executed by -ExecutePythonScript (private scope)
    run_ticked(json.load(open(mh.script_args()[0])), os.path.join(V1, 'logs', f'render-result-{int(time.time())}.json'))
