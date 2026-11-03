# UE side of the ColorChecker light calibration (runs inside UE; host: metahuman/light_calibration.py).
# usage (via launch.py): chart_probe.py <plan.json>
# A matte 24-patch chart (one Engine plane per patch, default-lit material, roughness 1, specular 0, linear base colour =
# the ColorChecker reflectances) is placed per probe facing the reference camera and captured with the reference capture
# settings (render_batch.make_capture_actor) in:
#   studio   - the MetaHuman 'Studio' lighting environment used for the reference renders
#   key/fill/rim/replica - environment lights, sky light and reflection captures hidden; the vhuman studio rig replicated
#              as equal-area RectLights (one at a time, then all) with the plan's intensities
# Writes <out>/ue/<condition>__<probe>.exr and env_actors.json (what the Studio level contains and what was hidden).
import json
import os
import time
import traceback

import unreal

import mh_common as mh
import render_batch as rb

CALIB = '/Game/MHGen/Calib'
PLANE = '/Engine/BasicShapes/Plane.Plane'


def chart_material():
    path = f'{CALIB}/M_ChartPatch'
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools(); mel = unreal.MaterialEditingLibrary
    mat = tools.create_asset('M_ChartPatch', CALIB, unreal.Material, unreal.MaterialFactoryNew())
    color = mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -400, 0)
    color.set_editor_property('parameter_name', 'Color')
    mel.connect_material_property(color, 'RGB', unreal.MaterialProperty.MP_BASE_COLOR)
    for prop, value, y in ((unreal.MaterialProperty.MP_ROUGHNESS, 1., 200), (unreal.MaterialProperty.MP_SPECULAR, 0., 300),
                           (unreal.MaterialProperty.MP_METALLIC, 0., 400)):
        c = mel.create_material_expression(mat, unreal.MaterialExpressionConstant, -400, y); c.set_editor_property('r', value)
        mel.connect_material_property(c, '', prop)
    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def patch_instances(mat, albedo):
    tools = unreal.AssetToolsHelpers.get_asset_tools(); mel = unreal.MaterialEditingLibrary
    out = []
    for i, rgb in enumerate(albedo):
        path = f'{CALIB}/MI_Patch{i:02d}'
        mic = unreal.load_asset(path) if unreal.EditorAssetLibrary.does_asset_exist(path) else \
            tools.create_asset(f'MI_Patch{i:02d}', CALIB, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        mel.set_material_instance_parent(mic, mat)
        mel.set_material_instance_vector_parameter_value(mic, 'Color', unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.))
        mel.update_material_instance(mic)
        out.append(mic)
    return out


def spawn_patches(mics, patch_m):
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem); plane = unreal.load_asset(PLANE)
    actors = []
    for i, mic in enumerate(mics):
        a = eas.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        c = a.static_mesh_component; c.set_mobility(unreal.ComponentMobility.MOVABLE)
        c.set_static_mesh(plane); c.set_material(0, mic); c.set_cast_shadow(False)
        a.set_actor_scale3d(unreal.Vector(patch_m, patch_m, 1.))      # Engine plane: 100 cm, normal +Z
        a.set_actor_label(f'chart_patch_{i:02d}')
        actors.append(a)
    return actors


def place_patches(actors, probe):
    rot = unreal.MathLibrary.make_rot_from_zx(unreal.Vector(*probe['normal']), unreal.Vector(*probe['right']))
    for a, p in zip(actors, probe['patches']):
        a.set_actor_location_and_rotation(unreal.Vector(*p), rot, False, True)


def describe(world):
    rows = []
    for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
        row = dict(label=a.get_actor_label(), cls=a.get_class().get_name(), location=[a.get_actor_location().x, a.get_actor_location().y, a.get_actor_location().z])
        comps = []
        for c in a.get_components_by_class(unreal.LightComponentBase):
            d = dict(cls=c.get_class().get_name(), visible=c.is_visible(), intensity=float(c.get_editor_property('intensity')))
            for k in ('intensity_units', 'source_width', 'source_height', 'source_radius', 'source_type', 'light_color'):
                try:
                    d[k] = str(c.get_editor_property(k))
                except Exception:   # noqa: BLE001
                    pass
            rot = c.get_world_rotation(); d['rotation'] = [rot.roll, rot.pitch, rot.yaw]
            loc = c.get_world_location(); d['location'] = [loc.x, loc.y, loc.z]
            comps.append(d)
        if comps:
            row['lights'] = comps
        rows.append(row)
    return rows


def steps(plan, results):
    out = os.path.join(plan['out'], 'ue'); os.makedirs(out, exist_ok=True)
    world = rb.load_environment('Studio')
    size = int(plan['size'])
    unreal.SystemLibrary.execute_console_command(world, 'r.Streaming.FullyLoadUsedTextures 1')
    cap, comp, rt = rb.make_capture_actor(world, size)
    mat = chart_material(); mics = patch_instances(mat, plan['albedo_linear'])
    patches = spawn_patches(mics, plan['patch_m'])
    yield from rb.settle(world)
    results['env_actors'] = describe(world)
    files = []

    def shoot_all(cond):
        for name, probe in plan['probes'].items():
            place_patches(patches, probe['ue'])
            yield
            files.append(rb.shoot(world, comp, rt, dict(ue=probe['ue']['camera']), unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR, False, out, f'{cond}__{name}'))

    yield from shoot_all('studio')
    lights = rb.spawn_replica(plan['replica'])
    results['hidden'] = rb.hide_environment(world, keep={a.get_actor_label() for a in patches + [cap] + [c.get_owner() for c, _ in lights.values()]})
    for cond in ('key', 'fill', 'rim', 'replica'):
        for name, (c, cd) in lights.items():
            c.set_intensity(cd if cond in (name, 'replica') else 0.)
        yield from rb.settle(world)
        yield from shoot_all(cond)
    results['files'] = files
    for a in patches + [c.get_owner() for c, _ in lights.values()] + [cap]:
        a.destroy_actor()


def run_ticked(plan, result_path):
    results = {}; gen = steps(plan, results); state = {}

    def tick(_dt):
        if state.get('busy') or state.get('done'):
            return
        state['busy'] = True
        try:
            next(gen)
        except StopIteration:
            state['done'] = True
            unreal.unregister_slate_post_tick_callback(state['handle'])
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)
            mh.write_json(os.path.join(plan['out'], 'ue', 'env_actors.json'), dict(actors=results.get('env_actors'), hidden=results.get('hidden')))
            mh.finish(result_path, dict(files=results.get('files', [])), ok=True)
        except Exception:      # noqa: BLE001
            state['done'] = True
            unreal.unregister_slate_post_tick_callback(state['handle'])
            tb = traceback.format_exc(); mh.log(tb)
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)
            mh.finish(result_path, dict(error=tb), ok=False)
        finally:
            state['busy'] = False

    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    state['handle'] = unreal.register_slate_post_tick_callback(tick)


if __name__ != 'chart_probe':     # executed by -ExecutePythonScript (private scope)
    _plan = json.load(open(mh.script_args()[0]))
    run_ticked(_plan, os.path.join(_plan['out'], 'ue_result.json'))
