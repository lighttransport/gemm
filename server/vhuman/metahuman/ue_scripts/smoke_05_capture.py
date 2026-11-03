# Smoke 05: load the MetaHuman lighting environment, spawn the preview actor and capture the thumbnail-style camera to
# linear EXR (final colour HDR + base colour) through a SceneCapture2D. Writes smoke/05_capture.json. Runs inside UE.
import math
import os

import unreal

import mh_common as mh

OUT = os.path.join(mh.OUT_ROOT, 'smoke', '05_capture.json')
IMG_DIR = os.path.join(mh.OUT_ROOT, 'smoke', '05_img')
BASE_LEVEL = '/MetaHumanCharacter/LightingEnvironments/L_BaseEnvironment'
STUDIO = '/MetaHumanCharacter/LightingEnvironments/Studio'
FOV = 18.0017


def load_environment(out):
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    cls = unreal.EditorAssetLibrary.find_asset_data(BASE_LEVEL)
    out['base_level_class'] = str(cls.asset_class_path.asset_name) if cls.is_valid() else None
    cls2 = unreal.EditorAssetLibrary.find_asset_data(STUDIO)
    out['studio_class'] = str(cls2.asset_class_path.asset_name) if cls2.is_valid() else None
    loaded = False
    if cls.is_valid() and 'World' in out['base_level_class']:
        loaded = les.load_level(BASE_LEVEL)
    elif cls2.is_valid() and 'World' in (out['studio_class'] or ''):
        loaded = les.load_level(STUDIO)
    out['level_loaded'] = bool(loaded)
    world = mh.editor_world()
    if loaded and cls2.is_valid() and 'World' in (out['studio_class'] or '') and BASE_LEVEL != STUDIO:
        try:
            streaming = unreal.EditorLevelUtils.add_level_to_world(world, STUDIO, unreal.LevelStreamingAlwaysLoaded)
            out['studio_streamed'] = streaming is not None
        except Exception as e:      # noqa: BLE001
            out['studio_stream_error'] = str(e)
    return mh.editor_world()


def capture(world, origin, rotation, fov, w, h, source, path, name):
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, origin, rotation)
    comp = cap.capture_component2d
    rt = unreal.RenderingLibrary.create_render_target2d(world, w, h, unreal.TextureRenderTargetFormat.RTF_RGBA16F, unreal.LinearColor(0, 0, 0, 1), False)
    comp.texture_target = rt
    comp.capture_source = source
    comp.fov_angle = fov
    comp.capture_every_frame = False
    comp.always_persist_rendering_state = True
    pps = comp.post_process_settings
    pps.override_auto_exposure_method = True
    pps.auto_exposure_method = unreal.AutoExposureMethod.AEM_MANUAL
    pps.override_auto_exposure_bias = True
    pps.auto_exposure_bias = 0.0
    pps.override_bloom_intensity = True
    pps.bloom_intensity = 0.0
    pps.override_vignette_intensity = True
    pps.vignette_intensity = 0.0
    comp.post_process_settings = pps
    comp.capture_scene()
    unreal.RenderingLibrary.export_render_target(world, rt, path, f'{name}.exr')
    cap.destroy_actor()
    return os.path.join(path, f'{name}.exr')


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    out = {}
    world = load_environment(out)
    character = mh.create_character('/Game/MHGen/Smoke', 'Smoke05')
    sub = mh.subsystem()
    with mh.Editing(character):
        actor = sub.spawn_meta_human_actor(character=character)
        if actor is None:
            raise RuntimeError('spawn_meta_human_actor returned None')
        origin, extent = actor.get_actor_bounds(False)
        out['actor_bounds'] = dict(origin=[origin.x, origin.y, origin.z], extent=[extent.x, extent.y, extent.z])
        # face framing as the thumbnail renderer: look at the head from the front; try the head component bounds first
        head_centre = None
        for comp in actor.get_components_by_class(unreal.SkeletalMeshComponent):
            if 'face' in comp.get_name().lower() or 'head' in comp.get_name().lower():
                head_centre, head_extent, _ = unreal.SystemLibrary.get_component_bounds(comp)
                break
        if head_centre is None:
            head_centre = unreal.Vector(origin.x, origin.y, origin.z + extent.z * 0.8); head_extent = unreal.Vector(12, 12, 14)
        out['head_bounds'] = dict(origin=[head_centre.x, head_centre.y, head_centre.z], extent=[head_extent.x, head_extent.y, head_extent.z])
        radius = max(head_extent.x, head_extent.y, head_extent.z) * 0.8
        dist = radius / math.tan(math.radians(FOV / 2))
        shots = {}
        for tag, yaw in (('front_px', 180.0), ('front_mx', 0.0), ('side', 90.0)):
            # camera on a circle around the head centre, looking at it
            rad = math.radians(yaw)
            cam_pos = unreal.Vector(head_centre.x + dist * math.cos(rad), head_centre.y + dist * math.sin(rad), head_centre.z)
            rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, head_centre)
            shots[tag] = dict(camera=[cam_pos.x, cam_pos.y, cam_pos.z], yaw=yaw)
            for src, name in ((unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR, f'{tag}_final'), (unreal.SceneCaptureSource.SCS_BASE_COLOR, f'{tag}_basecolor')):
                try:
                    shots[tag][name] = capture(world, cam_pos, rot, FOV, 1024, 1024, src, IMG_DIR, name)
                except Exception as e:      # noqa: BLE001
                    shots[tag][name + '_error'] = str(e)
        out['shots'] = shots
        out['fov'] = FOV; out['distance_cm'] = dist
        actor.destroy_actor()
    mh.delete_asset(character)
    return out


mh.guarded(main, OUT)
