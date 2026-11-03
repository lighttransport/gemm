# Smoke 02: local skin texture synthesis (commit_skin_settings) for several face texture indices, and texture readback
# through a render target (route A). Writes smoke/02_skin.json and smoke/02_tex/<index>/<type>.exr. Runs inside UE.
import os

import unreal

import mh_common as mh

OUT = os.path.join(mh.OUT_ROOT, 'smoke', '02_skin.json')
TEX_DIR = os.path.join(mh.OUT_ROOT, 'smoke', '02_tex')
TOOLS_PKG = '/Game/MHGen/Tools'


def skin_settings(u, v, index, roughness=1.0):
    skin = unreal.MetaHumanCharacterSkinProperties()
    skin.u = u; skin.v = v
    skin.face_texture_index = index
    skin.body_texture_index = 0
    skin.roughness = roughness
    skin.show_top_underwear = True
    s = unreal.MetaHumanCharacterSkinSettings()
    s.skin = skin
    s.freckles = unreal.MetaHumanCharacterFrecklesProperties()
    s.accents = unreal.MetaHumanCharacterAccentRegions()
    try:
        s.texture_material_overrides.enable_texture_overrides = False
    except Exception:      # noqa: BLE001
        pass
    return s


def copy_material(name, sampler):
    """Unlit material: Emissive = texture parameter 'Tex' sampled with the given sampler type (raw values)."""
    path = f'{TOOLS_PKG}/{name}'
    mat = unreal.EditorAssetLibrary.load_asset(path)
    if mat is None:
        mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, TOOLS_PKG, unreal.Material, unreal.MaterialFactoryNew())
        mat.set_editor_property('shading_model', unreal.MaterialShadingModel.MSM_UNLIT)
        node = unreal.MaterialEditingLibrary.create_material_expression(mat, unreal.MaterialExpressionTextureSampleParameter2D, -400, 0)
        node.set_editor_property('parameter_name', 'Tex')
        node.set_editor_property('sampler_type', sampler)
        # the node's default texture must match the sampler type or the material does not compile
        default_tex = '/Engine/EngineMaterials/DefaultNormal' if sampler == unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL else '/Engine/EngineResources/DefaultTexture'
        node.set_editor_property('texture', unreal.EditorAssetLibrary.load_asset(default_tex))
        if sampler == unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL:
            # re-encode the decoded normal to [0,1] so the file holds the same bytes as a colour map
            mul = unreal.MaterialEditingLibrary.create_material_expression(mat, unreal.MaterialExpressionMultiply, -200, 0)
            half = unreal.MaterialEditingLibrary.create_material_expression(mat, unreal.MaterialExpressionConstant, -300, 150)
            half.set_editor_property('r', 0.5)
            add = unreal.MaterialEditingLibrary.create_material_expression(mat, unreal.MaterialExpressionAdd, -100, 0)
            unreal.MaterialEditingLibrary.connect_material_expressions(node, 'RGB', mul, 'A')
            unreal.MaterialEditingLibrary.connect_material_expressions(half, '', mul, 'B')
            unreal.MaterialEditingLibrary.connect_material_expressions(mul, '', add, 'A')
            unreal.MaterialEditingLibrary.connect_material_expressions(half, '', add, 'B')
            unreal.MaterialEditingLibrary.connect_material_property(add, '', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        else:
            unreal.MaterialEditingLibrary.connect_material_property(node, 'RGBA', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        unreal.MaterialEditingLibrary.recompile_material(mat)
        unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def texture_info(tex):
    info = dict(cls=tex.get_class().get_name())
    for prop in ('srgb', 'compression_settings', 'lod_group', 'never_stream'):
        try:
            info[prop] = str(tex.get_editor_property(prop))
        except Exception as e:      # noqa: BLE001
            info[prop] = f'err {e}'
    try:
        info['size'] = [tex.blueprint_get_size_x(), tex.blueprint_get_size_y()]
    except Exception:      # noqa: BLE001
        pass
    return info


def type_name(key):
    """EFaceTextureType enum value -> 'basecolor', 'normal_animated_wm1', ... (repr looks like 'BASECOLOR: 0>')."""
    n = getattr(key, 'name', None) or str(key).split('.')[-1].split(':')[0]
    return n.strip().lower()


def export_direct(tex, out_dir, name):
    """Route C: ImageWriteBlueprintLibrary reads the texture's resident mip 0 and writes it synchronously (PNG = exact bytes)."""
    os.makedirs(out_dir, exist_ok=True)
    opts = unreal.ImageWriteOptions()
    opts.format = unreal.DesiredImageFormat.PNG
    opts.compression_quality = 100
    opts.async_ = False
    path = os.path.join(out_dir, f'{name}.png')
    unreal.ImageWriteBlueprintLibrary.export_to_disk(tex, path, opts)
    return path


def readback(world, tex, out_dir, name, mats):
    """Draw the texture into a float render target and export it as EXR (raw sampled values)."""
    os.makedirs(out_dir, exist_ok=True)
    w, h = tex.blueprint_get_size_x(), tex.blueprint_get_size_y()
    rt = unreal.RenderingLibrary.create_render_target2d(world, w, h, unreal.TextureRenderTargetFormat.RTF_RGBA16F,
                                                        unreal.LinearColor(0, 0, 0, 1), False)
    comp = str(tex.get_editor_property('compression_settings'))
    mat = mats['normal'] if 'NORMAL' in comp.upper() else mats['linear']
    mid = unreal.MaterialLibrary.create_dynamic_material_instance(world, mat)
    mid.set_texture_parameter_value('Tex', tex)
    unreal.RenderingLibrary.draw_material_to_render_target(world, rt, mid)
    unreal.RenderingLibrary.export_render_target(world, rt, out_dir, f'{name}.exr')
    return os.path.join(out_dir, f'{name}.exr'), comp


def main():
    out = dict(indices={})
    world = mh.editor_world()
    mats = dict(linear=copy_material('M_CopyTexColor', unreal.MaterialSamplerType.SAMPLERTYPE_COLOR),
                normal=copy_material('M_CopyTexNormal', unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL))
    character = mh.create_character('/Game/MHGen/Smoke', 'Smoke02')
    sub = mh.subsystem()
    prev = None
    with mh.Editing(character):
        for u, v, idx in ((0.2, 0.7, 3), (0.5, 0.3, 50), (0.8, 0.5, 120)):
            character.preview_material_type = unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE
            s = skin_settings(u, v, idx)
            sub.commit_skin_settings(character=character, skin_settings=s)
            applied = character.get_editor_property('skin_settings')
            rec = dict(requested_index=idx, applied_index=int(applied.skin.face_texture_index), applied_u=float(applied.skin.u), textures={})
            texmap = character.get_editor_property('synthesized_face_textures')
            for key in texmap:
                tex = texmap[key]
                info = texture_info(tex)
                name = type_name(key)
                try:
                    p = export_direct(tex, os.path.join(TEX_DIR, str(idx)), name)
                    info['exported_direct'] = p if os.path.isfile(p) else f'missing {p}'
                except Exception as e:      # noqa: BLE001
                    info['direct_error'] = str(e)
                try:
                    path, comp = readback(world, tex, os.path.join(TEX_DIR, str(idx)), name + '_rt', mats)
                    info['exported'] = path if os.path.isfile(path) else f'missing {path}'
                except Exception as e:      # noqa: BLE001
                    info['export_error'] = str(e)
                rec['textures'][str(key)] = info
            out['indices'][idx] = rec
            prev = s
    out['n_texture_types'] = max((len(r['textures']) for r in out['indices'].values()), default=0)
    mh.delete_asset(character)
    return out


mh.guarded(main, OUT)
