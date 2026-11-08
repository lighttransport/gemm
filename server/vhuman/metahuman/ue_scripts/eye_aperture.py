"""Opaque mesh eye-aperture audit; separate from beauty references.

Temporarily replace head materials by black unlit and eye materials by white
unlit, hiding translucent shells, lacrimal edges and lashes. This isolates mesh
occlusion from refraction/blur. Overrides also remove original material vertex
/depth offsets, so these are not exact beauty boundaries. Restore afterwards.
"""
import json
import os
import time
import unreal
import mh_common as mh
import gen_batch as gb
import render_batch as rb


def flat_material(tag,name,value):
    package='/Game/MHGen/ApertureAudit';asset_name=tag+'_'+name
    if unreal.EditorAssetLibrary.does_asset_exist(package+'/'+asset_name):
        raise RuntimeError('temporary aperture material already exists')
    material=unreal.AssetToolsHelpers.get_asset_tools().create_asset(asset_name,package,unreal.Material,unreal.MaterialFactoryNew())
    material.set_editor_property('shading_model',unreal.MaterialShadingModel.MSM_UNLIT)
    material.set_editor_property('blend_mode',unreal.BlendMode.BLEND_OPAQUE)
    expression=unreal.MaterialEditingLibrary.create_material_expression(material,unreal.MaterialExpressionConstant3Vector)
    expression.set_editor_property('constant',unreal.LinearColor(value,value,value,1))
    if not unreal.MaterialEditingLibrary.connect_material_property(expression,'',unreal.MaterialProperty.MP_EMISSIVE_COLOR):
        raise RuntimeError('could not connect aperture material')
    errors=unreal.MaterialEditingLibrary.recompile_material(material)
    if errors:raise RuntimeError('aperture shader compile: '+str(errors))
    return material


def steps(batch,results):
    world=rb.load_environment('Studio');cap,comp,rt=rb.make_capture_actor(world,int(batch.get('size',1024)))
    lights=rb.spawn_replica(batch['lights'][batch['ids'][0]])
    rb.hide_environment(world,keep={cap.get_actor_label()}|{c.get_owner().get_actor_label() for c,_ in lights.values()})
    black=white=None;sub=mh.subsystem();tag=batch.get('tag','aperture_audit')
    try:
        black=flat_material(tag,'black',0);white=flat_material(tag,'white',1)
        yield from rb.settle(world)
        for sid in batch['ids']:
            root=os.path.join(rb.V1,'samples',sid);params=json.load(open(os.path.join(root,'params.json')));cams=json.load(open(os.path.join(root,'cameras.json')))['cameras']
            out=os.path.join(mh.OUT_ROOT,'eye_aperture',tag,sid);os.makedirs(out,exist_ok=True)
            character=mh.create_character('/Game/MHGen/ApertureAudit',sid+'_'+tag);actor=None;face=None;saved=[];hidden=[]
            try:
                character.preview_material_type=unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE
                with mh.Editing(character):
                    sub.set_face_model_coefficients(character=character,coefficients=[float(x) for x in params['coefficients']]);sub.commit_face_state(character=character)
                    sub.commit_skin_settings(character=character,skin_settings=gb.skin_settings_from_job(params['skin']));gb.apply_eyes(character,params['eyes'])
                    actor=sub.spawn_meta_human_actor(character=character);rb.place_replica(lights,batch['lights'][sid])
                    faces=[c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_name()=='Face']
                    if len(faces)!=1:raise RuntimeError('expected one Face component')
                    face=faces[0];face.set_forced_lod(1);slots=[str(x) for x in face.get_material_slot_names()]
                    eye_slots=[i for i,s in enumerate(slots) if s.lower().startswith(('eyeleft_','eyeright_'))]
                    if len(eye_slots)!=2:raise RuntimeError('expected two eye material slots')
                    comp.clear_show_only_components();comp.show_only_actor_components(actor,True)
                    yield from rb.settle(world);files=[]
                    for name in batch.get('cameras',['eye_left','eye_right']):
                        rb.shoot(world,comp,rt,cams[name],unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR,False,out,name+'__reference');files.append(name+'__reference.exr');yield
                    saved=[face.get_material(i) for i in range(len(slots))]
                    for i,slot in enumerate(slots):
                        face.set_material(i,white if i in eye_slots else black)
                        if slot.lower().startswith(('eyeshell_','eyeedge_','eyelashes_shader')):
                            face.show_material_section(i,i,False,0);hidden.append(i)
                    yield from rb.settle(world)
                    for name in batch.get('cameras',['eye_left','eye_right']):
                        rb.shoot(world,comp,rt,cams[name],unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR,True,out,name+'__geometry_mask');files.append(name+'__geometry_mask.exr');yield
                    record=dict(ok=True,id=sid,slots=slots,eye_slots=eye_slots,hidden_sections=hidden,files=files,size=int(batch.get('size',1024)),
                                method='opaque mesh aperture; black head/cartilage, white eyes; translucent shells/edges/lashes hidden',
                                engine_version=unreal.SystemLibrary.get_engine_version(),requested_rhi=os.environ.get('VHUMAN_UE_RHI'),finished=time.time(),
                                sha256={f:gb.sha256(os.path.join(out,f)) for f in files})
                    mh.write_json(os.path.join(out,'done.json'),record);results[sid]=record
            finally:
                if face:
                    for i,material in enumerate(saved):face.set_material(i,material)
                    for i in hidden:face.show_material_section(i,i,True,0)
                comp.clear_show_only_components()
                if actor:actor.destroy_actor()
                mh.delete_asset(character)
    finally:
        cap.destroy_actor()
        for component,_ in lights.values():component.get_owner().destroy_actor()
        for material in (white,black):
            if material:mh.delete_asset(material)


batch=json.load(open(mh.script_args()[0]));rb.batch_steps=steps
rb.run_ticked(batch,os.path.join(mh.OUT_ROOT,'eye_aperture',batch.get('tag','aperture_audit'),'result.json'))
