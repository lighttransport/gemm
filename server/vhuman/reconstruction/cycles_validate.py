"""Blender-only reload gate for packed signed maps and native motion samples."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np


def validate(out):
    out=Path(out);detail=np.load(out/'skin_detail.npz',allow_pickle=False)
    checks=[]
    for name,expected in [('skin_height_metres',detail['height_m'])]+[
            (f'wrinkle_height_{i}',row) for i,row in enumerate(detail['dynamic_height_m'])]:
        image=bpy.data.images[name]
        if image.packed_file is None:raise ValueError(f'{name}: unpacked physical map')
        pixels=np.empty(len(image.pixels),np.float32);image.pixels.foreach_get(pixels)
        actual=pixels.reshape(image.size[1],image.size[0],4)[:,:,0]
        error=float(np.max(abs(actual-expected[::-1])))
        if error>1e-8:raise ValueError(f'{name}: signed physical map changed by {error} m')
        checks.append(dict(name=name,max_error_m=error,minimum_m=float(actual.min())))
    assets=np.load(out/'scene_assets.npz',allow_pickle=False)
    expected_hair=assets['hair_curves'].reshape(-1,3)
    if len(expected_hair):
        hair=bpy.data.objects['hair'].data
        positions=np.empty(len(expected_hair)*3,np.float32)
        hair.attributes['position'].data.foreach_get('vector',positions)
        if np.max(abs(positions.reshape(-1,3)-expected_hair))>1e-7:
            raise ValueError('hair strand coordinates changed after scene reload')
    if (out/'hair_coverage.png').is_file():
        image=bpy.data.images['hair_coverage.png']
        if image.packed_file is None:raise ValueError('short-hair coverage is unpacked')
        # Reload source bytes independently and compare the packed Non-Color mask.
        source=bpy.data.images.load(str(out/'hair_coverage.png'),check_existing=False)
        source.colorspace_settings.name='Non-Color'
        original=np.empty(len(source.pixels),np.float32);source.pixels.foreach_get(original)
        actual=np.empty(len(image.pixels),np.float32);image.pixels.foreach_get(actual)
        if np.max(abs(actual-original))>1e-7:raise ValueError('packed hair coverage changed')
        bpy.data.images.remove(source)
    if 'skin_hair_crown' in assets:
        mesh=bpy.data.objects['skin'].data
        actual=np.empty(len(assets['skin_hair_crown']),np.float32)
        mesh.attributes['hair_crown_coverage'].data.foreach_get('value',actual)
        if np.max(abs(actual-assets['skin_hair_crown']))>1e-7:
            raise ValueError('native crown coverage changed after scene reload')
        uv=mesh.uv_layers['SourceCamera'];actual_uv=np.empty(len(uv.data)*2,np.float32)
        uv.data.foreach_get('uv',actual_uv)
        expected=assets['skin_source_uvs'].reshape(-1,2).copy();expected[:,1]=1-expected[:,1]
        if np.max(abs(actual_uv.reshape(-1,2)-expected))>1e-7:
            raise ValueError('source-camera hair UV changed after scene reload')
        if not mesh.uv_layers['UVMap'].active_render:raise ValueError('hair UV displaced skin atlas UV')
    request=json.loads((out/'request.json').read_text())
    eye_material_checked=False
    if (request.get('material_overrides') or {}).get('eye_per_side',0.) >= .5:
        for side in ('left','right'):
            eye=bpy.data.objects[f'{side}_cornea'];material=eye.data.materials[0]
            images=[n.image for n in material.node_tree.nodes if n.type=='TEX_IMAGE' and n.image is not None]
            # Blender stores packed texture paths as //relative paths; resolve them before pathlib handles them on Windows.
            bases=[im for im in images if Path(bpy.path.abspath(im.filepath)).name.startswith('eye_unified_')]
            if len(bases)!=1:raise ValueError(f'{side} eye: expected one unified base texture')
            is_right=Path(bpy.path.abspath(bases[0].filepath)).name.startswith('eye_unified_right')
            if is_right != (side=='right'):raise ValueError(f'{side} eye uses the wrong side texture')
            for image in images:
                if image.packed_file is None:raise ValueError(f'{side} eye texture is unpacked: {image.name}')
        eye_material_checked=True
    eye_split_checked=False
    if (request.get('material_overrides') or {}).get('eye_ue_split',0.) >= .5:
        for side in ('left','right'):
            mesh=bpy.data.objects[f'{side}_cornea'].data
            attr=mesh.attributes.get('iris_plane_normal')
            if attr is None:raise ValueError(f'{side} eye has no iris-plane normal')
            values=np.empty(len(mesh.vertices)*3,np.float32);attr.data.foreach_get('vector',values)
            normals=values.reshape(-1,3)
            if not np.allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-5):
                raise ValueError(f'{side} eye iris-plane normals are not unit vectors')
        if (request.get('material_overrides') or {}).get('eye_bilateral_select',0.) >= .5:
            right=json.loads((out/'eye_params_right.json').read_text())
            node=bpy.data.objects['right_cornea'].data.materials[0].node_tree.nodes.get('vhuman_iris_mask')
            if node is None:raise ValueError('right eye iris mask is missing')
            size=float(right['cornea']['size']);half=.5*float(right['cornea']['limbus_softness'])
            if not np.allclose([node.inputs['From Min'].default_value,node.inputs['From Max'].default_value],
                               [size-half,size+half],atol=1e-6):
                raise ValueError('right eye iris mask uses the wrong eye parameters')
            rough=bpy.data.objects['right_cornea'].data.materials[0].node_tree.nodes.get('vhuman_sclera_roughness')
            if rough is not None:
                left=json.loads((out/'eye_params.json').read_text());delta=size-float(left['cornea']['size'])
                if not np.allclose([rough.inputs['From Min'].default_value,rough.inputs['From Max'].default_value],
                                   [.19+delta,.23+delta],atol=1e-6):
                    raise ValueError('right eye roughness transition uses the wrong eye size')
        eye_split_checked=True
    if 'unified_key_caustic_strength' in (request.get('material_overrides') or {}):
        for side in ('left','right'):
            material=bpy.data.objects[f'{side}_cornea'].data.materials[0]
            if material.node_tree.nodes.get('vhuman_key_caustic') is None:
                raise ValueError(f'{side} eye key caustic node is missing')
    if request.get('motion'):
        motion=np.load(Path(request['motion'])/'motion.npz',allow_pickle=False)
        assets=np.load(out/'scene_assets.npz',allow_pickle=False)
        obj=bpy.data.objects['skin'];keys=obj.data.shape_keys.key_blocks
        if len(keys)!=len(motion['vertices'])+1:raise ValueError('native motion frame count changed')
        for frame in (0,len(motion['vertices'])-1):
            key=keys[f'GNM_{frame+1:03d}'];positions=np.empty(len(key.data)*3,np.float32)
            key.data.foreach_get('co',positions)
            expected=motion['vertices'][frame,assets['skin_native_ids']]
            if 'skin_bound_ids' in assets:
                corners=motion['vertices'][frame][assets['skin_bound_ids']].astype(float)
                x=corners[:,1]-corners[:,0];x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
                z=np.cross(x,corners[:,2]-corners[:,0]);z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
                frames=np.stack((x,np.cross(z,x),z),-1)
                bound=(corners*assets['skin_bound_weights'][:,:,None]).sum(1)+np.einsum('vij,vj->vi',frames,assets['skin_bound_offsets'])
                expected=np.concatenate((expected,bound))
            if np.max(abs(positions.reshape(-1,3)-expected))>1e-7:
                raise ValueError('native motion coordinates changed after scene reload')
    result=dict(passed=True,float_maps=checks,hair_points_checked=len(expected_hair),hair_mask_checked=(out/'hair_coverage.png').is_file(),native_crown_checked='skin_hair_crown' in assets,native_motion_checked=bool(request.get('motion')),
                eye_material_checked=eye_material_checked,eye_split_checked=eye_split_checked,
                storage='packed scene-linear 32-bit EXR')
    (out/'asset_validation.json').write_text(json.dumps(result,indent=2))
    print('VHUMAN_PACKED_ASSET_VALID',flush=True)


if __name__=='__main__':validate(sys.argv[sys.argv.index('--')+1])
