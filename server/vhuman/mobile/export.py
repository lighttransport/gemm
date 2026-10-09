"""Export a provenance-checked GNM capture and its prepared optical anatomy.

The portable GLB is a static reference. Native deformation/bindings are separate
and retain all original GNM coefficients, rather than renaming PCA as visemes.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import numpy as np
from PIL import Image
from ..eye.glb import GLBBuilder
from ..eye.geometry import Mesh
from ..rig.common import vertex_normals
from ..rig.gnm_model import GNMModel
from ..reconstruction.provenance import validate_candidate
from ..reconstruction.observations import sha256
from ..reconstruction.offline_assets import attachment_frames
from .native import write_model
from .profile import IPHONE12
from .materials import skin_textures


def hair_cards(paths,limit=1024,width=.00035):
    """Deterministic two-segment ribbons; roots retain source scalp bindings."""
    selected=np.linspace(0,len(paths)-1,min(limit,len(paths)),dtype=int)
    curves=paths[selected][:,[0,3,7]]
    if not len(curves):return np.zeros((0,3)),np.zeros((0,3),int),np.zeros((0,3,2)),selected
    tangent=curves[:,-1]-curves[:,0]
    side=np.cross(tangent,np.array([0.,0.,1.]))
    short=np.linalg.norm(side,axis=1)<1e-8
    side[short]=np.cross(tangent[short],np.array([0.,1.,0.]))
    side/=np.maximum(np.linalg.norm(side,axis=1,keepdims=True),1e-8)
    taper=np.array([1.,.7,.03])*width
    vertices=np.stack((curves-side[:,None]*taper[None,:,None],curves+side[:,None]*taper[None,:,None]),2).reshape(-1,3)
    faces=np.array([[0,1,2],[1,3,2],[2,3,4],[3,5,4]])[None]+np.arange(len(curves))[:,None,None]*6
    uv=np.array([[0,0],[1,0],[0,.5],[1,.5],[0,1],[1,1]])
    return vertices,faces.reshape(-1,3),np.tile(uv[np.array([[0,1,2],[1,3,2],[2,3,4],[3,5,4]])],(len(curves),1,1)),selected


def validate_package(folder):
    folder=Path(folder).resolve();manifest=json.loads((folder/'avatar.json').read_text())
    if manifest.get('schema')!='vhuman.mobile_avatar.v1':raise ValueError('unsupported mobile avatar')
    files=manifest.get('files',{})
    if not isinstance(files,dict) or not {'gnm.bin','avatar.glb','streams.bin','bindings.bin','controls.json'}<=files.keys():
        raise ValueError('incomplete mobile package')
    for name,record in files.items():
        path=(folder/name).resolve()
        if ('/' in name or '\\' in name or not path.is_relative_to(folder) or not path.is_file()
                or path.stat().st_size!=record['bytes'] or not 0<record['bytes']<=256*1024*1024
                or sha256(path)!=record['sha256']):
            raise ValueError('mobile asset path/checksum mismatch')
    return manifest


def export(candidate,out, *, scene,profile='iphone12'):
    if profile!='iphone12':raise ValueError('unsupported mobile profile')
    candidate,out,scene=map(lambda p:Path(p).resolve(),(candidate,out,scene))
    manifest=validate_candidate(candidate);description=json.loads((scene/'scene.json').read_text())
    if validate_candidate(description['candidate'])['geometry_sha256']!=manifest['geometry_sha256']:
        raise ValueError('prepared scene belongs to different geometry')
    if out.exists() and any(out.iterdir()):raise ValueError('mobile output must be empty')
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
    with np.load(scene/'scene_assets.npz',allow_pickle=False) as z:assets={k:z[k] for k in z.files}
    # A scene path alone is not sufficient provenance: inspect native positions.
    for p in description['parts']:
        if p['native']:
            name=p['name'];expected=g['full_captured'][0,assets[name+'_native_ids']]
            if name+'_bound_ids' in assets:
                corners=g['full_captured'][0][assets[name+'_bound_ids']].astype(float)
                bound=(corners*assets[name+'_bound_weights'][:,:,None]).sum(1)+np.einsum(
                    'vij,vj->vi',attachment_frames(corners),assets[name+'_bound_offsets'])
                expected=np.concatenate((expected,bound))
            np.testing.assert_allclose(assets[name+'_positions'],expected,atol=1e-7)
    out.mkdir(parents=True,exist_ok=True)
    model=GNMModel();native=write_model(out/'gnm.bin',model,g)
    skin_maps=skin_textures(candidate,scene,assets,description,out)
    b=GLBBuilder('vhuman mobile reference');materials={};images={}
    def texture(path):
        if str(path) not in images:
            im=Image.open(path);im.thumbnail((2048,2048),Image.Resampling.LANCZOS)
            images[str(path)]=b.texture(np.asarray(im),path.stem)
        return images[str(path)]
    def material(kind):
        if kind in materials:return materials[kind]
        colors={'teeth':[.72,.66,.52,1],'gums':[.28,.045,.055,1],'tongue':[.3,.07,.09,1],
                'cavity':[.32,.085,.085,1],'pupil':[.001,.001,.001,1],'lash':[.008,.005,.003,1],
                'hair':[*description['hair']['color_linear'],1],'glass':[1,1,1,.18],'tear':[1,1,1,.2],
                'cornea':[1,1,1,.08]}
        pbr=dict(baseColorFactor=colors.get(kind,[1,1,1,1]),metallicFactor=0.,roughnessFactor=.5)
        m=dict(name=kind,pbrMetallicRoughness=pbr)
        if kind=='skin':
            pbr['roughnessFactor']=1.
            pbr.update(baseColorTexture={'index':texture(skin_maps['basecolor'])},metallicRoughnessTexture={'index':texture(skin_maps['orm'])})
            m['normalTexture']={'index':texture(skin_maps['normal'])}
            m['extensions']={'KHR_materials_specular':{'specularColorFactor':[1,1,1], 'specularFactor':manifest['material']['f0']['value']/.04}}
        elif kind in ('iris','sclera'):
            pbr['baseColorTexture']={'index':texture(scene/(kind+'.png'))};pbr['roughnessFactor']=.25
        elif kind in ('glass','tear','cornea'):
            pbr['roughnessFactor']=.025;pbr['baseColorFactor']=[1,1,1,1]
            m['extensions']={'KHR_materials_ior':{'ior':1.376 if kind=='cornea' else (1.336 if kind=='tear' else 1.5)},
                'KHR_materials_transmission':{'transmissionFactor':1.}}
        if kind=='hair':m['doubleSided']=True;pbr['roughnessFactor']=.65
        materials[kind]=b.material(m);return materials[kind]
    parts=list(description['parts']);full=g['full_captured'][0]
    curves=assets['hair_curves'];v,t,uv,selected=hair_cards(curves,width=.0002 if description['hair']['short_hair'] else .0005)
    if len(selected):
        name='mobile_hair';parts.append(dict(name=name,material='hair',native=False,joint=None,surface_bound=True))
        anchors=np.repeat(assets['hair_root_triangle_ids'][selected],6,axis=0)
        weights=np.repeat(assets['hair_root_weights'][selected],6,axis=0)
        root=(full[anchors]*weights[:,:,None]).sum(1);frame=attachment_frames(full[anchors])
        assets.update({name+'_positions':v,name+'_triangles':t,name+'_uvs':uv,
            name+'_surface_ids':anchors,name+'_surface_weights':weights,
            name+'_surface_offsets':np.einsum('vji,vj->vi',frame,v-root)})
    records=[];bindings={};streams=[];triangles=0;contact={};render_maps={}
    for part in parts:
        name=part['name'];pos=assets[name+'_positions'];tri=assets[name+'_triangles'];uv=assets[name+'_uvs']
        normals=vertex_normals(pos,tri)
        # UV seams duplicate render vertices, but retain native/surface bindings.
        keys=np.column_stack((tri.ravel(),uv.reshape(-1,2)))
        unique,inverse=np.unique(keys,axis=0,return_inverse=True);mapping=unique[:,0].astype(int)
        mesh=Mesh(name,pos[mapping],normals[mapping],unique[:,1:],inverse.reshape(-1,3)).with_tangents()
        mi=b.mesh(mesh,material(part['material']));b.node(name,mesh=mi)
        streams.append(mesh)
        n=len(mapping);ids=np.zeros((n,6),np.uint32);weights=np.zeros((n,6),np.float32);offset=np.zeros((n,3),np.float32)
        joint=-1
        if part['native']:
            native_count=len(assets[name+'_native_ids']);direct=mapping<native_count
            ids[direct,0]=assets[name+'_native_ids'][mapping[direct]];weights[direct,0]=1
            if (~direct).any():
                # Appended refined-ear vertices: barycentric anchor + attachment-frame offset.
                bound=mapping[~direct]-native_count
                ids[~direct,:3]=assets[name+'_bound_ids'][bound];weights[~direct,:3]=assets[name+'_bound_weights'][bound]
                offset[~direct]=assets[name+'_bound_offsets'][bound]
        elif part.get('surface_bound'):
            src=assets[name+'_surface_ids'][mapping];w=assets[name+'_surface_weights'][mapping]
            ids[:,:src.shape[1]]=src;weights[:,:w.shape[1]]=w;offset=assets[name+'_surface_offsets'][mapping]
        else:
            joint=part.get('joint') if part.get('joint') is not None else 1
            offset=pos[mapping]-g['gnm_joint_positions'][joint]
        bindings[name+'_ids']=ids;bindings[name+'_weights']=weights;bindings[name+'_offset']=offset;render_maps[name]=mapping
        if name+'_contact_active' in assets:
            # Per-pose lip/arch contact deformer, evaluated in part-vertex space (render_to_part scatters seams).
            contact[name]=dict(render_to_part=mapping.tolist(),part_vertices=int(len(pos)),active=assets[name+'_contact_active'].tolist(),
                weight=np.round(assets[name+'_contact_weight'],6).tolist(),candidates=assets[name+'_contact_candidates'].tolist(),
                params=assets[name+'_contact_params'].tolist(),edges=assets[name+'_contact_edges'].tolist())
        center=np.asarray(part.get('rotation_center_offset',[0,0,0]),dtype=np.float32)
        if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('invalid fitted joint center')
        records.append(dict(name=name,mesh=mi,vertices=n,triangles=len(tri),joint=joint,native=part['native'],
            rotation_center_offset=center.tolist()))
        triangles+=len(tri)
    if triangles>IPHONE12['triangle_budgets'][0]:raise ValueError(f'mobile mesh exceeds triangle budget: {triangles}')
    b.write(out/'avatar.glb');np.savez_compressed(out/'bindings.npz',**bindings)
    with (out/'streams.bin').open('wb') as f:
        f.write(struct.pack('<8sI',b'VHMES001',len(streams)))
        for mesh in streams:
            name=mesh.name.encode();f.write(struct.pack('<III',len(name),len(mesh.positions),len(mesh.indices)))
            f.write(name)
            for array,dtype in ((mesh.positions,'<f4'),(mesh.normals,'<f4'),(mesh.uvs,'<f4'),(mesh.indices,'<u4')):
                f.write(np.asarray(array,dtype=dtype).tobytes())
    # Flat binary avoids a Python/NPZ dependency on device. Each record follows
    # glTF mesh order and shares its indexed vertex ordering.
    with (out/'bindings.bin').open('wb') as f:
        f.write(struct.pack('<8sI',b'VHBND002',len(records)))
        for p in records:
            name=p['name'];f.write(struct.pack('<Iii',p['vertices'],p['joint'],int(p['native'])))
            f.write(np.asarray(p['rotation_center_offset'],dtype='<f4').tobytes())
            for suffix,dtype in (('_ids','<u4'),('_weights','<f4'),('_offset','<f4')):
                f.write(np.asarray(bindings[name+suffix],dtype=dtype).tobytes())
    controls=dict(schema='vhuman.gnm_controls.v1',names=model.data['expression_names'].tolist(),
        reference=g['gnm_expressions'][0].tolist(),range=[-3,3],rotations='four native axis-angle joints in radians',
        speech_mapping='requires geometry-matched semantic-to-native mapping; PCA names are not visemes')
    (out/'controls.json').write_text(json.dumps(controls,indent=2))
    if (scene/'skin_detail.npz').is_file():
        shutil.copyfile(scene/'skin_detail.npz',out/'skin_detail.npz')
        shutil.copyfile(scene/'skin_detail.json',out/'skin_detail.json')
    result=dict(schema='vhuman.mobile_avatar.v1',profile=IPHONE12,source_geometry_sha256=manifest['geometry_sha256'],
        source_portrait_sha256=manifest['portrait_sha256'],native=native,parts=records,triangles=triangles,
        camera=description['camera'],material=manifest['material'],licenses=[dict(component='GNM v3',license='Apache-2.0',
        source='https://huggingface.co/google/gnm-v3')],texture_encoding='embedded PNG reference',
        validation=dict(device_measured=False,production_ready=False),
        limitations=['LOD0 reference export; reduced LODs and ASTC packaging pending',
            'GLB is a static reference; native player must apply bindings for motion',
            'corneal transparency is an approximation; mobile optical/SSS shaders require device review',
            'source image and generated-reference redistribution terms require separate receipts'])
    completion=manifest['material'].get('synthetic_completion')
    if completion:
        result['licenses'].append(dict(component='Generated skin appearance',license=completion['license'],
            generator=completion['generator'],source='generated_skin.json'))
        result['validation']['research_generated_skin']=True
        for name in ('skin_generated_support.png','generated_skin.json'):
            shutil.copyfile(candidate/name,out/name)
        # Retain evidence masks at their original resolution, independently of
        # the resized GLB texture. These are provenance assets, not shader maps.
        evidence=('ear_rebake_mask.png','local_color_edit_mask.png','skin_source_core_mask_0.png',
            'skin_strict_exclusion_0.png','skin_coverage.png','skin_confidence.png','skin_prior_transfer.png')
        result['appearance_evidence']={}
        for name in evidence:
            if (candidate/name).is_file():
                shutil.copyfile(candidate/name,out/name)
                result['appearance_evidence'][name]=sha256(candidate/name)
        refinement=completion.get('surface_refinement')
        if refinement:
            name='skin_projection_repair.png'
            if sha256(candidate/name)!=refinement['repair_mask_sha256']:
                raise ValueError('projection repair mask checksum mismatch')
            shutil.copyfile(candidate/name,out/name)
    if contact:
        (out/'oral_contact.json').write_text(json.dumps(dict(schema='vhuman.oral_contact.v1',geometry_sha256=manifest['geometry_sha256'],parts=contact)))
        result['oral_contact']=manifest['geometry'].get('oral_contact',{}).get('contact')
    if (scene/'oral_occlusion.json').is_file():
        # Optional lip-aperture mouth occlusion fitted for this geometry (native oral parts only).
        oral=json.loads((scene/'oral_occlusion.json').read_text())
        if oral.get('schema')!='vhuman.oral_occlusion.v1' or oral.get('geometry_sha256')!=manifest['geometry_sha256']:
            raise ValueError('oral occlusion belongs to another geometry')
        by_name={p['name']:p for p in description['parts']}
        for name,spec in oral['parts'].items():
            if name not in by_name or not by_name[name]['native']:raise ValueError('oral occlusion part is not native: '+name)
            if 'part_weights' in spec:
                # Part-vertex weight sets (refined parts with bound vertices) -> render-vertex order.
                for key in [k for k in list(spec) if k.startswith('part_') and k.endswith('weights')]:
                    if len(spec[key])!=len(assets[name+'_positions']):raise ValueError('oral occlusion does not cover part vertices: '+name)
                    spec['render_'+key[5:]]=np.asarray(spec.pop(key))[render_maps[name]].round(6).tolist()
            elif not set(assets[name+'_native_ids'].tolist())<=set(spec['native_ids']) or len(spec['weights'])!=len(spec['native_ids']):
                raise ValueError('oral occlusion does not cover part vertices: '+name)
        if not oral['rim'] or max(oral['rim'])>=len(g['full_captured'][0]):raise ValueError('invalid oral rim')
        (out/'oral_occlusion.json').write_text(json.dumps(oral))
        result['oral_occlusion']={k:oral[k] for k in ('model','metrics','limitations') if k in oral}
    result['files']={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'avatar.json').write_text(json.dumps(result,indent=2));validate_package(out)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',required=True)
    p.add_argument('--scene',required=True);p.add_argument('--out',required=True);p.add_argument('--profile',default='iphone12',choices=['iphone12'])
    a=p.parse_args();r=export(**vars(a));print(json.dumps(dict(out=a.out,triangles=r['triangles'],validation=r['validation']),indent=2))


if __name__=='__main__':main()
