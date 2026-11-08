"""Blender worker: physical materials and explicitly selected GPU head rendering.

Run through offline_render; this module intentionally imports only Blender and
its bundled NumPy, not the project's PyTorch environment.
"""
import json
import math
import sys
import time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector


def rotations_from_vectors(vectors):
    vectors=np.asarray(vectors)
    angle=np.sqrt(np.maximum((vectors*vectors).sum(-1),1e-8))
    axis=vectors/angle[...,None];x,y,z=np.moveaxis(axis,-1,0);zero=np.zeros_like(x)
    skew=np.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*vectors.shape[:-1],3,3)
    return np.eye(3)+np.sin(angle)[...,None,None]*skew+(1-np.cos(angle))[...,None,None]*(skew@skew)


def attachment_frames(points):
    x=points[...,1,:]-points[...,0,:];x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    z=np.cross(x,points[...,2,:]-points[...,0,:]);z/=np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
    return np.stack((x,np.cross(z,x),z),-1)


def native_part_positions(data, name, vertices):
    """Per-frame native part coordinates, plus any appended surface-bound vertices (refined ears)."""
    positions=vertices[...,data[name+'_native_ids'],:]
    if name+'_bound_ids' in data:
        ids=data[name+'_bound_ids'];weights=data[name+'_bound_weights']
        corners=vertices[...,ids,:]
        roots=(corners*weights[...,None]).sum(-2)
        bound=roots+np.einsum('...vij,vj->...vi',attachment_frames(corners),data[name+'_bound_offsets'])
        positions=np.concatenate((positions,bound.astype(positions.dtype)),axis=-2)
    if name+'_contact_active' in data:
        # Per-pose lip/arch contact deformer (numpy-only module, importable inside Blender).
        if __package__:
            from .contact_runtime import contact_deform
        else:
            sys.path.insert(0,str(Path(__file__).parent))
            from contact_runtime import contact_deform
        edges=data[name+'_contact_edges'];params=data[name+'_contact_params']
        spec=dict(active=data[name+'_contact_active'],weight=data[name+'_contact_weight'],candidates=data[name+'_contact_candidates'],
                  clearance=float(params[0]),max_move=float(params[1]),smoothing=[float(params[2]),int(params[3])],
                  neighbours=(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]]))
        frames=positions.reshape(-1,*positions.shape[-2:])
        native=vertices.reshape(-1,*vertices.shape[-2:])
        positions=np.stack([contact_deform(f,n,spec) for f,n in zip(frames,native)]).reshape(positions.shape).astype(positions.dtype)
    return positions


def animate(scene, request, config, data, objects):
    directory=Path(request['motion']);track=json.loads((directory/'motion.json').read_text())
    motion=np.load(directory/'motion.npz',allow_pickle=False)
    geometry=np.load(Path(config['candidate'])/'geometry.npz',allow_pickle=False)
    alignment=geometry['rotation'];rest_joints=data['rest_joints']
    local=rotations_from_vectors(motion['rotations']);world=local.copy()
    for joint,parent in [(1,0),(2,1),(3,1)]:world[:,joint]=world[:,parent]@local[:,joint]
    world=alignment[None,None]@world@alignment.T[None,None]
    scene.frame_start=1;scene.frame_end=track['frames'];scene.render.fps=round(track['fps'])
    detail=np.load(Path(request['out'])/'skin_detail.npz',allow_pickle=False)
    activations=np.clip((motion['expression']-detail['reference_expression'])@detail['coefficient_to_activation'].T,-1,1)
    for part in config['parts']:
        obj=objects[part['name']]
        if part['native']:
            positions=native_part_positions(data,part['name'],motion['vertices'])
        elif part.get('surface_bound'):
            name=part['name'];ids=data[name+'_surface_ids'];weights=data[name+'_surface_weights']
            roots=(motion['vertices'][:,ids]*weights[None,:,:,None]).sum(2)
            frames=attachment_frames(motion['vertices'][:,ids[:,:3]])
            positions=roots+np.einsum('fvij,vj->fvi',frames,data[name+'_surface_offsets'])
        elif part.get('joint') is not None:
            joint=part['joint'];p=data[part['name']+'_positions']-rest_joints[joint]
            positions=np.einsum('fij,vj->fvi',world[:,joint],p)+motion['joints'][:,joint,None]
            if 'rotation_center_offset' in part:
                if joint not in (2,3):
                    raise ValueError('shifted optical centers require a GNM eye joint')
                if __package__:
                    from .joint_motion import shifted_joint_positions
                else:
                    sys.path.insert(0,str(Path(__file__).parent))
                    from joint_motion import shifted_joint_positions
                positions=shifted_joint_positions(data[part['name']+'_positions'],
                    rest_joints[joint],motion['joints'][:,joint],world[:,joint],
                    part['rotation_center_offset'],world[:,1])
        else:
            positions=None
        if positions is not None:
            obj.shape_key_add(name='Basis')
            for i,vertices in enumerate(positions):
                key=obj.shape_key_add(name=f'GNM_{i+1:03d}')
                key.data.foreach_set('co',vertices.astype(np.float32).ravel())
                for frame,value in [(i,0),(i+1,1),(i+2,0)]:
                    key.value=value;key.keyframe_insert('value',frame=frame)
            action=obj.data.shape_keys.animation_data.action
            # Blender 4.5 channel bags replace the earlier action.fcurves API.
            for layer in action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        for curve in bag.fcurves:
                            for point in curve.keyframe_points:point.interpolation='LINEAR'
    native_names={p['name'] for p in config['parts'] if p['native'] or p.get('joint') is not None or p.get('surface_bound')}
    for name,obj in objects.items():
        if name in native_names:continue
        obj.rotation_mode='QUATERNION'
        for i in range(track['frames']):
            rotation=world[i,1]
            obj.rotation_quaternion=Matrix(rotation.tolist()).to_quaternion()
            obj.location=motion['joints'][i,1]-rotation@rest_joints[1]
            obj.keyframe_insert('rotation_quaternion',frame=i+1);obj.keyframe_insert('location',frame=i+1)
    for i,row in enumerate(activations):
        for j,value in enumerate(row):
            scene[f'wrinkle_{j}']=float(value);scene.keyframe_insert(f'["wrinkle_{j}"]',frame=i+1)
    scene['motion_provenance']=json.dumps(track)
    scene['hair_motion_limit']='strand roots follow rigid head pose; skin strain at hair roots is not fitted'
    scene.frame_set(request.get('frame',1))


def full_skin_graph(skin,node,bound,out,request):
    """Quality-candidate skin: region templates (roughness/specular/cavity/lips), two specular lobes (coat),
    random-walk skin subsurface, tiled micro-normal and true displacement. Maps come from skin_templates.build."""
    nodes,links=skin.node_tree.nodes,skin.node_tree.links
    maps=np.load(out/'skin_templates.npz',allow_pickle=False)
    tex={k:float_image(nodes,'template_'+k,maps[k]) for k in ('roughness','specular','cavity','lips','micro_strength')}
    def channel(name):
        sep=nodes.new('ShaderNodeSeparateColor');links.new(tex[name].outputs['Color'],sep.inputs['Color']);return sep.outputs['Red']
    rough,spec,cavity,lips,micro=[channel(k) for k in ('roughness','specular','cavity','lips','micro_strength')]
    def math(op,a,b=None,*,value=None,clamp=False):
        m=nodes.new('ShaderNodeMath');m.operation=op;m.use_clamp=clamp
        links.new(a,m.inputs[0]) if not isinstance(a,(int,float)) else m.inputs[0].__setattr__('default_value',a)
        if b is not None:
            links.new(b,m.inputs[1]) if not isinstance(b,(int,float)) else m.inputs[1].__setattr__('default_value',b)
        elif value is not None:m.inputs[1].default_value=value
        return m.outputs[0]
    # roughness = template*(1+.6 cavity), lips wetter: -0.12*lips
    r=math('MULTIPLY',rough,math('MULTIPLY_ADD',cavity,.6,value=1) if False else math('ADD',math('MULTIPLY',cavity,.6),1))
    # material_overrides (metahuman.compare_render --sweep; empty = the authored graph, no extra nodes):
    # roughness_scale, specular_scale, skin_coat (the .10 skin coat weight), subsurface_weight, subsurface_scale, subsurface_radius,
    # ear_sss_scale (ears: SSS weight 1, scale x k), dual_lobe (UE-style two specular lobes via the coat), normal_ear_scale (per-texel skin normal strength on the ears; skin_normal_map only), ear_roughness_scale (ears-mask roughness scale)
    mo=dict(request.get('material_overrides') or {})
    r=math('SUBTRACT',r,math('MULTIPLY',lips,.12),clamp=True)
    if 'ear_roughness_scale' in mo:
        # ears mask gets its own roughness scale: r x (s (1 - e) + s_ear e); s = roughness_scale (default 1)
        if 'ears' not in maps.files:raise RuntimeError('ear_roughness_scale needs skin_templates with an ears map; rebuild them')
        ear=float_image(nodes,'template_ears_rough',maps['ears']);esep=nodes.new('ShaderNodeSeparateColor');links.new(ear.outputs['Color'],esep.inputs['Color'])
        s=float(mo.get('roughness_scale',1.));se=float(mo['ear_roughness_scale'])
        r=math('MULTIPLY',r,math('ADD',math('MULTIPLY',esep.outputs['Red'],se-s),s),clamp=True)
    elif mo.get('roughness_scale',1.)!=1.:r=math('MULTIPLY',r,float(mo['roughness_scale']),clamp=True)
    if mo.get('peri_knobs'):
        # periocular sweep knobs (neutral defaults; peri_sweep_blender sets the named nodes): regional roughness scale and
        # colour tint on the orbital (lids) / infraorbital masks of skin_detail
        rm=np.load(out/'skin_detail.npz',allow_pickle=False)['region_masks']
        def region(name,mask):
            t=float_image(nodes,name,np.clip(mask,0,1).astype(np.float32));sp_=nodes.new('ShaderNodeSeparateColor')
            links.new(t.outputs['Color'],sp_.inputs['Color']);return sp_.outputs['Red']
        orb=region('peri_mask_orbital',rm[2]+rm[3]);inf=region('peri_mask_infraorbital',rm[4]+rm[5])
        def knob(name,value):
            v=nodes.new('ShaderNodeValue');v.name=v.label=name;v.outputs[0].default_value=value;return v.outputs[0]
        kl,ki=knob('peri_k_lid_rough',float(mo.get('peri_k_lid_rough',1.))),knob('peri_k_inf_rough',float(mo.get('peri_k_inf_rough',1.)))
        r=math('MULTIPLY',r,math('ADD',math('ADD',math('MULTIPLY',orb,math('SUBTRACT',kl,1.)),math('MULTIPLY',inf,math('SUBTRACT',ki,1.))),1.),clamp=True)
        base_link=next((l for l in links if l.to_socket==node.inputs['Base Color']),None)
        if base_link is not None:
            src=base_link.from_socket;links.remove(base_link)
            for tname,mask in (('peri_lid_tint',orb),('peri_inf_tint',inf)):
                tint=nodes.new('ShaderNodeRGB');tint.name=tint.label=tname
                tint.outputs[0].default_value=(*[float(x) for x in mo.get(tname,(1.,1.,1.))],1.)
                mul=nodes.new('ShaderNodeMix');mul.data_type='RGBA';mul.blend_type='MULTIPLY';mul.clamp_result=True
                links.new(mask,mul.inputs['Factor']);links.new(src,mul.inputs[6]);links.new(tint.outputs[0],mul.inputs[7]);src=mul.outputs[2]
            links.new(src,node.inputs['Base Color'])
    if mo.get('peri_contact_shading'):
        # Distance follows the skin mesh under motion; no opaque overlay or image blur.
        contact=nodes.new('ShaderNodeAttribute');contact.attribute_name='peri_contact_distance'
        ramp=nodes.new('ShaderNodeMapRange');ramp.interpolation_type='SMOOTHSTEP';ramp.clamp=True
        ramp.inputs['From Min'].default_value=0.
        width=float(mo.get('peri_contact_width_m',.0005))
        if not np.isfinite(width) or width<=0:raise ValueError('positive finite contact width required')
        ramp.inputs['From Max'].default_value=width
        ramp.inputs['To Min'].default_value=1.;ramp.inputs['To Max'].default_value=0.
        distance=contact.outputs['Fac']
        if mo.get('peri_contact_atlas'):
            atlas=np.load(out/'peri_contact_atlas.npz',allow_pickle=False)
            lower=float_image(nodes,'peri_lower_distance',atlas['lower_distance_m'])
            upper=float_image(nodes,'peri_upper_distance',atlas['upper_distance_m'])
            distance=math('MINIMUM',lower.outputs['Color'],upper.outputs['Color'])
        width_node=nodes.new('ShaderNodeValue');width_node.name=width_node.label='peri_contact_width_m'
        width_node.outputs[0].default_value=width;links.new(width_node.outputs[0],ramp.inputs['From Max'])
        links.new(distance,ramp.inputs['Value'])
        wet=ramp.outputs['Result']
        wet_node=nodes.new('ShaderNodeValue');wet_node.name=wet_node.label='peri_contact_wetness'
        wet_node.outputs[0].default_value=float(mo.get('peri_contact_wetness',.35))
        r=math('MULTIPLY',r,math('SUBTRACT',1.,math('MULTIPLY',wet,wet_node.outputs[0])),clamp=True)
    r_dual=r
    if mo.get('dual_lobe'):r=math('MULTIPLY',r,.8,clamp=True)   # UE SSP dual specular: lobes .8r / 1.1r, mix .5
    links.new(r,node.inputs['Roughness'])
    # specular level = template*(1-.7 cavity)
    sp=math('MULTIPLY',spec,math('SUBTRACT',1.,math('MULTIPLY',cavity,.7)),clamp=True)
    if mo.get('specular_scale',1.)!=1.:sp=math('MULTIPLY',sp,float(mo['specular_scale']),clamp=True)
    if mo.get('dual_lobe'):sp=math('MULTIPLY',sp,.5)   # half of F0 in the base lobe, the other half in the coat lobe
    links.new(sp,node.inputs['Specular IOR Level'])
    # second (sharper) lobe kept small so skin reads oily, not wet: coat .10(1-cavity)+.35 lips,
    # roughness .30 -> .12 on lips (the moist lip band)
    coat=math('ADD',math('MULTIPLY',math('SUBTRACT',1.,cavity),float(mo.get('skin_coat',.10))),math('MULTIPLY',lips,.35*float(request.get('lip_coat_scale',1.))),clamp=True);links.new(coat,node.inputs['Coat Weight'])
    cr=math('SUBTRACT',.30,math('MULTIPLY',lips,.30-float(request.get('lip_coat_roughness',.12))),clamp=True);links.new(cr,node.inputs['Coat Roughness'])
    node.inputs['Coat IOR'].default_value=1.4
    if mo.get('dual_lobe'):
        # second lobe = the coat: weight (1-.7 cavity) (+ lips), roughness 1.1r (lips keep their wet coat roughness),
        # IOR 1.27 -> F0 .014 = half of the skin's .028
        links.new(math('ADD',math('SUBTRACT',1.,math('MULTIPLY',cavity,.7)),math('MULTIPLY',lips,.35*float(request.get('lip_coat_scale',1.))),clamp=True),node.inputs['Coat Weight'])
        r11=math('MULTIPLY',r_dual,1.1,clamp=True)
        links.new(math('ADD',math('MULTIPLY',r11,math('SUBTRACT',1.,lips)),math('MULTIPLY',lips,float(request.get('lip_coat_roughness',.12))),clamp=True),node.inputs['Coat Roughness'])
        node.inputs['Coat IOR'].default_value=1.27
    if mo.get('peri_waterline_coat'):
        if not mo.get('peri_contact_shading'):raise ValueError('waterline coat needs contact shading')
        def wet_knob(name, default):
            value=nodes.new('ShaderNodeValue');value.name=value.label=name
            value.outputs[0].default_value=float(mo.get(name,default));return value.outputs[0]
        wet_coat=wet_knob('peri_waterline_coat',.7)
        wet_rough=wet_knob('peri_waterline_roughness',.045)
        coat_source=node.inputs['Coat Weight'].links[0].from_socket
        rough_source=node.inputs['Coat Roughness'].links[0].from_socket
        coat_ior=float(node.inputs['Coat IOR'].default_value)
        links.new(math('MAXIMUM',coat_source,math('MULTIPLY',wet,wet_coat)),node.inputs['Coat Weight'])
        links.new(math('ADD',math('MULTIPLY',rough_source,math('SUBTRACT',1.,wet)),math('MULTIPLY',wet,wet_rough)),node.inputs['Coat Roughness'])
        links.new(math('ADD',coat_ior,math('MULTIPLY',wet,1.336-coat_ior)),node.inputs['Coat IOR'])
    # random-walk skin subsurface (radii in metres, literature-scale skin profile)
    node.subsurface_method='RANDOM_WALK_SKIN'
    node.inputs['Subsurface Weight'].default_value=float(mo.get('subsurface_weight',.6));node.inputs['Subsurface Scale'].default_value=float(mo.get('subsurface_scale',.5))
    if mo.get('ear_sss_scale'):
        # thin-part transmission test: on the ears mask, SSS weight -> 1 and scale x k (UE adds a transmission profile)
        if 'ears' not in maps.files:raise RuntimeError('ear_sss_scale needs skin_templates with an ears map; rebuild them')
        ear=float_image(nodes,'template_ears_sss',maps['ears']);esep=nodes.new('ShaderNodeSeparateColor');links.new(ear.outputs['Color'],esep.inputs['Color'])
        e=esep.outputs['Red'];w0=float(mo.get('subsurface_weight',.6));s0=float(mo.get('subsurface_scale',.5));k=float(mo['ear_sss_scale'])
        links.new(math('ADD',math('MULTIPLY',e,1.-w0),w0,clamp=True),node.inputs['Subsurface Weight'])
        links.new(math('ADD',math('MULTIPLY',e,s0*(k-1.)),s0),node.inputs['Subsurface Scale'])
    if mo.get('peri_lid_sss'):
        # periocular study: thin eyelid tissue (orbital region masks) scatters further and fully (UE eyelids read pink-
        # translucent); weight -> 1, scale x k on the orbital masks
        rm=np.load(out/'skin_detail.npz',allow_pickle=False)['region_masks']
        lid=float_image(nodes,'peri_lid_sss',np.clip(rm[2]+rm[3],0,1).astype(np.float32));lsep=nodes.new('ShaderNodeSeparateColor')
        links.new(lid.outputs['Color'],lsep.inputs['Color']);l=lsep.outputs['Red']
        w0=float(mo.get('subsurface_weight',.6));s0=float(mo.get('subsurface_scale',.5));k=float(mo['peri_lid_sss'])
        links.new(math('ADD',math('MULTIPLY',l,1.-w0),w0,clamp=True),node.inputs['Subsurface Weight'])
        links.new(math('ADD',math('MULTIPLY',l,s0*(k-1.)),s0),node.inputs['Subsurface Scale'])
    if mo.get('peri_contact_shading'):
        # Compose with existing regional SSS links rather than replacing ear treatment.
        for socket_name, target in (('Subsurface Weight', float(mo.get('peri_contact_sss_weight',.8))),
                                    ('Subsurface Scale', float(mo.get('peri_contact_sss_scale',1.)))):
            socket=node.inputs[socket_name]
            old=next((link for link in links if link.to_socket==socket),None)
            source=old.from_socket if old else float(socket.default_value)
            if old:links.remove(old)
            target_node=nodes.new('ShaderNodeValue')
            target_node.name=target_node.label='peri_contact_sss_'+('weight' if socket_name=='Subsurface Weight' else 'scale')
            target_node.outputs[0].default_value=target
            mixed=math('ADD',math('MULTIPLY',source,math('SUBTRACT',1.,wet)),math('MULTIPLY',wet,target_node.outputs[0]))
            links.new(mixed,socket)
    if mo:skin['material_overrides']=json.dumps(mo)
    # subsurface_radius override: per-channel mean free path in mm (e.g. UE SSP_skin_unified 1.05/.41/.26)
    node.inputs['Subsurface Radius'].default_value=tuple(float(x)*1e-3 for x in mo['subsurface_radius']) if 'subsurface_radius' in mo else (.00367,.00137,.00068)
    node.inputs['Subsurface IOR'].default_value=1.4
    # tiled micro normal (pores), strength scaled by the region mask
    micro_img=image_node(nodes,out/('skin_micro_isotropic.png' if mo.get('skin_micro_isotropic') else 'skin_micro_normal.png'),linear=True)
    uvn=nodes.new('ShaderNodeUVMap');uvn.uv_map='UVMap';mapping=nodes.new('ShaderNodeMapping');mapping.inputs['Scale'].default_value=(12,12,1)
    links.new(uvn.outputs['UV'],mapping.inputs['Vector']);links.new(mapping.outputs['Vector'],micro_img.inputs['Vector'])
    preserve_lids=bool(mo.get('skin_normal_preserve_lids'))
    if preserve_lids:
        legacy_micro=image_node(nodes,out/'skin_micro_normal.png',linear=True)
        links.new(mapping.outputs['Vector'],legacy_micro.inputs['Vector'])
        if mo.get('peri_knobs'):
            protected_normal=math('ADD',orb,inf,clamp=True)
        else:
            rm=np.load(out/'skin_detail.npz',allow_pickle=False)['region_masks']
            protected_tex=float_image(nodes,'skin_normal_protected_regions',np.clip(rm[2:6].sum(0),0,1))
            protected_split=nodes.new('ShaderNodeSeparateColor')
            links.new(protected_tex.outputs['Color'],protected_split.inputs['Color'])
            protected_normal=protected_split.outputs['Red']
    nm=nodes.new('ShaderNodeNormalMap');nm.space='TANGENT';nm.uv_map='UVMap'
    skin_normal=out/'skin_normal_map.png'
    if request.get('skin_normal_map') and skin_normal.is_file():
        # per-texel skin normal (e.g. metahuman.normal_map: converted MetaHuman detail, Blender/OpenGL tangent convention)
        # combined with the tiled pores by a whiteout blend in tangent space: n = norm(a.xy + k b.xy, a.z b.z), with the
        # pore strength k = micro mask x .6 x micro_scale and the map strength s (xy scale) from skin_normal_strength
        def vec(op,a,b=None,*,scale=None):
            v=nodes.new('ShaderNodeVectorMath');v.operation=op;links.new(a,v.inputs[0])
            if b is not None:
                links.new(b,v.inputs[1]) if not isinstance(b,tuple) else v.inputs[1].__setattr__('default_value',b)
            if scale is not None:links.new(scale,v.inputs['Scale']) if not isinstance(scale,(int,float)) else v.inputs['Scale'].__setattr__('default_value',scale)
            return v.outputs[0] if op!='DOT_PRODUCT' else v.outputs['Value']
        detail_img=image_node(nodes,skin_normal,linear=True)
        a=vec('MULTIPLY_ADD',detail_img.outputs['Color'],(2,2,2));a.node.inputs[2].default_value=(-1,-1,-1)
        b=vec('MULTIPLY_ADD',micro_img.outputs['Color'],(2,2,2));b.node.inputs[2].default_value=(-1,-1,-1)
        sa,sb=nodes.new('ShaderNodeSeparateXYZ'),nodes.new('ShaderNodeSeparateXYZ');links.new(a,sa.inputs[0]);links.new(b,sb.inputs[0])
        k=math('MULTIPLY',micro,.6*float(request.get('micro_scale',1.)))
        legacy_k=k
        if float(mo.get('skin_pore_scale',1.))!=1.:
            k=math('MULTIPLY',k,float(mo['skin_pore_scale']))
        strength=float(request.get('skin_normal_strength',1.))
        st=strength
        if mo.get('peri_knobs'):
            st=math('MULTIPLY',st,knob('peri_k_normal',float(mo.get('peri_k_normal',1.))))
            regional=math('ADD',1.,math('ADD',math('MULTIPLY',orb,math('SUBTRACT',knob('peri_k_lid_normal',float(mo.get('peri_k_lid_normal',1.))),1.)),
                math('MULTIPLY',inf,math('SUBTRACT',knob('peri_k_inf_normal',float(mo.get('peri_k_inf_normal',1.))),1.))))
            st=math('MULTIPLY',st,regional)
            micro_knob=knob('peri_k_micro',float(mo.get('peri_k_micro',1.)))
            k=math('MULTIPLY',k,micro_knob)
            if preserve_lids:legacy_k=math('MULTIPLY',legacy_k,micro_knob)
        if mo.get('normal_ear_scale',1.)!=1.:
            # fade the per-texel map on the ears: s * (1 - (1 - ear_scale) * ear_weight), ear weight from skin_templates
            if 'ears' not in maps.files:raise RuntimeError('normal_ear_scale needs skin_templates with an ears map; rebuild them')
            ear=float_image(nodes,'template_ears',maps['ears']);esep=nodes.new('ShaderNodeSeparateColor');links.new(ear.outputs['Color'],esep.inputs['Color'])
            st=math('MULTIPLY',math('SUBTRACT',1.,math('MULTIPLY',esep.outputs['Red'],1.-float(mo['normal_ear_scale']))),st)
        if preserve_lids:
            legacy_b=vec('MULTIPLY_ADD',legacy_micro.outputs['Color'],(2,2,2));legacy_b.node.inputs[2].default_value=(-1,-1,-1)
            legacy_split=nodes.new('ShaderNodeSeparateXYZ');links.new(legacy_b,legacy_split.inputs[0])
            legacy_combine=nodes.new('ShaderNodeCombineXYZ')
            for axis in ('X','Y'):
                links.new(math('ADD',math('MULTIPLY',sa.outputs[axis],st),math('MULTIPLY',legacy_split.outputs[axis],legacy_k)),legacy_combine.inputs[axis])
            links.new(math('MULTIPLY',sa.outputs['Z'],legacy_split.outputs['Z']),legacy_combine.inputs['Z'])
            legacy_tangent=vec('NORMALIZE',legacy_combine.outputs[0])
        if mo.get('skin_normal_weighted_blend'):
            # Scale each layer's slopes, then normalize BEFORE whiteout. At
            # zero strength the layer is exactly (0,0,1), including its Z.
            def weighted_channels(channels, strength):
                weighted=nodes.new('ShaderNodeCombineXYZ')
                for axis in ('X','Y'):
                    links.new(math('MULTIPLY',channels.outputs[axis],strength),weighted.inputs[axis])
                links.new(channels.outputs['Z'],weighted.inputs['Z'])
                split=nodes.new('ShaderNodeSeparateXYZ')
                links.new(vec('NORMALIZE',weighted.outputs[0]),split.inputs[0])
                return split
            sa=weighted_channels(sa,st);sb=weighted_channels(sb,k)
            st=k=1.
        c=nodes.new('ShaderNodeCombineXYZ')
        links.new(math('ADD',math('MULTIPLY',sa.outputs['X'],st),math('MULTIPLY',sb.outputs['X'],k)),c.inputs['X'])
        links.new(math('ADD',math('MULTIPLY',sa.outputs['Y'],st),math('MULTIPLY',sb.outputs['Y'],k)),c.inputs['Y'])
        links.new(math('MULTIPLY',sa.outputs['Z'],sb.outputs['Z']),c.inputs['Z'])
        n=vec('NORMALIZE',c.outputs[0])
        if preserve_lids:
            n=vec('NORMALIZE',vec('ADD',vec('SCALE',legacy_tangent,scale=protected_normal),
                vec('SCALE',n,scale=math('SUBTRACT',1.,protected_normal))))
        col=vec('MULTIPLY_ADD',n,(.5,.5,.5));col.node.inputs[2].default_value=(.5,.5,.5)
        links.new(col,nm.inputs['Color']);nm.inputs['Strength'].default_value=1.
        skin['skin_normal_map']=json.dumps(dict(path=str(skin_normal),strength=strength,
            blend='regional whiteout; legacy orbital/infraorbital calibration' if preserve_lids else
                'normalized weighted whiteout with tiled pores' if mo.get('skin_normal_weighted_blend') else 'whiteout with tiled pores'))
    else:
        source=micro_img.outputs['Color']
        if preserve_lids:
            mix=nodes.new('ShaderNodeMix');mix.data_type='RGBA'
            links.new(protected_normal,mix.inputs['Factor']);links.new(source,mix.inputs[6]);links.new(legacy_micro.outputs['Color'],mix.inputs[7]);source=mix.outputs[2]
        links.new(source,nm.inputs['Color']);links.new(math('MULTIPLY',micro,.6*float(request.get('micro_scale',1.))),nm.inputs['Strength'])
    links.new(nm.outputs['Normal'],node.inputs['Normal'])
    # dynamic facial colour (hemoglobin prior): compressed regions flush, stretched regions blanch.
    # hb = sum_r gain_r * mask_r * wrinkle_r  (wrinkle_r = regional compression in -1..1 driven by motion)
    detail=np.load(out/'skin_detail.npz',allow_pickle=False);masks=detail['region_masks']
    gains={'left_cheek_region':.6,'right_cheek_region':.6,'nose_region':.4,'forehead_region':.3,'chin_region':.3,
           'left_infraorbital_region':.3,'right_infraorbital_region':.3,'middle_brow_region':.3}
    scene=bpy.context.scene;hb=None
    for i,name in enumerate(('forehead_region','middle_brow_region','left_orbital_region','right_orbital_region',
        'left_infraorbital_region','right_infraorbital_region','nose_region','left_cheek_region','right_cheek_region',
        'upper_lip_region','lower_lip_region','chin_region')):
        gain=gains.get(name,.15)
        tex_mask=float_image(nodes,f'hemoglobin_mask_{i}',(masks[i]*gain).astype(np.float32))
        sep=nodes.new('ShaderNodeSeparateColor');links.new(tex_mask.outputs['Color'],sep.inputs['Color'])
        drive=nodes.new('ShaderNodeMath');drive.operation='MULTIPLY';links.new(sep.outputs['Red'],drive.inputs[0])
        driver=drive.inputs[1].driver_add('default_value').driver;driver.expression='activation'
        variable=driver.variables.new();variable.name='activation';variable.targets[0].id_type='SCENE'
        variable.targets[0].id=scene;variable.targets[0].data_path=f'["wrinkle_{i}"]'
        hb=drive.outputs[0] if hb is None else math('ADD',hb,drive.outputs[0])
    # colour factor = 1 + k*hb*(-.10,.14,.12) with k=.5 (authored); applied multiplicatively after the appearance deltas
    scale=nodes.new('ShaderNodeVectorMath');scale.operation='SCALE';scale.inputs[0].default_value=(-.10,.14,.12)
    links.new(math('MULTIPLY',hb,.5),scale.inputs['Scale'])
    one=nodes.new('ShaderNodeVectorMath');one.operation='ADD';one.inputs[1].default_value=(1,1,1);links.new(scale.outputs[0],one.inputs[0])
    current=node.inputs['Base Color'].links[0].from_socket if node.inputs['Base Color'].links else None
    tint=nodes.new('ShaderNodeVectorMath');tint.operation='MULTIPLY'
    if current is not None:links.new(current,tint.inputs[0])
    links.new(one.outputs[0],tint.inputs[1])
    clampv=nodes.new('ShaderNodeVectorMath');clampv.operation='MINIMUM';clampv.inputs[1].default_value=(1,1,1)
    links.new(tint.outputs[0],clampv.inputs[0]);links.new(clampv.outputs[0],node.inputs['Base Color'])
    # true displacement with a wider clamp for the wrinkle library
    bound.inputs['Min'].default_value=-.0015;bound.inputs['Max'].default_value=.0015
    for n in nodes:   # displacement_scale: height-map amplitude of the true displacement (1 = authored)
        if n.type=='DISPLACEMENT':n.inputs['Scale'].default_value=float(request.get('displacement_scale',1.))
    skin.displacement_method='DISPLACEMENT'
    skin['quality_shader']=json.dumps(dict(templates=str(out/'skin_templates.json'),subsurface='random_walk_skin',
        coat='second specular lobe',micro_normal='tiled procedural pores',displacement='true, adaptive'))


def full_mouth_materials(materials,out,request):
    """Quality-candidate teeth (two-layer enamel/dentin approximation on a VITA shade), gums (wet coat, cervical
    darkening) and tongue (papillae bump, tip-to-root colour, wet coat, red subsurface). Colours are priors
    unless out/teeth.json carries a measured shade."""
    shade=dict(shade='A2',body_rgb=(.60,.486,.329),cervical_rgb=(.559,.419,.252),incisal_rgb=(.593,.526,.411),source='prior only')
    teeth_file=out/'teeth.json'
    if teeth_file.is_file():shade=json.loads(teeth_file.read_text()).get('colors',shade)
    def fresh(name):
        mat=materials[name];nodes=mat.node_tree.nodes;links=mat.node_tree.links
        node=nodes.get('Principled BSDF');return mat,nodes,links,node
    # teeth
    mat,nodes,links,node=fresh('teeth')
    tip=nodes.new('ShaderNodeAttribute');tip.attribute_name='enamel_tip'
    ramp=nodes.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.15;ramp.color_ramp.elements[0].color=(*shade['cervical_rgb'],1)
    ramp.color_ramp.elements[1].position=.85;ramp.color_ramp.elements[1].color=(*shade['incisal_rgb'],1)
    mid=ramp.color_ramp.elements.new(.5);mid.color=(*shade['body_rgb'],1)
    links.new(tip.outputs['Fac'],ramp.inputs['Fac']);links.new(ramp.outputs['Color'],node.inputs['Base Color'])
    node.inputs['Roughness'].default_value=.18;node.inputs['IOR'].default_value=1.58
    node.inputs['Coat Weight'].default_value=1.;node.inputs['Coat Roughness'].default_value=.08;node.inputs['Coat IOR'].default_value=1.5
    node.subsurface_method='RANDOM_WALK';node.inputs['Subsurface Weight'].default_value=.35;node.inputs['Subsurface Scale'].default_value=1.
    node.inputs['Subsurface Radius'].default_value=(.0010,.0007,.0005)
    glow=float(request.get('teeth_glow',0.))
    if glow>0:   # stand-in for mouth bounce light the path tracer cannot gather behind the lips: faint shade-coloured emission
        em=nodes.new('ShaderNodeEmission');em.inputs['Color'].default_value=(*shade['body_rgb'],1);em.inputs['Strength'].default_value=glow
        mixer=nodes.new('ShaderNodeAddShader');out_node=nodes['Material Output']
        links.new(node.outputs['BSDF'],mixer.inputs[0]);links.new(em.outputs['Emission'],mixer.inputs[1]);links.new(mixer.outputs['Shader'],out_node.inputs['Surface'])
    # translucent incisal edge: transmission rises with enamel_tip^2 (dentin core blocks it elsewhere)
    sq=nodes.new('ShaderNodeMath');sq.operation='POWER';sq.inputs[1].default_value=2.;links.new(tip.outputs['Fac'],sq.inputs[0])
    tr=nodes.new('ShaderNodeMath');tr.operation='MULTIPLY';tr.inputs[1].default_value=.18;links.new(sq.outputs[0],tr.inputs[0])
    links.new(tr.outputs[0],node.inputs['Transmission Weight'])
    mat['quality_shader']=json.dumps(dict(shade=shade['shade'],source=shade.get('source'),model='enamel/dentin two-layer approximation; authored'))
    # gums
    mat,nodes,links,node=fresh('gums')
    node.inputs['Base Color'].default_value=(.55,.22,.20,1);node.inputs['Roughness'].default_value=.35
    node.inputs['Coat Weight'].default_value=.5;node.inputs['Coat Roughness'].default_value=.08
    node.subsurface_method='RANDOM_WALK';node.inputs['Subsurface Weight'].default_value=.3;node.inputs['Subsurface Radius'].default_value=(.0012,.0004,.0003)
    ao=nodes.new('ShaderNodeAmbientOcclusion');ao.inputs['Distance'].default_value=.004;ao.samples=8
    dark=nodes.new('ShaderNodeMixRGB');dark.blend_type='MULTIPLY';dark.inputs['Fac'].default_value=.6
    dark.inputs['Color1'].default_value=(.55,.22,.20,1);links.new(ao.outputs['Color'],dark.inputs['Color2'])
    links.new(dark.outputs['Color'],node.inputs['Base Color'])   # cervical line: occlusion against the crowns
    # tongue
    mat,nodes,links,node=fresh('tongue')
    axis=nodes.new('ShaderNodeAttribute');axis.attribute_name='tongue_axis'
    ramp=nodes.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.0;ramp.color_ramp.elements[0].color=(.52,.12,.10,1)
    ramp.color_ramp.elements[1].position=1.;ramp.color_ramp.elements[1].color=(.50,.30,.26,1)   # pale coating toward the root
    links.new(axis.outputs['Fac'],ramp.inputs['Fac']);links.new(ramp.outputs['Color'],node.inputs['Base Color'])
    node.inputs['Roughness'].default_value=.45;node.inputs['Coat Weight'].default_value=.4;node.inputs['Coat Roughness'].default_value=.12
    node.subsurface_method='RANDOM_WALK';node.inputs['Subsurface Weight'].default_value=.5;node.inputs['Subsurface Radius'].default_value=(.0012,.0003,.0002)
    noise=nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=600.;noise.inputs['Detail'].default_value=3.
    bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.35;bump.inputs['Distance'].default_value=.0002
    links.new(noise.outputs['Fac'],bump.inputs['Height']);links.new(bump.outputs['Normal'],node.inputs['Normal'])   # filiform papillae
    mat['quality_shader']='authored papillae/coating priors'


def image_node(nodes,path, *, linear=False):
    node=nodes.new('ShaderNodeTexImage')
    node.image=bpy.data.images.load(str(path),check_existing=True)
    if linear:node.image.colorspace_settings.name='Non-Color'
    node.image.pack()
    return node


def float_image(nodes,name,values):
    h,w=values.shape[:2]
    image=bpy.data.images.new(name,width=w,height=h,float_buffer=True)
    rgba=np.ones((h,w,4),np.float32)
    rgba[:,:,:3]=values[::-1,:,None] if values.ndim==2 else values[::-1]
    image.colorspace_settings.name='Non-Color'
    # Generated float images are not serialized reliably by Image.pack().
    # Save raw signed EXR, reload as a file image and pack the actual bytes.
    path=Path(bpy.context.preferences.filepaths.temporary_directory)/(name+'.exr')
    # Image.save() can retain the generated buffer's PNG encoder despite an
    # EXR file_format property. Explicit render settings select the EXR encoder;
    # EXR output stays scene-linear and preserves signed data (no display LUT).
    image.pixels.foreach_set(rgba.ravel());image.update()
    settings=bpy.context.scene.render.image_settings
    previous=(settings.file_format,settings.color_depth,settings.color_mode)
    settings.file_format='OPEN_EXR';settings.color_depth='32';settings.color_mode='RGBA'
    image.save_render(str(path),scene=bpy.context.scene)
    settings.file_format,settings.color_depth,settings.color_mode=previous
    bpy.data.images.remove(image)
    image=bpy.data.images.load(str(path),check_existing=True);image.name=name
    image.colorspace_settings.name='Non-Color';image.file_format='OPEN_EXR';image.pack()
    node=nodes.new('ShaderNodeTexImage');node.image=image
    return node


def material(name,color,roughness=.4,transmission=0,ior=1.4):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    node=mat.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value=(*color,1)
    node.inputs['Roughness'].default_value=roughness
    node.inputs['IOR'].default_value=ior
    node.inputs['Transmission Weight'].default_value=transmission
    return mat,node


def eye2_layered(scene,objects,data,out,eye_mo):
    """Fresh physically layered eye (server/vhuman/eye2, opt-in eye_model_layered2): SSS sclera shell with a wet coat,
    closed refractive cornea/aqueous volume, displaced iris inside it and a dark pupil cup. Assets eye2_<side>.* are
    baked into the staged base by eye2.bake. Placement: sphere fit of the template sclera, axis through its cornea apex."""
    made=[]
    for side in ('left','right'):
        info_path=out/f'eye2_{side}.json'
        if not info_path.is_file():raise RuntimeError(f'eye_model_layered2 requires staged {info_path.name}')
        info=json.loads(info_path.read_text());mesh_data=np.load(out/f'eye2_{side}.npz')
        sclera_positions=np.asarray(data[f'{side}_sclera_positions'],np.float64)
        design=np.column_stack((2.*sclera_positions,np.ones(len(sclera_positions))))
        sol=np.linalg.lstsq(design,(sclera_positions*sclera_positions).sum(axis=1),rcond=None)[0]
        centre=sol[:3];radius=math.sqrt(sol[3]+centre@centre)
        z=np.asarray(data[f'{side}_cornea_positions'][0],np.float64)-centre;z/=np.linalg.norm(z)
        up=np.array([0.,1.,0.]);x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x)
        k=radius/float(info['dims']['R']);rot=np.column_stack((x,y,z))
        offset=eye_mo.get(f'eye_offset_{side}_m')
        if offset is not None:centre=centre+np.asarray(offset,np.float64)
        centre=centre-z*float(eye_mo.get('peri_eye_back',0.))   # periocular study: set the eyeball deeper into the socket
        for tear_name in (f'{side}_tearline',f'{side}_tearline_upper'):
            # tear lines sit at the lid-eye contact: move them back with the eye (else they float over the gap and refract dark)
            if objects.get(tear_name) is not None and float(eye_mo.get('peri_eye_back',0.)):
                objects[tear_name].location=tuple(np.asarray(objects[tear_name].location)-z*float(eye_mo['peri_eye_back'])*float(eye_mo.get('peri_tear_follow',.7)))
        for name in (f'{side}_cornea',f'{side}_sclera',f'{side}_iris',f'{side}_pupil_cup'):
            if objects.get(name) is not None:objects[name].hide_render=True
        pre=f'eye2_{side}'
        def tex(nt,suffix,linear=False):
            return image_node(nt.nodes,out/f'{pre}_{suffix}.png',linear=linear)
        # sclera: SSS base under a clear wet coat whose normal follows the tear-film/conjunctiva relief
        sclera,sn=material(f'{pre}_sclera',(1,1,1),float(eye_mo.get('eye2_sclera_roughness',.35)),0,1.376);nt=sclera.node_tree
        albedo=tex(nt,'sclera_albedo')
        # UE-style painted limbus: iris outer colour feathers into the sclera texture (eye2_blend, per vertex)
        blend=nt.nodes.new('ShaderNodeAttribute');blend.attribute_type='GEOMETRY';blend.attribute_name='eye2_blend'
        limbus_mix=nt.nodes.new('ShaderNodeMix');limbus_mix.data_type='RGBA'
        iris_rgb=[float(c) for c in info['measured'].get('rim',info['measured']['outer'])]
        limbus_mix.inputs[6].default_value=(*iris_rgb,1.)
        if float(eye_mo.get('eye2_limbus_paint',0.)) >= .5:nt.links.new(blend.outputs['Fac'],limbus_mix.inputs['Factor'])
        else:limbus_mix.inputs['Factor'].default_value=1.   # veil: sclera colour fading in over the iris edge (alpha only)
        nt.links.new(albedo.outputs['Color'],limbus_mix.inputs[7])
        color_out=limbus_mix.outputs[2]
        if float(eye_mo.get('peri_eye_ao',0.)) > 0.:
            # non-physical, labelled: UE darkens/blurs the eye under the lids with a translucent occlusion shell; here
            # the AO of the surrounding lids (2 mm) darkens the sclera/veil base colour
            ao=nt.nodes.new('ShaderNodeAmbientOcclusion');ao.samples=8;ao.inputs['Distance'].default_value=float(eye_mo.get('peri_eye_ao_dist',.002))*k
            aof=nt.nodes.new('ShaderNodeMapRange');aof.clamp=True
            if eye_mo.get('peri_contact_ao_smooth',eye_mo.get('peri_contact_shading') and not eye_mo.get('peri_contact_atlas')):aof.interpolation_type='SMOOTHSTEP'
            aof.name='peri_eye_ao';aof.inputs['To Min'].default_value=1.-float(eye_mo['peri_eye_ao']);aof.inputs['To Max'].default_value=1.
            nt.links.new(ao.outputs['AO'],aof.inputs['Value'])
            dark=nt.nodes.new('ShaderNodeVectorMath');dark.operation='SCALE'
            nt.links.new(color_out,dark.inputs[0]);nt.links.new(aof.outputs['Result'],dark.inputs['Scale']);color_out=dark.outputs[0]
        nt.links.new(color_out,sn.inputs['Base Color'])
        sn.subsurface_method='RANDOM_WALK';sn.inputs['Subsurface Weight'].default_value=float(eye_mo.get('eye2_sclera_sss',0.))
        sn.inputs['Subsurface Radius'].default_value=(1.,.45,.3);sn.inputs['Subsurface Scale'].default_value=.001*k*float(eye_mo.get('eye2_sss_scale',1.))
        sn.inputs['Coat Weight'].default_value=float(eye_mo.get('eye2_coat',1.))
        sn.inputs['Coat Roughness'].default_value=float(eye_mo.get('eye2_coat_roughness',.035));sn.inputs['Coat IOR'].default_value=1.376
        height=tex(nt,'sclera_height',True);bump=nt.nodes.new('ShaderNodeBump')
        bump.inputs['Distance'].default_value=float(info['sclera_bump_m'])*k*float(eye_mo.get('eye2_bump',1.))
        nt.links.new(height.outputs['Color'],bump.inputs['Height'])
        nt.links.new(bump.outputs['Normal'],sn.inputs['Coat Normal'])
        base_bump=nt.nodes.new('ShaderNodeBump');base_bump.inputs['Distance'].default_value=bump.inputs['Distance'].default_value*.5
        nt.links.new(height.outputs['Color'],base_bump.inputs['Height']);nt.links.new(base_bump.outputs['Normal'],sn.inputs['Normal'])
        # no sclera relief over the iris-side veil (bumped coat glints over the iris read as sparkles)
        bump_fade=nt.nodes.new('ShaderNodeMath');bump_fade.operation='POWER';bump_fade.inputs[1].default_value=3.
        nt.links.new(blend.outputs['Fac'],bump_fade.inputs[0])
        nt.links.new(bump_fade.outputs[0],bump.inputs['Strength']);nt.links.new(bump_fade.outputs[0],base_bump.inputs['Strength'])
        # one smooth outer shell: clear cornea (glass) -> sclera across a soft limbus (per-vertex eye2_alpha)
        alpha=nt.nodes.new('ShaderNodeAttribute');alpha.attribute_type='GEOMETRY';alpha.attribute_name='eye2_alpha'
        glass=nt.nodes.new('ShaderNodeBsdfPrincipled');glass.inputs['Transmission Weight'].default_value=1.
        glass.inputs['Roughness'].default_value=float(eye_mo.get('eye2_cornea_roughness',.0));glass.inputs['IOR'].default_value=float(info.get('ior',1.376))
        surf=nt.nodes.new('ShaderNodeMixShader')
        opacity=alpha if float(eye_mo.get('eye2_limbus_paint',0.)) >= .5 else blend
        nt.links.new(opacity.outputs['Fac'],surf.inputs[0]);nt.links.new(glass.outputs[0],surf.inputs[1]);nt.links.new(sn.outputs[0],surf.inputs[2])
        # inner wall over the sclera = pigmented choroid/uvea (dark), so no light leaks through the limbal wedge
        geom=nt.nodes.new('ShaderNodeNewGeometry');uvea=nt.nodes.new('ShaderNodeBsdfDiffuse')
        uvea.inputs['Color'].default_value=(.02,.012,.01,1.)
        inner_fac=nt.nodes.new('ShaderNodeMath');inner_fac.operation='MULTIPLY'
        nt.links.new(geom.outputs['Backfacing'],inner_fac.inputs[0]);nt.links.new(opacity.outputs['Fac'],inner_fac.inputs[1])
        inner=nt.nodes.new('ShaderNodeMixShader')
        nt.links.new(inner_fac.outputs[0],inner.inputs[0]);nt.links.new(surf.outputs[0],inner.inputs[1]);nt.links.new(uvea.outputs[0],inner.inputs[2])
        out_shader=inner.outputs[0]
        if float(eye_mo.get('eye2_cornea_shadow',0.)) < .5:
            # UE EyeBxDF lights the iris without corneal attenuation: shadow rays pass the clear cornea
            # (camera/indirect rays still refract). eye2_cornea_shadow 1 = physical shadowing (+ MNEE caustics).
            lp=nt.nodes.new('ShaderNodeLightPath');inv=nt.nodes.new('ShaderNodeMath');inv.operation='SUBTRACT';inv.inputs[0].default_value=1.
            nt.links.new(opacity.outputs['Fac'],inv.inputs[1])
            sf=nt.nodes.new('ShaderNodeMath');sf.operation='MULTIPLY';nt.links.new(lp.outputs['Is Shadow Ray'],sf.inputs[0]);nt.links.new(inv.outputs[0],sf.inputs[1])
            clear=nt.nodes.new('ShaderNodeBsdfTransparent');sh=nt.nodes.new('ShaderNodeMixShader')
            nt.links.new(sf.outputs[0],sh.inputs[0]);nt.links.new(out_shader,sh.inputs[1]);nt.links.new(clear.outputs[0],sh.inputs[2]);out_shader=sh.outputs[0]
        nt.links.new(out_shader,nt.nodes.get('Material Output').inputs['Surface'])
        # posterior aqueous closure (cylinder + disc behind the iris): same dielectric
        cornea,cn=material(f'{pre}_chamber',(1,1,1),0.,1,float(info.get('ior',1.376)))
        # iris: diffuse stroma with fibre relief (mesh displacement + bump), weak scattering
        iris,iris_node=material(f'{pre}_iris',(1,1,1),.55,0,1.4);nt=iris.node_tree
        ia=tex(nt,'iris_albedo');nt.links.new(ia.outputs['Color'],iris_node.inputs['Base Color'])
        iris_node.inputs['Specular IOR Level'].default_value=.15
        ih=tex(nt,'iris_height',True);ib=nt.nodes.new('ShaderNodeBump')
        ib.inputs['Distance'].default_value=float(info['iris_height_amp_m'])*k*float(eye_mo.get('eye2_iris_bump',.5))
        nt.links.new(ih.outputs['Color'],ib.inputs['Height']);nt.links.new(ib.outputs['Normal'],iris_node.inputs['Normal'])
        iris_node.subsurface_method='RANDOM_WALK';iris_node.inputs['Subsurface Weight'].default_value=float(eye_mo.get('eye2_iris_sss',0.))
        iris_node.inputs['Subsurface Radius'].default_value=(1.,.5,.25);iris_node.inputs['Subsurface Scale'].default_value=.0002*k
        pupil,pupil_node=material(f'{pre}_pupil',(.002,.002,.002),.9);pupil_node.inputs['Specular IOR Level'].default_value=0.
        for part,mat in (('shell',sclera),('chamber',cornea),('iris',iris),('pupil_cup',pupil)):
            positions=mesh_data[f'{part}_positions'].astype(np.float64)
            world=centre+k*positions@rot.T
            triangles=mesh_data[f'{part}_triangles'];uvs=mesh_data[f'{part}_uvs']
            mesh=bpy.data.meshes.new(f'{pre}_{part}');mesh.from_pydata(world.tolist(),[],triangles.tolist());mesh.update()
            uv=mesh.uv_layers.new(name='UVMap');uv.data.foreach_set('uv',uvs[triangles].reshape(-1).astype(np.float32))
            mesh.materials.append(mat)
            for key in ('alpha','blend'):
                if f'{part}_{key}' in mesh_data.files:
                    attr=mesh.attributes.new(f'eye2_{key}','FLOAT','POINT');attr.data.foreach_set('value',mesh_data[f'{part}_{key}'].astype(np.float32))
            for polygon in mesh.polygons:polygon.use_smooth=True
            obj=bpy.data.objects.new(f'{pre}_{part}',mesh);bpy.context.collection.objects.link(obj)
            obj['eye2_axis']=[float(v) for v in z]   # sweep tools move the eye along its optical axis
            # manifold next-event estimation: refracted key light through the cornea onto the iris
            if part=='chamber' and float(eye_mo.get('eye2_cornea_shadow',0.)) < .5:obj.visible_shadow=False
            for holder in (getattr(obj,'cycles',None),obj):
                if part in ('shell','chamber') and holder is not None and hasattr(holder,'is_caustics_caster'):holder.is_caustics_caster=True;made.append('mnee_caster');break
                if part=='iris' and holder is not None and hasattr(holder,'is_caustics_receiver'):holder.is_caustics_receiver=True;made.append('mnee_receiver');break
            objects[f'{pre}_{part}']=obj;made.append(obj.name)
        if float(eye_mo.get('peri_socket',0.)) >= .5:
            # conjunctival sac lining: dark mucosa shell (radius 1.06 R) behind the visible eye, open over the front
            # (theta < 75 deg) so it never covers the eyeball; lid gaps of a deeper-set eye show tissue, not void
            th=np.linspace(math.radians(75),math.pi,24);ph=np.linspace(0,2*math.pi,49)
            T_,P_=np.meshgrid(th,ph,indexing='ij')
            local=1.06*float(info['dims']['R'])*np.stack([np.sin(T_)*np.cos(P_),np.sin(T_)*np.sin(P_),np.cos(T_)],-1).reshape(-1,3)
            world=centre+k*local@rot.T
            r_,c_=np.meshgrid(np.arange(len(th)-1),np.arange(len(ph)-1),indexing='ij');a_=(r_*len(ph)+c_).ravel()
            tris=np.concatenate([np.stack([a_,a_+len(ph),a_+1],-1),np.stack([a_+1,a_+len(ph),a_+len(ph)+1],-1)])
            sock_mesh=bpy.data.meshes.new(f'{pre}_socket');sock_mesh.from_pydata(world.tolist(),[],tris.tolist());sock_mesh.update()
            if 'eye2_socket' not in bpy.data.materials:
                sm,snode=material('eye2_socket',(.16,.05,.05),.5);snode.inputs['Specular IOR Level'].default_value=.2
            sock_mesh.materials.append(bpy.data.materials['eye2_socket'])
            for polygon in sock_mesh.polygons:polygon.use_smooth=True
            sobj=bpy.data.objects.new(f'{pre}_socket',sock_mesh);bpy.context.collection.objects.link(sobj)
            objects[f'{pre}_socket']=sobj;made.append(sobj.name)
        if float(eye_mo.get('peri_lid_veil',0.)) > 0.:
            # soft lid->eye boundary (UE's translucent eye-occlusion shell softens the contact): a ribbon laid on the
            # eyeball from each lid margin toward the eye centre, lid-tissue colour fading to transparent (non-physical)
            width=float(eye_mo.get('peri_lid_veil_width',.0008));opacity=float(eye_mo['peri_lid_veil'])
            Rw=k*float(info['dims']['R'])+.00015
            if 'eye2_lid_veil' not in bpy.data.materials:
                vm=bpy.data.materials.new('eye2_lid_veil');vm.use_nodes=True;vnt=vm.node_tree
                vnt.nodes.remove(vnt.nodes.get('Principled BSDF'))
                va=vnt.nodes.new('ShaderNodeAttribute');va.attribute_type='GEOMETRY';va.attribute_name='veil_alpha'
                vd=vnt.nodes.new('ShaderNodeBsdfDiffuse');vd.inputs['Color'].default_value=(.16,.075,.065,1.)
                vt=vnt.nodes.new('ShaderNodeBsdfTransparent');vmix=vnt.nodes.new('ShaderNodeMixShader')
                vnt.links.new(va.outputs['Fac'],vmix.inputs[0]);vnt.links.new(vt.outputs[0],vmix.inputs[1]);vnt.links.new(vd.outputs[0],vmix.inputs[2])
                vnt.links.new(vmix.outputs[0],vnt.nodes.get('Material Output').inputs['Surface'])
            for margin_name in (f'{side}_tearline',f'{side}_tearline_upper'):
                key=margin_name+'_positions'
                if key not in data.files:continue
                ring=np.asarray(data[key],np.float64).reshape(-1,8,3).mean(1)
                if objects.get(margin_name) is not None:ring=ring+np.asarray(objects[margin_name].location)
                u=np.linspace(0,len(ring)-1,6*(len(ring)-1)+1);i0=np.floor(u).astype(int).clip(0,len(ring)-2);f_=u-i0
                m=ring[i0]*(1-f_)[:,None]+ring[i0+1]*f_[:,None]
                dirs=centre-m;dirs-=np.outer(dirs@z,z);dirs/=np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-12)
                ts=np.linspace(0,1,7);rows=[]
                for t in ts:
                    p=m+dirs*(t*width);v=p-centre;rows.append(centre+v/np.linalg.norm(v,axis=1,keepdims=True)*Rw)
                pts=np.concatenate(rows);n=len(m)
                r_,c_=np.meshgrid(np.arange(len(ts)-1),np.arange(n-1),indexing='ij');a_=(r_*n+c_).ravel()
                tris=np.concatenate([np.stack([a_,a_+1,a_+n],-1),np.stack([a_+1,a_+n+1,a_+n],-1)])
                vmesh=bpy.data.meshes.new(f'{pre}_veil_{margin_name}');vmesh.from_pydata(pts.tolist(),[],tris.tolist());vmesh.update()
                al=vmesh.attributes.new('veil_alpha','FLOAT','POINT')
                al.data.foreach_set('value',np.repeat(opacity*(1-ts)**2,n).astype(np.float32))
                vmesh.materials.append(bpy.data.materials['eye2_lid_veil'])
                for polygon in vmesh.polygons:polygon.use_smooth=True
                vobj=bpy.data.objects.new(f'{pre}_veil_{margin_name}',vmesh);bpy.context.collection.objects.link(vobj)
                vobj.visible_shadow=False;objects[vobj.name]=vobj;made.append(vobj.name)
    scene.cycles.transmission_bounces=max(scene.cycles.transmission_bounces,16)
    scene.cycles.max_bounces=max(scene.cycles.max_bounces,24);scene.cycles.glossy_bounces=max(scene.cycles.glossy_bounces,8)
    scene.cycles.caustics_refractive=True;scene.cycles.caustics_reflective=True
    scene.cycles.blur_glossy=float(eye_mo.get('eye2_filter_glossy',.3))
    scene['eye2_layered']=json.dumps(dict(objects=made,model='physical layered eye v2 smooth shell (server/vhuman/eye2)'))
    return made


def curve_object(name,paths,mat,radius, *, cyclic=False):
    paths=np.asarray(paths,np.float32)
    if not len(paths):return None
    if paths.ndim==2:paths=paths[None]
    # Native hair Curves are ray-traced as strands, not beveled mesh tubes.
    if not cyclic and hasattr(bpy.data,'hair_curves'):
        data=bpy.data.hair_curves.new(name)
        data.add_curves([len(p) for p in paths])
        data.attributes['position'].data.foreach_set('vector',paths.reshape(-1))
        size=data.attributes.get('radius') or data.attributes.new('radius','FLOAT','POINT')
        values=np.tile(np.linspace(radius,radius*.08,paths.shape[1]),len(paths)).astype(np.float32)
        size.data.foreach_set('value',values)
    else:
        data=bpy.data.curves.new(name,'CURVE');data.dimensions='3D';data.bevel_depth=radius;data.bevel_resolution=3
        for path in paths:
            spline=data.splines.new('POLY');spline.points.add(len(path)-1)
            spline.points.foreach_set('co',np.column_stack((path,np.ones(len(path)))).ravel())
            spline.use_cyclic_u=cyclic
    obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj)
    data.materials.append(mat)
    return obj


# ColorChecker calibration (2026-11-01, color_chart.py, chart 2 cm in front of the face, -1.5 EV Standard view):
# the authored lamp rig read 1.36 EV too bright (exposure -2.86 needed), so its energies are scaled by 2^-1.36 and
# -1.5 EV now shows the neutral patches at their reflectance. The default HDRI (interior.exr) read -0.82 EV with a warm
# cast; its environment is multiplied by the neutralising gains and 2^0.68. Other HDRIs are left as given.
STUDIO_ENERGY_SCALE=2.0**-1.3632
# Studio lamp rig (name, position m, authored power W, disk diameter m); each disk aims at the world origin. Shared with
# metahuman/chart_probe_blender.py (UE replica + ColorChecker light calibration). Studio world: grey .1 at strength .15.
STUDIO_LIGHTS=(('key',(-.35,.35,.45),35,.35),('fill',(.35,.1,.35),12,.3),('rim',(.2,.3,-.3),20,.25))
STUDIO_WORLD=(.1,.15)
CALIBRATED_HDRI={'e945ff5c1ddd7a3aaf05e9fb5c3bc9cb93c5518414febd44ed2c394e013f0cbd':(.8351,1.,1.4483,2.0**.6840)}


def add_color_chart(scene,camera_obj,data,out,lighting):
    """24-patch matte ColorChecker 2 cm in front of the face, facing the camera, under the scene's lights.
    Writes chart.json (patch names, reference linear albedo, pixel centres) for color_chart.calibrate_render."""
    if __package__:
        from . import color_chart
    else:
        sys.path.insert(0,str(Path(__file__).parent));import color_chart
    from bpy_extras.object_utils import world_to_camera_view
    m=np.asarray(camera_obj.matrix_world);origin=m[:3,3];right,up,back=m[:3,0],m[:3,1],m[:3,2];forward=-back
    skin=np.asarray(data['skin_positions'],float).reshape(-1,3)
    depth=float(((skin-origin)@forward).min())-.02
    centre=origin+forward*depth
    centres,size=color_chart.layout()
    albedo=color_chart.reference_linear();patches=[]
    rotation=Matrix(np.stack([right,up,back],1).tolist()).to_euler()
    for i,((name,rgb),(cx,cy)) in enumerate(zip(color_chart.PATCHES,centres)):
        mesh=bpy.data.meshes.new(f'chart_{i:02d}');h=size/2
        mesh.from_pydata([(-h,-h,0),(h,-h,0),(h,h,0),(-h,h,0)],[],[(0,1,2,3)])
        obj=bpy.data.objects.new(f'chart_{i:02d}',mesh);bpy.context.collection.objects.link(obj)
        position=centre+right*cx+up*cy;obj.location=position.tolist();obj.rotation_euler=rotation
        mat=bpy.data.materials.new(f'chart_{i:02d}');mat.use_nodes=True;bsdf=mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value=(*[float(v) for v in albedo[i]],1.);bsdf.inputs['Roughness'].default_value=1.
        if 'Specular IOR Level' in bsdf.inputs:bsdf.inputs['Specular IOR Level'].default_value=0.
        mesh.materials.append(mat)
        res=(scene.render.resolution_x,scene.render.resolution_y)
        c=world_to_camera_view(scene,camera_obj,Vector(position.tolist()))
        e=world_to_camera_view(scene,camera_obj,Vector((position+right*h).tolist()))
        patches.append(dict(name=name,srgb8=list(rgb),albedo_linear=[float(v) for v in albedo[i]],
                            pixel_center=[float(c.x*res[0]),float((1-c.y)*res[1])],pixel_half_size=float(abs(e.x-c.x)*res[0])))
    (out/'chart.json').write_text(json.dumps(dict(lighting=lighting,depth_m=depth,patches=patches),indent=2))


def run(request):
    started=time.perf_counter();out=Path(request['out']);out.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.temporary_directory=str(out/'cache')
    (out/'cache').mkdir(exist_ok=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES'
    if request['device'] in ('hip','cuda','optix'):
        preferences=bpy.context.preferences.addons['cycles'].preferences
        device_type=request['device'].upper()
        preferences.compute_device_type=device_type;preferences.get_devices()
        devices=[d for d in preferences.devices if d.type==device_type]
        if not devices:raise RuntimeError(f'Cycles found no {device_type} device; CPU rendering must be requested explicitly')
        for device in preferences.devices:device.use=device==devices[request.get('gpu_index',0)]
        if hasattr(preferences,'use_hiprt'):preferences.use_hiprt=False
        scene.cycles.device='GPU';device_name=devices[request.get('gpu_index',0)].name
    else:
        scene.cycles.device='CPU';device_name='CPU'
    presets={'draft':(512,32),'final':(1024,256),'hero':(2048,1024),'crop':(1024,1024)}
    resolution,samples=presets[request['preset']]
    resolution=int(request.get('resolution') or resolution);samples=int(request.get('samples') or samples)
    scene.cycles.samples=samples
    if (request.get('material_overrides') or {}).get('skin_lighting_profile'):
        scene.cycles.use_auto_tile=True;scene.cycles.tile_size=256
    scene.cycles.use_denoising=True;scene.cycles.denoiser='OPENIMAGEDENOISE'
    if hasattr(scene.cycles,'denoising_use_gpu'):scene.cycles.denoising_use_gpu=False
    scene.cycles.max_bounces=12;scene.cycles.transmission_bounces=8
    if request['preset']=='hero':scene.cycles.max_bounces=16;scene.cycles.transmission_bounces=12
    scene.render.resolution_x=resolution;scene.render.resolution_y=resolution;scene.render.resolution_percentage=100
    scene.render.film_transparent=True
    scene.view_settings.exposure=request.get('exposure',-1.5)
    for transform in ('Standard',):
        try:scene.view_settings.view_transform=transform;scene.view_settings.look='None';break
        except TypeError:pass
    noisy_output=None
    if request.get('noisy',True):
        # One render yields both: the denoised beauty through the group output and the raw 'Noisy Image'
        # pass through a File Output node (Blender 5 compositor node group API).
        view_layer=bpy.context.view_layer;view_layer.cycles.denoising_store_passes=True
        group=bpy.data.node_groups.new('beauty_compositor','CompositorNodeTree');scene.compositing_node_group=group
        if hasattr(scene,'use_nodes'):scene.use_nodes=True
        group.interface.new_socket('Image',in_out='OUTPUT',socket_type='NodeSocketColor')
        layers=group.nodes.new('CompositorNodeRLayers');output=group.nodes.new('NodeGroupOutput')
        group.links.new(layers.outputs['Image'],output.inputs[0])
        files=group.nodes.new('CompositorNodeOutputFile');files.directory=str(out);files.file_name='beauty_noisy'
        files.format.file_format='OPEN_EXR_MULTILAYER';files.format.color_depth='16'   # the 5.2 File Output node only offers multilayer EXR
        if not len(files.file_output_items):files.file_output_items.new('RGBA','Image')
        group.links.new(layers.outputs['Noisy Image'],files.inputs[0])
        noisy_output='beauty_noisy.exr'
    scene['skin_lighting_prior']=json.dumps(dict(sss_weight=request.get('sss_weight',.08),exposure=request.get('exposure',-1.5),status='authored studio preset'))
    config=json.loads((out/'scene.json').read_text());candidate=Path(config['candidate'])
    data=np.load(out/'scene_assets.npz',allow_pickle=False)
    materials={}
    skin,node=material('skin',(.5,.3,.2),config['material']['roughness']['value'],ior=
        (1+math.sqrt(config['material']['f0']['value']))/(1-math.sqrt(config['material']['f0']['value'])))
    nodes,links=skin.node_tree.nodes,skin.node_tree.links
    color=image_node(nodes,candidate/'skin_basecolor.png');links.new(color.outputs['Color'],node.inputs['Base Color'])
    if (out/'expression_appearance.npz').is_file():
        appearance=np.load(out/'expression_appearance.npz',allow_pickle=False)
        current_color=color.outputs['Color']
        for i,values in enumerate(appearance['linear_albedo_delta']):
            texture=float_image(nodes,f'expression_color_{i}',values)
            scale=nodes.new('ShaderNodeVectorMath');scale.operation='SCALE'
            driver=scale.inputs['Scale'].driver_add('default_value').driver
            driver.expression='activation';variable=driver.variables.new();variable.name='activation'
            variable.targets[0].id_type='SCENE';variable.targets[0].id=scene;variable.targets[0].data_path=f'["wrinkle_{i}"]'
            links.new(texture.outputs['Color'],scale.inputs[0])
            add=nodes.new('ShaderNodeVectorMath');add.operation='ADD'
            links.new(current_color,add.inputs[0]);links.new(scale.outputs[0],add.inputs[1]);current_color=add.outputs[0]
        lower=nodes.new('ShaderNodeVectorMath');lower.operation='MAXIMUM';lower.inputs[1].default_value=(0,0,0)
        upper=nodes.new('ShaderNodeVectorMath');upper.operation='MINIMUM';upper.inputs[1].default_value=(1,1,1)
        links.new(current_color,lower.inputs[0]);links.new(lower.outputs[0],upper.inputs[0])
        links.new(upper.outputs[0],node.inputs['Base Color'])
    confidence=image_node(nodes,candidate/'skin_confidence.png',linear=True)
    confidence.label='Observed source confidence; hidden texels remain inferred'
    orm=image_node(nodes,candidate/'skin_orm.png',linear=True);channels=nodes.new('ShaderNodeSeparateColor')
    links.new(orm.outputs['Color'],channels.inputs['Color']);links.new(channels.outputs['Green'],node.inputs['Roughness'])
    node.inputs['Subsurface Weight'].default_value=request.get('sss_weight',.08)
    node.inputs['Subsurface Scale'].default_value=1
    node.inputs['Subsurface Radius'].default_value=config['material']['sss']['radii_m']
    detail=np.load(out/'skin_detail.npz',allow_pickle=False)
    height=float_image(nodes,'skin_height_metres',detail['height_m'])
    current=height.outputs['Color']
    for i,values in enumerate(detail['dynamic_height_m']):
        texture=float_image(nodes,f'wrinkle_height_{i}',values)
        multiply=nodes.new('ShaderNodeMath');multiply.operation='MULTIPLY'
        scene[f'wrinkle_{i}']=0.
        driver=multiply.inputs[1].driver_add('default_value').driver
        driver.expression='activation'
        variable=driver.variables.new();variable.name='activation';variable.targets[0].id_type='SCENE'
        variable.targets[0].id=scene;variable.targets[0].data_path=f'["wrinkle_{i}"]'
        links.new(texture.outputs['Color'],multiply.inputs[0])
        add=nodes.new('ShaderNodeMath');add.operation='ADD';links.new(current,add.inputs[0]);links.new(multiply.outputs[0],add.inputs[1]);current=add.outputs[0]
    bound=nodes.new('ShaderNodeClamp');bound.inputs['Min'].default_value=-.0005;bound.inputs['Max'].default_value=.0005
    links.new(current,bound.inputs['Value'])
    displacement=nodes.new('ShaderNodeDisplacement');displacement.inputs['Midlevel'].default_value=0;displacement.inputs['Scale'].default_value=1
    links.new(bound.outputs[0],displacement.inputs['Height'])
    links.new(displacement.outputs[0],nodes.get('Material Output').inputs['Displacement'])
    skin.displacement_method='BOTH';materials['skin']=skin
    full=request.get('quality','baseline')=='full';ablate=set(request.get('ablate',[]))
    skin_full=full and '2c' not in ablate and (out/'skin_templates.npz').is_file()
    if skin_full:full_skin_graph(skin,node,bound,out,request)
    # eye material overrides (metahuman.compare_render EYE_SWEEP): sclera/iris/cornea roughness, cornea IOR, cornea
    # Specular IOR Level, iris_/sclera_gain (texture x gain), iris_/sclera_sat (HSV saturation), cornea_shadow, iris_specular (Specular IOR Level), eye_offset_<side>_m, iris_emit / iris_emission_strength / cornea_camera (diagnostics); absent = the authored values.
    eye_mo=dict(request.get('material_overrides') or {})
    for name,color,roughness in [('teeth',(.65,.6,.5),.25),('gums',(.35,.08,.07),.35),
        ('tongue',(.4,.09,.08),.4),('cavity',(.33,.09,.09),.45),('sclera',(.7,.65,.6),.25),
        ('iris',(.12,.065,.035),.55),('frame',(.12,.045,.025),.25),('pupil',(.001,.001,.001),1),('lash',(.025,.018,.012),.45)]:
        materials[name],principled=material(name,color,float(eye_mo.get(f'{name}_roughness',roughness)) if name in ('sclera','iris') else roughness)
        if name=='iris' and 'iris_specular' in eye_mo:principled.inputs['Specular IOR Level'].default_value=float(eye_mo['iris_specular'])
        if name=='iris' and 'iris_normal_strength' in eye_mo:
            normal_path=out/'iris_normal.png'
            if not normal_path.is_file():raise RuntimeError('iris_normal_strength requires staged iris_normal.png')
            nt=materials[name].node_tree;tex=image_node(nt.nodes,normal_path,linear=True)
            normal=nt.nodes.new('ShaderNodeNormalMap');normal.inputs['Strength'].default_value=float(eye_mo['iris_normal_strength'])
            nt.links.new(tex.outputs['Color'],normal.inputs['Color']);nt.links.new(normal.outputs['Normal'],principled.inputs['Normal'])
        if name=='sclera' and 'sclera_sss_weight' in eye_mo:
            principled.subsurface_method='RANDOM_WALK'
            principled.inputs['Subsurface Weight'].default_value=float(eye_mo['sclera_sss_weight'])
            principled.inputs['Subsurface Radius'].default_value=(.0015,.0006,.0004)
            principled.inputs['Subsurface Scale'].default_value=1.
        if name in ('gums','tongue'):
            principled.inputs['Subsurface Weight'].default_value=.1
            principled.inputs['Subsurface Radius'].default_value=(.0008,.0004,.0002)
            principled.inputs['Subsurface Scale'].default_value=1
        if name in ('sclera','iris'):
            texture=image_node(materials[name].node_tree.nodes,out/('sclera.png' if name=='sclera' else 'iris.png'))
            color_out=texture.outputs['Color']
            if f'{name}_gain' in eye_mo:   # iris base-colour brightness test (absent = no extra node)
                gain=materials[name].node_tree.nodes.new('ShaderNodeVectorMath');gain.operation='SCALE'
                gain.inputs['Scale'].default_value=float(eye_mo[f'{name}_gain'])
                materials[name].node_tree.links.new(color_out,gain.inputs[0]);color_out=gain.outputs['Vector']
            if f'{name}_sat' in eye_mo:   # iris saturation test (absent = no extra node)
                hsv=materials[name].node_tree.nodes.new('ShaderNodeHueSaturation')
                hsv.inputs['Saturation'].default_value=float(eye_mo[f'{name}_sat'])
                materials[name].node_tree.links.new(color_out,hsv.inputs['Color']);color_out=hsv.outputs['Color']
            if name=='sclera' and 'sclera_vessel_strength' in eye_mo:
                vessel_path=out/'sclera_vessels.png'
                if not vessel_path.is_file():raise RuntimeError('sclera_vessel_strength requires staged sclera_vessels.png')
                nt=materials[name].node_tree; tex=image_node(nt.nodes,vessel_path,linear=True)
                sep=nt.nodes.new('ShaderNodeSeparateColor');nt.links.new(tex.outputs['Color'],sep.inputs['Color'])
                strength=nt.nodes.new('ShaderNodeMath');strength.operation='MULTIPLY';strength.use_clamp=True
                nt.links.new(sep.outputs['Red'],strength.inputs[0]);strength.inputs[1].default_value=float(eye_mo['sclera_vessel_strength'])
                tint=nt.nodes.new('ShaderNodeMixRGB');tint.blend_type='MULTIPLY';tint.inputs['Color2'].default_value=(.72,.16,.14,1.)
                nt.links.new(strength.outputs[0],tint.inputs['Fac']);nt.links.new(color_out,tint.inputs['Color1']);color_out=tint.outputs[0]
            materials[name].node_tree.links.new(color_out,principled.inputs['Base Color'])
            if name=='iris' and 'iris_emission_strength' in eye_mo:
                # Optional diagnostic: add a controlled amount of iris-coloured fill while retaining surface shading.
                materials[name].node_tree.links.new(color_out,principled.inputs['Emission Color'])
                principled.inputs['Emission Strength'].default_value=float(eye_mo['iris_emission_strength'])
            if name=='iris' and 'iris_emit' in eye_mo:   # diagnostic: iris = emission of its texture x iris_emit (no shading)
                nt=materials[name].node_tree;emit=nt.nodes.new('ShaderNodeEmission')
                emit.inputs['Strength'].default_value=float(eye_mo['iris_emit'])
                nt.links.new(color_out,emit.inputs['Color']);nt.links.new(emit.outputs[0],nt.nodes.get('Material Output').inputs['Surface'])
    if full and '2e' not in ablate:full_mouth_materials(materials,out,request)
    materials['cornea'],cornea=material('cornea',(1,1,1),float(eye_mo.get('cornea_roughness',.025)),1,
                                        float(eye_mo.get('cornea_ior',config['optical_eyes']['ior'])))
    if 'cornea_specular' in eye_mo:cornea.inputs['Specular IOR Level'].default_value=float(eye_mo['cornea_specular'])
    if eye_mo.get('eye_unified',0.) >= .5:
        # UE's eye shading model shades a single opaque surface with a separate iris normal;
        # this diagnostic uses a unified albedo and a dielectric highlight, not a glass cap.
        trans_comp=float(eye_mo.get('sclera_transmission_compensation',0.))
        if trans_comp not in (0.,.5,1.):raise ValueError('sclera_transmission_compensation must be 0, .5 or 1')
        unified_path=out/('eye_unified_clean_ring_soft.png' if eye_mo.get('iris_ring_soft',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_clean_ring_none.png' if eye_mo.get('iris_ring_none',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_clean_fibers_fit_trans05.png' if eye_mo.get('iris_fiber_chroma_fit',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_clean_fibers_trans05.png' if eye_mo.get('iris_fiber_chroma',0.) > 0. and trans_comp == .5 else
                          'eye_unified_clean_swap_trans05.png' if eye_mo.get('iris_swap_regions',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_clean_bands_trans05.png' if eye_mo.get('iris_radial_bands',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_clean_adaptive_trans05.png' if eye_mo.get('sclera_clean_base',0.) >= .5 and eye_mo.get('iris_adaptive_contrast',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_clean_trans05.png' if eye_mo.get('sclera_clean_base',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_adaptive_trans05.png' if eye_mo.get('iris_adaptive_contrast',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_layered_trans05.png' if eye_mo.get('iris_layered_calibrate',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_radial_trans05.png' if eye_mo.get('iris_radial_calibrate',0.) >= .5 and trans_comp == .5 else
                          'eye_unified_trans05.png' if trans_comp == .5 else
                          'eye_unified_trans1.png' if trans_comp == 1. else
                          'eye_unified_calibrated.png' if eye_mo.get('iris_calibrate',0.) >= .5 else
                          'eye_unified_smooth.png' if 'iris_angular_smoothing' in eye_mo else
                          'eye_unified_soft.png' if 'iris_blend_softness_floor' in eye_mo else 'eye_unified.png')
        if not unified_path.is_file():raise RuntimeError('eye_unified requires staged eye_unified.png')
        unified,unode=material('eye_unified',(1,1,1),float(eye_mo.get('cornea_roughness',.075)),0,
                               float(eye_mo.get('cornea_ior',1.41)))
        nt=unified.node_tree;tex=image_node(nt.nodes,unified_path)
        color_out=tex.outputs['Color']
        if 'sclera_tissue_strength' in eye_mo:
            tissue_path=out/'eye_tissue_factor.png'
            if not tissue_path.is_file():raise RuntimeError('sclera_tissue_strength requires staged eye_tissue_factor.png')
            tissue=image_node(nt.nodes,tissue_path,linear=True)
            scale=nt.nodes.new('ShaderNodeVectorMath');scale.operation='SCALE';scale.inputs['Scale'].default_value=2.
            nt.links.new(tissue.outputs['Color'],scale.inputs[0])
            mix=nt.nodes.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY'
            mix.inputs['Fac'].default_value=float(eye_mo['sclera_tissue_strength'])
            nt.links.new(color_out,mix.inputs['Color1']);nt.links.new(scale.outputs['Vector'],mix.inputs['Color2']);color_out=mix.outputs[0]
        if 'sclera_microtexture' in eye_mo or 'sclera_surface_microtexture' in eye_mo:
            spherical='sclera_surface_microtexture' in eye_mo
            micro_strength=float(eye_mo['sclera_surface_microtexture'] if spherical else eye_mo['sclera_microtexture'])
            if spherical:
                if micro_strength not in (.05,.1):raise ValueError('sclera_surface_microtexture must be .05 or .1')
                micro_path=out/('eye_surface_fine.png' if eye_mo.get('sclera_surface_fine',0.) >= .5 else
                               'eye_surface05.png' if micro_strength == .05 else 'eye_surface10.png')
            else:
                if micro_strength not in (.1,.2):raise ValueError('sclera_microtexture must be .1 or .2')
                micro_path=out/('eye_micro10.png' if micro_strength == .1 else 'eye_micro20.png')
            if not micro_path.is_file():raise RuntimeError('sclera_microtexture requires staged microtexture')
            micro=image_node(nt.nodes,micro_path,linear=True)
            scale=nt.nodes.new('ShaderNodeVectorMath');scale.operation='SCALE';scale.inputs['Scale'].default_value=2.
            nt.links.new(micro.outputs['Color'],scale.inputs[0])
            mix=nt.nodes.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs['Fac'].default_value=1.
            nt.links.new(color_out,mix.inputs['Color1']);nt.links.new(scale.outputs['Vector'],mix.inputs['Color2']);color_out=mix.outputs[0]
        if 'sclera_vessel_all_strength' in eye_mo:
            vessel_path=out/('sclera_vessels_selfsafe.png' if eye_mo.get('sclera_vessel_selfsafe',0.) >= .5 else
                             'sclera_vessels_nocross.png' if eye_mo.get('sclera_vessel_nocross',0.) >= .5 else
                             'sclera_vessels_guided.png' if eye_mo.get('sclera_vessel_guided',0.) >= .5 else
                             'sclera_vessels_organic_dense.png' if eye_mo.get('sclera_vessel_organic_dense',0.) >= .5 else
                             'sclera_vessels_organic.png' if eye_mo.get('sclera_vessel_organic',0.) >= .5 else
                             'sclera_vessels_sparse.png' if eye_mo.get('sclera_vessel_density',0.) >= .5 else
                             'sclera_vessels_lines.png' if eye_mo.get('sclera_vessel_lines_only',0.) >= .5 else
                             'sclera_vessels_dense.png' if eye_mo.get('sclera_dense_vessels',0.) >= .5 else
                             'sclera_vessels_irregular.png' if eye_mo.get('sclera_irregular_vessels',0.) >= .5 else
                             'sclera_vessels_curvy.png' if eye_mo.get('sclera_curvy_vessels',0.) >= .5 else 'sclera_vessels_all.png')
            if not vessel_path.is_file():raise RuntimeError('sclera_vessel_all_strength requires staged sclera_vessels_all.png')
            mask=image_node(nt.nodes,vessel_path,linear=True)
            strength=nt.nodes.new('ShaderNodeMath');strength.operation='MULTIPLY';strength.use_clamp=True
            nt.links.new(mask.outputs['Color'],strength.inputs[0]);strength.inputs[1].default_value=float(eye_mo['sclera_vessel_all_strength'])
            tint=nt.nodes.new('ShaderNodeMixRGB');tint.blend_type='MULTIPLY';tint.inputs['Color2'].default_value=(.72,.16,.14,1.)
            nt.links.new(strength.outputs[0],tint.inputs['Fac']);nt.links.new(color_out,tint.inputs['Color1']);color_out=tint.outputs[0]
        nt.links.new(color_out,unode.inputs['Base Color'])
        if eye_mo.get('eye_basecolor_only',0.) >= .5:
            emit=nt.nodes.new('ShaderNodeEmission')
            nt.links.new(color_out,emit.inputs['Color'])
            nt.links.new(emit.outputs['Emission'],nt.nodes.get('Material Output').inputs['Surface'])
        roughness_out=None;surface_normal=None
        if 'unified_sclera_roughness' in eye_mo:
            # The angular UV layout places the corneal edge at radius .2.
            uvnode=nt.nodes.new('ShaderNodeUVMap');dist=nt.nodes.new('ShaderNodeVectorMath');dist.operation='DISTANCE'
            nt.links.new(uvnode.outputs['UV'],dist.inputs[0]);dist.inputs[1].default_value=(.5,.5,0.)
            ramp=nt.nodes.new('ShaderNodeMapRange');ramp.name='vhuman_sclera_roughness';ramp.clamp=True
            ramp.inputs['From Min'].default_value=.19;ramp.inputs['From Max'].default_value=.23
            ramp.inputs['To Min'].default_value=float(eye_mo.get('cornea_roughness',.075))
            ramp.inputs['To Max'].default_value=float(eye_mo['unified_sclera_roughness'])
            nt.links.new(dist.outputs['Value'],ramp.inputs['Value']);nt.links.new(ramp.outputs['Result'],unode.inputs['Roughness'])
            roughness_out=ramp.outputs['Result']
        if 'unified_shell_normal_strength' in eye_mo:
            shell_normal_path=out/'eye_shell_normal.png'
            if not shell_normal_path.is_file():raise RuntimeError('unified_shell_normal_strength requires staged eye_shell_normal.png')
            ntex=image_node(nt.nodes,shell_normal_path,linear=True);nmap=nt.nodes.new('ShaderNodeNormalMap')
            nmap.inputs['Strength'].default_value=float(eye_mo['unified_shell_normal_strength'])
            nt.links.new(ntex.outputs['Color'],nmap.inputs['Color']);nt.links.new(nmap.outputs['Normal'],unode.inputs['Normal'])
            surface_normal=nmap.outputs['Normal']
        if 'unified_surface_bump_distance' in eye_mo:
            bump_distance=float(eye_mo['unified_surface_bump_distance'])
            if not 0. <= bump_distance <= .001:raise ValueError('unified_surface_bump_distance must be between 0 and .001 m')
            bump_path=out/'eye_surface_fine.png'
            if not bump_path.is_file():raise RuntimeError('unified_surface_bump_distance requires staged eye_surface_fine.png')
            bumptex=image_node(nt.nodes,bump_path,linear=True);bump=nt.nodes.new('ShaderNodeBump')
            bump.inputs['Distance'].default_value=bump_distance
            nt.links.new(bumptex.outputs['Color'],bump.inputs['Height'])
            if eye_mo.get('unified_surface_bump_patchy',0.) >= .5:
                patch_path=out/'eye_surface05.png'
                if not patch_path.is_file():raise RuntimeError('patchy surface bump requires staged eye_surface05.png')
                patchtex=image_node(nt.nodes,patch_path,linear=True);patch=nt.nodes.new('ShaderNodeMapRange');patch.clamp=True
                patch.inputs['From Min'].default_value=.49;patch.inputs['From Max'].default_value=.53
                nt.links.new(patchtex.outputs['Color'],patch.inputs['Value']);nt.links.new(patch.outputs['Result'],bump.inputs['Strength'])
            if surface_normal is not None:nt.links.new(surface_normal,bump.inputs['Normal'])
            nt.links.new(bump.outputs['Normal'],unode.inputs['Normal']);surface_normal=bump.outputs['Normal']
        if 'sclera_sss_weight' in eye_mo:
            unode.subsurface_method='RANDOM_WALK'
            unode.inputs['Subsurface Weight'].default_value=float(eye_mo['sclera_sss_weight'])
            radius_scale=float(eye_mo.get('unified_sss_radius_scale',1.))
            unode.inputs['Subsurface Radius'].default_value=(.0015*radius_scale,.0006*radius_scale,.0004*radius_scale)
            unode.inputs['Subsurface Scale'].default_value=1.
        if 'unified_key_caustic_strength' in eye_mo and eye_mo.get('eye_ue_split',0.) < .5:
            raise ValueError('key caustic approximation requires eye_ue_split')
        if eye_mo.get('eye_ue_split',0.) >= .5:
            # Approximate UE EyeBxDF's separate iris diffuse and surface specular normals.
            # Cycles has no per-light custom BxDF in node materials, so the UE caustic term is not reproduced here.
            eye_params=json.loads((out/'eye_params.json').read_text());size=float(eye_params['cornea']['size'])
            half=.5*float(eye_params['cornea']['limbus_softness'])
            uvnode=nt.nodes.new('ShaderNodeUVMap');dist=nt.nodes.new('ShaderNodeVectorMath');dist.operation='DISTANCE'
            nt.links.new(uvnode.outputs['UV'],dist.inputs[0]);dist.inputs[1].default_value=(.5,.5,0.)
            iris_mask=nt.nodes.new('ShaderNodeMapRange');iris_mask.name='vhuman_iris_mask';iris_mask.clamp=True
            iris_mask.inputs['From Min'].default_value=size-half;iris_mask.inputs['From Max'].default_value=size+half
            iris_mask.inputs['To Min'].default_value=1.;iris_mask.inputs['To Max'].default_value=0.
            nt.links.new(dist.outputs['Value'],iris_mask.inputs['Value'])
            geom=nt.nodes.new('ShaderNodeNewGeometry');plane=nt.nodes.new('ShaderNodeAttribute');plane.attribute_name='iris_plane_normal'
            surface_normal=surface_normal or geom.outputs['Normal']
            inv=nt.nodes.new('ShaderNodeMath');inv.operation='SUBTRACT';inv.inputs[0].default_value=1.
            nt.links.new(iris_mask.outputs['Result'],inv.inputs[1])
            left=nt.nodes.new('ShaderNodeVectorMath');left.operation='SCALE'
            right=nt.nodes.new('ShaderNodeVectorMath');right.operation='SCALE'
            nt.links.new(surface_normal,left.inputs[0]);nt.links.new(inv.outputs[0],left.inputs['Scale'])
            nt.links.new(plane.outputs['Vector'],right.inputs[0]);nt.links.new(iris_mask.outputs['Result'],right.inputs['Scale'])
            add=nt.nodes.new('ShaderNodeVectorMath');add.operation='ADD'
            nt.links.new(left.outputs['Vector'],add.inputs[0]);nt.links.new(right.outputs['Vector'],add.inputs[1])
            normal=nt.nodes.new('ShaderNodeVectorMath');normal.operation='NORMALIZE';nt.links.new(add.outputs['Vector'],normal.inputs[0])
            nt.links.new(normal.outputs['Vector'],unode.inputs['Normal']);unode.inputs['Specular IOR Level'].default_value=0.
            if 'unified_key_caustic_strength' in eye_mo:
                # Approximate the UE EyeBxDF caustic for the studio key only.
                # This scales the iris albedo and therefore also affects indirect
                # light; it cannot reproduce per-light shadowing or area integration.
                strength=float(eye_mo['unified_key_caustic_strength'])
                if not 0. <= strength <= 1. or request.get('lighting','studio') != 'studio':
                    raise ValueError('key caustic requires studio lighting and strength in [0, 1]')
                def math_node(op,a,b):
                    n=nt.nodes.new('ShaderNodeMath');n.operation=op
                    for index,value in enumerate((a,b)):
                        if isinstance(value,(int,float)):n.inputs[index].default_value=float(value)
                        else:nt.links.new(value,n.inputs[index])
                    return n.outputs[0]
                light_vec=nt.nodes.new('ShaderNodeVectorMath');light_vec.operation='SUBTRACT'
                light_vec.inputs[0].default_value=STUDIO_LIGHTS[0][1];nt.links.new(geom.outputs['Position'],light_vec.inputs[1])
                light_dir=nt.nodes.new('ShaderNodeVectorMath');light_dir.operation='NORMALIZE';nt.links.new(light_vec.outputs['Vector'],light_dir.inputs[0])
                def dot(a,b):
                    n=nt.nodes.new('ShaderNodeVectorMath');n.operation='DOT_PRODUCT';nt.links.new(a,n.inputs[0]);nt.links.new(b,n.inputs[1]);return n.outputs['Value']
                iris_nol=math_node('MAXIMUM',dot(plane.outputs['Vector'],light_dir.outputs['Vector']),0.)
                power=math_node('SUBTRACT',12.,math_node('MULTIPLY',11.,iris_nol))
                caustic_n=nt.nodes.new('ShaderNodeVectorMath');caustic_n.operation='SCALE'
                nt.links.new(geom.outputs['Normal'],caustic_n.inputs[0]);caustic_n.inputs['Scale'].default_value=-.25
                iris_n=nt.nodes.new('ShaderNodeVectorMath');iris_n.operation='SCALE'
                nt.links.new(plane.outputs['Vector'],iris_n.inputs[0]);iris_n.inputs['Scale'].default_value=.75
                caustic_sum=nt.nodes.new('ShaderNodeVectorMath');caustic_sum.operation='ADD'
                nt.links.new(caustic_n.outputs['Vector'],caustic_sum.inputs[0]);nt.links.new(iris_n.outputs['Vector'],caustic_sum.inputs[1])
                caustic_norm=nt.nodes.new('ShaderNodeVectorMath');caustic_norm.operation='NORMALIZE';nt.links.new(caustic_sum.outputs['Vector'],caustic_norm.inputs[0])
                caustic_dot=math_node('MAXIMUM',dot(caustic_norm.outputs['Vector'],light_dir.outputs['Vector']),0.)
                caustic=math_node('ADD',.8,math_node('MULTIPLY',.2,math_node('MULTIPLY',math_node('ADD',power,1.),math_node('POWER',caustic_dot,power))))
                factor=math_node('ADD',1.,math_node('MULTIPLY',strength,math_node('MULTIPLY',iris_mask.outputs['Result'],math_node('SUBTRACT',caustic,1.))))
                tint=nt.nodes.new('ShaderNodeVectorMath');tint.name='vhuman_key_caustic';tint.operation='SCALE';nt.links.new(color_out,tint.inputs[0]);nt.links.new(factor,tint.inputs['Scale'])
                nt.links.new(tint.outputs['Vector'],unode.inputs['Base Color'])
            gloss=nt.nodes.new('ShaderNodeBsdfGlossy');gloss.inputs['Roughness'].default_value=float(eye_mo.get('cornea_roughness',.075))
            nt.links.new(surface_normal,gloss.inputs['Normal'])
            if roughness_out is not None:nt.links.new(roughness_out,gloss.inputs['Roughness'])
            fresnel=nt.nodes.new('ShaderNodeFresnel');fresnel.inputs['IOR'].default_value=float(eye_mo.get('cornea_ior',1.41))
            nt.links.new(surface_normal,fresnel.inputs['Normal'])
            combined=nt.nodes.new('ShaderNodeMixShader');nt.links.new(fresnel.outputs['Fac'],combined.inputs[0])
            nt.links.new(unode.outputs['BSDF'],combined.inputs[1]);nt.links.new(gloss.outputs['BSDF'],combined.inputs[2])
            nt.links.new(combined.outputs[0],nt.nodes.get('Material Output').inputs['Surface'])
        if eye_mo.get('eye_per_side',0.) >= .5:
            # Copy the graph so the right eye may use independent fitted iris and sclera textures.
            right_material=unified.copy();right_material.name='eye_unified_right'
            if eye_mo.get('eye_bilateral_select',0.) >= .5 and eye_mo.get('eye_ue_split',0.) >= .5:
                right_params=json.loads((out/'eye_params_right.json').read_text())
                right_size=float(right_params['cornea']['size']);right_half=.5*float(right_params['cornea']['limbus_softness'])
                right_mask=right_material.node_tree.nodes.get('vhuman_iris_mask')
                if right_mask is None:raise RuntimeError('right eye is missing its iris mask')
                right_mask.inputs['From Min'].default_value=right_size-right_half
                right_mask.inputs['From Max'].default_value=right_size+right_half
                right_roughness=right_material.node_tree.nodes.get('vhuman_sclera_roughness')
                if right_roughness is not None:
                    left_size=float(json.loads((out/'eye_params.json').read_text())['cornea']['size'])
                    right_roughness.inputs['From Min'].default_value=.19+right_size-left_size
                    right_roughness.inputs['From Max'].default_value=.23+right_size-left_size
            replacements={'eye_unified_calibrated.png':'eye_unified_right.png',
                          'eye_unified_clean_bands_trans05.png':'eye_unified_right_bilateral.png',
                          'eye_unified_clean_ring_none.png':'eye_unified_right_bilateral.png',
                          'eye_unified_trans05.png':'eye_unified_right_trans05.png',
                          'eye_unified_radial_trans05.png':'eye_unified_right_radial_trans05.png',
                          'eye_unified_layered_trans05.png':'eye_unified_right_layered_trans05.png',
                          'eye_tissue_factor.png':'eye_tissue_factor_right.png',
                          'sclera_vessels_all.png':'sclera_vessels_all_right.png',
                          'sclera_vessels_curvy.png':'sclera_vessels_curvy_right.png',
                          'sclera_vessels_irregular.png':'sclera_vessels_irregular_right.png',
                          'sclera_vessels_dense.png':'sclera_vessels_dense_right.png',
                          'sclera_vessels_lines.png':'sclera_vessels_lines_right.png',
                          'sclera_vessels_sparse.png':'sclera_vessels_sparse_right.png',
                          'sclera_vessels_organic.png':'sclera_vessels_organic_right.png',
                          'sclera_vessels_organic_dense.png':'sclera_vessels_organic_dense_right.png',
                          'sclera_vessels_guided.png':'sclera_vessels_guided_right.png',
                          'sclera_vessels_nocross.png':'sclera_vessels_nocross_right.png',
                          'sclera_vessels_selfsafe.png':'sclera_vessels_selfsafe_right.png',
                          'eye_shell_normal.png':'eye_shell_normal_right.png',
                          'eye_surface05.png':'eye_surface05_right.png',
                          'eye_surface_fine.png':'eye_surface_fine_right.png'}
            replaced_base=False
            for texture_node in right_material.node_tree.nodes:
                if texture_node.type!='TEX_IMAGE' or texture_node.image is None:continue
                source=Path(texture_node.image.filepath).name
                if source not in replacements:raise RuntimeError(f'no right-eye texture mapping for {source}')
                path=out/replacements[source]
                if not path.is_file():raise RuntimeError(f'eye_per_side requires staged {path.name}')
                replacement=bpy.data.images.load(str(path),check_existing=True)
                replacement.colorspace_settings.name=texture_node.image.colorspace_settings.name
                replacement.pack();texture_node.image=replacement
                if source.startswith('eye_unified_'):replaced_base=True
            if not replaced_base:raise RuntimeError('eye_per_side could not find the right-eye base color texture')
            materials['eye_unified_right']=right_material
        materials['cornea']=materials['sclera']=unified
    materials['glass'],_=material('glass',(1,1,1),.025,1,1.5)
    materials['tear'],_=material('tear_film',(1,1,1),.02,1,1.336)
    # periocular study: caruncle / plica = wet pink mucosa (thin SSS + clear tear coat)
    materials['caruncle'],cnode=material('caruncle',(.38,.14,.13),.45)
    cnode.subsurface_method='RANDOM_WALK';cnode.inputs['Subsurface Weight'].default_value=.6
    cnode.inputs['Subsurface Radius'].default_value=(1.,.35,.25);cnode.inputs['Subsurface Scale'].default_value=.0006
    cnode.inputs['Coat Weight'].default_value=.5;cnode.inputs['Coat Roughness'].default_value=.12;cnode.inputs['Coat IOR'].default_value=1.336
    materials['hat'],principled=material('hat',(.015,.018,.025),.7)
    hattex=image_node(materials['hat'].node_tree.nodes,out/'accessory_source.png')
    materials['hat'].node_tree.links.new(hattex.outputs['Color'],principled.inputs['Base Color'])
    materials['hat_cloth'],_=material('inferred_navy_cap_cloth',(.015,.018,.025),.7)
    if (out/'hair_coverage.png').is_file():
        # One opaque surface avoids transparent shells and mesh-cut hairline steps.
        nodes=skin.node_tree.nodes;links=skin.node_tree.links
        coverage=image_node(nodes,out/'hair_coverage.png',linear=True)
        uv_source=nodes.new('ShaderNodeUVMap');uv_source.uv_map='SourceCamera'
        links.new(uv_source.outputs['UV'],coverage.inputs['Vector'])
        crown=nodes.new('ShaderNodeAttribute');crown.attribute_name='hair_crown_coverage'
        combine=nodes.new('ShaderNodeMath');combine.operation='MAXIMUM'
        links.new(coverage.outputs['Color'],combine.inputs[0]);links.new(crown.outputs['Fac'],combine.inputs[1])
        hair_base=nodes.new('ShaderNodeBsdfPrincipled');hair_base.label='Opaque short hair undercoat prior'
        hair_base.inputs['Base Color'].default_value=(*config['hair']['color_linear'],1)
        hair_base.inputs['Roughness'].default_value=.85;hair_base.inputs['IOR'].default_value=1.2
        mix=nodes.new('ShaderNodeMixShader')
        links.new(combine.outputs[0],mix.inputs[0]);links.new(node.outputs[0],mix.inputs[1])
        links.new(hair_base.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],nodes.get('Material Output').inputs['Surface'])
    objects={}
    for part in config['parts']:
        name=part['name']
        if eye_mo.get('eye_unified',0.) >= .5 and name in ('left_sclera','right_sclera'):
            continue
        if (('_tearline' in name) and eye_mo.get('peri_no_tearline',0.)>=.5) or \
           (name.endswith('_lid_margin') and eye_mo.get('peri_margin',0.)<.5) or \
           (part['material']=='caruncle' and not name.endswith('_lid_margin') and eye_mo.get('peri_caruncle',0.)<.5) or \
           (name.endswith('_tearline_upper') and eye_mo.get('peri_meniscus',0.)<.5):
            continue   # periocular study parts staged into a shared base: only the variants that enable them render them
        positions=data[name+'_positions'];triangles=data[name+'_triangles'];uvs=data[name+'_uvs']
        if eye_mo.get('eye_unified',0.) >= .5 and name in ('left_cornea','right_cornea'):
            other=name.replace('_cornea','_sclera')
            joined=np.concatenate((positions,data[other+'_positions']))
            triangles=np.concatenate((triangles,data[other+'_triangles']+len(positions)))
            positions,remap=np.unique(joined,axis=0,return_inverse=True)
            triangles=remap[triangles]
            uvs=np.concatenate((uvs,data[other+'_uvs']))
        mesh=bpy.data.meshes.new(name)
        mesh.from_pydata(positions.tolist(),[],triangles.tolist());mesh.update()
        uv=mesh.uv_layers.new(name='UVMap');values=uvs.reshape(-1,2).copy();values[:,1]=1-values[:,1]
        uv.data.foreach_set('uv',values.ravel())
        if eye_mo.get('eye_ue_split',0.) >= .5 and name in ('left_cornea','right_cornea'):
            other=name.replace('_cornea','_sclera');sclera_positions=np.asarray(data[other+'_positions'],np.float64)
            design=np.column_stack((2.*sclera_positions,np.ones(len(sclera_positions))))
            centre=np.linalg.lstsq(design,(sclera_positions*sclera_positions).sum(axis=1),rcond=None)[0][:3]
            axis=np.asarray(data[name+'_positions'][0],np.float64)-centre;axis/=np.linalg.norm(axis)
            attr=mesh.attributes.new('iris_plane_normal','FLOAT_VECTOR','POINT')
            attr.data.foreach_set('vector',np.broadcast_to(axis,(len(positions),3)).astype(np.float32).ravel())
        for key in ('enamel_tip','tongue_axis'):
            if name+'_'+key in data:
                attribute=mesh.attributes.new(key,'FLOAT','POINT');attribute.data.foreach_set('value',data[name+'_'+key].astype(np.float32))
        if name=='skin' and 'skin_profile_weight' in data:
            field=mesh.attributes.new('skin_profile_weight','FLOAT','POINT')
            field.data.foreach_set('value',data['skin_profile_weight'])
        if name=='skin' and eye_mo.get('peri_contact_shading'):
            if __package__:
                from .periocular import distance_to_segments
            else:
                sys.path.insert(0,str(Path(__file__).parent))
                from periocular import distance_to_segments
            paths=[]
            for side in ('left','right'):
                for margin in ('tearline','tearline_upper'):
                    key=f'{side}_{margin}_positions'
                    if key in data.files:
                        paths.append(np.asarray(data[key]).reshape(-1,8,3).mean(1))
            if not paths:raise RuntimeError('contact shading requires staged lid margins')
            points=np.asarray(data['skin_positions'])
            distances=np.minimum.reduce([distance_to_segments(points,path) for path in paths])
            field=mesh.attributes.new('peri_contact_distance','FLOAT','POINT')
            field.data.foreach_set('value',distances)
        if name=='skin' and 'skin_hair_crown' in data:
            field=mesh.attributes.new('hair_crown_coverage','FLOAT','POINT')
            field.data.foreach_set('value',data['skin_hair_crown'])
            source=mesh.uv_layers.new(name='SourceCamera');values=data['skin_source_uvs'].reshape(-1,2).copy();values[:,1]=1-values[:,1]
            source.data.foreach_set('uv',values.ravel())
            # Preserve atlas UV as default for albedo and signed physical maps.
            mesh.uv_layers.active_index=0;mesh.uv_layers[0].active_render=True
        obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
        use_right=eye_mo.get('eye_per_side',0.) >= .5 and name in ('right_cornea','right_sclera')
        mesh.materials.append(materials['eye_unified_right'] if use_right else materials[part['material']])
        objects[name]=obj
        for polygon in mesh.polygons:polygon.use_smooth=part['material'] not in ('glass',)
        for side in ('left','right'):   # eye_offset_<side>_m [dx,dy,dz]: move the eyeball parts (not lids/tearline)
            if f'eye_offset_{side}_m' in eye_mo and name in (f'{side}_cornea',f'{side}_sclera',f'{side}_iris',f'{side}_pupil_cup'):
                obj.location=tuple(float(x) for x in eye_mo[f'eye_offset_{side}_m'])
        if part['material'] in ('cornea','tear') and eye_mo.get('cornea_shadow',1.)<.5:
            obj.visible_shadow=False   # cornea_shadow 0: shadow rays pass the glass, so the iris gets direct light (no caustic-only lighting)
        if part['material'] in ('cornea','tear') and eye_mo.get('cornea_camera',1.)<.5:
            obj.visible_camera=False   # diagnostic: camera rays skip the glass (no cornea reflection/refraction in view)
        if part['material']=='skin':
            modifier=obj.modifiers.new('skin_subdivision','SUBSURF');modifier.levels=1;modifier.render_levels=2
            if skin_full and hasattr(modifier,'use_adaptive_subdivision'):
                # True displacement: dice to ~1 px (hero) / 2 px at render time instead of a fixed 2 levels.
                modifier.use_adaptive_subdivision=True;modifier.render_levels=3
                scene.cycles.dicing_rate=1. if request['preset']=='hero' else 2.;scene.cycles.preview_dicing_rate=8
                # Close-up cameras (ear/eye/mouth) dice the visible region at 1 px but kept the off-screen head
                # dense too and exceeded 14 GB on a 9070 XT: 2 px on-screen (still sub-mm at that distance), coarse
                # off-screen geometry and a hard subdivision cap.
                if str(request.get('camera','')).startswith(('ear_','eye_','mouth')):scene.cycles.dicing_rate=max(scene.cycles.dicing_rate,2.)
                scene.cycles.offscreen_dicing_scale=16.;scene.cycles.max_subdivisions=8
        if part['material']=='glass':
            modifier=obj.modifiers.new('lens_thickness_prior','SOLIDIFY');modifier.thickness=.001
    hair=bpy.data.materials.new('source_hair_color_prior');hair.use_nodes=True
    nodes=hair.node_tree.nodes;nodes.remove(nodes.get('Principled BSDF'))
    shader=nodes.new('ShaderNodeBsdfHairPrincipled');shader.parametrization='COLOR'
    shader.inputs['Color'].default_value=(*config['hair']['color_linear'],1)
    shader.inputs['Roughness'].default_value=.6 if config['hair']['short_hair'] else .35
    hair.node_tree.links.new(shader.outputs[0],nodes.get('Material Output').inputs[0])
    hair_object=curve_object('hair',data['hair_curves'],hair,.000035)
    if hair_object:objects['hair']=hair_object
    for name in config['curves']:
        tear=name.endswith('_tearline')
        objects[name]=curve_object(name,data[name],materials['tear' if tear else 'frame'],.00012 if tear else .001,
                                  cyclic=name.endswith('_frame'))
    if eye_mo.get('eye_unified',0.) >= .5:
        for side in ('left','right'):
            for part in ('iris','pupil_cup'):
                if objects.get(f'{side}_{part}'):objects[f'{side}_{part}'].hide_render=True
    if eye_mo.get('eye_model_layered2',0.) >= .5:eye2_layered(scene,objects,data,out,eye_mo)
    if eye_mo.get('eye_only',0.) >= .5:
        # Isolate the optical meshes for diagnostics, including their shadows and indirect contribution.
        keep={f'{side}_{part}' for side in ('left','right') for part in ('cornea','sclera','iris','pupil_cup')}
        keep|={name for name in objects if name.startswith('eye2_')}
        for name,obj in objects.items():
            if obj is not None and name not in keep:obj.hide_render=True
    camera=config['camera'];width,height=config['source_size']
    camera_data=bpy.data.cameras.new('portrait_camera');camera_obj=bpy.data.objects.new('portrait_camera',camera_data)
    bpy.context.collection.objects.link(camera_obj);scene.camera=camera_obj
    camera_data.sensor_fit='VERTICAL';camera_data.sensor_height=36;camera_data.lens=36*camera['focal']/height
    camera_data.shift_x=(width/2-camera['cx'])/height;camera_data.shift_y=(camera['cy']-height/2)/height
    rotation=np.asarray(camera['rotation']).T
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=camera['origin'];camera_obj.matrix_world=Matrix(transform.tolist())
    preset_name=request.get('camera','portrait');camera_record=dict(preset=preset_name,yaw=request.get('yaw',0),pitch=request.get('pitch',0))
    if preset_name!='portrait' or request.get('yaw') or request.get('pitch'):
        if __package__:
            from . import render_cameras
        else:
            sys.path.insert(0,str(Path(__file__).parent));import render_cameras
        eyes={side:data[f'{side}_cornea_positions'] for side in ('left','right') if f'{side}_cornea_positions' in data}
        ears={side:data[f'ear_{side}_positions'] for side in ('left','right') if f'ear_{side}_positions' in data}
        points=render_cameras.targets(data['skin_positions'],eyes=eyes,ears=ears)
        name='front' if preset_name=='portrait' else preset_name
        origin,rows,focal=render_cameras.place(name,camera['origin'],camera['rotation'],camera['focal'],points,
            yaw=request.get('yaw') if (request.get('yaw') or name=='front') else None,
            pitch=request.get('pitch') if (request.get('pitch') or name=='front') else None,
            distance_scale=request.get('distance_scale'),focal_scale=request.get('focal_scale'))
        # Close-ups aim at the part centre: drop the portrait's principal-point shift so the target is centred.
        if render_cameras.CAMERAS[name][4]!='head':camera_data.shift_x=0;camera_data.shift_y=0
        camera_data.lens=36*focal/height
        transform[:3,:3]=np.asarray(rows).T;transform[:3,3]=origin;camera_obj.matrix_world=Matrix(transform.tolist())
        camera_record.update(target=render_cameras.CAMERAS[name][4],origin=[float(v) for v in origin],focal_px=float(focal),
                             rotation=np.asarray(rows).tolist(),targets={k:[float(x) for x in v] for k,v in points.items()})
    explicit=request.get('explicit_camera')
    if explicit:   # metahuman.compare_render: exact shared pinhole camera, centred principal point
        camera_data.shift_x=0;camera_data.shift_y=0;camera_data.lens=36*explicit['focal_px']/explicit['size'][1]
        transform[:3,:3]=np.asarray(explicit['rotation']).T;transform[:3,3]=explicit['origin'];camera_obj.matrix_world=Matrix(transform.tolist())
        camera_record=dict(preset='explicit',origin=explicit['origin'],rotation=explicit['rotation'],focal_px=explicit['focal_px'],size=explicit['size'])
    camera_data.clip_start=.01;camera_data.clip_end=10
    lighting=request.get('lighting','studio')
    for name,position,power,size in ([] if lighting=='hdri' else STUDIO_LIGHTS):
        if lighting=='left':power={'key':45,'fill':2,'rim':8}[name]
        elif lighting=='right':
            position=(-position[0],position[1],position[2]);power={'key':45,'fill':2,'rim':8}[name]
        elif lighting=='rim':power={'key':3,'fill':3,'rim':55}[name]
        light=bpy.data.lights.new(name,'AREA');light.energy=power*STUDIO_ENERGY_SCALE
        if eye_mo.get('eye_model_layered2',0.) >= .5 and hasattr(light,'cycles') and hasattr(light.cycles,'is_caustics_light'):
            light.cycles.is_caustics_light=True   # MNEE: refracted light through the cornea onto the iris
        # The UE replica uses equal-area square RectLights; match their reflected shape for the eye diagnostic.
        light.shape='SQUARE' if eye_mo.get('eye_rect_lights',0.) >= .5 else 'DISK'
        light.size=size*math.sqrt(math.pi)/2 if light.shape=='SQUARE' else size
        obj=bpy.data.objects.new(name,light);bpy.context.collection.objects.link(obj);obj.location=position
        direction=(Vector((0,0,0))-obj.location).normalized()
        if eye_mo.get('eye_rect_align',0.) >= .5 and light.shape=='SQUARE':
            # vhuman's world-up is +Y (UE's +Z), while Blender's track helper uses +Z world-up.
            z=-direction;up=Vector((0,1,0))
            if abs(z.dot(up)) > .999:up=Vector((1,0,0))
            x=up.cross(z).normalized();y=z.cross(x).normalized()
            obj.rotation_euler=Matrix((x,y,z)).transposed().to_euler()
        else:
            obj.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()
    mouth_fill=float(request.get('mouth_fill',0.))
    oral=[objects[n] for n in ('mouth_cavity','upper_teeth_and_gums_teeth','upper_teeth_and_gums_gums','lower_teeth_and_gums_teeth',
                               'lower_teeth_and_gums_gums','tongue') if n in objects]
    if mouth_fill>0 and skin_full and 'mouth_cavity' in objects:   # quality candidates only: baseline A keeps its flat mouth
        # Light that reaches the teeth through the lips and cheeks. The head is a thin shell with no flesh behind the lips, so path
        # tracing gathers almost nothing in the mouth (the runtime oral-GI model in tmp/vhuman-mouth-gi measures ~2-5% indirect
        # irradiance there, but real mouths are lit by translucent tissue): a small warm point light at the cavity centroid,
        # light-linked to the oral parts only so it never lights the lips or face.
        centre=np.mean([v.co[:] for v in objects['mouth_cavity'].data.vertices],0)
        light=bpy.data.lights.new('mouth_fill','POINT');light.energy=mouth_fill;light.shadow_soft_size=.01
        light.color=tuple(float(c) for c in request.get('mouth_fill_color',(1.,.62,.48)))
        fill=bpy.data.objects.new('mouth_fill',light);bpy.context.collection.objects.link(fill);fill.location=tuple(float(c) for c in centre)
        receivers=bpy.data.collections.new('mouth_fill_receivers')
        for o in oral:receivers.objects.link(o)
        fill.light_linking.receiver_collection=receivers
        scene['mouth_fill']=json.dumps(dict(energy_w=mouth_fill,position=[float(c) for c in centre],model='translucent-tissue stand-in'))
    world=bpy.data.worlds.new('studio' if lighting!='hdri' else 'hdri');scene.world=world;world.use_nodes=True
    if lighting=='hdri':
        hdri=request['hdri'];wn=world.node_tree.nodes;wl=world.node_tree.links
        environment=wn.new('ShaderNodeTexEnvironment');environment.image=bpy.data.images.load(hdri['path'],check_existing=True)
        environment.image.pack()
        mapping=wn.new('ShaderNodeMapping');coords=wn.new('ShaderNodeTexCoord')
        mapping.inputs['Rotation'].default_value=(0,0,np.deg2rad(hdri.get('rotation_deg',0)))
        wl.new(coords.outputs['Generated'],mapping.inputs['Vector']);wl.new(mapping.outputs['Vector'],environment.inputs['Vector'])
        cal=CALIBRATED_HDRI.get(hdri.get('sha256'))
        if cal:   # neutralise the measured cast and exposure of the default HDRI (color_chart)
            tint=wn.new('ShaderNodeMix');tint.data_type='RGBA';tint.blend_type='MULTIPLY';tint.inputs['Factor'].default_value=1.
            tint.inputs[7].default_value=(cal[0]*cal[3],cal[1]*cal[3],cal[2]*cal[3],1.)   # RGBA 'B' (Mix has one A/B pair per type)
            wl.new(environment.outputs['Color'],tint.inputs[6]);wl.new(tint.outputs[2],wn['Background'].inputs['Color'])
            world['hdri_calibration']=json.dumps(dict(gains=cal[:3],scale=cal[3],source='color_chart 2026-11-01'))
        else:
            wl.new(environment.outputs['Color'],wn['Background'].inputs['Color'])
        wn['Background'].inputs['Strength'].default_value=hdri.get('strength',1.)
        world['hdri_sha256']=hdri['sha256'];world['hdri_license']=hdri.get('license','')
    else:
        world.node_tree.nodes['Background'].inputs['Color'].default_value=(*(STUDIO_WORLD[0],)*3,1)
        world.node_tree.nodes['Background'].inputs['Strength'].default_value=STUDIO_WORLD[1]
    scene['vhuman_provenance']='Single-portrait reconstruction; hidden anatomy, material amplitudes and accessories contain priors'
    scene['inferred_artifacts']=json.dumps(dict(accessories=config['accessories'],hair=config['hair'],
        detail=config['detail'],material=config['material']))
    if request.get('motion'):animate(scene,request,config,data,objects)
    if request.get('chart'):add_color_chart(scene,camera_obj,data,out,lighting)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'head.blend'))
    if request.get('scene_only',False):return
    scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='16'
    scene.render.filepath=str(out/'beauty.exr');bpy.ops.render.render(write_still=True)
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8'
    bpy.data.images['Render Result'].save_render(str(out/'beauty.png'),scene=scene)
    if noisy_output:
        # The File Output node appends the frame number; keep a stable name beside the denoised beauty.
        written=sorted(out.glob('beauty_noisy*.exr'))
        if written and written[-1].name!=noisy_output:written[-1].replace(out/noisy_output)
        if not (out/noisy_output).is_file():raise RuntimeError('noisy beauty output missing')
    result=dict(device=device_name,backend=request['device'],resolution=resolution,samples=scene.cycles.samples,
                seconds=time.perf_counter()-started,blend='head.blend',exr='beauty.exr',preview='beauty.png',noisy_exr=noisy_output,
                lighting=lighting,hdri=request.get('hdri'),camera=camera_record,preset=request['preset'],
                quality=request.get('quality','baseline'),ablate=request.get('ablate',[]),
                yaw=request.get('yaw',0),frame=scene.frame_current,exposure=scene.view_settings.exposure,sss_weight=request.get('sss_weight',.08))
    (out/'render_result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    path=Path(sys.argv[sys.argv.index('--')+1])
    run(json.loads(path.read_text()))
