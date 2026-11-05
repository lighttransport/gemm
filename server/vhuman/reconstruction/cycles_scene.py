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
    # normal_ear_scale (per-texel skin normal strength on the ears; skin_normal_map only)
    mo=dict(request.get('material_overrides') or {})
    r=math('SUBTRACT',r,math('MULTIPLY',lips,.12),clamp=True)
    if mo.get('roughness_scale',1.)!=1.:r=math('MULTIPLY',r,float(mo['roughness_scale']),clamp=True)
    links.new(r,node.inputs['Roughness'])
    # specular level = template*(1-.7 cavity)
    sp=math('MULTIPLY',spec,math('SUBTRACT',1.,math('MULTIPLY',cavity,.7)),clamp=True)
    if mo.get('specular_scale',1.)!=1.:sp=math('MULTIPLY',sp,float(mo['specular_scale']),clamp=True)
    links.new(sp,node.inputs['Specular IOR Level'])
    # second (sharper) lobe kept small so skin reads oily, not wet: coat .10(1-cavity)+.35 lips,
    # roughness .30 -> .12 on lips (the moist lip band)
    coat=math('ADD',math('MULTIPLY',math('SUBTRACT',1.,cavity),float(mo.get('skin_coat',.10))),math('MULTIPLY',lips,.35*float(request.get('lip_coat_scale',1.))),clamp=True);links.new(coat,node.inputs['Coat Weight'])
    cr=math('SUBTRACT',.30,math('MULTIPLY',lips,.30-float(request.get('lip_coat_roughness',.12))),clamp=True);links.new(cr,node.inputs['Coat Roughness'])
    node.inputs['Coat IOR'].default_value=1.4
    # random-walk skin subsurface (radii in metres, literature-scale skin profile)
    node.subsurface_method='RANDOM_WALK_SKIN'
    node.inputs['Subsurface Weight'].default_value=float(mo.get('subsurface_weight',.6));node.inputs['Subsurface Scale'].default_value=float(mo.get('subsurface_scale',.5))
    if mo:skin['material_overrides']=json.dumps(mo)
    # subsurface_radius override: per-channel mean free path in mm (e.g. UE SSP_skin_unified 1.05/.41/.26)
    node.inputs['Subsurface Radius'].default_value=tuple(float(x)*1e-3 for x in mo['subsurface_radius']) if 'subsurface_radius' in mo else (.00367,.00137,.00068)
    node.inputs['Subsurface IOR'].default_value=1.4
    # tiled micro normal (pores), strength scaled by the region mask
    micro_img=image_node(nodes,out/'skin_micro_normal.png',linear=True)
    uvn=nodes.new('ShaderNodeUVMap');uvn.uv_map='UVMap';mapping=nodes.new('ShaderNodeMapping');mapping.inputs['Scale'].default_value=(12,12,1)
    links.new(uvn.outputs['UV'],mapping.inputs['Vector']);links.new(mapping.outputs['Vector'],micro_img.inputs['Vector'])
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
        k=math('MULTIPLY',micro,.6*float(request.get('micro_scale',1.)));strength=float(request.get('skin_normal_strength',1.))
        st=strength
        if mo.get('normal_ear_scale',1.)!=1.:
            # fade the per-texel map on the ears: s * (1 - (1 - ear_scale) * ear_weight), ear weight from skin_templates
            if 'ears' not in maps.files:raise RuntimeError('normal_ear_scale needs skin_templates with an ears map; rebuild them')
            ear=float_image(nodes,'template_ears',maps['ears']);esep=nodes.new('ShaderNodeSeparateColor');links.new(ear.outputs['Color'],esep.inputs['Color'])
            st=math('MULTIPLY',math('SUBTRACT',1.,math('MULTIPLY',esep.outputs['Red'],1.-float(mo['normal_ear_scale']))),strength)
        c=nodes.new('ShaderNodeCombineXYZ')
        links.new(math('ADD',math('MULTIPLY',sa.outputs['X'],st),math('MULTIPLY',sb.outputs['X'],k)),c.inputs['X'])
        links.new(math('ADD',math('MULTIPLY',sa.outputs['Y'],st),math('MULTIPLY',sb.outputs['Y'],k)),c.inputs['Y'])
        links.new(math('MULTIPLY',sa.outputs['Z'],sb.outputs['Z']),c.inputs['Z'])
        n=vec('NORMALIZE',c.outputs[0]);col=vec('MULTIPLY_ADD',n,(.5,.5,.5));col.node.inputs[2].default_value=(.5,.5,.5)
        links.new(col,nm.inputs['Color']);nm.inputs['Strength'].default_value=1.
        skin['skin_normal_map']=json.dumps(dict(path=str(skin_normal),strength=strength,blend='whiteout with tiled pores'))
    else:
        links.new(micro_img.outputs['Color'],nm.inputs['Color']);links.new(math('MULTIPLY',micro,.6*float(request.get('micro_scale',1.))),nm.inputs['Strength'])
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
    # Specular IOR Level, iris_gain (iris texture x gain); absent = the authored values.
    eye_mo=dict(request.get('material_overrides') or {})
    for name,color,roughness in [('teeth',(.65,.6,.5),.25),('gums',(.35,.08,.07),.35),
        ('tongue',(.4,.09,.08),.4),('cavity',(.33,.09,.09),.45),('sclera',(.7,.65,.6),.25),
        ('iris',(.12,.065,.035),.55),('frame',(.12,.045,.025),.25),('pupil',(.001,.001,.001),1),('lash',(.025,.018,.012),.45)]:
        materials[name],principled=material(name,color,float(eye_mo.get(f'{name}_roughness',roughness)) if name in ('sclera','iris') else roughness)
        if name in ('gums','tongue'):
            principled.inputs['Subsurface Weight'].default_value=.1
            principled.inputs['Subsurface Radius'].default_value=(.0008,.0004,.0002)
            principled.inputs['Subsurface Scale'].default_value=1
        if name in ('sclera','iris'):
            texture=image_node(materials[name].node_tree.nodes,out/('sclera.png' if name=='sclera' else 'iris.png'))
            color_out=texture.outputs['Color']
            if name=='iris' and 'iris_gain' in eye_mo:   # iris base-colour brightness test (absent = no extra node)
                gain=materials[name].node_tree.nodes.new('ShaderNodeVectorMath');gain.operation='SCALE'
                gain.inputs['Scale'].default_value=float(eye_mo['iris_gain'])
                materials[name].node_tree.links.new(color_out,gain.inputs[0]);color_out=gain.outputs['Vector']
            materials[name].node_tree.links.new(color_out,principled.inputs['Base Color'])
    if full and '2e' not in ablate:full_mouth_materials(materials,out,request)
    materials['cornea'],cornea=material('cornea',(1,1,1),float(eye_mo.get('cornea_roughness',.025)),1,
                                        float(eye_mo.get('cornea_ior',config['optical_eyes']['ior'])))
    if 'cornea_specular' in eye_mo:cornea.inputs['Specular IOR Level'].default_value=float(eye_mo['cornea_specular'])
    materials['glass'],_=material('glass',(1,1,1),.025,1,1.5)
    materials['tear'],_=material('tear_film',(1,1,1),.02,1,1.336)
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
        name=part['name'];mesh=bpy.data.meshes.new(name)
        mesh.from_pydata(data[name+'_positions'].tolist(),[],data[name+'_triangles'].tolist());mesh.update()
        uv=mesh.uv_layers.new(name='UVMap');values=data[name+'_uvs'].reshape(-1,2).copy();values[:,1]=1-values[:,1]
        uv.data.foreach_set('uv',values.ravel())
        for key in ('enamel_tip','tongue_axis'):
            if name+'_'+key in data:
                attribute=mesh.attributes.new(key,'FLOAT','POINT');attribute.data.foreach_set('value',data[name+'_'+key].astype(np.float32))
        if name=='skin' and 'skin_hair_crown' in data:
            field=mesh.attributes.new('hair_crown_coverage','FLOAT','POINT')
            field.data.foreach_set('value',data['skin_hair_crown'])
            source=mesh.uv_layers.new(name='SourceCamera');values=data['skin_source_uvs'].reshape(-1,2).copy();values[:,1]=1-values[:,1]
            source.data.foreach_set('uv',values.ravel())
            # Preserve atlas UV as default for albedo and signed physical maps.
            mesh.uv_layers.active_index=0;mesh.uv_layers[0].active_render=True
        obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj);mesh.materials.append(materials[part['material']])
        objects[name]=obj
        for polygon in mesh.polygons:polygon.use_smooth=part['material'] not in ('glass',)
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
        light=bpy.data.lights.new(name,'AREA');light.energy=power*STUDIO_ENERGY_SCALE;light.shape='DISK';light.size=size
        obj=bpy.data.objects.new(name,light);bpy.context.collection.objects.link(obj);obj.location=position
        obj.rotation_euler=(Vector((0,0,0))-obj.location).to_track_quat('-Z','Y').to_euler()
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
