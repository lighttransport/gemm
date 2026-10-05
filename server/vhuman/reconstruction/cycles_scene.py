"""Blender worker: physical materials and AMD HIP offline head rendering.

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
            positions=motion['vertices'][:,data[part['name']+'_native_ids']]
        elif part.get('surface_bound'):
            name=part['name'];ids=data[name+'_surface_ids'];weights=data[name+'_surface_weights']
            roots=(motion['vertices'][:,ids]*weights[None,:,:,None]).sum(2)
            frames=attachment_frames(motion['vertices'][:,ids[:,:3]])
            positions=roots+np.einsum('fvij,vj->fvi',frames,data[name+'_surface_offsets'])
        elif part.get('joint') is not None:
            joint=part['joint'];p=data[part['name']+'_positions']-rest_joints[joint]
            positions=np.einsum('fij,vj->fvi',world[:,joint],p)+motion['joints'][:,joint,None]
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


def run(request):
    started=time.perf_counter();out=Path(request['out']);out.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.temporary_directory=str(out/'cache')
    (out/'cache').mkdir(exist_ok=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES'
    if request['device']=='hip':
        preferences=bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type='HIP';preferences.get_devices()
        devices=[d for d in preferences.devices if d.type=='HIP']
        if not devices:raise RuntimeError('Cycles found no HIP device; CPU rendering must be requested explicitly')
        for device in preferences.devices:device.use=device==devices[request.get('gpu_index',0)]
        if hasattr(preferences,'use_hiprt'):preferences.use_hiprt=False
        scene.cycles.device='GPU';device_name=devices[request.get('gpu_index',0)].name
    else:
        scene.cycles.device='CPU';device_name='CPU'
    scene.cycles.samples=32 if request['preset']=='draft' else 256
    scene.cycles.use_denoising=True;scene.cycles.denoiser='OPENIMAGEDENOISE'
    if hasattr(scene.cycles,'denoising_use_gpu'):scene.cycles.denoising_use_gpu=False
    scene.cycles.max_bounces=12;scene.cycles.transmission_bounces=8
    resolution=512 if request['preset']=='draft' else 1024
    scene.render.resolution_x=resolution;scene.render.resolution_y=resolution;scene.render.resolution_percentage=100
    scene.render.film_transparent=True
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
    node.inputs['Subsurface Weight'].default_value=.2
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
    for name,color,roughness in [('teeth',(.65,.6,.5),.25),('gums',(.35,.08,.07),.35),
        ('tongue',(.4,.09,.08),.4),('cavity',(.04,.009,.009),.65),('sclera',(.7,.65,.6),.25),
        ('iris',(.12,.065,.035),.55),('frame',(.12,.045,.025),.25),('pupil',(.001,.001,.001),1),('lash',(.025,.018,.012),.45)]:
        materials[name],principled=material(name,color,roughness)
        if name in ('gums','tongue'):
            principled.inputs['Subsurface Weight'].default_value=.1
            principled.inputs['Subsurface Radius'].default_value=(.0008,.0004,.0002)
            principled.inputs['Subsurface Scale'].default_value=1
        if name in ('sclera','iris'):
            texture=image_node(materials[name].node_tree.nodes,out/('sclera.png' if name=='sclera' else 'iris.png'))
            materials[name].node_tree.links.new(texture.outputs['Color'],principled.inputs['Base Color'])
    materials['cornea'],_=material('cornea',(1,1,1),.025,1,config['optical_eyes']['ior'])
    materials['glass'],_=material('glass',(1,1,1),.025,1,1.5)
    materials['tear'],_=material('tear_film',(1,1,1),.02,1,1.336)
    materials['hat'],principled=material('hat',(.015,.018,.025),.7)
    hattex=image_node(materials['hat'].node_tree.nodes,out/'accessory_source.png')
    materials['hat'].node_tree.links.new(hattex.outputs['Color'],principled.inputs['Base Color'])
    materials['hat_cloth'],_=material('inferred_navy_cap_cloth',(.015,.018,.025),.7)
    objects={}
    for part in config['parts']:
        name=part['name'];mesh=bpy.data.meshes.new(name)
        mesh.from_pydata(data[name+'_positions'].tolist(),[],data[name+'_triangles'].tolist());mesh.update()
        uv=mesh.uv_layers.new(name='UVMap');values=data[name+'_uvs'].reshape(-1,2).copy();values[:,1]=1-values[:,1]
        uv.data.foreach_set('uv',values.ravel())
        obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj);mesh.materials.append(materials[part['material']])
        objects[name]=obj
        for polygon in mesh.polygons:polygon.use_smooth=part['material'] not in ('glass',)
        if part['material']=='skin':
            modifier=obj.modifiers.new('skin_subdivision','SUBSURF');modifier.levels=1;modifier.render_levels=2
        if part['material']=='glass':
            modifier=obj.modifiers.new('lens_thickness_prior','SOLIDIFY');modifier.thickness=.001
    hair=bpy.data.materials.new('gray_hair_prior');hair.use_nodes=True
    nodes=hair.node_tree.nodes;nodes.remove(nodes.get('Principled BSDF'))
    shader=nodes.new('ShaderNodeBsdfHairPrincipled');shader.parametrization='COLOR'
    shader.inputs['Color'].default_value=(.35,.34,.31,1)
    shader.inputs['Roughness'].default_value=.35
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
    if request.get('yaw'):
        angle=np.deg2rad(request['yaw']);orbit=rotations_from_vectors([0,angle,0])
        pivot=np.array([0,.02,0])
        transform[:3,:3]=orbit@transform[:3,:3];transform[:3,3]=orbit@(transform[:3,3]-pivot)+pivot
        camera_obj.matrix_world=Matrix(transform.tolist())
    camera_data.clip_start=.01;camera_data.clip_end=10
    for name,position,power,size in [('key',(-.35,.35,.45),35,.35),('fill',(.35,.1,.35),12,.3),('rim',(.2,.3,-.3),20,.25)]:
        lighting=request.get('lighting','studio')
        if lighting=='left':power={'key':45,'fill':2,'rim':8}[name]
        elif lighting=='right':
            position=(-position[0],position[1],position[2]);power={'key':45,'fill':2,'rim':8}[name]
        elif lighting=='rim':power={'key':3,'fill':3,'rim':55}[name]
        light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
        obj=bpy.data.objects.new(name,light);bpy.context.collection.objects.link(obj);obj.location=position
        obj.rotation_euler=(Vector((0,0,0))-obj.location).to_track_quat('-Z','Y').to_euler()
    world=bpy.data.worlds.new('studio');scene.world=world;world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.1,.1,.1,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.15
    scene['vhuman_provenance']='Single-portrait reconstruction; hidden anatomy, material amplitudes and accessories contain priors'
    scene['inferred_artifacts']=json.dumps(dict(accessories=config['accessories'],hair=config['hair'],
        detail=config['detail'],material=config['material']))
    if request.get('motion'):animate(scene,request,config,data,objects)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'head.blend'))
    scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='16'
    scene.render.filepath=str(out/'beauty.exr');bpy.ops.render.render(write_still=True)
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8'
    bpy.data.images['Render Result'].save_render(str(out/'beauty.png'),scene=scene)
    result=dict(device=device_name,backend=request['device'],resolution=resolution,samples=scene.cycles.samples,
                seconds=time.perf_counter()-started,blend='head.blend',exr='beauty.exr',preview='beauty.png',
                lighting=request.get('lighting','studio'),yaw=request.get('yaw',0),frame=scene.frame_current)
    (out/'render_result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    path=Path(sys.argv[sys.argv.index('--')+1])
    run(json.loads(path.read_text()))
