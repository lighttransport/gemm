"""Blender-only worker; NPZ is the exact numeric bridge, USD is the interchange asset."""
import hashlib
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def to_blender(p):
    return np.asarray(p)[...,[0,2,1]]*np.array([1,-1,1])


def run(request):
    started=time.monotonic();out=Path(request['out']);candidate=Path(request['candidate'])
    for path,key in [(candidate/'geometry.npz','geometry_sha256'),(candidate/'skin_basecolor.png','basecolor_sha256'),(out/'surface.npz','surface_sha256')]:
        if digest(path)!=request[key]:raise ValueError('changed Blender input: '+str(path))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.temporary_directory=str(out/'cache')
    scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1.
    data=np.load(candidate/'geometry.npz',allow_pickle=False)
    positions=to_blender(data['captured'][0]);tri=data['triangles'];uv=data['triangle_uvs'].copy();uv[...,1]=1-uv[...,1]
    mesh=bpy.data.meshes.new('GNM_skin');mesh.from_pydata(positions.tolist(),[],tri.tolist());mesh.update()
    mesh.uv_layers.new(name='UVMap').data.foreach_set('uv',uv.reshape(-1))
    for face in mesh.polygons:face.use_smooth=True
    obj=bpy.data.objects.new('Skin',mesh);bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active=obj;obj.select_set(True)
    obj['source_geometry_sha256']=request['geometry_sha256'];obj['appearance_is_synthetic']=True
    mat=bpy.data.materials.new('Skin');mat.use_nodes=True;mesh.materials.append(mat)
    nodes=mat.node_tree.nodes;links=mat.node_tree.links;bsdf=nodes.get('Principled BSDF')
    bsdf.inputs['Roughness'].default_value=.55;bsdf.inputs['IOR'].default_value=1.4
    def texture(name,raw=False):
        node=nodes.new('ShaderNodeTexImage');node.image=bpy.data.images.load(str(candidate/name));node.image.pack()
        if raw:node.image.colorspace_settings.name='Non-Color'
        return node
    color=texture('skin_basecolor.png');links.new(color.outputs['Color'],bsdf.inputs['Base Color'])
    original_pixels=np.empty(len(color.image.pixels),np.float32);color.image.pixels.foreach_get(original_pixels)
    normal=texture('skin_normal.png',True);mapping=nodes.new('ShaderNodeNormalMap');links.new(normal.outputs['Color'],mapping.inputs['Color']);links.new(mapping.outputs['Normal'],bsdf.inputs['Normal'])
    orm=texture('skin_orm.png',True);separate=nodes.new('ShaderNodeSeparateColor');links.new(orm.outputs['Color'],separate.inputs['Color']);links.new(separate.outputs['Green'],bsdf.inputs['Roughness'])
    result=dict(schema='vhuman.blender_texture_result.v1',blender=bpy.app.version_string,geometry_sha256=request['geometry_sha256'],basecolor_sha256=request['basecolor_sha256'])
    if request['audit']:
        samples=np.load(out/'surface.npz',allow_pickle=False);points=to_blender(samples['points']);origin=Vector(to_blender(samples['camera_origin']))
        tree=BVHTree.FromPolygons(positions.tolist(),tri.tolist(),all_triangles=True)
        errors=np.full(len(points),np.inf,np.float32)
        for i,point in enumerate(points):
            direction=Vector(point)-origin;length=direction.length
            hit,_,_,distance=tree.ray_cast(origin,direction.normalized(),length+.002)
            if hit is not None:errors[i]=abs(distance-length)
        visible=errors<.00025
        np.savez_compressed(out/'visibility.npz',visible=visible,ray_error_m=errors)
        observed=samples['observed']
        result['visibility']=dict(tolerance_m=.00025,samples=len(points),observed=int(observed.sum()),
            observed_occluded=int((observed&~visible).sum()),visibility_sha256=digest(out/'visibility.npz'))
    # Blender Z-up is converted to the project's Y-up USD convention.
    bpy.ops.wm.usd_export(filepath=str(out/'head.usdc'),selected_objects_only=True,export_materials=True,
        export_textures_mode='NEW',relative_paths=True,convert_orientation=True,
        export_global_forward_selection='NEGATIVE_Z',export_global_up_selection='Y')
    result['usd_sha256']=digest(out/'head.usdc')
    scene.render.engine='CYCLES';scene.cycles.samples=request['samples'];scene.cycles.use_denoising=True
    if request['device']!='CPU':
        pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type=request['device'];pref.get_devices()
        devices=[d for d in pref.devices if d.type==request['device']]
        if not devices:raise RuntimeError('requested GPU device unavailable')
        for device in pref.devices:device.use=device==devices[0]
        scene.cycles.device='GPU';result['device']=devices[0].name+' '+request['device']
    else:scene.cycles.device='CPU';result['device']='CPU'
    scene.render.resolution_x=scene.render.resolution_y=request['resolution'];scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.view_settings.view_transform='AgX';scene.view_settings.look='None'
    scene.view_settings.exposure=-2.
    result['color_management']=dict(view='AgX',exposure_ev=-2.,lighting='fixed three-area studio; authored, not estimated')
    scene.world=bpy.data.worlds.new('Studio');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.3,.3,.3,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.3
    centre=(positions.min(0)+positions.max(0))*.5;extent=float(np.ptp(positions,axis=0).max())
    def point_at(item):item.rotation_euler=(Vector(centre)-item.location).to_track_quat('-Z','Y').to_euler()
    for name,offset,power,size in [('Key',(-.4,-.6,.6),35.,.5),('Fill',(.4,-.2,.3),15.,.5),('Rim',(.1,.5,.5),25.,.4)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
        item=bpy.data.objects.new(name,light);bpy.context.collection.objects.link(item);item.location=Vector(centre)+Vector(offset);point_at(item)
    camera=bpy.data.cameras.new('Review');camera.type='ORTHO';camera.ortho_scale=extent*1.12
    cam=bpy.data.objects.new('Review',camera);bpy.context.collection.objects.link(cam);scene.camera=cam
    renders=[]
    for name,direction in [('front',(0,-1,0)),('right',(1,0,0)),('back',(0,1,0)),('left',(-1,0,0)),('top',(0,.001,1))]:
        cam.location=Vector(centre)+Vector(direction)*extent*3;point_at(cam)
        if request['render']:
            scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True);renders.append(name+'.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'head.blend'))
    # Import USD into an empty scene and verify actual coordinates and corner UVs.
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.wm.usd_import(filepath=str(out/'head.usdc'))
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if len(meshes)!=1:raise ValueError('USD round trip changed mesh count')
    imported=meshes[0];actual=np.array([imported.matrix_world@v.co for v in imported.data.vertices])
    if actual.shape!=positions.shape or len(imported.data.polygons)!=len(tri):raise ValueError('USD topology size changed')
    # USD preserves vertex and face ordering for this unmodified triangulated mesh.
    err=float(np.max(abs(actual-positions)))
    actual_tri=np.array([p.vertices[:] for p in imported.data.polygons]);actual_uv=np.array([p.uv[:] for p in imported.data.uv_layers.active.data]).reshape(uv.shape)
    uv_err=float(np.max(abs(actual_uv-uv)))
    if err>1e-6 or not np.array_equal(actual_tri,tri) or uv_err>1e-6:raise ValueError(f'USD round trip changed geometry/UVs: {err}, {uv_err}')
    images=[n.image for n in imported.data.materials[0].node_tree.nodes if n.type=='TEX_IMAGE' and n.image and 'basecolor' in n.image.name]
    if len(images)!=1:raise ValueError('USD basecolor texture missing')
    texture_path=Path(bpy.path.abspath(images[0].filepath)).resolve()
    if not texture_path.is_relative_to(out) or not texture_path.is_file():raise ValueError('USD texture is not portable')
    actual_pixels=np.empty(len(images[0].pixels),np.float32);images[0].pixels.foreach_get(actual_pixels)
    if actual_pixels.shape!=original_pixels.shape:raise ValueError('USD texture resolution changed')
    texture_error=float(np.max(abs(actual_pixels-original_pixels)))
    if texture_error>1e-6:raise ValueError('USD texture pixels changed')
    result.update(usd_roundtrip=dict(max_coordinate_error_m=err,max_uv_error=uv_err,triangles_unchanged=True,
        max_texture_error=texture_error,texture=str(texture_path.relative_to(out))),renders=renders,seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':run(json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text()))
