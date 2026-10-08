"""Run with Blender --background --python this_file -- --scene DIR --candidate DIR --out DIR."""
import bpy,json,hashlib,sys,argparse,shutil
from pathlib import Path
import numpy as np
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from server.vhuman.reconstruction.usd_materials import capture,restore,verify
parser=argparse.ArgumentParser(description='Export static complete anatomy to USD with verified Blender material sidecar')
parser.add_argument('--scene',required=True,help='offline_render directory with head.blend and scene.json')
parser.add_argument('--candidate',required=True,help='same geometry, optionally refined material')
parser.add_argument('--out',required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
scene_source,candidate,out=(Path(p).resolve() for p in (args.scene,args.candidate,args.out))
if out.exists() and any(out.iterdir()):raise ValueError('output directory must be empty')
config=json.loads((scene_source/'scene.json').read_text());original=Path(config['candidate'])
manifest=json.loads((candidate/'manifest.json').read_text());original_manifest=json.loads((original/'manifest.json').read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if (digest(candidate/'geometry.npz')!=manifest['geometry_sha256']
        or manifest['geometry_sha256']!=original_manifest['geometry_sha256']
        or digest(original/'geometry.npz')!=original_manifest['geometry_sha256']):
    raise ValueError('render scene and candidate geometry must match')
if manifest['portrait_sha256']!=original_manifest['portrait_sha256']:raise ValueError('portrait mismatch')
for name in ('skin_orm.png','skin_specular.png','skin_normal.png'):
    if digest(candidate/name)!=digest(original/name):raise ValueError('only basecolor refinement is supported; rebuild offline scene for other map changes')
out.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(scene_source/'head.blend'))
material=bpy.data.materials['skin'];nodes=[n for n in material.node_tree.nodes if n.type=='TEX_IMAGE' and n.image and 'skin_basecolor' in n.image.name];assert len(nodes)==1
nodes[0].image=bpy.data.images.load(str(candidate/'skin_basecolor.png'),check_existing=False);nodes[0].image.pack()
# Scene meshes are GNM Y-up. Express them in Blender Z-up before requesting
# the standard Blender-to-Y-up USD conversion.
axis=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)));snapshots={};materials={};modifiers={}
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    if obj.type!='MESH':continue
    obj.select_set(True);obj.matrix_world=axis@obj.matrix_world
    mesh=obj.data
    modifiers[obj.name]=[dict(name=m.name,type=m.type,levels=m.levels,render_levels=m.render_levels) for m in obj.modifiers if m.type=='SUBSURF']
    if any(m.type!='SUBSURF' for m in obj.modifiers):raise ValueError('unsupported object modifier in static export')
    snapshots[obj.name]=dict(vertices=np.array([obj.matrix_world@v.co for v in mesh.vertices]),triangles=np.array([p.vertices[:] for p in mesh.polygons]),uv=np.array([v.uv[:] for v in mesh.uv_layers.active.data]) if mesh.uv_layers.active else np.zeros((len(mesh.loops),2)))
    materials[obj.name]=[m.name for m in mesh.materials]
capture(list({m.name:m for obj in bpy.context.scene.objects if obj.select_get() for m in obj.data.materials}.values()),out)
bpy.ops.wm.usd_export(filepath=str(out/'head.usdc'),selected_objects_only=True,export_materials=True,export_subdivision='IGNORE',export_textures_mode='NEW',relative_paths=True,convert_orientation=True,export_global_forward_selection='NEGATIVE_Z',export_global_up_selection='Y',generate_materialx_network=True)
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.wm.usd_import(filepath=str(out/'head.usdc'),import_materials=True,import_textures_mode='IMPORT_NONE')
restore(out)
material_verification=verify(out)
reports=[]
for obj in bpy.context.scene.objects:
    if obj.type!='MESH':continue
    ref=snapshots[obj.name];mesh=obj.data;v=np.array([obj.matrix_world@x.co for x in mesh.vertices]);t=np.array([p.vertices[:] for p in mesh.polygons]);uv=np.array([x.uv[:] for x in mesh.uv_layers.active.data]) if mesh.uv_layers.active else np.zeros((len(mesh.loops),2))
    row=dict(name=obj.name,vertices=len(v),triangles=len(t),position_shape_same=v.shape==ref['vertices'].shape,topology_same=np.array_equal(t,ref['triangles']),uv_shape_same=uv.shape==ref['uv'].shape,materials=[m.name for m in mesh.materials],expected_materials=materials[obj.name])
    if row['position_shape_same']:row['max_position_error_m']=float(np.max(abs(v-ref['vertices'])))
    if row['uv_shape_same']:row['max_uv_error']=float(np.max(abs(uv-ref['uv'])))
    if (not row['position_shape_same'] or not row['topology_same'] or not row['uv_shape_same']
         or row['max_position_error_m']>1e-6 or row['max_uv_error']>1e-6
         or row['materials']!=row['expected_materials']):raise ValueError('USD roundtrip changed anatomy: '+obj.name)
    for spec in modifiers[obj.name]:
        modifier=obj.modifiers.new(spec['name'],spec['type']);modifier.levels=spec['levels'];modifier.render_levels=spec['render_levels']
    reports.append(row)
shaders={}
for m in bpy.data.materials:
    if not m.use_nodes:continue
    shader=next((n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
    if shader:shaders[m.name]={k:float(shader.inputs[k].default_value) for k in ('Roughness','IOR','Transmission Weight','Alpha','Subsurface Weight')}
if len(reports)!=len(snapshots):raise ValueError('USD roundtrip changed mesh count')
report=dict(schema='vhuman.full_anatomy_usd.v1',static=True,up_axis='Y',units='metres',
    source_scene_sha256=digest(scene_source/'head.blend'),scene_arrays_sha256=digest(scene_source/'scene_assets.npz'),candidate_geometry_sha256=manifest['geometry_sha256'],
    candidate_basecolor_sha256=digest(candidate/'skin_basecolor.png'),
    material_sidecar_sha256=digest(out/'blender_materials.json'),subdivision_modifiers=modifiers,
    limitations=['Static capture only; native expression/skeletal animation is not serialized',
                 'USD-only shader import may omit subsurface/displacement; use sidecar restoration in Blender'],
    material_verification=material_verification,meshes=reports,expected_meshes=len(snapshots),shaders=shaders,usd_sha256=hashlib.sha256((out/'head.usdc').read_bytes()).hexdigest())
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'reimported.blend'))

shutil.copyfile(candidate/'geometry.npz',out/'geometry.npz')
shutil.copyfile(scene_source/'scene_assets.npz',out/'scene_assets.npz')
shutil.copyfile(candidate/'manifest.json',out/'candidate_manifest.json')
