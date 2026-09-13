"""Prepare a native GLB for the repository's common CPU preview renderer."""
import argparse
import io
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
from PIL import Image
from safetensors.numpy import save_file
p=argparse.ArgumentParser()
p.add_argument('path',type=Path)
p.add_argument('--renderer',type=Path,default=Path(__file__).resolve().parent/'.cache/preview_render')
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
raw=a.path.read_bytes();length,kind=struct.unpack_from('<II',raw,12)
assert kind==0x4e4f534a
scene=json.loads(raw[20:20+length]);blob=raw[28+length:]
def accessor(index):
    item=scene['accessors'][index];view=scene['bufferViews'][item['bufferView']]
    width={'SCALAR':1,'VEC2':2,'VEC3':3}[item['type']]
    dtype={5126:'<f4',5125:'<u4'}[item['componentType']]
    return np.frombuffer(blob,dtype=dtype,count=item['count']*width,offset=view.get('byteOffset',0)+item.get('byteOffset',0)).reshape(-1,width).copy()
primitive=scene['meshes'][0]['primitives'][0];attributes=primitive['attributes']
uv=accessor(attributes['TEXCOORD_0'])
# glTF image coordinates have V downward; common preview samples V upward.
uv[:,1]=1-uv[:,1]
vertices=accessor(attributes['POSITION'])
center=(vertices.min(0)+vertices.max(0))*.5
scale=100/np.ptp(vertices,axis=0).max()
# The common preview ray/triangle test has an absolute determinant epsilon.
# Scale only this disposable preview copy so 1/1024 triangles remain visible.
vertices=(vertices-center)*scale
(a.output_dir/'preview_transform.json').write_text(json.dumps(dict(center=center.tolist(),scale=float(scale))))
mesh=a.output_dir/'preview.safetensors'
save_file({'vertices':vertices,'faces':accessor(primitive['indices']).reshape(-1,3).astype(np.int32),
           'vertex_normals':accessor(attributes['NORMAL']),'uvs':uv},str(mesh))
material=scene['materials'][primitive['material']]['pbrMetallicRoughness']
image=scene['images'][scene['textures'][material['baseColorTexture']['index']]['source']]
view=scene['bufferViews'][image['bufferView']];offset=view.get('byteOffset',0)
texture=a.output_dir/'base.png'
Image.open(io.BytesIO(blob[offset:offset+view['byteLength']])).save(texture)
for yaw in [0,90,180,270]:
    subprocess.run([str(a.renderer.resolve()),'--in',str(mesh),'--texture',str(texture),'--out',str(a.output_dir/f'view-{yaw}.png'),
                    '-w','512','-h','512','--yaw',str(yaw),'--pitch','10','--frame-scale','2.1'],check=True)
views=[Image.open(a.output_dir/f'view-{yaw}.png').convert('RGB') for yaw in [0,90,180,270]]
canvas=Image.new('RGB',(1024,1024))
for i,view in enumerate(views):canvas.paste(view,((i%2)*512,(i//2)*512))
canvas.save(a.output_dir/'views.png')
print(a.output_dir/'views.png',flush=True)
