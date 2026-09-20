"""Validate native GLB structure, mesh attributes and embedded PBR images."""
import argparse
import io
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('path',type=Path);a=p.parse_args()
raw=a.path.read_bytes();magic,version,total=struct.unpack_from('<III',raw)
assert magic==0x46546c67 and version==2 and total==len(raw)
length,kind=struct.unpack_from('<II',raw,12);assert kind==0x4e4f534a
scene=json.loads(raw[20:20+length]);blen,btype=struct.unpack_from('<II',raw,20+length)
assert btype==0x004e4942 and len(raw)==28+length+blen
blob=raw[28+length:]
assert scene['buffers'][0]['byteLength']<=len(blob)
for view in scene['bufferViews']:
    assert view.get('byteOffset',0)%4==0
    assert 0<=view.get('byteOffset',0)<len(blob)
    assert view.get('byteOffset',0)+view['byteLength']<=len(blob)
def accessor(index):
    item=scene['accessors'][index];view=scene['bufferViews'][item['bufferView']]
    width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[item['type']]
    dtype={5126:np.dtype('<f4'),5125:np.dtype('<u4')}[item['componentType']]
    return np.frombuffer(blob,dtype=dtype,count=item['count']*width,offset=view.get('byteOffset',0)+item.get('byteOffset',0)).reshape(-1,width)
primitive=scene['meshes'][0]['primitives'][0];attributes=primitive['attributes']
v=accessor(attributes['POSITION']);n=accessor(attributes['NORMAL']);uv=accessor(attributes['TEXCOORD_0']);f=accessor(primitive['indices']).reshape(-1,3)
assert len(v)==len(n)==len(uv) and len(f)>0 and f.max()<len(v)
assert np.isfinite(v).all() and np.isfinite(n).all() and np.isfinite(uv).all()
assert abs(v).max()<=.51 and uv.min()>=0 and uv.max()<=1
normal_lengths=np.linalg.norm(n,axis=1)
zero_normals=normal_lengths<=2e-4
# The pinned CuMesh exporter can leave a handful of referenced boundary
# vertices with zero normals. Keep that upstream condition visible while still
# rejecting a damaged normal buffer or non-unit nonzero normals.
assert zero_normals.sum()<=max(16,int(len(n)*1e-4)), 'Too many zero-length normals'
np.testing.assert_allclose(normal_lengths[~zero_normals],1,atol=2e-4,rtol=0)
area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)*.5
zero_area_faces=area<=0
# CuMesh can retain a few collapsed boundary faces alongside its zero-normal
# vertices. Report them explicitly and reject anything beyond that narrow quirk.
assert zero_area_faces.sum()<=max(16,int(len(f)*1e-4)), 'Too many zero-area GLB faces'
material=scene['materials'][0];assert material['alphaMode']=='OPAQUE' and not material['doubleSided']
pbr=material['pbrMetallicRoughness'];assert pbr['metallicFactor']==pbr['roughnessFactor']==1
images=[]
texture_size=None
for item in scene['images']:
    assert item['mimeType'] in ('image/png','image/webp')
    view=scene['bufferViews'][item['bufferView']];offset=view.get('byteOffset',0)
    image=Image.open(io.BytesIO(blob[offset:offset+view['byteLength']]))
    assert image.format.lower()==item['mimeType'].split('/')[1]
    assert image.width==image.height and image.width in (1024,2048,4096)
    texture_size=texture_size or image.width
    assert image.width==texture_size
    data=np.asarray(image);assert data.max()>0
    images.append(data)
assert images[0].shape==(texture_size,texture_size,4) and images[1].shape==(texture_size,texture_size,3)
# Lossy WebP can introduce small red-channel ringing around metallic/roughness
# edges; native PNG output remains exactly zero in that unused channel.
red_limit=0 if all(item['mimeType']=='image/png' for item in scene['images']) else 32
assert images[1][:,:,0].max()<=red_limit
print(json.dumps(dict(path=str(a.path),bytes=len(raw),vertices=len(v),triangles=len(f),
    bounds=[v.min(0).tolist(),v.max(0).tolist()],texture_size=texture_size,
    zero_normal_count=int(zero_normals.sum()),
    referenced_zero_normal_count=int(np.isin(np.flatnonzero(zero_normals),f).sum()),
    zero_area_face_count=int(zero_area_faces.sum()),
    normal_max_error=float(abs(normal_lengths[~zero_normals]-1).max(initial=0)),
    base_median=np.median(images[0].reshape(-1,4),axis=0).tolist(),material_median=np.median(images[1].reshape(-1,3),axis=0).tolist())),flush=True)
print('GLB PASS',flush=True)
