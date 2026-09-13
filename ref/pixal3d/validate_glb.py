"""Validate native GLB structure, mesh attributes and embedded 4096 PBR images."""
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
np.testing.assert_allclose(np.linalg.norm(n,axis=1),1,atol=2e-4,rtol=0)
area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)*.5
assert (area>0).all(),'Zero-area GLB faces'
material=scene['materials'][0];assert material['alphaMode']=='OPAQUE' and not material['doubleSided']
pbr=material['pbrMetallicRoughness'];assert pbr['metallicFactor']==pbr['roughnessFactor']==1
images=[]
for item in scene['images']:
    assert item['mimeType']=='image/png'
    view=scene['bufferViews'][item['bufferView']];offset=view.get('byteOffset',0)
    image=Image.open(io.BytesIO(blob[offset:offset+view['byteLength']]))
    assert image.size==(4096,4096)
    data=np.asarray(image);assert data.max()>0
    images.append(data)
assert images[0].shape==(4096,4096,4) and images[1].shape==(4096,4096,3)
assert (images[1][:,:,0]==0).all()
print(json.dumps(dict(path=str(a.path),bytes=len(raw),vertices=len(v),triangles=len(f),
    bounds=[v.min(0).tolist(),v.max(0).tolist()],texture_size=4096,normal_max_error=float(abs(np.linalg.norm(n,axis=1)-1).max()),
    base_median=np.median(images[0].reshape(-1,4),axis=0).tolist(),material_median=np.median(images[1].reshape(-1,3),axis=0).tolist())),flush=True)
print('GLB PASS',flush=True)
