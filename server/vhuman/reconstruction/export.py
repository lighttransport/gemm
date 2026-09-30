"""Map explicit skin semantics to glTF without changing the rig format."""
import json
import io
import shutil
from PIL import Image
import numpy as np


def apply_material(skin, baked, subject, out):
    """Intensity is a linear alpha channel, per KHR_materials_specular."""
    manifest = json.loads((subject.folder/'skin_material.json').read_text())
    skin['gltf']['pbrMetallicRoughness']['metallicFactor'] = 0.
    # Explicit intensity; confidence/coverage remain sidecars, never AO.
    f0 = manifest['f0']['value']
    intensity = np.asarray(Image.open(baked['files']['rig_specular.png']).convert('L'))
    packed = np.full((*intensity.shape,4),255,np.uint8)
    packed[:,:,3] = intensity
    buffer = io.BytesIO();Image.fromarray(packed).save(buffer,format='PNG')
    skin['images'][3] = buffer.getvalue()
    skin['gltf']['extensions'] = {'KHR_materials_specular': {'specularFactor':1.,
        'specularColorFactor':[1.,1.,1.], 'specularTexture':{'index':3}}}
    skin['gltf']['extras'] = {'vhuman_skin_material':'vhuman.skin_material.v1',
        'sss':manifest['sss'], 'roughness_status':manifest['roughness']['status'], 'dielectric_f0':f0,
        'usd_specular_approximation':'constant prior F0; intensity texture and SSS remain sidecars'}
    shutil.copyfile(subject.folder/'skin_material.json',out/'skin_material.json')
    # Bake coverage/confidence separately through the same spatial/UV mapping.
    # bake.py emits these when present, keeping linear sampling and zero far confidence.


def bake_model_atlas(run, out, res):
    """Identical source topology/UVs: copy maps without a nearest-surface resnap."""
    from .reference import srgb_to_linear,linear_to_srgb
    files={}
    for channel in ('basecolor','orm','normal','specular','coverage','confidence'):
        path=run/f'skin_{channel}.png'
        with Image.open(path) as im:
            if im.size!=(res,res):
                if channel=='basecolor':
                    data=srgb_to_linear(np.asarray(im.convert('RGB'))/255.)
                    linear=np.stack([np.asarray(Image.fromarray(data[:,:,k].astype(np.float32)).resize((res,res),Image.Resampling.BILINEAR)) for k in range(3)],-1)
                    im=Image.fromarray(np.uint8(np.clip(linear_to_srgb(linear)*255+.5,0,255)))
                else:
                    im=im.resize((res,res),Image.Resampling.NEAREST if channel=='coverage' else Image.Resampling.BILINEAR)
            name=f'rig_{channel}.png';im.save(out/name);files[name]=str(out/name)
    return dict(files=files,stats=dict(res=res,method='identical model UVs; no surface projection',mean_transfer_mm=0,p95_transfer_mm=0))
