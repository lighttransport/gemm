"""Recheck a native video against an existing independent official reference.

Reuse requires identical model manifests, prompt, prepared portrait, dimensions,
preset and captured initial noise. Every reference tensor's recorded checksum is
verified. No inference is run; this permits precise regression checks without
recomputing an unchanged official chain. Both component and frame gates apply.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ref.hunyuan_video15.compare import compare, compare_frames
from ref.hunyuan_video15.convert_native import convert

NAMES = ('qwen_hidden','siglip_hidden','vae_encoded','dit_first','latent_final','vae_decoded')
PIN = '60783e704160023913bee78f0b47036d393d4dfa'

def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def audit(reference_run, reference_manifest, native, manifest):
    source=json.loads((reference_run/'parity.json').read_text())
    original=json.loads(reference_manifest.read_text())
    current=json.loads(manifest.read_text())
    if source['scope']!='independent_official_component_chain_with_matched_noise' or source['upstream_revision']!=PIN:
        raise ValueError('requires a pinned independent official component reference')
    if checksum(reference_manifest)!=source['generation_manifest_sha256']:
        raise ValueError('reference generation manifest checksum changed')
    for key in ('model','prompt','task','preset','frames','width','height','vision_profile','steps','cfg','flow_shift'):
        if original[key]!=current[key]:
            raise ValueError(f'reference reuse requires identical {key}')
    if ((current['task'],current['preset'],current['width'],current['height'])!=('i2v','fast12',480,848) or current['frames'] not in (81,121)):
        raise ValueError('unsupported pipeline dimensions/profile')
    if (current['steps'],current['cfg'],current['flow_shift'])!=(12,1,7):
        raise ValueError('unexpected fast12 sampler recipe')
    if checksum(manifest.parent/'input.png')!=source['image_sha256']:
        raise ValueError('prepared portrait differs')
    if checksum(native/'noise_input.f32')!=source['noise_sha256']:
        raise ValueError('initial noise differs')
    reference=reference_run/'reference'
    for name in NAMES:
        if checksum(reference/f'{name}.npy')!=source['reference_output_sha256'][name]:
            raise ValueError(f'reference checksum mismatch: {name}')
    convert(native)
    results=compare(reference,native,NAMES)
    frames=compare_frames(reference,native)
    frames_pass=all(v['pass'] for v in frames)
    return {'scope':'independent_official_component_chain_with_reused_reference',
            'upstream_revision':PIN,'reference_report_sha256':checksum(reference_run/'parity.json'),
            'reference_generation_manifest_sha256':checksum(reference_manifest),
            'generation_manifest_sha256':checksum(manifest),
            'reference_precision':source['reference_precision'],
            'shared_inputs':['prepared_portrait','prompt','initial_noise'],
            'reference_output_sha256':source['reference_output_sha256'],
            'native_output_sha256':{name:checksum(native/f'{name}.npy') for name in NAMES},
            'results':results,'frame_errors':frames,'frames_pass':frames_pass,
            'pass':all(v['pass'] for v in results.values()) and frames_pass}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-run',type=Path,required=True)
    ap.add_argument('--reference-generation-manifest',type=Path,required=True)
    ap.add_argument('--native',type=Path,required=True)
    ap.add_argument('--generation-manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=audit(args.reference_run,args.reference_generation_manifest,args.native,args.generation_manifest)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pass':result['pass'],'results':result['results'],
                      'failed_frames':[v for v in result['frame_errors'] if not v['pass']]},indent=2))
    return 0 if result['pass'] else 1

if __name__=='__main__':
    raise SystemExit(main())
