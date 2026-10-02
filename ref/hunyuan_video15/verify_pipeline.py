"""Independent component-chain parity for a captured native fast12 I2V video (81 or 121 frames).

Reference Qwen/Google SigLIP default to FP32; --encoder-dtype float16 checks
the upstream activation precision independently.
VAE and DiT weights use FP16. Only initial noise is shared. Reference image/text
conditioning, all Euler steps and decoded pixels are computed independently.
This validates the assembled official components, not upstream generate.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ref.hunyuan_video15.compare import compare, compare_frames
from ref.hunyuan_video15.convert_native import convert

def checksum(path):
    result=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--native',type=Path,required=True,help='HV15_DUMP_DIR of a complete native run')
    ap.add_argument('--generation-manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument("--encoder-dtype", choices=("float32", "float16"), default="float32")
    args=ap.parse_args()
    generation=json.loads(args.generation_manifest.read_text())
    if ((generation['task'],generation['preset'],generation['width'],generation['height'])!=('i2v','fast12',480,848) or generation['frames'] not in (81,121)):
        raise ValueError('requires a captured fast12 480x848 I2V run with 81 or 121 frames')
    if generation['vision_profile']!='google_siglip_so400m_14_384' or '"' in generation['prompt']:
        raise ValueError('requires Google SigLIP and ordinary unquoted prompts')
    if args.threads<1:
        raise ValueError('threads must be positive')
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    convert(args.native)
    image=args.generation_manifest.resolve().parent/'input.png'
    model=args.model.resolve()
    common=['--model',str(model),'--upstream',str(args.upstream.resolve())]
    reference=out/'reference'
    reference.mkdir()
    stage='initialization'
    reports={}
    def run(name,script,argv):
        nonlocal stage
        stage=name
        print(f'PIPELINE_COMPONENT {name}',flush=True)
        target=out/name
        with (out/f'{name}.log').open('w') as log:
            completed=subprocess.run([sys.executable,str(ROOT/'ref/hunyuan_video15'/script),*argv,'--out',str(target)],
                                     stdout=log,stderr=subprocess.STDOUT)
        if (target/'parity.json').is_file():
            reports[name]=json.loads((target/'parity.json').read_text())
        completed.check_returncode()
        return target/'reference'
    report={'scope':'independent_official_component_chain_with_matched_noise',
            'upstream_revision':'60783e704160023913bee78f0b47036d393d4dfa',
            'generation_manifest_sha256':checksum(args.generation_manifest),
            'image_sha256':checksum(image),'noise_sha256':checksum(args.native/'noise_input.f32'),
            'shared_inputs':['prepared_portrait','prompt','initial_noise'],
            'reference_precision':{'qwen':args.encoder_dtype+'_cpu','google_siglip':args.encoder_dtype+'_cpu',
                                   'vae':'float16_autocast','dit_weights':'float16',
                                   'dit_conditioning_and_latents':'float32'},
            'quoted_glyphs':False,'pass':False}
    try:
        qwen=run('qwen','verify_qwen.py',['--model',str(model),'--config',str(model/'reference_configs/qwen'),
                 '--actual',str(args.native.resolve()),'--generation-manifest',str(args.generation_manifest.resolve()),
                 '--threads',str(args.threads),'--reference-dtype',args.encoder_dtype])
        vision=run('siglip','verify_siglip.py',['--model',str(model),'--image',str(image),
                   '--actual',str(args.native.resolve()),'--reference-device','cpu','--reference-dtype',args.encoder_dtype])
        vae=run('vae_encode','verify_vae.py',common+['--config',str(model/'vae/config.json'),'--image',str(image),
                 '--portrait','--encode-only','--actual',str(args.native.resolve()),'--reference-dtype','float16'])
        for name,source in [('qwen_hidden',qwen),('siglip_hidden',vision),('vae_encoded',vae)]:
            shutil.copyfile(source/f'{name}.npy',reference/f'{name}.npy')
        dit=run('denoise','verify_dit.py',common+['--config',str(model/'transformer/480p_i2v_step_distilled/config.json'),
                 '--native',str(args.native.resolve()),'--generation-manifest',str(args.generation_manifest.resolve()),
                 '--reference-inputs',str(reference),'--reference-dtype','float16','--full-denoise','--final-only'])
        for name in ('dit_first','latent_final'):
            shutil.copyfile(dit/f'{name}.npy',reference/f'{name}.npy')
        decode=run('vae_decode','verify_vae_decode.py',common+['--config',str(model/'vae/config.json'),
                    '--latent',str(args.native.resolve()),'--actual',str(args.native.resolve()),
                    '--frames',str(generation['frames']),
                    '--reference-latent-npy',str(dit/'latent_final.npy'),'--reference-dtype','float16'])
        shutil.copyfile(decode/'vae_decoded.npy',reference/'vae_decoded.npy')
        names=['qwen_hidden','siglip_hidden','vae_encoded','dit_first','latent_final','vae_decoded']
        report['results']=compare(reference,args.native,names)
        report['frame_errors']=compare_frames(reference,args.native)
        report['frames_pass']=all(v['pass'] for v in report['frame_errors'])
        report['reference_output_sha256']={name:checksum(reference/f'{name}.npy') for name in names}
        report['pass']=all(v['pass'] for v in report['results'].values()) and report['frames_pass']
    except subprocess.CalledProcessError as error:
        report['failed_stage']=stage
        report['exit_code']=error.returncode
        if stage=='vae_decode' and stage in reports:
            report['frame_errors']=reports[stage]['frame_errors']
            report['frames_pass']=reports[stage]['frames_pass']
        print(f'PIPELINE_FAILED {stage}: see {out/stage} and {out/(stage+".log")}',flush=True)
    report['components']=reports
    (out/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['pass'] else 1

if __name__=='__main__':
    raise SystemExit(main())
