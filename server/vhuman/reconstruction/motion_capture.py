"""Generate short identity-conditioned motion probes through native HIP adapters."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from .. import gpu
from ..face_assets import asset_path
from ..face_parsing import FaceParser
from ..native_landmarks import FaceLandmarker
from ..video_backend import select
from .observations import sha256
from .temporal import fit_clip

IDENTITY='Keep the same person, age, glasses, hat, skin and hairstyle. Keep the camera and illumination fixed. Silent portrait video. '
ACTIONS={
    'neutral_validation':'Maintain a relaxed neutral face, looking into the camera.',
    'smile_validation':'Develop a warm smile with raised cheeks and visible upper teeth.',
    'surprise_validation':'Raise the eyebrows and open the mouth in surprise.',
    'head_turn':'Turn the head gently about 20 degrees to the left while keeping the expression relaxed.',
    'blink':'Close both eyelids completely for a natural blink, then reopen the eyes. Keep the head still.',
    'gaze':'Keep the head still and move only the eyes gently to the left.',
    'pucker':'Keep the head still and pucker the lips forward with the mouth closed.'}
from ..rig.expression_catalog import EMOTIONS
ACTIONS.update(EMOTIONS)


def capture(candidate,out, *, backend='h3-fl2va',actions=None,preset='fast5',frames=None,seed=43,iterations=200):
    import cv2
    candidate,out=Path(candidate),Path(out);adapter=select(backend)
    if backend not in ('wan','h3','h3-fl2va','hv15-rocm'):raise ValueError('motion probes require a native ROCm I2V adapter')
    frames=frames or (81 if backend=='hv15-rocm' else 9 if backend=='wan' else 22)
    if frames not in adapter.frames or preset not in adapter.presets:raise ValueError('unsupported backend frame count/preset')
    names=list(actions or ('neutral_validation','head_turn','blink','gaze','pucker'))
    if not names or any(name not in ACTIONS for name in names):raise ValueError('unknown motion probe')
    out.mkdir(parents=True,exist_ok=True)
    size=(480,848 if backend=='hv15-rocm' else 832)
    reference=ImageOps.fit(Image.open(candidate/'portrait.png').convert('RGB'),size)
    path=out/'reference.png'
    if path.exists() and not np.array_equal(np.asarray(Image.open(path).convert('RGB')),np.asarray(reference)):
        raise ValueError('existing motion reference belongs to another portrait')
    reference.save(path)
    parser=FaceParser();results={}
    for index,name in enumerate(names):
        folder=out/name;folder.mkdir(exist_ok=True)
        prompt=IDENTITY+ACTIONS[name];clip=folder/'video/clip.mp4'
        if not clip.exists():
            print(f'{name}: native {backend} generation',flush=True)
            options=dict(image=path,prompt=prompt,out=folder/'video',preset=preset,frames=frames,
                         seed=seed+index,device=gpu.device_index(),allow_experimental=True)
            if backend.startswith('h3'):options['vram_budget_mib']=12288
            receipt=adapter.generate(**options)
            receipt['source_portrait_sha256']=sha256(path)
            (folder/'video/adapter_receipt.json').write_text(json.dumps(receipt,indent=2))
        generation=json.loads((folder/'video/manifest.json').read_text())
        expected_prompt=('The person in <Picture 1>. ' if backend=='h3' else '')+prompt
        if generation.get('seed')!=seed+index or generation.get('frames')!=frames or generation.get('prompt')!=expected_prompt:
            raise ValueError('existing generation settings changed')
        if not (folder/'observations.json').exists():
            cap=cv2.VideoCapture(str(clip));fps=cap.get(cv2.CAP_PROP_FPS);rows=[];labels=[]
            try:
                with FaceLandmarker(asset_path('mediapipe')) as tracker:
                    while True:
                        ok,bgr=cap.read()
                        if not ok:break
                        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);faces=tracker.detect(rgb)
                        rows.append(dict(frame=len(rows),visible=len(faces)==1,
                            landmarks=faces[0]['landmarks'].tolist() if len(faces)==1 else None,
                            measured_controls=faces[0]['blendshapes'] if len(faces)==1 else None))
                        labels.append(parser(rgb))
            finally:cap.release()
            if len(rows)!=frames:raise ValueError('generated frame count changed')
            np.savez_compressed(folder/'face_parsing.npz',labels=np.stack(labels))
            (folder/'observations.json').write_text(json.dumps(dict(source_sha256=sha256(clip),
                width=size[0],height=size[1],fps=fps,frames=frames,observations=rows),indent=2))
        motion=folder/'native_motion'
        if (motion/'motion.json').is_file():
            existing=json.loads((motion/'motion.json').read_text())
            if 'skin_min_oriented_area_ratio' not in existing:
                legacy=folder/'native_motion.legacy'
                if legacy.exists():raise ValueError('legacy motion already archived; inspect the incomplete probe')
                motion.rename(legacy)
        if not (motion/'motion.json').exists():
            with gpu.device_session(2048):
                results[name]=fit_clip(candidate,folder,motion,iterations=iterations,device=f'cuda:{gpu.device_index()}')
        else:results[name]=json.loads((motion/'motion.json').read_text())
        (out/'probes.json').write_text(json.dumps(dict(backend=backend,preset=preset,seed=seed,results=results),indent=2))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate');parser.add_argument('--out',required=True)
    parser.add_argument('--backend',choices=('wan','h3','h3-fl2va','hv15-rocm'),default='h3-fl2va')
    parser.add_argument('--actions',default='neutral_validation,head_turn,blink,gaze,pucker')
    parser.add_argument('--preset',default='fast5');parser.add_argument('--frames',type=int)
    parser.add_argument('--seed',type=int,default=43);parser.add_argument('--iterations',type=int,default=200)
    args=parser.parse_args();args.actions=args.actions.split(',')
    with gpu.execution('rocm',gpu.device_index()):capture(**vars(args))


if __name__=='__main__':main()
