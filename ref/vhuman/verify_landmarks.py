"""Independent pinned TFLite tensor and MediaPipe image-pipeline checks.

This offline oracle requires ai-edge-litert and optionally MediaPipe. Native
inference and asset export require neither package.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from server.vhuman.native_landmarks import Graph, FaceLandmarker, MODELS
from server.vhuman.landmark_assets import export


def verify(task, output, images, mediapipe=False):
    from ai_edge_litert.interpreter import Interpreter
    from PIL import Image
    from unittest.mock import patch
    output.mkdir(parents=True,exist_ok=True)
    assets=output/'assets';export(task,assets)
    archive=zipfile.ZipFile(task)
    reports=[]
    class Oracle:
        def __init__(self,path,threads=4):
            self.model=Interpreter(model_content=archive.read(path.stem+'.tflite'),num_threads=threads)
            self.model.allocate_tensors()
            self.input=self.model.get_input_details()[0]
        def __call__(self,value):
            self.model.set_tensor(self.input['index'],np.asarray(value,np.float32).reshape(self.input['shape']))
            self.model.invoke()
            return np.concatenate([self.model.get_tensor(o['index']).ravel() for o in self.model.get_output_details()])
        def close(self):pass
    for name in MODELS:
        path=assets/(name+'.bin');oracle=Oracle(path);native=Graph(path)
        for seed in (7,19):
            x=np.random.default_rng(seed).uniform(-1 if name=='face_detector' else 0,1,oracle.input['shape']).astype(np.float32)
            started=time.monotonic();a=native(x);elapsed=time.monotonic()-started;b=oracle(x)
            error=np.abs(a-b)
            passed=bool(np.allclose(a,b,atol=.003,rtol=5e-4))
            reports.append(dict(model=name,seed=seed,max_error=float(error.max()),mean_error=float(error.mean()),seconds=elapsed,passed=passed))
            np.savez(output/f'{name}-{seed}.npz',input=x,native=a,reference=b)
        native.close()
    image_reports=[]
    with FaceLandmarker(task,assets=assets) as native:
        with patch('server.vhuman.native_landmarks.Graph',Oracle):
            reference=FaceLandmarker(task,assets=assets)
        for path in images:
            rgb=np.asarray(Image.open(path).convert('RGB'))
            a=native.detect(rgb);b=reference.detect(rgb)
            report=dict(image=str(path),native_faces=len(a),oracle_faces=len(b),passed=len(a)==len(b))
            if len(a)==len(b)==1:
                error=np.abs(a[0]['landmarks']-b[0]['landmarks'])
                blend=np.abs(np.array(list(a[0]['blendshapes'].values()))-list(b[0]['blendshapes'].values()))
                report.update(max_normalized_error=float(error.max()),blendshape_max_error=float(blend.max()))
                report['passed'] &= bool(error.max()<1e-4 and blend.max()<1e-3)
            if mediapipe:
                import types
                sys.modules.setdefault('mediapipe.tasks.python.audio',types.ModuleType('mediapipe.tasks.python.audio'))
                import mediapipe as mp
                options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=2,output_face_blendshapes=True)
                with mp.tasks.vision.FaceLandmarker.create_from_options(options) as tracker:
                    result=tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                report['mediapipe_faces']=len(result.face_landmarks)
                report['passed'] &= len(a)==len(result.face_landmarks)
                if len(a)==len(result.face_landmarks)==1:
                    points=np.array([[v.x,v.y,v.z] for v in result.face_landmarks[0]])
                    delta=a[0]['landmarks']-points
                    pixel=np.linalg.norm(delta[:,:2]*[rgb.shape[1],rgb.shape[0]],axis=1)
                    blend=np.abs(np.array(list(a[0]['blendshapes'].values()))-[c.score for c in result.face_blendshapes[0]])
                    report.update(mediapipe_pixel_max=float(pixel.max()),mediapipe_pixel_mean=float(pixel.mean()),mediapipe_blendshape_max=float(blend.max()))
                    report['passed'] &= bool(pixel.max()<.5 and blend.max()<.01)
                    np.savez(output/(path.stem+'-landmarks.npz'),native=a[0]['landmarks'],reference=points)
            image_reports.append(report)
        reference.close()
    report=dict(tensors=reports,images=image_reports,passed=all(r['passed'] for r in reports+image_reports))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--images',type=Path,nargs='*',default=[])
    parser.add_argument('--mediapipe',action='store_true')
    raise SystemExit(0 if verify(**vars(parser.parse_args())) else 1)
