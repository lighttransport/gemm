"""Pinned MediaPipe face model inference through the repository C++ executor.

NumPy/OpenCV implement image sampling and detector postprocessing. No inference
framework is imported. Models and the landmark index/name mapping originate from
Google MediaPipe (Apache-2.0); see the source attribution beside the mapping.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import threading
import numpy as np
from .landmark_assets import TASK_SHA256, MODELS, export

ROOT = Path(__file__).resolve().parents[2]
# Mapping from MediaPipe v0.10.21 face_blendshapes_graph.cc, Apache-2.0.
# https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc
SUBSET = [0,1,4,5,6,7,8,10,13,14,17,21,33,37,39,40,46,52,53,54,55,58,61,63,65,66,67,70,78,80,
          81,82,84,87,88,91,93,95,103,105,107,109,127,132,133,136,144,145,146,148,149,150,152,
          153,154,155,157,158,159,160,161,162,163,168,172,173,176,178,181,185,191,195,197,234,
          246,249,251,263,267,269,270,276,282,283,284,285,288,291,293,295,296,297,300,308,310,
          311,312,314,317,318,321,323,324,332,334,336,338,356,361,362,365,373,374,375,377,378,
          379,380,381,382,384,385,386,387,388,389,390,397,398,400,402,405,409,415,454,466,
          468,469,470,471,472,473,474,475,476,477]
BLENDSHAPES = ['_neutral','browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight',
              'cheekPuff','cheekSquintLeft','cheekSquintRight','eyeBlinkLeft','eyeBlinkRight','eyeLookDownLeft',
              'eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight',
              'eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight',
              'jawForward','jawLeft','jawOpen','jawRight','mouthClose','mouthDimpleLeft','mouthDimpleRight',
              'mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft',
              'mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight',
              'mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft',
              'mouthSmileRight','mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight',
              'noseSneerLeft','noseSneerRight']


class Graph:
    def __init__(self, path, threads=4):
        self.handle = None
        self.lock = threading.RLock()
        library = ROOT/'cpu/vhuman/libvhuman_landmarks.so'
        if not library.is_file():
            raise RuntimeError('build with make -C cpu/vhuman libvhuman_landmarks.so')
        self.lib = C.CDLL(str(library))
        for name, args, result in (
            ('vh_face_open', [C.c_char_p, C.c_int], C.c_void_p),
            ('vh_face_error', [], C.c_char_p),
            ('vh_face_close', [C.c_void_p], None),
            ('vh_face_size', [C.c_void_p,C.c_int], C.c_size_t),
            ('vh_face_run', [C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t], C.c_int)):
            fn = getattr(self.lib,name); fn.argtypes=args; fn.restype=result
        self.handle = self.lib.vh_face_open(str(path).encode(),threads)
        if not self.handle:
            raise RuntimeError(self.lib.vh_face_error().decode())
        self.input_size = self.lib.vh_face_size(self.handle,0)
        self.output_size = self.lib.vh_face_size(self.handle,1)

    def __call__(self, value):
        value = np.ascontiguousarray(value,np.float32)
        if value.size != self.input_size or not np.isfinite(value).all():
            raise ValueError('invalid native landmark network input')
        with self.lock:
            if not self.handle:
                raise RuntimeError('native landmark network closed')
            out = np.empty(self.output_size,np.float32)
            if self.lib.vh_face_run(self.handle,value.ctypes.data,value.size,out.ctypes.data,out.size):
                raise RuntimeError(self.lib.vh_face_error().decode())
        return out

    def close(self):
        with self.lock:
            if self.handle:
                self.lib.vh_face_close(self.handle); self.handle=None

    def __del__(self):
        self.close()


def sample(image, center, size, angle, resolution, zero_border=False):
    import cv2
    corners = cv2.boxPoints((tuple(map(float,center)),tuple(map(float,size)),float(angle*180/np.pi)))
    target = np.array([[0,resolution],[0,0],[resolution,0],[resolution,resolution]],np.float32)
    matrix = cv2.getPerspectiveTransform(corners,target)
    crop = cv2.warpPerspective(image,matrix,(resolution,resolution),flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT if zero_border else cv2.BORDER_REPLICATE)
    return crop.astype(np.float32)/255


def detections(output, threshold=.5):
    boxes, logits = output[:896*16].reshape(896,16), output[896*16:]
    anchors=[]
    for grid,repeat in ((16,2),(8,6)):
        for y in range(grid):
            for x in range(grid):
                anchors.extend([[(x+.5)/grid,(y+.5)/grid]]*repeat)
    anchors=np.asarray(anchors,np.float32)
    scores=1/(1+np.exp(-np.clip(logits,-80,80)))
    ids=np.flatnonzero(scores>=threshold)
    ids=ids[np.argsort(-scores[ids],kind='stable')]
    decoded=boxes.copy()/128
    decoded[:,:2]+=anchors
    decoded[:,4:]+=np.tile(anchors,(1,6))
    rects=np.concatenate((decoded[:,:2]-decoded[:,2:4]/2,decoded[:,:2]+decoded[:,2:4]/2),axis=1)
    result=[]
    while len(ids):
        top=ids[0]; area=np.maximum(0,rects[:,2:]-rects[:,:2]).prod(1)
        intersection=np.maximum(0,np.minimum(rects[top,2:],rects[ids,2:])-np.maximum(rects[top,:2],rects[ids,:2])).prod(1)
        iou=intersection/np.maximum(area[top]+area[ids]-intersection,1e-12)
        selected=iou>.3;selected[0]=True
        overlap=ids[selected];weights=scores[overlap]
        result.append((np.sum(decoded[overlap]*weights[:,None],axis=0)/weights.sum(),float(scores[top])))
        ids=ids[~selected]
    return result


class FaceLandmarker:
    def __init__(self, task, *, assets=None, threads=4):
        self.graphs=[]
        task=Path(task)
        if hashlib.sha256(task.read_bytes()).hexdigest()!=TASK_SHA256:
            raise ValueError('verified MediaPipe task required')
        assets=Path(assets) if assets else task.parent/'native-face-v1'
        if not (assets/'manifest.json').is_file():
            export(task,assets)
        manifest=json.loads((assets/'manifest.json').read_text())
        if manifest.get('format')!='vhuman.native_landmarks.v1' or manifest.get('task_sha256')!=TASK_SHA256:
            raise ValueError('invalid native landmark receipt')
        try:
            for name in MODELS:
                path=assets/(name+'.bin')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['models'][name]['sha256']:
                    raise ValueError('native landmark asset checksum mismatch')
                self.graphs.append(Graph(path,threads))
        except Exception:
            self.close();raise

    def detect(self, image, *, blendshapes=True):
        image=np.asarray(image)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<2 or image.shape[0]*image.shape[1]>4_000_000:
            raise ValueError('expected RGB8 image up to 4M pixels')
        if len(self.graphs)!=3:
            raise RuntimeError('landmarker closed')
        h,w=image.shape[:2];side=max(h,w)
        tensor=sample(image,(w/2,h/2),(side,side),0,128,True)*2-1
        found=[]
        for box,score in detections(self.graphs[0](tensor))[:2]:
            center=box[:2]*side+np.array([(w-side)/2,(h-side)/2])
            size=box[2:4]*side*1.5
            if min(size)<=0:
                continue
            eye=box[6:8]-box[4:6];angle=np.arctan2(eye[1],eye[0])
            raw=self.graphs[1](sample(image,center,size,angle,256))
            presence=float(1/(1+np.exp(-np.clip(raw[1434],-80,80))))
            if presence<.5:
                continue
            points=raw[:1434].reshape(478,3).copy()/256
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            points[:,:2]=((points[:,:2]-.5)*size)@rotation.T+center
            points[:,2]*=size[0]
            values=self.graphs[2](points[SUBSET,:2]) if blendshapes else None
            points/=np.array([w,h,w])
            found.append(dict(landmarks=points,blendshapes={} if values is None else dict(zip(BLENDSHAPES,map(float,values))),
                              presence=presence,detection_score=score))
        return found

    def close(self):
        for graph in self.graphs:graph.close()
        self.graphs=[]

    def __enter__(self):return self
    def __exit__(self,*unused):self.close()
