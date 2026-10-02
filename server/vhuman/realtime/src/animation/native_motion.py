"""Native causal motion adapter. Runtime requires NumPy and a C shared library."""
import ctypes as C
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .export_native import sha256

ROOT=Path(__file__).resolve().parents[5]


class MotionAdapter:
    def __init__(self, checkpoint, revision, device='cpu', allow_diagnostic=False, *, library=None):
        self.handle=None
        if device!='cpu':raise ValueError('the native motion adapter currently runs on CPU')
        source=Path(checkpoint);bundle=source if source.is_dir() else source.with_suffix('.native')
        if not (bundle/'native.json').is_file():
            raise ValueError('Native motion assets missing; run python -m server.vhuman.realtime.src.animation.export_native CHECKPOINT')
        data=json.loads((bundle/'native.json').read_text())
        if data.get('format')!='vhuman.native_motion.v1' or data.get('trained') is not True or data.get('tts_revision')!=revision:
            raise ValueError('checkpoint is untrained, incompatible, or uses different TTS weights')
        if data.get('purpose')!='production' and not (allow_diagnostic and data.get('purpose')=='diagnostic'):
            raise ValueError('diagnostic motion checkpoint requires explicit diagnostic mode')
        if source.is_file() and sha256(source)!=data['source_sha256']:raise ValueError('native motion export is stale')
        weights=bundle/'motion.safetensors'
        if sha256(weights)!=data['weights_sha256']:raise ValueError('motion weights checksum mismatch')
        names=data['names'];hidden=data['hidden_size'];bounds=np.asarray(data['ranges'],np.float32)
        if (type(hidden) is not int or not 1<=hidden<=16384 or not isinstance(names,list) or
                not 1<=len(names)<=512 or any(not isinstance(n,str) or not n for n in names) or
                len(set(names))!=len(names) or bounds.shape!=(len(names),2) or
                not np.isfinite(bounds).all() or (bounds[:,0]>bounds[:,1]).any()):
            raise ValueError('invalid motion metadata')
        self.model=SimpleNamespace(names=tuple(names),hidden_size=hidden)
        self.text_feed=data.get('text_feed','incremental')
        if self.text_feed not in ('incremental','full'):raise ValueError('unsupported text feed')
        path=Path(library) if library else ROOT/'cpu/vhuman/libvhuman_motion.so'
        if not path.is_file():raise RuntimeError('Build native motion with make -C cpu/vhuman libvhuman_motion.so')
        self.lib=C.CDLL(str(path.resolve()))
        self.lib.vh_motion_open.argtypes=[C.c_char_p,C.c_int,C.c_int];self.lib.vh_motion_open.restype=C.c_void_p
        self.lib.vh_motion_close.argtypes=[C.c_void_p];self.lib.vh_motion_close.restype=None
        self.lib.vh_motion_reset.argtypes=[C.c_void_p];self.lib.vh_motion_reset.restype=None
        self.lib.vh_motion_step.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.POINTER(C.c_int32),C.POINTER(C.c_float)]
        self.lib.vh_motion_step.restype=C.c_int
        self.handle=self.lib.vh_motion_open(str(weights.resolve()).encode(),hidden,len(names))
        if not self.handle:raise ValueError('invalid native motion tensors')
        self.device,self.revision='cpu',revision
        self.reset(0)

    def reset(self, epoch):
        if type(epoch) is not int or epoch<0:raise ValueError('invalid epoch')
        if not self.handle:raise RuntimeError('motion adapter is closed')
        self.lib.vh_motion_reset(self.handle)
        self.epoch,self.expected=epoch,0

    def push(self, features):
        from ..pipeline.protocol import MotionFrame
        if not self.handle:raise RuntimeError('motion adapter is closed')
        if features.epoch!=self.epoch:return []
        if features.model_revision!=self.revision or features.sample_start!=self.expected:
            raise ValueError('noncontiguous or incompatible TTS features')
        h=np.asarray(features.hidden);codes=np.asarray(features.codes)
        if (h.shape!=(self.model.hidden_size,) or not np.isfinite(h).all() or codes.shape!=(16,) or
                codes.dtype.kind not in 'iu' or (codes<0).any() or (codes>=2048).any()):
            raise ValueError('invalid motion feature arrays')
        h=np.ascontiguousarray(h,np.float32);codes=np.ascontiguousarray(codes,np.int32)
        values=np.empty((8,len(self.model.names)),np.float32)
        if self.lib.vh_motion_step(self.handle,h.ctypes.data_as(C.POINTER(C.c_float)),
                                  codes.ctypes.data_as(C.POINTER(C.c_int32)),values.ctypes.data_as(C.POINTER(C.c_float))):
            raise ValueError('native motion rejected nonfinite features/state')
        self.expected+=1920
        return [MotionFrame(self.epoch,features.sample_start+i*240,v) for i,v in enumerate(values)]

    def close(self):
        if self.handle:self.lib.vh_motion_close(self.handle);self.handle=None

    def __del__(self):
        self.close()
