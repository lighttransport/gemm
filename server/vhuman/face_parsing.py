"""Pinned BiSeNet parsing through repository C++/GEMM, without ONNX Runtime."""
import json
from pathlib import Path
import numpy as np
from .face_assets import asset_path, sha256, PARSING_SHA256

LABELS = ('background', 'skin', 'left_brow', 'right_brow', 'left_eye', 'right_eye',
          'glasses', 'left_ear', 'right_ear', 'earring', 'nose', 'mouth', 'upper_lip',
          'lower_lip', 'neck', 'necklace', 'clothes', 'hair', 'hat')


class FaceParser:
    def __init__(self, model=None, *, threads=4, assets=None):
        from .native_landmarks import Graph
        from .parsing_assets import export
        model = Path(model or asset_path('parsing'))
        if sha256(model) != PARSING_SHA256:
            raise ValueError('verified face parsing weights required')
        if type(threads) is not int or not 1<=threads<=256: raise ValueError('threads must be 1..256')
        native=Path(assets) if assets is not None else model.with_suffix('.native.bin')
        receipt=native.with_name(native.name+'.json')
        if not receipt.is_file():export(model,native)
        record=json.loads(receipt.read_text())
        if (record.get('format')!='vhuman.native_parsing.v1' or record.get('source_sha256')!=PARSING_SHA256
                or record.get('sha256')!=sha256(native)):
            raise ValueError('native face parsing asset checksum mismatch')
        self.graph=Graph(native,threads)

    def __call__(self, rgb):
        return self.predict(rgb)[0]

    def predict(self, rgb):
        """Return labels and softmax confidence without changing legacy callers."""
        from .native_face import resize
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<2 or rgb.shape[0]*rgb.shape[1]>4000000:
            raise ValueError('expected RGB8 image up to 4M pixels')
        image = resize(rgb,(512,512),quantize=True) / 255
        image = (image - [.485, .456, .406]) / [.229, .224, .225]
        value = self.graph(image.transpose(2,0,1)[None]).reshape(1,19,512,512)
        labels = value[0].argmax(0).astype(np.uint8)
        logits = value[0] - value[0].max(0, keepdims=True)
        probability = np.exp(logits)
        confidence = probability.max(0) / probability.sum(0)
        size = (rgb.shape[1], rgb.shape[0])
        return (resize(labels,size,nearest=True).astype(np.uint8),resize(confidence,size))

    def close(self):self.graph.close()
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()
