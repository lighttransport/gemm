"""Pinned BiSeNet face parsing, explicitly executed on CPU."""
import numpy as np
from .face_assets import asset_path, sha256, PARSING_SHA256

LABELS = ('background', 'skin', 'left_brow', 'right_brow', 'left_eye', 'right_eye',
          'glasses', 'left_ear', 'right_ear', 'earring', 'nose', 'mouth', 'upper_lip',
          'lower_lip', 'neck', 'necklace', 'clothes', 'hair', 'hat')


class FaceParser:
    def __init__(self, model=None):
        import onnxruntime as ort
        model = model or asset_path('parsing')
        if sha256(model) != PARSING_SHA256:
            raise ValueError('verified face parsing weights required')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])

    def __call__(self, rgb):
        return self.predict(rgb)[0]

    def predict(self, rgb):
        """Return labels and softmax confidence without changing legacy callers."""
        import cv2
        rgb = np.asarray(rgb, dtype=np.uint8)
        image = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        image = (image - [.485, .456, .406]) / [.229, .224, .225]
        value = self.session.run(None, {self.session.get_inputs()[0].name:
                                      image.transpose(2, 0, 1)[None].astype(np.float32)})[0]
        labels = value[0].argmax(0).astype(np.uint8)
        logits = value[0] - value[0].max(0, keepdims=True)
        probability = np.exp(logits)
        confidence = probability.max(0) / probability.sum(0)
        size = (rgb.shape[1], rgb.shape[0])
        return (cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST),
                cv2.resize(confidence, size, interpolation=cv2.INTER_LINEAR))
