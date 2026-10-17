"""Read-only weight audit for the ROCm virtual-human workflow."""
import argparse
import json
from pathlib import Path
from server.model_paths import MODELS_ROOT

ROOT = Path(__file__).resolve().parents[2]


def audit(models_root=MODELS_ROOT, klein3_model='/mnt/nvme02/model/flux3-klein'):
    models = Path(models_root)
    required = {
        'qwen-image-2.1': models / 'qimg-21/model_index.json',
        'qwen-int8': models / 'qimg-21-fast/int8-smooth-a0.6/manifest.json',
        'pixal3d': models / 'Pixal3D/ckpts',
        'dinov3': models / 'dinov3-vitl16/model.safetensors',
        'moge2-native': models / 'moge-2-vitl/native/native.json',
        'moge2-backbone': models / 'moge-2-vitl/native/dinov2.safetensors',
        'moge2-heads': models / 'moge-2-vitl/native/heads.safetensors',
        'rmbg2': models / 'RMBG-2.0',
        'sam3': models / 'sam3/sam3.model.safetensors',
        'clip-bpe': models / 'clip-bpe/vocab.json',
        'sam3d-body': models / 'sam3d-body/safetensors/sam3d_body_dinov3_decoder.safetensors',
        'mhr': models / 'sam3d-body/safetensors/sam3d_body_mhr_jit.safetensors',
        'mhr-constants': models / 'sam3d-body/safetensors/sam3d_body_mhr_jit.json',
        'mhr-rig': models / 'sam3d-body/safetensors/sam3d_body_mhr_jit_rig.safetensors',
        'mhr-rig-manifest': models / 'sam3d-body/safetensors/sam3d_body_mhr_jit_rig.json',
        'tts': models / 'speech/Qwen3-TTS-12Hz-1.7B-CustomVoice',
        'ja-aligner': models / 'speech/japanese-wav2vec2-large-hiragana-ctc/ja_align.safetensors',
        'naf': ROOT / 'ref/pixal3d/weights/naf_release.safetensors',
        'gnm': models / 'vhuman-rig-models/gnm-v3/gnm_head.npz',
        'ict': models / 'vhuman-rig-models/ict-facekit/source/FaceXModel/generic_neutral_mesh.obj',
    }
    optional = {
        'portrait-depth': models / 'vhuman-rig-models/depth-anything-v2-small/native/native.json',
        'speech-emotion': models / 'vhuman/sensevoice-small-q8.gguf',
        'video-face-landmarks': models / 'vhuman-rig-models/face_landmarker.task',
    }
    def describe(paths):
        return {key: {'path': str(path), 'present': path.exists()} for key, path in paths.items()}
    klein = Path(klein3_model)
    stages = {
        'qwen-portraits': [required['qwen-image-2.1'], required['qwen-int8']],
        'klein3-portraits': [klein/'variants/fp8r/model.safetensors',
                            klein/'vae/diffusion_pytorch_model.safetensors',
                            klein/'text_encoder/model.safetensors.index.json'],
        'portrait-reconstruction': [required['gnm'], optional['video-face-landmarks'],
                                    models/'vhuman-rig-models/face-parsing/resnet18.onnx'],
        'pixal3d': [required['pixal3d'], required['dinov3']],
        'native-camera': [required['moge2-native'], required['moge2-backbone'], required['moge2-heads']],
        'body-rig': [required['mhr-rig'], required['mhr-rig-manifest']],
    }
    return {'models_root': str(models), 'required': describe(required), 'optional': describe(optional),
            'stages': {key: {'ready': all(p.exists() for p in paths),
                             'missing': [str(p) for p in paths if not p.exists()]}
                       for key, paths in stages.items()},
            'missing_required': [str(p) for p in required.values() if not p.exists()],
            'missing_optional': [str(p) for p in optional.values() if not p.exists()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', type=Path, default=MODELS_ROOT)
    parser.add_argument('--klein3-model', type=Path, default=Path('/mnt/nvme02/model/flux3-klein'))
    args = parser.parse_args()
    result = audit(args.models_root, args.klein3_model)
    print(json.dumps(result, indent=2))
    return bool(result['missing_required'])


if __name__ == '__main__':
    raise SystemExit(main())
