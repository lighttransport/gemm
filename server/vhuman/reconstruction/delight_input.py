"""Delit texture sources (stage 2a): Qwen-Image-Edit-2511 removes lighting from the views used for baking.

The delit copies are used ONLY as texture sources (pipeline --texture-images); landmarks, cameras and geometry
keep the original photographs. The conservative SH/low-order inverse lighting inside materials.bake_portrait
then acts on the residual low-frequency shading left after delighting. Adapter: the Light_restoration LoRA
(trained for Edit-2509) if it binds and passes the compatibility check, else text-prompted delighting.
Acceptance per view: ArcFace >= arcface_min against the original, skin-mask L* std reduced by >= min_shading_drop
(fraction) and skin median hue drift <= max_hue_shift degrees. Rejected views keep the original.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .view_acceptance import ArcFace, srgb_to_lab

LORAS = {
    'v2': '/mnt/nvme01/models/loras/qie2509-light-restoration-v2.safetensors',
    'v1': '/mnt/nvme01/models/loras/qie2509-light-restoration-v1.safetensors',
}
PROMPT = ('Remove all lighting and shadows from this photo of the person: flat, even, shadowless studio lighting as '
          'if lit by a softbox from every direction, no specular highlights, no cast shadows, no light gradients '
          'across the skin. Keep the exact same person, pose, framing, hair, skin colour and every skin detail; '
          'keep the plain light grey background. Photorealistic, sharp.')
NEGATIVE = 'shadows, highlights, dramatic lighting, rim light, blur, different person, makeup, retouching, text'
THRESHOLDS = dict(arcface_min=.6, min_shading_drop=.2, max_hue_shift_deg=6.)


def skin_stats(rgb, parser):
    from ..face_parsing import LABELS
    labels, confidence = parser.predict(rgb)
    skin = (labels == LABELS.index('skin')) & (confidence > .6)
    if skin.sum() < 500:
        return None
    lab = srgb_to_lab(np.asarray(rgb)[skin])
    med = np.median(lab, 0)
    return dict(l_std=float(lab[:, 0].std()), l_mean=float(lab[:, 0].mean()), median_lab=med.tolist(),
                hue_deg=float(np.degrees(np.arctan2(med[2], med[1]))), pixels=int(skin.sum()))


def delight(inputs, out, *, lora='v2', steps=20, seed=4242, offload_blocks=1, parsing_model=None, thresholds=THRESHOLDS,
            editor=None, arcface=None, only_yaws=None):
    from ..face_parsing import FaceParser
    from .qwen_edit_backend import make_editor
    inputs = Path(inputs); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(inputs.read_text())
    views = [v for v in bundle['views'] if not v.get('held_out')]
    if only_yaws is not None:
        views = [v for v in views if v['yaw_deg'] in only_yaws]
    parser = FaceParser(model=parsing_model); arcface = arcface or ArcFace()
    lora_spec = None if lora == 'none' else [f'{LORAS[lora]}:1.0:light_restoration_{lora}']
    report = dict(format='vhuman.delight_input.v1', inputs=str(inputs.resolve()), adapter=lora, prompt=PROMPT, negative=NEGATIVE,
                  steps=steps, seed=seed, thresholds=thresholds, views=[], texture_images={})
    if editor is None:
        try:
            editor = make_editor('gguf', offload_blocks=offload_blocks, loras=lora_spec)
            report['adapter_status'] = 'bound' if lora_spec else 'prompt only'
        except Exception as exc:
            report['adapter_status'] = f'LoRA failed to bind ({exc}); prompt only'
            editor = make_editor('gguf', offload_blocks=offload_blocks)
    report['editor'] = editor.metadata
    for view in views:
        src = inputs.parent / view['file']
        original = np.asarray(Image.open(src).convert('RGB'))
        target = out / f"{view['yaw_deg']}.png"
        if not target.is_file():
            result, seconds = editor([Image.fromarray(original)], PROMPT, steps=steps, seed=seed + int(view['yaw_deg']) % 97, cfg=4., size=1024, negative=NEGATIVE)
            Image.fromarray(result).save(target)
        delit = np.asarray(Image.open(target).convert('RGB'))
        e0, _, _ = arcface.embed(original); e1, _, _ = arcface.embed(delit)
        sim = arcface.similarity(e0, e1)
        s0, s1 = skin_stats(original, parser), skin_stats(delit, parser)
        row = dict(yaw_deg=view['yaw_deg'], source=str(src), output=str(target), arcface=sim, original_skin=s0, delit_skin=s1)
        failures = []
        if sim is None or sim < thresholds['arcface_min']:
            failures.append(f'arcface {sim}')
        if s0 and s1:
            drop = 1 - s1['l_std'] / max(s0['l_std'], 1e-6)
            hue = abs((s1['hue_deg'] - s0['hue_deg'] + 180) % 360 - 180)
            row.update(shading_drop=drop, hue_shift_deg=hue)
            if drop < thresholds['min_shading_drop']:
                failures.append(f'shading drop {drop:.2f}')
            if hue > thresholds['max_hue_shift_deg']:
                failures.append(f'hue shift {hue:.1f}')
        else:
            failures.append('skin mask missing')
        row.update(failures=failures, accepted=not failures)
        if not failures:
            report['texture_images'][str(view['yaw_deg'])] = str(target.resolve())
        report['views'].append(row)
    (out / 'delight.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    (out / 'texture_images.json').write_text(json.dumps(report['texture_images'], indent=2) + '\n')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--inputs', required=True, help='generated_face_inputs.v1 json (front + side views)')
    ap.add_argument('--out', required=True); ap.add_argument('--lora', choices=('v2', 'v1', 'none'), default='v2')
    ap.add_argument('--steps', type=int, default=20); ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--offload-blocks', type=int, default=1); ap.add_argument('--parsing-model')
    ap.add_argument('--yaws', type=int, nargs='*', help='restrict to these yaw angles')
    a = ap.parse_args(argv)
    report = delight(a.inputs, a.out, lora=a.lora, steps=a.steps, seed=a.seed, offload_blocks=a.offload_blocks,
                     parsing_model=a.parsing_model, only_yaws=a.yaws)
    print(report['adapter_status'])
    for v in report['views']:
        print(v['yaw_deg'], 'arcface', v['arcface'], 'drop', v.get('shading_drop'), 'hue', v.get('hue_shift_deg'), 'ACCEPT' if v['accepted'] else 'REJECT ' + '; '.join(v['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
