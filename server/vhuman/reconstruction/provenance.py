"""Bind reconstruction evidence to its subject, independently of model hashes."""
import json
from pathlib import Path
from .observations import pixel_sha256, sha256


def portrait_record(image):
    return dict(schema='vhuman.portrait_provenance.v1',
                source_sha256=sha256(image), pixel_sha256=pixel_sha256(image))


def validate_candidate(candidate):
    candidate=Path(candidate)
    manifest=json.loads((candidate/'manifest.json').read_text())
    if manifest.get('format')!='vhuman.reconstruction.v1' or manifest.get('face_model')!='gnm_v3':
        raise ValueError('complete GNM reconstruction required')
    if manifest.get('geometry_sha256')!=sha256(candidate/'geometry.npz'):
        raise ValueError('candidate geometry hash mismatch')
    if manifest.get('portrait_sha256')!=sha256(candidate/'portrait.png'):
        raise ValueError('candidate portrait hash mismatch')
    completion = manifest.get('material', {}).get('synthetic_completion')
    if completion is not None:
        if (not isinstance(completion, dict)
                or completion.get('schema') != 'vhuman.synthetic_skin_completion.v1'):
            raise ValueError('invalid synthetic completion report')
        try:
            report = json.loads((candidate/'generated_skin.json').read_text())
            basecolor_hash = sha256(candidate/'skin_basecolor.png')
        except (OSError, ValueError) as error:
            raise ValueError('missing or invalid synthetic completion assets') from error
        if report != completion:
            raise ValueError('synthetic completion report differs from manifest')
        if completion.get('basecolor_sha256') != basecolor_hash:
            raise ValueError('synthetic completion basecolor hash mismatch')
        if ('source_geometry_sha256' in completion
                and completion['source_geometry_sha256'] != manifest['geometry_sha256']):
            raise ValueError('synthetic completion geometry hash mismatch')
    return manifest


def verify_rig_portrait(rig, image):
    """Legacy rigs are checked against their actual adjacent stored portrait."""
    rig = Path(rig)
    definition = json.loads((rig/'rig.json').read_text())
    expected = definition.get('portrait_provenance')
    if expected is None:
        stored = rig.parent/'portrait.png'
        if not stored.is_file():
            raise ValueError('rig lacks source portrait provenance; rebuild from the requested portrait')
        expected = portrait_record(stored)
    actual = portrait_record(image)
    if expected.get('source_sha256') != actual['source_sha256'] and expected.get('pixel_sha256') != actual['pixel_sha256']:
        raise ValueError('rig source portrait does not match video portrait; rebuild the subject rig first')
    return actual
