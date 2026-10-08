"""Build a local, evidence-linked review of the face quality iteration artifacts.

Usage: python -m server.vhuman.reconstruction.quality_review --work DIR
The existing experiment directories remain authoritative; this does not promote
an anatomy candidate or rebuild the older browser avatar.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import zipfile

from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(work):
    work = Path(work).resolve()
    out = work / 'review'
    media, downloads = out / 'media', out / 'downloads'
    media.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(exist_ok=True)
    sources = {}

    def record(relative):
        path = work / relative
        if not path.is_file():
            raise FileNotFoundError('review evidence missing: ' + str(path))
        sources[relative] = dict(sha256=digest(path), bytes=path.stat().st_size)
        return path

    def read(relative):
        return json.loads(record(relative).read_text())

    def link(relative):
        record(relative)
        return '../' + relative

    def pair(key, label, before, after, crop=None):
        paths = []
        sizes = []
        for suffix, source in [('before', before), ('after', after)]:
            with Image.open(record(source)) as original:
                image = original.crop(crop) if crop else original.copy()
                sizes.append(image.size)
                target = media / f'{key}_{suffix}.png'
                image.save(target)
                paths.append('media/' + target.name)
        if sizes[0] != sizes[1]:
            raise ValueError('comparison image dimensions differ: ' + key)
        return dict(label=label, before=paths[0], after=paths[1], width=sizes[0][0], height=sizes[0][1])

    def archive(directory, slug, title):
        receipt = read(directory + '/portable.validation.json')
        report = read(directory + '/report.json')
        if not receipt.get('passed') or not report.get('passed'):
            raise ValueError('only verified interchange bundles belong in review downloads')
        if (receipt['usd_sha256'] != report['usd_sha256']
                or receipt['sample_checks'] != report['meshes'] * report['samples']):
            raise ValueError('fresh import receipt does not match the bundle report')
        root = work / directory
        for filename, field in [('head.usdc', 'usd_sha256'),
                                ('blender_materials.json', 'material_sidecar_sha256'),
                                ('blender_material_animation.json', 'material_animation_sidecar_sha256')]:
            path = record(directory + '/' + filename)
            if digest(path) != report[field]:
                raise ValueError('bundle content changed: ' + str(path))
        members = ['head.usdc', 'portable.blend', 'portable.validation.json', 'report.json',
                   'blender_materials.json', 'blender_material_animation.json',
                   'candidate_manifest.json', 'source_motion.json']
        paths = [root / name for name in members]
        if 'candidate_evidence_sha256' in report:
            from .usd_candidate_evidence import verify as verify_evidence
            verify_evidence(root)
            paths.append(root/'candidate_evidence.json')
        for folder in ('textures', 'shader_assets', 'candidate_evidence'):
            paths.extend(sorted(p for p in (root / folder).rglob('*') if p.is_file()))
        target = downloads / (slug + '.zip')
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for path in paths:
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError('external bundle member: ' + str(path))
                record(str(path.relative_to(work)))
                bundle.write(path, slug + '/' + str(path.relative_to(root)))
            bundle.writestr(slug + '/READ_ME.txt',
                title + '\n\nOpen portable.blend and keep this directory intact.\n'
                'The Blender cache uses //head.usdc. Shader animation uses the included Blender sidecar.\n'
                'Generic USD viewers may show static shader values.\n'
                'Interchange verification does not imply anatomical contact acceptance.\n'
                'Read source_motion.json and report.json for provenance and limitations.\n')
        return dict(title=title, href='downloads/' + target.name, bytes=target.stat().st_size,
                    sha256=digest(target), mesh_checks=receipt['sample_checks'],
                    shader_checks=receipt['material_animation_verification']['checks'],
                    evidence='../' + directory + '/portable.validation.json')

    skin = read('completed_illumination_0.5/generated_skin.json')
    lid = read('lid_camera_fit_joint/contact_audit.json')
    lid_projection = read('lid_camera_fit_joint/projection_validation.json')
    tongue = read('native_tongue_motion_late_contacts/visibility_comparison.json')
    gaze = read('gaze_usd_animated_materials/portable.validation.json')
    bundles = [archive('combined_lid_fit_usd_materials', 'eyelid-fit', 'Eyelid fitting candidate'),
               archive('native_tongue_motion_usd', 'tongue-motion', 'Tongue articulation candidate'),
               archive('gaze_usd_animated_materials', 'gaze-wrinkles', 'Gaze with restored wrinkle animation')]
    eye_reduction = 100 * (1 - lid['totals']['fit_shell_pairs'] / lid['totals']['baseline_shell_pairs'])
    cards = [dict(id='skin', title='Skin lighting correction', status='Selected for these previews',
        description='The half-strength correction improves the broad tone transition. It revises albedo estimates within photographed support; the original portrait file is unchanged.',
        metrics=[f"{skin['photographed_texels_changed']:,} photo-supported albedo texels revised", 'Geometry and auxiliary material maps unchanged'],
        limit='Broad scalp mottling remains. This is an illumination estimate, not measured reflectance.',
        evidence=[('Appearance measurements', link('illumination_comparison/appearance_metrics.json')),
                  ('Albedo provenance', link('completed_illumination_0.5/generated_skin.json'))],
        views=[pair('skin_' + view, label, f'illumination_comparison/{view}_baseline.png',
                    f'illumination_comparison/{view}_half.png')
               for view, label in [('front', 'Front'), ('side', 'Side'), ('raking', 'Alternate light')]]),
        dict(id='eyes', title='Eyelid–globe fitting', status='Experimental anatomy',
        description='A bounded depth correction preserves the source-view lid projection while reducing sampled eye-shell crossings.',
        metrics=[f'{eye_reduction:.1f}% fewer sampled shell-crossing pairs',
                 f"Source landmark shift ≤ {lid_projection['maximum_landmark_projection_shift_px']:.5f} px"],
        limit='Residual eye-shell contacts and small side-view closure gaps remain. No full contact acceptance.',
        evidence=[('Contact audit', link('lid_camera_fit_joint/contact_audit.json')),
                  ('Area and interpolation check', link('lid_camera_fit_joint/geometry_validation.json'))],
        bundle=0, views=[pair('eyes_' + view, label, f'lid_camera_fit_joint_render/{view}_baseline.png',
                              f'lid_camera_fit_joint_render/{view}_fit.png', (250, 280, 780, 580))
                        for view, label in [('open', 'Open'), ('closed', 'Closed'), ('closed_side', 'Closed, side')]]),
        dict(id='tongue', title='Tongue articulation', status='Experimental anatomy',
        description='Two native tongue controls activate only for wide mouth openings. The source pose and non-tongue geometry stay unchanged.',
        metrics=[f"Sampled visible contacts: {tongue['baseline_sum']} → {tongue['fit_sum']}",
                 f"{len(tongue['samples'])} views checked; {len(tongue['worsened'])} worsened"],
        limit='Hidden contacts remain. Two intermediate poses add one hidden self-crossing each. This motion is an authored prior, not observed subject motion.',
        evidence=[('Visibility comparison', link('native_tongue_motion_late_contacts/visibility_comparison.json')),
                  ('Motion recipe', link('native_tongue_motion_late/recipe.json'))], bundle=1,
        views=[pair('tongue_' + str(frame), label, f'native_tongue_motion_render/{frame:02d}_baseline.png',
                    f'native_tongue_motion_render/{frame:02d}_fit.png')
               for frame, label in [(6, 'Opening'), (11, 'Wide open'), (22, 'Later pose')]]),
        dict(id='wrinkles', title='Wrinkle animation survives export', status='Interchange verified',
        description='The Blender import restores sampled shader values instead of freezing them at the first frame.',
        metrics=[f"{gaze['material_animation_verification']['checks']} shader checks; maximum value error {gaze['material_animation_verification']['maximum_error']:.3g}",
                 f"{gaze['material_animation_verification']['channels']} varying wrinkle channels retained"],
        limit='Generic USD readers may still use static shader values. Between samples, restored Blender values interpolate linearly.',
        evidence=[('Fresh import receipt', link('gaze_usd_animated_materials/portable.validation.json')),
                  ('Matched render measurements', link('gaze_usd_material_comparison/metrics.json'))], bundle=2,
        views=[pair('wrinkles', 'Gaze pose', 'gaze_usd_material_comparison/usd_static.png',
                    'gaze_usd_material_comparison/usd_animated.png')]),
        dict(id='teeth', title='Tooth appearance', status='Optional shading',
        description='Warm enamel and root-to-tip variation were compared with the original material. Tooth geometry and spacing are unchanged.',
        metrics=['Static tooth attribute and shader roundtrip verified'],
        limit='These are artist-selected appearance priors. They do not resolve dental/gum intersections.',
        evidence=[('Appearance measurements', link('tooth_material_trial/appearance_metrics.json')),
                  ('Shader and attribute receipt', link('tooth_gradient_usd_baked/report.json'))],
        views=[pair('teeth_' + view, label, f'tooth_material_trial/{view}_baseline.png',
                    f'tooth_material_trial/{view}_gradient.png', (330, 560, 710, 800))
               for view, label in [('smile', 'Smile'), ('open', 'Open mouth')]]),
        dict(id='mucosa', title='Tongue surface appearance', status='Optional shading',
        description='UV-bound fine bump and mild color/coat variation add a small surface change without moving geometry.',
        metrics=['933 vertices and UVs unchanged in static USD roundtrip'],
        limit='The visible change is small at these views. This is authored detail, not recovered microanatomy.',
        evidence=[('Appearance measurements', link('tongue_material_trial/metrics.json')),
                  ('Shader and UV receipt', link('tongue_material_usd/report.json'))],
        views=[pair('mucosa_' + str(yaw), label, f'tongue_material_trial/{yaw}_baseline.png',
                    f'tongue_material_trial/{yaw}_color.png') for yaw, label in [(0, 'Front'), (30, 'Side')]])]
    data = dict(cards=cards, bundles=bundles)
    encoded = json.dumps(data).replace('<', '\\u003c')
    device = work.parent / 'vhuman-blender/player/device-test.html'
    device_link = html.escape(os.path.relpath(device, out)) if device.is_file() else ''
    page = TEMPLATE.replace('__DATA__', encoded).replace('__DEVICE__', device_link)
    (out / 'index.html').write_text(page)
    manifest = dict(schema='vhuman.quality_review.v1', work=str(work),
        comparisons=len(cards), image_pairs=sum(len(card['views']) for card in cards),
        bundles=bundles, source_files=sources,
        limitations=['Review of offline Blender candidates; no new anatomy promotion',
                     'Existing browser/device page uses the previous avatar package',
                     'No physical phone/tablet test was performed'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'READ_ME.txt').write_text(
        'Open index.html directly, or serve the parent tmp directory on localhost.\n'
        'From the repository root: python3 -m http.server 8797 --bind 127.0.0.1 --directory tmp\n'
        'Then open http://127.0.0.1:8797/vhuman-quality8h/review/\n'
        'Keep each downloaded archive directory intact for relative USD caches.\n'
        'Experimental anatomy is not globally accepted; use the linked measurements and limits.\n')
    return manifest


TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Virtual human · Face quality review</title>
<style>
:root{color-scheme:dark;font:16px/1.5 system-ui,sans-serif;background:#14171c;color:#e8edf2}*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:32px 24px 60px}h1{font-size:clamp(1.8rem,4vw,2.7rem);line-height:1.15;margin:8px 0 18px}h2{font-size:1.25rem;margin:6px 0 12px}h3{margin:0;font-size:1.05rem}p{margin:8px 0 14px}.muted{color:#aebac8}.eyebrow{letter-spacing:.12em;text-transform:uppercase;font-size:.75rem;color:#80c6cb}.intro{max-width:870px}nav{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0}a{color:#9bdbde;text-underline-offset:3px}nav a,button,select{border:1px solid #45515f;border-radius:8px;padding:8px 12px;background:#222a33;color:#edf4fa;font:inherit}button,select{cursor:pointer}button:focus-visible,a:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #80d7df;outline-offset:3px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{border:1px solid #35414e;border-radius:14px;background:#1b222b;overflow:hidden;scroll-margin-top:18px}.text{padding:20px}.badge{display:inline-block;font-size:.75rem;color:#e8c98e;background:#3a3325;padding:3px 8px;border-radius:5px;margin-bottom:9px}.badge.verified{color:#a7ded5;background:#203d39}.comparison{position:relative;background:#0c1015;overflow:hidden}.comparison img{position:absolute;width:100%;height:100%;inset:0;object-fit:contain}.comparison .after{clip-path:inset(0 0 0 50%)}.divider{position:absolute;left:50%;height:100%;width:2px;background:#fff8;pointer-events:none}.labels{position:absolute;inset:8px 8px auto;display:flex;justify-content:space-between;font-size:.7rem;pointer-events:none}.labels span{background:#000b;padding:3px 6px;border-radius:4px}.controls{padding:12px 20px;border-bottom:1px solid #35414e}.row{display:flex;flex-wrap:wrap;align-items:center;gap:8px}.row select{margin-left:auto;max-width:100%}button{font-size:.8rem;padding:6px 9px}input[type=range]{width:100%;accent-color:#91d9dc;margin:12px 0 0}.stats{display:flex;flex-wrap:wrap;gap:7px;font-size:.82rem;margin:12px 0}.stats span{background:#26303b;padding:6px 9px;border-radius:5px}.limit{font-size:.9rem;color:#cfbd9e}details{font-size:.88rem}summary{cursor:pointer}details a{display:block;margin:7px 0}.download{display:block;border-top:1px solid #35414e;padding-top:12px;margin-top:16px}.bundle-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.bundle{background:#202934;border:1px solid #45515f;border-radius:10px;padding:18px}.bundle p{font-size:.88rem}.section{margin-top:40px}.remaining{border-left:3px solid #c8a86f;padding:6px 20px;color:#d5dee6}footer{margin-top:30px;border-top:1px solid #35414e;padding-top:20px;font-size:.88rem}small{font-size:.8rem} @media(max-width:760px){main{padding:24px 12px}.grid,.bundle-grid{grid-template-columns:1fr}.text{padding:16px}.row select{margin-left:0}.controls{padding:12px 16px}}
</style>
<main><header class="intro"><div class="eyebrow">Virtual human · Local review</div><h1>Face, eyes and mouth</h1><p>Matched renders and portable scenes from the quality iterations. Use the split control to compare an earlier render with its candidate.</p><p class="muted">The lighting correction is selected for these previews. Anatomy changes remain experimental where contacts are unresolved. Body, clothing and hair are outside this review.</p></header>
<nav aria-label="Review sections"><a href="#skin">Skin</a><a href="#eyes">Eyelids</a><a href="#tongue">Tongue motion</a><a href="#wrinkles">Wrinkle export</a><a href="#teeth">Tooth shading</a><a href="#bundles">Portable scenes</a></nav>
<div id="cards" class="grid"></div>
<section id="bundles" class="section"><h2>Portable Blender / USD bundles</h2><p class="muted">Download and extract a complete bundle, then open <strong>portable.blend</strong>. Keep its directory intact. Geometry and shader interchange checks are separate from anatomical acceptance.</p><div id="downloads" class="bundle-grid"></div></section>
<section class="section"><h2>Still unresolved</h2><div class="remaining"><p>Small eyelid gaps and residual globe contacts; tongue-root, cavity and dental/gum intersections; broad skin mottling under some views and lighting.</p><p>The new offline anatomy candidates have not replaced the existing live browser avatar. A physical phone/tablet result is still pending.</p></div></section>
<footer><a id="device" href="__DEVICE__">Open the prepared device test page</a><p class="muted">That page uses the previous browser avatar package. No phone or tablet was connected for this work. Serve the parent <code>tmp/</code> directory for this link to work over HTTP.</p><a href="manifest.json">Review manifest and source hashes</a></footer></main>
<script id="review-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('review-data').textContent);
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
for(const card of data.cards){
 const section=el('section','card');section.id=card.id;const top=el('div','text');top.append(el('span','badge'+(card.status.includes('verified')?' verified':''),card.status),el('h2','',card.title),el('p','',card.description));section.append(top);
 const comparison=el('div','comparison'),before=el('img','before'),after=el('img','after'),divider=el('div','divider');before.alt='Earlier render: '+card.title;after.alt='Candidate render: '+card.title;const labels=el('div','labels');labels.append(el('span','','Earlier render'),el('span','','Candidate'));comparison.append(before,after,divider,labels);section.append(comparison);
 const controls=el('div','controls'),row=el('div','row'),range=el('input');range.type='range';range.min='0';range.max='100';range.value='50';range.setAttribute('aria-label',card.title+' comparison split');
 const setSplit=value=>{range.value=value;after.style.clipPath=`inset(0 0 0 ${value}%)`;divider.style.left=value+'%';section.dataset.split=value;};
 for(const [label,value] of [['Earlier',100],['Split',50],['Candidate',0]]){const b=el('button','',label);b.type='button';b.onclick=()=>setSplit(value);row.append(b);}
 const select=el('select');select.setAttribute('aria-label',card.title+' view');for(const [i,view] of card.views.entries()){const o=el('option','',view.label);o.value=i;select.append(o);}row.append(select);controls.append(row,range);section.append(controls);range.oninput=()=>setSplit(range.value);
 const load=()=>{const v=card.views[Number(select.value)];before.src=v.before;after.src=v.after;comparison.style.aspectRatio=v.width+'/'+v.height;section.dataset.view=select.value;};select.onchange=load;load();setSplit(50);
 const bottom=el('div','text'),stats=el('div','stats');card.metrics.forEach(s=>stats.append(el('span','',s)));bottom.append(stats,el('p','limit',card.limit));const detail=el('details');detail.append(el('summary','','Measurements and provenance'));for(const [title,href] of card.evidence){const a=el('a','',title);a.href=href;detail.append(a);}bottom.append(detail);
 if(card.bundle!==undefined){const bundle=data.bundles[card.bundle],a=el('a','download','Download portable scene · '+Math.round(bundle.bytes/1048576)+' MiB');a.href=bundle.href;a.download='';bottom.append(a);}section.append(bottom);document.getElementById('cards').append(section);
}
for(const bundle of data.bundles){const box=el('article','bundle'),a=el('a','','Download ZIP · '+Math.round(bundle.bytes/1048576)+' MiB');a.href=bundle.href;a.download='';box.append(el('h3','',bundle.title),el('p','',`${bundle.mesh_checks} mesh checks · ${bundle.shader_checks} shader checks`),a);const receipt=el('a','','Verification receipt');receipt.href=bundle.evidence;const p=el('p');p.append(receipt);box.append(p);document.getElementById('downloads').append(box);}
if(!document.getElementById('device').getAttribute('href'))document.getElementById('device').hidden=true;
window.vhumanQualityReview={ready:true,cards:data.cards.length,bundles:data.bundles.length};
</script></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True)
    args = parser.parse_args()
    result = build(args.work)
    print(json.dumps({key: result[key] for key in ('comparisons', 'image_pairs')}))
