"""Export a local photo-only landmark/silhouette annotation kit, no model overlay."""
import argparse
import json
from pathlib import Path
from . import observations
from .artifacts import artifact_path
NAMES=['eye_right','eye_left','nose_tip','menton','mouth_right','mouth_left','upper_lip','lower_lip']
PAGE='''<!doctype html><meta charset="utf-8"><title>Independent face annotations</title>
<style>body{font:16px sans-serif;max-width:1000px;margin:24px auto}canvas{max-width:100%;background:#222}button,select,input{padding:8px;margin:4px}#status{min-height:2em}</style>
<h1>Photo-only face annotation</h1><p>Use the subject's anatomical right/left. Click the eight named landmarks, then trace the visible skin outline. No fitted geometry or detector points are shown. Mark the outline of visible skin, excluding hair and clothing. These annotations remain estimates; record uncertainty in notes.</p>
<label>Rater <input id="rater"></label><label>Notes <input id="notes"></label>
<select id="view"></select><select id="point"></select><button id="undo">Undo outline point</button><button id="reset">Reset this view</button>
<p id="status"></p><canvas id="canvas"></canvas><br><button id="save">Download observations.json</button>
<p>Place the downloaded JSON beside this page and images before running evaluation. Annotate without inspecting predictions. Keep this set separate from tuning data.</p>
<script>
const doc=__DOC__, names=__NAMES__, records=doc.views.map(()=>({anchors:{},silhouette:[],notes:''}));
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d'),image=new Image();
doc.views.forEach((v,i)=>$('view').add(new Option(`View ${i+1}`,i)));
names.concat(['skin outline']).forEach(n=>$('point').add(new Option(n,n)));
function draw(){ctx.drawImage(image,0,0);const r=records[+$('view').value];ctx.strokeStyle='#00ff88';ctx.lineWidth=Math.max(2,canvas.width/400);ctx.fillStyle='#00ff88';
Object.entries(r.anchors).forEach(([n,p])=>{ctx.beginPath();ctx.arc(...p.xy,canvas.width/180,0,Math.PI*2);ctx.stroke();ctx.fillText(n,p.xy[0]+8,p.xy[1]);});
ctx.beginPath();r.silhouette.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();$('status').textContent=`${Object.keys(r.anchors).length}/8 landmarks; ${r.silhouette.length} outline points. Selected: ${$('point').value}`;}
function load(){const i=+$('view').value;image.src=doc.views[i].image;$('notes').value=records[i].notes;}
image.onload=()=>{canvas.width=image.width;canvas.height=image.height;draw();};$('view').onchange=load;$('point').onchange=draw;
$('notes').oninput=()=>records[+$('view').value].notes=$('notes').value;
canvas.onclick=e=>{const box=canvas.getBoundingClientRect(),p=[(e.clientX-box.left)*canvas.width/box.width,(e.clientY-box.top)*canvas.height/box.height],r=records[+$('view').value],n=$('point').value;
if(n==='skin outline'){if(r.silhouette.length>=256)return;r.silhouette.push(p);}else{r.anchors[n]={xy:p,weight:1};const index=names.indexOf(n);$('point').selectedIndex=Math.min(index+1,8);}draw();};
$('undo').onclick=()=>{records[+$('view').value].silhouette.pop();draw();};$('reset').onclick=()=>{records[+$('view').value]={anchors:{},silhouette:[],notes:''};draw();};
$('save').onclick=()=>{if(!$('rater').value.trim()||records.some(r=>names.some(n=>!r.anchors[n])||r.silhouette.length<3)){alert('Provide rater, eight landmarks and an outline for every view.');return;}
const result=JSON.parse(JSON.stringify(doc));result.views.forEach((v,i)=>Object.assign(v,records[i]));result.provenance={...result.provenance,annotations:'human photo-only review; no model or detector overlay',rater:$('rater').value.trim(),annotation_time:new Date().toISOString(),source_annotations_used:false};
const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='observations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};load();
</script>'''


def run(source,out):
    source,out=Path(source),artifact_path(out);doc=observations.load(source)
    out.mkdir(parents=True,exist_ok=False);views=[]
    for i,view in enumerate(doc['views']):
        name=f'view_{i}.png';from PIL import Image
        Image.open(view['image_path']).convert('RGB').save(out/name)
        row={k:v for k,v in view.items() if k in ('size','camera','frame_id','camera_id','timestamp_s','split')}
        row.update(image=name,sha256=observations.sha256(out/name),anchors={});views.append(row)
    clean=dict(format=observations.FORMAT,views=views,provenance=dict(source_observations_sha256=observations.sha256(source),
               source_annotations_used=False,annotation_status='pending human work'))
    encoded=json.dumps(clean).replace('<','\\u003c')
    (out/'index.html').write_text(PAGE.replace('__DOC__',encoded).replace('__NAMES__',json.dumps(NAMES)))
    (out/'pending.json').write_text(json.dumps(clean,indent=2)+'\n');return dict(status='pending human annotation',views=len(views),page=str(out/'index.html'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.observations,a.out)))

if __name__=='__main__':main()
