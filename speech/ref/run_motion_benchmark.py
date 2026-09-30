"""Run local Japanese WAV and TTS clips through production speech-to-rig jobs.

No clips are downloaded. Diagnostics test consistency, not perceptual lip-sync;
independent human review remains required because ReazonSpeech has no face GT.
"""
import argparse
import csv
import json
import sys
import threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from server.vhuman.reconstruction.artifacts import artifact_path
from server.vhuman.reconstruction.observations import sha256
from server.vhuman.rig.speech import speech_job
from server.vhuman.service import EyeService
from speech.ref.eval_rig_motion import diagnostics


def run(clips,work,head,out,backend='cpu',limit=12,tts=False):
    clips,out=Path(clips),artifact_path(out);doc=json.loads((clips/'manifest.json').read_text())
    if not 1<=limit<=48:raise ValueError('limit must be 1..48')
    out.mkdir(parents=True,exist_ok=False);service=EyeService(Path(work));rows=[]
    rig=service.rig_file(head,'rig.json');rig_hash=sha256(rig)
    for clip in doc['clips'][:limit]:
        wav=(clips/clip['wav']).resolve()
        if not wav.is_relative_to(clips.resolve()):raise ValueError('clip WAV escapes manifest directory')
        requests=[('wav',dict(wav=str(wav),transcript=clip['transcription']))]
        if tts:requests.append(('tts',dict(text=clip['transcription'])))
        for source,request in requests:
            print(f"{clip['id']} {source}",flush=True)
            result=speech_job(service,dict(head_id=head,seed=7,**request),lambda *x:None,threading.Event(),backend=backend,allow_wav=True)
            take=service.take_file(head,result['id'],'manifest.json').parent
            score=diagnostics(take)
            rows.append(dict(clip_id=clip['id'],source=source,take=str(take.resolve()),source_wav_sha256=sha256(wav),
                             produced_wav_sha256=sha256(take/'audio.wav'),alignment_sha256=sha256(take/'align.json'),diagnostics=score))
            (out/'progress.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    with (out/'review.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['clip_id','source','take','rater','lip_sync_1_5','closures_1_5','naturalness_1_5','notes'])
        for row in rows:writer.writerow([row['clip_id'],row['source'],row['take'],'','','','',''])
    report=dict(format='vhuman.japanese_speech_benchmark.v1',source_manifest_sha256=sha256(clips/'manifest.json'),rig_sha256=rig_hash,
        clips=len(doc['clips'][:limit]),takes=rows,backend=backend,visual_ground_truth=False,perceptual_gate_passed=False,
        consistency_gate_passed=all(r['diagnostics']['face_track_max_error']<=1e-5 and r['diagnostics']['neutral_final_frame'] for r in rows),
        limitations=['ReazonSpeech provides audio/transcripts, no synchronized facial reference',
                     'alignment-based closure scores are consistency checks, not independent timing accuracy',
                     'review.csv requires human ratings; no automatic perceptual quality claim'])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('clips','work','head','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu');p.add_argument('--limit',type=int,default=12);p.add_argument('--tts',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.clips,a.work,a.head,a.out,a.backend,a.limit,a.tts),ensure_ascii=False))

if __name__=='__main__':main()
