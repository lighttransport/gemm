"""Carry independent states through all DiT blocks using separate <55 s replays."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cuda.hunyuan_video15_native.generate import atomic_json, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-fixture',type=Path,required=True)
    p.add_argument('--reference-fixture',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'tmp/hv15-native/opt-build/replay_probe')
    p.add_argument('--reference-python',type=Path,default=ROOT/'tmp/qimg21-ref-venv/bin/python')
    p.add_argument('--native-pid',type=int)
    p.add_argument('--native-run',type=Path,default=ROOT/'tmp/hv15-native/full-quality-i2v-v1')
    args=p.parse_args()
    if args.out.exists(): raise ValueError('chain output must be new')
    args.out.mkdir(parents=True)
    report=dict(status='running',full_pipeline_acceptance=False,segments=[],initial_runner_sha256=digest(args.runner))
    def write(): atomic_json(args.out/'chain.json',report)
    def run(command):
        subprocess.run([str(x) for x in command],check=True,timeout=58)
    def gpu(command,receipt):
        run([sys.executable,ROOT/'ref/hunyuan_video15_native/short_run.py','--out',receipt]+
            (['--native-pid',str(args.native_pid),'--native-run',args.native_run] if args.native_pid else [])+['--']+command)
    fixtures=[args.native_fixture,args.reference_fixture]
    replay=ROOT/'ref/hunyuan_video15_native/replay.py'
    write()
    try:
        while True:
            cases=[json.loads((f/'case.json').read_text()) for f in fixtures]
            for key in ('kind','index','blocks','checkpoint','profile'):
                if cases[0].get(key)!=cases[1].get(key): raise ValueError('chain fixture mismatch: '+key)
            case=cases[0];index=case['index'];count=case.get('blocks',1)
            if case['kind'] not in ('dit_block','dit_pair'): raise ValueError('DiT fixtures required')
            outputs=[args.out/f'{index}-{backend}' for backend in ('native','reference')]
            report.update(index=index,phase='native');write()
            gpu([args.runner,fixtures[0],outputs[0],'candidate'],outputs[0].with_suffix('.json'))
            report['phase']='reference';write()
            gpu([args.reference_python,replay,'reference','--fixture',fixtures[1],'--out',outputs[1]],outputs[1].with_suffix('.json'))
            run([sys.executable,replay,'compare','--fixture',fixtures[1],'--out',outputs[1],'--actual',outputs[0]])
            report['segments'].append(dict(first=index,blocks=count,parity_sha256=digest(outputs[0]/'parity.json'),
                runner_sha256=json.loads(outputs[0].with_suffix('.json').read_text())['executable_sha256'],
                native_receipt_sha256=digest(outputs[0].with_suffix('.json')),
                reference_receipt_sha256=digest(outputs[1].with_suffix('.json')),
                native_fixture_sha256=digest(fixtures[0]/'case.json'),reference_fixture_sha256=digest(fixtures[1]/'case.json')))
            print(f'PASS independent blocks {index}..{index+count-1}',flush=True);write()
            if index+count==54: break
            next_fixtures=[args.out/f'{index+count}-{backend}-fixture' for backend in ('native','reference')]
            for fixture,output,next_fixture in zip(fixtures,outputs,next_fixtures):
                run([sys.executable,replay,'advance','--fixture',fixture,'--actual',output,'--out',next_fixture])
            fixtures=next_fixtures
        report.update(status='pass',phase='complete')
    except BaseException as error:
        report.update(status='failed',error=str(error));raise
    finally: write()


if __name__=='__main__':main()
