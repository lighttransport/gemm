#!/usr/bin/env python3
"""Run the reproducible mixed-shape and moduli-sweep GPU campaign."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / 'bench_ozaki'
SHAPES = [
    ('square',512,512,512), ('square',1024,1024,1024),
    ('square',2048,2048,2048), ('square',4096,4096,4096),
    ('square',8192,8192,8192), ('tall',8192,512,2048),
    ('wide',512,8192,2048), ('long_k',512,512,16384),
    ('short_k',4096,4096,256), ('irregular',1009,1537,2039),
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default=str(ROOT/'results.csv'))
    p.add_argument('--iters',type=int,default=30)
    p.add_argument('--warmups',type=int,default=5)
    p.add_argument('--min-sample-ms',type=float,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--int8-variant',type=int,default=27)
    p.add_argument('--fp8-variant',type=int,default=27)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--quick',action='store_true',help='small campaign for testing')
    args=p.parse_args()
    if not BENCH.exists():p.error('build bench_ozaki first')
    shapes=SHAPES[:2] if args.quick else SHAPES
    tasks=[]
    for family,m,n,k in shapes:
        for backend in ('int8','fp8'):
            tasks.append((family,m,n,k,backend,15 if backend=='int8' else 11,'fast','uniform'))
    counts=(8,11,15,20) if args.quick else range(2,21)
    for backend in ('int8','fp8'):
        for count in counts:tasks.append(('sweep',512,512,512,backend,count,'fast','uniform'))
    for backend in ('int8','fp8'):
        for scaling in ('fast','accurate'):
            distributions=('wide','subnormal-input') if args.quick else ('wide','cancellation','zero','subnormal-input','subnormal-output','huge')
            for distribution in distributions:
                counts_for_stress=((15,) if backend=='int8' else (11,)) if args.quick else ((15,20) if backend=='int8' else (11,20))
                for count in counts_for_stress:
                    tasks.append(('stress',33,31,64,backend,count,scaling,distribution))
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    done=set();fields=None
    if args.resume and dest.exists():
        with dest.open(newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames
            for r in reader:
                expected_variant=args.int8_variant if r['backend']=='int8' else args.fp8_variant
                if (int(r['device'])!=args.device or int(r['seed'])!=args.seed or int(r['iters'])!=args.iters or int(r['warmups'])!=args.warmups or
                    float(r['min_sample_ms'])!=args.min_sample_ms or int(r['variant'])!=expected_variant):
                    p.error('resume settings do not match the existing results')
                done.add((r['family'],int(r['m']),int(r['n']),int(r['k']),r['backend'],int(r['moduli']),r['scaling'],r['distribution']))
    else:
        dest.write_text('')
    failures=[];start=time.monotonic()
    with dest.open('a',newline='',buffering=1) as out:
        for index,task in enumerate(tasks,1):
            if task in done:continue
            family,m,n,k,backend,count,scaling,distribution=task
            cmd=[str(BENCH),'--device',str(args.device),'--m',str(m),'--n',str(n),'--k',str(k),'--backend',backend,
                 '--moduli',str(count),'--scaling',scaling,'--distribution',distribution,
                 '--variant',str(args.int8_variant if backend=='int8' else args.fp8_variant),
                 '--iters',str(args.iters),'--warmups',str(args.warmups),
                 '--min-sample-ms',str(args.min_sample_ms),'--seed',str(args.seed)]
            print(f'[{index}/{len(tasks)}] {family} {backend} {m}x{n}x{k} moduli={count} {scaling} {distribution}',flush=True)
            completed=subprocess.run(cmd,text=True,capture_output=True,env=os.environ.copy())
            if completed.stderr:print(completed.stderr,file=sys.stderr,end='',flush=True)
            rows=list(csv.DictReader(completed.stdout.splitlines()))
            if len(rows)!=1:
                failures.append((task,completed.returncode));print('ERROR: missing measurement',flush=True);continue
            row=rows[0];row['family']=family;row['quality_pass']='1' if completed.returncode==0 else '0'
            if fields is None:
                fields=['family','quality_pass']+list(rows[0].keys())[:-2]
                writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
            else:writer=csv.DictWriter(out,fieldnames=fields)
            writer.writerow(row)
            print(f"  {float(row['ms']):.4f} ms, native {float(row['native_ms']):.4f} ms, error {float(row['sampled_norm_error']):.3g}, quality={row['quality_pass']}, elapsed {time.monotonic()-start:.0f}s",flush=True)
    if failures:
        print(f'{len(failures)} cases produced no measurement: {failures}',file=sys.stderr);return 1
    print(f'Wrote {dest} in {time.monotonic()-start:.0f}s',flush=True);return 0
if __name__=='__main__':sys.exit(main())
