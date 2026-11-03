#!/usr/bin/env python3
"""Measure 32 WMMA tile variants with equal weight to six shape families."""
import argparse
import csv
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
SHAPES=[('square',2048,2048,2048),('tall',2048,256,1024),('wide',256,2048,1024),
        ('long_k',256,256,4096),('short_k',2048,2048,128),('irregular',1009,1537,2039)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'tuning.csv'));p.add_argument('--trials',type=int,default=2);args=p.parse_args()
    rows=[];dest=Path(args.output)
    with dest.open('w',newline='',buffering=1) as f:
        writer=None
        for trial in range(args.trials):
            for family,m,n,k in SHAPES:
                for backend in ('int8','fp8'):
                    print(f'trial {trial+1}/{args.trials} {family} {backend}: {m}x{n}x{k}',flush=True)
                    proc=subprocess.Popen([str(ROOT/'tune_ozaki'),str(m),str(n),str(k),backend],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
                    reader=csv.DictReader(proc.stdout)
                    for item in reader:
                        item['family']=family;item['trial']=trial;rows.append(item)
                        if writer is None:
                            writer=csv.DictWriter(f,fieldnames=['family','trial']+reader.fieldnames);writer.writeheader()
                        writer.writerow(item)
                        if int(item['variant'])>=0 and int(item['variant'])%8==7:print(f"  through variant {item['variant']}",flush=True)
                    stderr=proc.stderr.read();ret=proc.wait()
                    if stderr:print(stderr,file=sys.stderr,end='')
                    if ret:print(f'ERROR: tuning process exited {ret}',file=sys.stderr);return 1
    for backend in ('int8','fp8'):
        candidates=[]
        for variant in range(-1,32):
            selected=[r for r in rows if r['backend']==backend and int(r['variant'])==variant]
            if len(selected)!=len(SHAPES)*args.trials or any(float(r['max_abs_diff'])!=0 for r in selected):continue
            speed=[]
            for family,*_ in SHAPES:
                base=[float(x['median_ms']) for x in rows if x['family']==family and x['backend']==backend and int(x['variant'])==-1]
                measured=[float(x['median_ms']) for x in selected if x['family']==family]
                speed.append(statistics.median(base)/statistics.median(measured))
            candidates.append((math.exp(sum(math.log(x) for x in speed)/len(speed)),variant))
        if not candidates:print(f'ERROR: no valid variants for {backend}',file=sys.stderr);return 1
        score,variant=max(candidates);print(f'{backend} selected variant {variant}, geometric mean speedup vs default {score:.3f}',flush=True)
    print(f'Wrote {dest}');return 0
if __name__=='__main__':sys.exit(main())
