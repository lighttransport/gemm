#!/usr/bin/env python3
"""Generate SVG charts and an equal-family summary from results.csv."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
TMP=ROOT.parent.parent/'tmp'/'gemmmul8'/'plots'

def gp(script):
    env=os.environ.copy();env['TMPDIR']=str(TMP)
    subprocess.run(['gnuplot'],input=script,text=True,cwd=ROOT,env=env,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default=str(ROOT/'results.csv'));args=p.parse_args()
    with open(args.input,newline='') as f: rows=list(csv.DictReader(f))
    TMP.mkdir(parents=True,exist_ok=True)
    shapes=[r for r in rows if r['family'] not in ('sweep','stress')]
    families=[]
    for r in shapes:
        key=(r['family'],r['m'],r['n'],r['k'])
        if key not in families:families.append(key)
    with (TMP/'perf.dat').open('w') as f:
        for family,m,n,k in families:
            v={r['backend']:r for r in shapes if (r['family'],r['m'],r['n'],r['k'])==(family,m,n,k)}
            if len(v)!=2:continue
            f.write(f'"{family}_{m}x{n}x{k}" {v["int8"]["speedup"]} {v["fp8"]["speedup"]}\n')
    gp('''set terminal svg size 1500,600 dynamic enhanced font 'sans,11'
set output 'performance.svg'
set title 'Complete FP64 GEMM pipeline versus native rocBLAS DGEMM'
set ylabel 'Native FP64 latency / Ozaki-II latency'
set yrange [0:*]
set style data histograms
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.85
set xtics rotate by -35 right font ',9'
set grid ytics
set key left top
set arrow 1 from graph 0, first 1 to graph 1, first 1 nohead dt 2 lc rgb '#555555'
plot '../../tmp/gemmmul8/plots/perf.dat' using 2:xtic(1) title 'INT8' lc rgb '#1769aa', '' using 3 title 'FP8' lc rgb '#d05a19'
''')
    for backend in ('int8','fp8'):
        with (TMP/(backend+'.dat')).open('w') as f:
            for r in rows:
                if r['family']=='sweep' and r['backend']==backend:
                    error=float(r['sampled_norm_error']);speed=float(r['speedup'])
                    if error>0 and math.isfinite(error) and speed>0:f.write(f'{speed} {error} {r["moduli"]}\n')
    gp('''set terminal svg size 1050,650 dynamic enhanced font 'sans,12'
set output 'precision_frontier.svg'
set title '512-cubed precision / speed frontier (labels: moduli)'
set xlabel 'Native FP64 latency / Ozaki-II latency'
set ylabel 'Maximum sampled normalized componentwise error vs MPFR'
set logscale y
set grid
set key right bottom
set arrow 1 from graph 0, first 3.5527136788e-15 to graph 1, first 3.5527136788e-15 nohead dt 2 lc rgb '#555555'
plot '../../tmp/gemmmul8/plots/int8.dat' using 1:2 with points pt 7 ps 1.2 lc rgb '#1769aa' title 'INT8', \\
     '' using 1:2:(sprintf('%d',$3)) with labels offset 0.6,0.4 textcolor rgb '#1769aa' notitle, \\
     '../../tmp/gemmmul8/plots/fp8.dat' using 1:2 with points pt 5 ps 1.2 lc rgb '#d05a19' title 'FP8', \\
     '' using 1:2:(sprintf('%d',$3)) with labels offset 0.6,-0.6 textcolor rgb '#d05a19' notitle
''')
    for backend in ('int8','fp8'):
        fam_scores=[]
        for family in dict.fromkeys(r['family'] for r in shapes):
            x=[float(r['speedup']) for r in shapes if r['family']==family and r['backend']==backend]
            if x:fam_scores.append(math.exp(sum(map(math.log,x))/len(x)))
        if fam_scores:print(f'{backend}: equal-family geometric-mean speedup {math.exp(sum(map(math.log,fam_scores))/len(fam_scores)):.3f}x')
    print('Wrote performance.svg and precision_frontier.svg')
if __name__=='__main__':main()
