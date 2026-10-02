"""Summarize numerically accepted replay pairs; never claim full video acceptance."""
import argparse
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cuda.hunyuan_video15_native.generate import atomic_json,digest


def samples(values,warm):
    if not values or any(not 0 < v < float('inf') for v in values):
        raise ValueError('invalid replay timing samples')
    if warm and len(values)<3:raise ValueError('warm comparison requires a cold and two warm samples')
    selected=values[1:] if warm else values
    return dict(values=selected,median=statistics.median(selected),minimum=min(selected),maximum=max(selected))


def pair(entry):
    native,reference=[Path(entry[name]) for name in ('native','reference')]
    parity_path=native/'parity.json';parity=json.loads(parity_path.read_text())
    results=parity.get('results',{})
    if (parity.get('pass_all') is not True or parity.get('full_pipeline_acceptance') is not False
        or not results or any(v.get('pass') is not True for v in results.values())
        or any(v.get('pass_all') is not True for v in parity.get('frames',[]))):
        raise ValueError('an accepted bounded parity report is required')
    for field,suffix in (('outputs','.f32'),('output_metadata','.json')):
        hashes=parity.get(field,{})
        if set(hashes)!=set(results):raise ValueError('parity report lacks complete artifact hashes; rerun compare')
        for name in results:
            for backend,folder in (('native',native),('reference',reference)):
                if hashes[name].get(backend)!=digest(folder/(name+suffix)):
                    raise ValueError('stale parity artifact: '+backend+' '+name+suffix)
    for backend,folder in (('native',native),('reference',reference)):
        if parity.get('timing_sha256',{}).get(backend)!=digest(folder/'timing.json'):
            raise ValueError('stale or unbound replay timing: '+backend)
        receipt=folder.with_suffix('.json')
        expected=parity.get('receipt_sha256',{}).get(backend)
        if expected!=(digest(receipt) if receipt.is_file() else None):
            raise ValueError('stale replay execution receipt: '+backend)
    timings=[json.loads((folder/'timing.json').read_text()) for folder in (native,reference)]
    warm=entry.get('warm',True)
    n,r=[samples(t['wall_seconds'],warm) for t in timings]
    receipts=[]
    for backend,folder,timing in zip(('native','reference'),(native,reference),timings):
        path=folder.with_suffix('.json')
        if path.is_file():
            receipt=json.loads(path.read_text())
            if receipt['status']!='pass' or not 0<receipt['elapsed_seconds']<60:
                raise ValueError('replay execution failed or exceeded one minute')
            peak=receipt.get('sampled_peak_vram_mib')
            if peak is not None and not 0<=peak<=14336:raise ValueError('invalid VRAM sample or budget exceeded')
            receipts.append(dict(path=str(path),sha256=digest(path),elapsed_seconds=receipt['elapsed_seconds'],
                                 sampled_peak_vram_mib=receipt.get('sampled_peak_vram_mib'),
                                 executable_sha256=receipt.get('executable_sha256')))
        else:
            if backend=='native' or timing.get('device')!='cpu_fp32':
                raise ValueError('GPU replay requires a bounded execution receipt')
            receipts.append(None)
    return dict(label=entry['label'],scope='warm_component' if warm else 'staged_block_segment',
                native=n,reference=r,native_over_reference=n['median']/r['median'],
                on_par_or_faster=n['median']<=r['median'],parity=parity['results'],
                frames_checked=len(parity.get('frames',[])),parity_sha256=digest(parity_path),
                timing_sha256=[digest(folder/'timing.json') for folder in (native,reference)],receipts=receipts,
                native_metrics=timings[0]['metrics'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    entries=json.loads(args.spec.read_text())
    if not entries:raise ValueError('at least one replay pair required')
    result=dict(schema='hv15n.replay_performance.v1',full_pipeline_acceptance=False,
                timing_policy='forward wall time with synchronization; setup/load and experiment totals separate',
                results=[pair(entry) for entry in entries])
    atomic_json(args.out,result)
    for entry in result['results']:
        print(f"{entry['label']}: native {entry['native']['median']:.6f}s / reference {entry['reference']['median']:.6f}s; ratio {entry['native_over_reference']:.3f}; parity PASS")


if __name__=='__main__':main()
