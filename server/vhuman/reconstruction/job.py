"""Subprocess reconstruction jobs with cancellation and candidate isolation."""
import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from .. import gpu
from ..rig.job import DEFAULT_PYTHON, MIN_FREE_MIB
from ..service import ROOT


def reconstruction_job(service, request, progress, cancel, *, python=None, mock=False, direct=False):
    py = Path(python or DEFAULT_PYTHON)
    if not py.is_file():
        raise ValueError('rig interpreter missing')
    run_id = uuid.uuid4().hex[:16]
    if direct:
        image = Path(request.get('portrait','')).resolve()
        if not image.is_file():
            raise ValueError('portrait must be an existing local image')
        # Direct heads are published only when reconstruction and rig succeed.
        hid = uuid.uuid4().hex[:12]
        folder = service.work/'heads'/f'.{hid}.partial'
    else:
        hid = request.get('head_id')
        folder = service.head_file(hid,'head.json').parent
    model = request.get('face_model','gnm_v3')
    if model not in ('gnm_v3','ict_facekit_light'):
        raise ValueError('reconstruction requires GNM or ICT topology')
    profile = request.get('profile','full')
    if profile not in ('geometry','material','full'):
        raise ValueError('invalid reconstruction profile')
    res, iterations = int(request.get('res',512)),int(request.get('iterations',80))
    if res not in (256,512,1024) or not 10<=iterations<=300:
        raise ValueError('res must be 256/512/1024; iterations 10..300')
    roughness,f0 = float(request.get('roughness',.55)),float(request.get('f0',.028))
    if not .08<=roughness<=1. or not .005<=f0<=.04:
        raise ValueError('invalid roughness/F0 prior')
    count = int(request.get('gaussians',0))
    if count not in (0,2000,8000,20000):
        raise ValueError('gaussians must be 0/2000/8000/20000')
    cmd = [str(py),'-m','server.vhuman.reconstruction.pipeline',str(folder),'--run-id',run_id,
           '--profile',profile,'--face-model',model,'--res',str(res),'--iterations',str(iterations),
           '--roughness',str(roughness),'--f0',str(f0)]
    if request.get('build_rig',True):
        cmd.append('--rig')
    for key,flag in [('observations','--observations'),('depth_installation','--depth-installation')]:
        if request.get(key):
            cmd.extend([flag,str(request[key])])
    if count:
        cmd.extend(['--gaussians',str(count)])
    if direct:
        cmd.extend(['--portrait',str(image)])
    tail = []
    try:
        if cancel.is_set():
            raise gpu.Cancelled('cancelled')
        if direct:
            folder.mkdir(parents=True)
        progress(.02,'waiting for reconstruction device')
        with gpu.device_session(MIN_FREE_MIB,cancel,lock_path=service.work/'mock-gpu.lock' if mock else gpu.LOCK_PATH,
                                check_memory=not mock and gpu.gpu_status() is not None):
            progress(.05,'fitting portrait and baking candidate materials')
            proc = subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4'))
            stop = threading.Event()
            def watch():
                while not stop.wait(.25):
                    if cancel.is_set() and proc.poll() is None:
                        proc.terminate()
            watcher = threading.Thread(target=watch,daemon=True)
            watcher.start()
            try:
                for line in proc.stdout:
                    tail = (tail+[line.strip()])[-12:]
                rc = proc.wait()
            finally:
                stop.set()
                watcher.join()
        if cancel.is_set():
            raise gpu.Cancelled('cancelled')
        if rc:
            raise RuntimeError('reconstruction failed: '+' | '.join(tail[-5:]))
        if direct:
            meta = dict(id=hid,subject='direct portrait',source='user portrait + statistical face model',
                        reconstruction_run=run_id)
            (folder/'head.json').write_text(json.dumps(meta,indent=2))
            folder.replace(service.work/'heads'/hid)
            folder = service.work/'heads'/hid
        if cancel.is_set():
            raise gpu.Cancelled('cancelled')
        progress(1.,'candidate ready')
        base = f'/v1/heads/{hid}/reconstruction/{run_id}/'
        return dict(id=hid,run_id=run_id,manifest_url=base+'manifest.json',glb_url=base+'rig/rig.glb' if (folder/'reconstruction'/run_id/'rig/rig.glb').is_file() else None,
                    viewer_url=f'/rig?head={hid}&reconstruction={run_id}' if (folder/'reconstruction'/run_id/'rig/rig.glb').is_file() else None)
    except BaseException:
        shutil.rmtree(folder/'reconstruction'/f'.{run_id}.partial',ignore_errors=True)
        shutil.rmtree(folder/'reconstruction'/run_id,ignore_errors=True)
        if direct:
            shutil.rmtree(folder,ignore_errors=True)
        raise
