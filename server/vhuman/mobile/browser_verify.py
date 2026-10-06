"""Run the local WebGL2 player in Chromium and compare WASM with native C++."""
import argparse
import asyncio
import base64
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import queue
import time
from urllib.request import urlopen,Request
import numpy as np
from .native import Native,build
from ..test_browser import Cdp,find_chrome


@contextmanager
def speech_fixture(package):
    """Silent timed PCM + native motion fixture; this is not a TTS quality test."""
    from websockets.asyncio.server import serve
    from .stream import connection_handler
    from . import wire
    from ..realtime.src.pipeline.protocol import AudioChunk
    reference=np.asarray(json.loads((package/'controls.json').read_text())['reference'])
    ready=queue.Queue();stop=threading.Event()
    async def provider(request,epoch):
        for sequence in range(25):
            position=sequence*1920
            expression=reference.copy();expression[:12]=np.clip(expression[:12]+.04*np.sin(sequence+np.arange(12)),-3,3)
            yield wire.pose(epoch,position,expression)
            yield wire.audio(AudioChunk(epoch,sequence,position,np.zeros(1920)))
            await asyncio.sleep(.08)
    async def run():
        async with serve(connection_handler(package,provider),'127.0.0.1',0) as server:
            ready.put(server.sockets[0].getsockname()[1])
            while not stop.is_set():await asyncio.sleep(.05)
    thread=threading.Thread(target=lambda:asyncio.run(run()),daemon=True);thread.start()
    try:yield ready.get(timeout=10)
    finally:stop.set();thread.join(timeout=5)


def verify(player,out,hardware=False):
    player,out=Path(player).resolve(),Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if not find_chrome():raise RuntimeError('Chromium not installed')
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(player)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    chrome=cdp=None
    try:
        profile=out/'chrome-profile';profile.mkdir(exist_ok=True)
        gpu=['--use-gl=angle','--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface'] if hardware else ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']
        environment=dict(os.environ,TMPDIR=str(out));environment.pop('DISPLAY',None)
        with (out/'chrome.log').open('w') as log:
            chrome=subprocess.Popen([find_chrome(),'--headless=new','--ozone-platform=headless','--no-sandbox','--no-first-run','--no-default-browser-check','--autoplay-policy=no-user-gesture-required',
                '--ignore-gpu-blocklist','--window-size=1100,900','--remote-debugging-port=0',f'--user-data-dir={profile}',*gpu,'about:blank'],
                stdout=log,stderr=subprocess.STDOUT,env=environment)
            active=profile/'DevToolsActivePort';deadline=time.monotonic()+30
            while not active.exists():
                if time.monotonic()>deadline or chrome.poll() is not None:raise RuntimeError((out/'chrome.log').read_text()[-3000:])
                time.sleep(.1)
            port=active.read_text().splitlines()[0]
            target=json.loads(urlopen(Request(f'http://127.0.0.1:{port}/json/new',method='PUT')).read())
            cdp=Cdp(target['webSocketDebuggerUrl'])
            for domain in ('Runtime.enable','Log.enable','Page.enable'):cdp.call(domain)
            cdp.call('Page.navigate',{'url':f'http://127.0.0.1:{server.server_port}/'})
            cdp.wait_for('window.vhuman && (vhuman.ready || vhuman.errors.length)',timeout=60)
            errors=cdp.evaluate('vhuman.errors')
            if errors:raise RuntimeError(errors)
            reference=json.loads((player/'avatar/controls.json').read_text())['reference']
            from ..eye.glb import GLB
            from ..reconstruction.offline_assets import attachment_frames
            manifest=json.loads((player/'avatar/avatar.json').read_text());glb=GLB.load(player/'avatar/avatar.glb')
            bindings=np.load(player/'avatar/bindings.npz',allow_pickle=False)
            errors=[];binding_errors=[];timings=[];activation_errors=[]
            with Native(build(out/'native'),player/'avatar/gnm.bin') as native:
                for index,yaw in enumerate((0,.35,-.35)):
                    expression=np.asarray(reference).copy()
                    expression[:24]=np.clip(expression[:24]+.1*np.sin(np.arange(24)+index),-3,3)
                    times=cdp.evaluate(f'vhuman.setPose({json.dumps(expression.tolist())},{yaw})')
                    actual=np.asarray(cdp.evaluate('vhuman.nativeVertices()')).reshape(-1,3)
                    rotations=np.zeros((4,3));rotations[0,1]=yaw
                    expected=native.evaluate(expression,rotations)
                    errors.append(np.linalg.norm(actual-expected,axis=1)*1000);timings.append(times)
                    for part in manifest['parts']:
                        name=part['name'];ids=bindings[name+'_ids'];weights=bindings[name+'_weights'];offset=bindings[name+'_offset']
                        if part['native']:want=expected[ids[:,0]]
                        elif part['joint']>=0:
                            primitive=glb.doc['meshes'][part['mesh']]['primitives'][0]
                            rest=glb.accessor(primitive['attributes']['POSITION']);matrix,translation=native.joint_transform(part['joint'])
                            want=rest@matrix.T+translation
                        else:
                            root=(expected[ids]*weights[:,:,None]).sum(1)
                            want=root+np.einsum('vij,vj->vi',attachment_frames(expected[ids[:,:3]]),offset)
                        rendered=np.asarray(cdp.evaluate(f'vhuman.renderVertices({json.dumps(name)})')).reshape(-1,3)
                        binding_errors.extend(np.linalg.norm(rendered-want,axis=1)*1000)
                    if (player/'detail/detail.json').is_file():
                        detail=json.loads((player/'detail/detail.json').read_text())
                        delta=expression-np.asarray(detail['reference']);prior=np.asarray(detail['prior'])@delta
                        target=np.clip(np.asarray(detail['weights'])@np.r_[delta,prior**2] if detail['selected_driver']=='trained' else prior,-1,1)
                        activation_errors.extend(abs(np.asarray(cdp.evaluate('vhuman.activations'))-target))
                    cdp.wait_for('vhuman.frames>5')
                    shot=cdp.call('Page.captureScreenshot',{'format':'png'})
                    (out/f'pose-{index}.png').write_bytes(base64.b64decode(shot['data']))
            # Rendering tests: relighting and detail must change pixels while
            # leaving the native pose untouched.
            def canvas():return cdp.evaluate('document.querySelector("canvas").toDataURL()')
            before=canvas();frame=cdp.evaluate('vhuman.frames')
            cdp.evaluate('document.getElementById("lighting").value="side";document.getElementById("lighting").dispatchEvent(new Event("change"))')
            cdp.wait_for(f'vhuman.frames>{frame+1}')
            if canvas()==before:raise AssertionError('relighting did not change the rendered frame')
            before=canvas();frame=cdp.evaluate('vhuman.frames')
            cdp.evaluate('document.getElementById("detail").checked=false;document.getElementById("detail").dispatchEvent(new Event("change"))')
            cdp.wait_for(f'vhuman.frames>{frame+1}')
            if canvas()==before:raise AssertionError('skin detail toggle did not change pixels')
            # Inspection cameras must reveal new surfaces without changing the
            # native fitted geometry or expression.
            native_before=cdp.evaluate('vhuman.nativeVertices()')
            cdp.evaluate('document.getElementById("lighting").value="studio";document.getElementById("lighting").dispatchEvent(new Event("change"))')
            for view in ('left','right','rear','crown','front'):
                before=canvas();frame=cdp.evaluate('vhuman.frames')
                cdp.evaluate(f'vhuman.setView({json.dumps(view)})')
                cdp.wait_for(f'vhuman.frames>{frame+1}')
                if canvas()==before:raise AssertionError('review camera did not change the frame: '+view)
                if cdp.evaluate('vhuman.nativeVertices()')!=native_before:raise AssertionError('review camera changed native geometry')
                shot=cdp.call('Page.captureScreenshot',{'format':'png'})
                (out/f'view-{view}.png').write_bytes(base64.b64decode(shot['data']))
            with speech_fixture(player/'avatar') as speech_port:
                cdp.evaluate(f'vhuman.speech.connect("ws://127.0.0.1:{speech_port}")')
                cdp.wait_for('vhuman.speech.ready',timeout=10)
                cdp.evaluate('vhuman.speech.command("こんにちは","ja")')
                cdp.wait_for('vhuman.speech.ended && vhuman.speech.position >= 48000',timeout=15)
                audio=cdp.evaluate('({samples:vhuman.speech.position,underruns:vhuman.speech.underruns,rate:vhuman.speech.context.sampleRate})')
                cdp.evaluate('vhuman.speech.command("cancel test","en")')
                cdp.wait_for('vhuman.speech.accepted>0 && vhuman.speech.epoch===1')
                cdp.evaluate('vhuman.speech.command()')
                cdp.wait_for('vhuman.speech.epoch===2 && vhuman.speech.position===0')
                cdp.evaluate('vhuman.speech.close()')
            # Measure steady rendering separately from readback/startup stalls.
            first=cdp.evaluate('({frames:vhuman.frames,time:performance.now()})')
            cdp.wait_for(f'performance.now()>{first["time"]+3000}')
            last=cdp.evaluate('({frames:vhuman.frames,time:performance.now()})')
            cdp.evaluate('document.getElementById("detail").checked=true;document.getElementById("detail").dispatchEvent(new Event("change"));document.getElementById("sweep").click()')
            animation_start=cdp.evaluate('({frames:vhuman.frames,poses:vhuman.poseId,time:performance.now()})')
            cdp.wait_for(f'performance.now()>{animation_start["time"]+3000}')
            animation_end=cdp.evaluate('({frames:vhuman.frames,poses:vhuman.poseId,time:performance.now()})')
            cdp.evaluate('document.getElementById("sweep").click()')
            driver=cdp.evaluate('(()=>{let g=document.querySelector("canvas").getContext("webgl2"),e=g.getExtension("WEBGL_debug_renderer_info");return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)})()')
            console=[e for e in cdp.console_errors() if 'favicon.ico' not in e]
            if console:raise AssertionError(console)
            result=dict(p95_vertex_mm=float(np.percentile(errors,95)),max_vertex_mm=float(np.max(errors)),
                p95_binding_mm=float(np.percentile(binding_errors,95)),max_activation_error=max(activation_errors,default=0),
                poses=len(errors),timings=timings,renderer=driver,display_fps=(last['frames']-first['frames'])*1000/(last['time']-first['time']),
                animated_display_fps=(animation_end['frames']-animation_start['frames'])*1000/(animation_end['time']-animation_start['time']),
                animated_pose_fps=(animation_end['poses']-animation_start['poses'])*1000/(animation_end['time']-animation_start['time']),
                audio=audio,frame_checks=['relighting','dynamic detail','ear/rear/crown cameras','audio sample clock','cancellation'],
                passed=bool(float(np.percentile(errors,95))<.25 and float(np.percentile(binding_errors,95))<.25 and max(activation_errors,default=0)<1e-5))
            (out/'verification.json').write_text(json.dumps(result,indent=2))
            if not result['passed']:raise AssertionError(result)
            return result
    finally:
        if cdp:cdp.close()
        if chrome:
            chrome.terminate()
            try:chrome.wait(timeout=5)
            except subprocess.TimeoutExpired:chrome.kill();chrome.wait()
        server.shutdown();server.server_close();thread.join(timeout=2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--player',required=True);p.add_argument('--out',required=True)
    p.add_argument('--hardware',action='store_true');print(json.dumps(verify(**vars(p.parse_args())),indent=2))


if __name__=='__main__':main()
