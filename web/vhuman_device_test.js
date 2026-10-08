const $=id=>document.getElementById(id),frame=$('viewer');
let running=false,report=null,hidden=false;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const percentile=(values,p)=>values.slice().sort((a,b)=>a-b)[Math.min(values.length-1,Math.floor(values.length*p))]??null;
document.addEventListener('visibilitychange',()=>{if(running&&document.hidden)hidden=true;});
async function ready(){
    const start=performance.now();
    while(!frame.contentWindow?.vhuman?.ready){
        const errors=frame.contentWindow?.vhuman?.errors;if(errors?.length)throw Error(errors.join('; '));
        if(performance.now()-start>180000)throw Error('Avatar loading timed out');
        await sleep(100);
    }
    return frame.contentWindow.vhuman;
}
function visual(){return Object.fromEntries(['seams','identity','lighting'].map(k=>[k,$('visual-'+k).checked]));}
function show(){
    if(!report)return;report.visual_review=visual();report.visual_review_complete=Object.values(report.visual_review).every(Boolean);
    $('result').textContent=JSON.stringify(report,null,2);
}
async function run({phaseMs=Number($('duration').value)*1000}={}){
    if(running)throw Error('A device test is already running');
    if(!Number.isFinite(phaseMs)||phaseMs<1000||phaseMs>60000)throw Error('Invalid test duration');
    running=true;hidden=document.hidden;report=null;$('run').disabled=true;$('download').disabled=true;
    let state,doc;
    try{
        state=await ready();doc=frame.contentDocument;
        const response=await fetch('./config.json');if(!response.ok)throw Error('Missing player config');const config=await response.json();
        const canvas=doc.getElementById('viewport'),gl=canvas.getContext('webgl2'),debug=gl.getExtension('WEBGL_debug_renderer_info');
        report={schema:'vhuman.device_test.v1',started_at:new Date().toISOString(),device_model:$('model').value.trim()||'unspecified',
            user_agent:navigator.userAgent,platform:navigator.platform,hardware_concurrency:navigator.hardwareConcurrency,
            device_memory_gib:navigator.deviceMemory??null,device_pixel_ratio:devicePixelRatio,secure_context:isSecureContext,
            package_sha256:config.package_sha256,detail_sha256:config.detail_sha256,
            renderer:debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),
            phases:[],not_tested:['speech/audio','native geometry parity','physical device identity verification'],visual_review:visual()};
        const change=(id,value)=>{const node=doc.getElementById(id);if(node.type==='checkbox')node.checked=value;else node.value=value;node.dispatchEvent(new frame.contentWindow.Event('change'));};
        const animate=value=>{if(state.animate!==value)doc.getElementById('sweep').click();};
        await state.setPose(state.reference,0);change('detail',true);
        for(const phase of ['static','animated','relighting','detail_off']){
            $('status').textContent='Measuring '+phase.replace('_',' ')+'…';animate(phase!=='static');change('detail',phase!=='detail_off');
            const begin={time:performance.now(),frames:state.frames,poses:state.poseId};const worker=[],update=[];let lastPose=state.poseId,step=0;
            while(performance.now()-begin.time<phaseMs){
                if(phase==='relighting'){
                    state.setView(['front','right','rear','left','crown'][step%5]);change('lighting',['studio','side','soft'][step%3]);
                }
                await sleep(250);step++;
                if(hidden||document.hidden)throw Error('Test interrupted: keep the page visible');
                if(state.errors.length)throw Error(state.errors.join('; '));
                if(gl.isContextLost())throw Error('WebGL context lost');
                if(state.poseId!==lastPose){worker.push(state.workerMs);update.push(state.updateMs);lastPose=state.poseId;}
            }
            const seconds=(performance.now()-begin.time)/1000;
            report.phases.push({name:phase,seconds,display_fps:(state.frames-begin.frames)/seconds,pose_fps:(state.poseId-begin.poses)/seconds,
                worker_p95_ms:percentile(worker,.95),update_p95_ms:percentile(update,.95),pose_timing_samples:worker.length,
                render_size:[canvas.width,canvas.height]});
        }
        report.errors=state.errors.slice();report.interrupted=false;
        report.timing_pass=report.phases.every(p=>p.display_fps>=24&&(p.name==='static'||p.pose_fps>=20));
        report.thresholds={display_fps_min:24,animated_pose_fps_min:20};
        $('status').textContent=report.timing_pass?'Timing checks passed. Complete the visual review and download results.':'Timing checks did not meet the target. Download results for diagnosis.';
    }catch(error){
        report=report||{schema:'vhuman.device_test.v1',phases:[]};report.errors=[String(error)];report.interrupted=hidden;report.timing_pass=false;
        $('status').textContent=String(error);
    }finally{
        if(state){state.animate=false;state.setView('front');if(doc){doc.getElementById('sweep').textContent='Play pose sweep';doc.getElementById('lighting').value='studio';doc.getElementById('lighting').dispatchEvent(new frame.contentWindow.Event('change'));}}
        running=false;$('run').disabled=false;$('download').disabled=!report;show();
    }
    return report;
}
$('run').onclick=()=>run();
for(const name of ['seams','identity','lighting'])$('visual-'+name).onchange=show;
$('download').onclick=()=>{
    show();const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const link=document.createElement('a');link.href=url;link.download='vhuman-device-test.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
window.vhumanDeviceTest={run,get report(){return report;}};
ready().then(()=>$('status').textContent='Avatar ready. Start the test when the device is ready.').catch(error=>$('status').textContent=String(error));
