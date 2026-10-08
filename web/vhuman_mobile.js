import * as THREE from './three.module.js';
import {GLTFLoader} from './GLTFLoader.js';
import {AvatarStream} from './vhuman_mobile_stream.js';
import {sha256} from './vhuman_mobile_hash.js';

const $=id=>document.getElementById(id), status=$('status');
const state={ready:false,pending:false,animate:false,frames:0,poseId:0,workerMs:0,updateMs:0,detail:true,errors:[]};
window.vhuman=state;
function fail(error){state.errors.push(String(error));$('error').textContent=String(error);console.error(error);}
async function bytes(url,digest,size){
    const response=await fetch(url);if(!response.ok)throw Error(`Unable to load ${url}`);
    const data=await response.arrayBuffer();
    if(size!==undefined&&data.byteLength!==size)throw Error(`Size mismatch: ${url}`);
    if(digest){
        const hash=await sha256(data);
        if(hash!==digest)throw Error(`Checksum mismatch: ${url}`);
    }
    return data;
}
const json=data=>JSON.parse(new TextDecoder().decode(data));
function readBindings(buffer,records){
    const view=new DataView(buffer);let cursor=0;
    const u32=()=>{const value=view.getUint32(cursor,true);cursor+=4;return value;};
    const i32=()=>{const value=view.getInt32(cursor,true);cursor+=4;return value;};
    const version=new TextDecoder().decode(new Uint8Array(buffer,0,8));
    if(!['VHBND001','VHBND002'].includes(version))throw Error('Invalid bindings');cursor=8;
    if(u32()!==records.length)throw Error('Binding count mismatch');
    const array=(Type,count)=>{if(cursor+count*4>buffer.byteLength)throw Error('Truncated bindings');const result=new Type(buffer,cursor,count);cursor+=count*4;return result;};
    const result=records.map(record=>{
        const count=u32(),joint=i32(),native=i32();
        if(count!==record.vertices||joint!==record.joint||Boolean(native)!==record.native)throw Error('Binding metadata mismatch');
        const center=version==='VHBND002'?array(Float32Array,3):new Float32Array(3);
        if(center.some(v=>!Number.isFinite(v)))throw Error('Invalid fitted center');
        const ids=array(Uint32Array,count*6),weights=array(Float32Array,count*6),offset=array(Float32Array,count*3);
        if(ids.some(v=>v>=17821)||weights.some(v=>!Number.isFinite(v)||v<0||v>1.0001)||offset.some(v=>!Number.isFinite(v)))throw Error('Invalid binding value');
        return {...record,ids,weights,offset,center};
    });
    if(cursor!==buffer.byteLength)throw Error('Trailing binding data');return result;
}
function attach(positions,joints,part){
    const destination=part.mesh.geometry.attributes.position.array;
    for(let i=0;i<part.vertices;i++){
        const p=i*3,b=i*6;
        if(part.joint>=0){
            const a=part.joint*12,x=part.rest[p],y=part.rest[p+1],z=part.rest[p+2];
            const d=part.center;
            for(let k=0;k<3;k++)destination[p+k]=joints[a+k*3]*(x-d[0])+joints[a+k*3+1]*(y-d[1])+joints[a+k*3+2]*(z-d[2])+joints[a+9+k]
                +joints[12+k*3]*d[0]+joints[12+k*3+1]*d[1]+joints[12+k*3+2]*d[2];
        }else if(part.native){destination.set(positions.subarray(part.ids[b]*3,part.ids[b]*3+3),p);}
        else{
            const a=part.ids[b]*3,c=part.ids[b+1]*3,d=part.ids[b+2]*3;
            let xx=positions[c]-positions[a],xy=positions[c+1]-positions[a+1],xz=positions[c+2]-positions[a+2];
            const dx=positions[d]-positions[a],dy=positions[d+1]-positions[a+1],dz=positions[d+2]-positions[a+2];
            let zx=xy*dz-xz*dy,zy=xz*dx-xx*dz,zz=xx*dy-xy*dx;
            const xn=Math.hypot(xx,xy,xz),zn=Math.hypot(zx,zy,zz);
            if(xn<1e-10||zn<1e-10){xx=0;xy=0;xz=1;zx=0;zy=0;zz=1;}
            else{xx/=xn;xy/=xn;xz/=xn;zx/=zn;zy/=zn;zz/=zn;}
            const yx=zy*xz-zz*xy,yy=zz*xx-zx*xz,yz=zx*xy-zy*xx;
            let rx=0,ry=0,rz=0;
            for(let k=0;k<6;k++){const index=part.ids[b+k]*3,w=part.weights[b+k];rx+=positions[index]*w;ry+=positions[index+1]*w;rz+=positions[index+2]*w;}
            const ox=part.offset[p],oy=part.offset[p+1],oz=part.offset[p+2];
            destination[p]=rx+xx*ox+yx*oy+zx*oz;destination[p+1]=ry+xy*ox+yy*oy+zy*oz;destination[p+2]=rz+xz*ox+yz*oy+zz*oz;
        }
    }
    part.mesh.geometry.attributes.position.needsUpdate=true;
}
function normals(parts,shared){
    shared.fill(0);
    for(const part of parts){
        const geometry=part.mesh.geometry,p=geometry.attributes.position.array,n=geometry.attributes.normal.array,t=geometry.index.array;
        n.fill(0);
        for(let i=0;i<t.length;i+=3){
            const a=t[i]*3,b=t[i+1]*3,c=t[i+2]*3;
            const x=p[b]-p[a],y=p[b+1]-p[a+1],z=p[b+2]-p[a+2],u=p[c]-p[a],v=p[c+1]-p[a+1],w=p[c+2]-p[a+2];
            const nx=y*w-z*v,ny=z*u-x*w,nz=x*v-y*u;
            for(let k=0;k<3;k++){const index=t[i+k]*3;n[index]+=nx;n[index+1]+=ny;n[index+2]+=nz;
                if(part.native){const q=part.ids[t[i+k]*6]*3;shared[q]+=nx;shared[q+1]+=ny;shared[q+2]+=nz;}}
        }
    }
    for(const part of parts){const n=part.mesh.geometry.attributes.normal.array;
        for(let i=0;i<part.vertices;i++){const p=i*3,q=part.ids[i*6]*3,source=part.native?shared:n,index=part.native?q:p;
            const length=Math.hypot(source[index],source[index+1],source[index+2])||1;
            n[p]=source[index]/length;n[p+1]=source[index+1]/length;n[p+2]=source[index+2]/length;}
        part.mesh.geometry.attributes.normal.needsUpdate=true;
    }
}
function wrinkleShader(material,detail,slopes){
    const quantized=detail.slope_encoding==='rg8_snorm_bias128';
    const texture=new THREE.DataArrayTexture(slopes,detail.resolution,detail.resolution,12);
    texture.format=THREE.RGFormat;texture.type=quantized?THREE.UnsignedByteType:THREE.FloatType;
    texture.minFilter=texture.magFilter=quantized?THREE.LinearFilter:THREE.NearestFilter;
    texture.generateMipmaps=false;texture.needsUpdate=true;
    const activation=new Float32Array(12);
    material.onBeforeCompile=shader=>{
        shader.uniforms.wrinkleSlopes={value:texture};shader.uniforms.wrinkleActivation={value:activation};
        shader.fragmentShader='uniform highp sampler2DArray wrinkleSlopes;\nuniform float wrinkleActivation[12];\n'+shader.fragmentShader;
        shader.fragmentShader=shader.fragmentShader.replace('#include <normal_fragment_maps>',`#include <normal_fragment_maps>
            vec2 wrinkleSlope=vec2(0.0);
            for(int region=0;region<12;region++){
                vec2 slope=texture(wrinkleSlopes,vec3(vNormalMapUv,float(region))).rg;
                ${quantized?`slope=(slope*255.0-128.0)*${Number(detail.slope_limit/127).toFixed(12)};`:''}
                wrinkleSlope+=slope*wrinkleActivation[region];
            }
            wrinkleSlope=clamp(wrinkleSlope,vec2(-0.5),vec2(0.5));
            normal=normalize(normal - tbn[0]*wrinkleSlope.x - tbn[1]*wrinkleSlope.y);
        `);
    };
    material.customProgramCacheKey=()=> 'vhuman-strain-v2-'+quantized;material.needsUpdate=true;
    return expression=>{
        const delta=expression.map((v,i)=>v-detail.reference[i]);
        const prior=detail.prior.map(row=>row.reduce((sum,v,i)=>sum+v*delta[i],0));
        const feature=[...delta,...prior.map(v=>v*v)];
        for(let region=0;region<12;region++)activation[region]=state.detail?Math.max(-1,Math.min(1,detail.selected_driver==='trained'?detail.weights[region].reduce((sum,v,i)=>sum+v*feature[i],0):prior[region])):0;
        state.activations=Array.from(activation);
    };
}

async function main(){
    const context=$('viewport').getContext('webgl2',{antialias:true,alpha:false,preserveDrawingBuffer:true});
    if(!context)throw Error('WebGL2 is required');
    const renderer=new THREE.WebGLRenderer({canvas:$('viewport'),context,antialias:true});
    renderer.setPixelRatio(1);renderer.setClearColor(0x181b1e);renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;
    const config=await (await fetch('./config.json')).json();
    const manifest=json(await bytes(config.package+'avatar.json',config.package_sha256));
    if(manifest.schema!=='vhuman.mobile_avatar.v1')throw Error('Unsupported avatar');
    if(manifest.material?.synthetic_completion){
        const note=document.createElement('p');note.className='muted';
        note.textContent='Unseen skin includes AI-generated texture. Photographed skin is preserved.';
        status.insertAdjacentElement('afterend',note);
    }
    if(config.skin_review==='skin-review/review.html'){
        const link=document.createElement('a');link.href=config.skin_review;link.textContent='Compare multiview skin bake';
        link.style.color='#d6b789';link.target='_blank';link.rel='noopener';status.insertAdjacentElement('afterend',link);
    }
    const buffers={};
    for(const [name,file] of Object.entries(manifest.files)){
        if(name.includes('/')||name.includes('\\')||file.bytes>256*1024*1024)throw Error('Invalid asset filename/size');
        status.textContent=`Checking ${name}…`;
        const data=await bytes(config.package+name,file.sha256,file.bytes);
        if(['gnm.bin','avatar.glb','bindings.bin','controls.json'].includes(name))buffers[name]=data;
    }
    const controls=json(buffers['controls.json']), parts=readBindings(buffers['bindings.bin'],manifest.parts);
    const speech=new AvatarStream(config.package_sha256,manifest.source_geometry_sha256,controls.reference,message=>{$('speech-status').textContent=message;});
    state.speech=speech;
    if(!globalThis.isSecureContext){
        $('connect').disabled=true;$('speak').disabled=true;$('stop').disabled=true;
        $('speech-status').textContent='Speech audio requires HTTPS or localhost. Visual preview works over LAN HTTP.';
    }
    $('connect').onclick=async()=>{try{state.animate=false;await speech.connect($('server').value);}catch(error){speech.error(error.message);}};
    $('speak').onclick=()=>{try{state.animate=false;speech.command($('text').value,$('language').value);}catch(error){$('speech-status').textContent=error.message;}};
    $('stop').onclick=()=>{try{speech.command();poseDirty=true;}catch(error){$('speech-status').textContent=error.message;}};
    const gltf=await new GLTFLoader().parseAsync(buffers['avatar.glb'],'');
    const scene=new THREE.Scene();scene.add(gltf.scene);
    for(const part of parts){
        part.mesh=gltf.scene.getObjectByName(part.name);
        if(!part.mesh?.isMesh||part.mesh.geometry.attributes.position.count!==part.vertices)throw Error('Mesh binding mismatch');
        part.mesh.frustumCulled=false;part.rest=part.mesh.geometry.attributes.position.array.slice();
        part.mesh.geometry.deleteAttribute('tangent');
        part.mesh.geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
    }
    const box=new THREE.Box3().setFromObject(gltf.scene),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
    const camera=new THREE.PerspectiveCamera(35,1,.01,10);camera.position.copy(center).add(new THREE.Vector3(0,0,Math.max(size.y*1.8,.35)));camera.lookAt(center);
    const reviewViews={front:[0,0],left:[-85,10],right:[85,10],rear:[180,20],crown:[0,75]};
    state.setView=name=>{
        if(!reviewViews[name])throw Error('Unknown review view');
        const [yaw,pitch]=reviewViews[name].map(v=>v*Math.PI/180),radius=Math.max(size.y*1.8,.35);
        camera.position.copy(center).add(new THREE.Vector3(Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)).multiplyScalar(radius));
        camera.lookAt(center);$('view').value=name;
    };
    $('view').onchange=()=>state.setView($('view').value);
    const key=new THREE.DirectionalLight(0xffefd9,3.2);key.position.set(-.4,.5,1);
    const fill=new THREE.DirectionalLight(0xcbdfff,1.1);fill.position.set(.7,.1,1);
    const ambient=new THREE.HemisphereLight(0xd6e4ff,0x5f4837,1.2);scene.add(key,fill,ambient);
    $('lighting').onchange=()=>{const mode=$('lighting').value;key.position.set(mode==='side'?-1:-.4,.5,mode==='side'?.2:1);key.intensity=mode==='soft'?1.2:3.2;fill.intensity=mode==='side'?.25:1.1;};
    let updateDetail=()=>{};
    if(config.detail){
        const detail=json(await bytes(config.detail+'detail.json',config.detail_sha256));
        if(detail.package_sha256!==config.package_sha256||detail.regions.length!==12)throw Error('Skin detail identity mismatch');
        if(detail.slope_encoding&&detail.slope_encoding!=='rg8_snorm_bias128')throw Error('Unsupported wrinkle encoding');
        const filename=detail.slope_file||'wrinkle_slopes.f32',file=detail.files[filename];
        const payload=await bytes(config.detail+filename,file.sha256,file.bytes);
        const slopes=detail.slope_encoding==='rg8_snorm_bias128'?new Uint8Array(payload):new Float32Array(payload);
        if(slopes.length!==12*detail.resolution**2*2||slopes.some(v=>!Number.isFinite(v)))throw Error('Invalid wrinkle texture');
        updateDetail=wrinkleShader(parts.find(p=>p.name==='skin').mesh.material,detail,slopes);
        state.detailMetrics=detail.metrics;
        state.detailDriver=detail.selected_driver;
    }else{$('detail').disabled=true;}
    const worker=new Worker('./vhuman_mobile_worker.js',{type:'module'}),shared=new Float32Array(17821*3);
    let latestPose=null,latestExpression=controls.reference.slice(),sweepStarted=0,lastReport=performance.now(),frames=0,poseDirty=false;
    let id=0;
    function submit(expression=controls.reference,yaw=Number($('yaw').value),rotations=null,translation=[0,0,0]){
        if(state.pending)return false;
        const pose=new Float32Array(398);pose.set(expression);if(rotations)pose.set(rotations,383);else pose[384]=yaw;pose.set(translation,395);
        latestExpression=Array.from(expression);state.pending=true;
        worker.postMessage({type:'pose',pose,id:++id},[pose.buffer]);return true;
    }
    state.submitPose=submit;
    state.setPose=async(expression,yaw=0,rotations=null,translation=[0,0,0])=>{
        state.animate=false;
        while(state.pending)await new Promise(resolve=>setTimeout(resolve,10));
        submit(expression,yaw,rotations,translation);const wanted=id;
        while(state.poseId<wanted){if(state.errors.length)throw Error(state.errors.at(-1));await new Promise(resolve=>setTimeout(resolve,10));}
        return {workerMs:state.workerMs,updateMs:state.updateMs};
    };
    worker.onerror=error=>fail(error.message);
    worker.onmessage=({data})=>{
        if(data.type==='error'){state.pending=false;fail(data.message);return;}
        if(data.type==='ready'){submit();return;}
        if(data.type==='pose'){
            latestPose=data;const start=performance.now();
            for(const part of parts)attach(data.positions,data.joints,part);normals(parts,shared);updateDetail(latestExpression);
            state.updateMs=performance.now()-start;state.workerMs=data.milliseconds;state.poseId=data.id;state.pending=false;
            state.ready=true;state.reference=controls.reference;state.vertexSample=Array.from(data.positions.slice(0,30));
            for(const name of ['sweep','reset'])$(name).disabled=false;
            for(const name of ['connect','speak','stop'])$(name).disabled=!globalThis.isSecureContext;
        }
    };
    state.nativeVertices=()=>Array.from(latestPose?.positions||[]);
    state.renderVertices=name=>Array.from(parts.find(p=>p.name===name).mesh.geometry.attributes.position.array);
    worker.postMessage({type:'init',weights:buffers['gnm.bin']},[buffers['gnm.bin']]);
    $('yaw').oninput=()=>{state.animate=false;poseDirty=true;};
    $('detail').onchange=()=>{state.detail=$('detail').checked;updateDetail(latestExpression);};
    $('sweep').onclick=()=>{state.animate=!state.animate;sweepStarted=performance.now();$('sweep').textContent=state.animate?'Pause sweep':'Play pose sweep';};
    $('reset').onclick=()=>{state.animate=false;$('yaw').value=0;state.setView('front');$('sweep').textContent='Play pose sweep';poseDirty=true;};
    document.addEventListener('visibilitychange',()=>{if(document.hidden){state.animate=false;speech.close();}});
    window.addEventListener('pagehide',()=>{worker.terminate();speech.close();});
    let lastDraw=0;
    function frame(now){
        requestAnimationFrame(frame);if(document.hidden||now-lastDraw<1000/30)return;
        lastDraw+=Math.floor((now-lastDraw)/(1000/30))*(1000/30);
        const rect=$('viewport').getBoundingClientRect(),scale=Math.min(1,720/rect.width,1280/rect.height),w=Math.max(1,Math.round(rect.width*scale)),h=Math.max(1,Math.round(rect.height*scale));
        if(renderer.domElement.width!==w||renderer.domElement.height!==h){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
        if(state.animate){
            const t=(now-sweepStarted)/1000,x=controls.reference.map((v,i)=>Math.max(-3,Math.min(3,v+(i<24?.1*Math.sin(t*1.4+i):0))));
            submit(x,.4*Math.sin(t*.7));
        }
        if(poseDirty&&state.ready&&!state.pending){submit();poseDirty=false;}
        if(!state.animate&&state.ready&&!state.pending){const pose=speech.sample();if(pose)submit(pose.expression,0,pose.rotations,pose.translation);}
        renderer.render(scene,camera);state.frames++;frames++;
        if(now-lastReport>1000){
            state.fps=frames*1000/(now-lastReport);frames=0;lastReport=now;
            status.textContent=`WebGL2 • ${manifest.triangles.toLocaleString()} triangles\nDisplay ${state.fps.toFixed(1)} FPS\nWASM pose ${state.workerMs.toFixed(1)} ms\nBindings + normals ${state.updateMs.toFixed(1)} ms\n${state.detailMetrics?(state.detailDriver==='trained'?'Trained strain driver · held-out checked':'Analytic strain driver · fit rejected'):'Static skin detail'}\nReference build · device quality unverified`;
        }
    }
    requestAnimationFrame(frame);
}
main().catch(fail);
