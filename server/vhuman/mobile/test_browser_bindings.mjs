import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

// Exercise the shipped browser functions without initializing WebGL or a DOM.
const source=readFileSync(new URL('../../../web/vhuman_mobile.js',import.meta.url),'utf8');
const context=vm.createContext({TextDecoder,DataView,Uint8Array,Uint32Array,Float32Array});
vm.runInContext(source.slice(source.indexOf('function readBindings('),source.indexOf('function normals(')),context);
function fixture(version,center=[.002,0,0]){
    const buffer=new ArrayBuffer(12+12+(version===2?12:0)+60),view=new DataView(buffer);
    new Uint8Array(buffer,0,8).set(new TextEncoder().encode(`VHBND00${version}`));
    view.setUint32(8,1,true);view.setUint32(12,1,true);view.setInt32(16,2,true);
    if(version===2)center.forEach((v,i)=>view.setFloat32(24+i*4,v,true));
    return buffer;
}
test('binding versions retain fitted centers and reject nonfinite centers',()=>{
    const records=[{vertices:1,joint:2,native:false}];
    assert.equal(context.readBindings(fixture(1),records)[0].center[0],0);
    assert.ok(Math.abs(context.readBindings(fixture(2),records)[0].center[0]-.002)<1e-9);
    assert.throws(()=>context.readBindings(fixture(2,[NaN,0,0]),records),/fitted center/);
});
test('fitted globe center stays fixed under gaze and follows the head',()=>{
    const d=[.002,.001,0],joints=new Float32Array(48);
    // Head turns 90 degrees around Z, eye turns another 90 degrees.
    joints.set([0,-1,0,1,0,0,0,0,1,.03,.04,.05],12);
    joints.set([-1,0,0,0,-1,0,0,0,1,.03,.04,.05],24);
    const destination=new Float32Array(3),part={vertices:1,joint:2,rest:d,center:d,
        mesh:{geometry:{attributes:{position:{array:destination}}}}};
    context.attach(new Float32Array(0),joints,part);
    // A point at the fitted center moves only with the head, independent of gaze.
    [.029,.042,.05].forEach((v,i)=>assert.ok(Math.abs(destination[i]-v)<1e-8));
});
