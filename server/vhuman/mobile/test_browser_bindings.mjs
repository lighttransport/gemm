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
test('native parts with appended surface-bound vertices (refined ears)',()=>{
    // One native part: vertex 0 copies native 3; vertex 1 is bound to triangle (0,1,2) plus a normal offset.
    const buffer=new ArrayBuffer(12+12+12+2*(24+24+12)),view=new DataView(buffer);let c=0;
    new Uint8Array(buffer,0,8).set(new TextEncoder().encode('VHBND002'));c=8;
    view.setUint32(c,1,true);c+=4;view.setUint32(c,2,true);c+=4;view.setInt32(c,-1,true);c+=4;view.setInt32(c,1,true);c+=4;c+=12;
    [3,0,0,0,0,0, 0,1,2,0,0,0].forEach(v=>{view.setUint32(c,v,true);c+=4;});
    [1,0,0,0,0,0, .2,.3,.5,0,0,0].forEach(v=>{view.setFloat32(c,v,true);c+=4;});
    [0,0,0, 0,0,.001].forEach(v=>{view.setFloat32(c,v,true);c+=4;});
    const [record]=context.readBindings(buffer,[{vertices:2,joint:-1,native:true}]);
    assert.deepEqual(Array.from(record.direct),[1,0]);
    const positions=new Float32Array([0,0,0, 1,0,0, 0,1,0, 7,8,9]),destination=new Float32Array(6);
    context.attach(positions,new Float32Array(48),{...record,mesh:{geometry:{attributes:{position:{array:destination}}}}});
    assert.deepEqual(Array.from(destination.slice(0,3)),[7,8,9]);
    [.3,.5,.001].forEach((v,i)=>assert.ok(Math.abs(destination[3+i]-v)<1e-6));
});
