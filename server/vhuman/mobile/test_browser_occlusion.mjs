import test from 'node:test';
import assert from 'node:assert/strict';
import {polygonFormFactor,apertureFrame,apertureShadow} from '../../../web/vhuman_mobile_occlusion.js';

const disk=(r,z,n=128)=>{const p=new Float64Array(n*3);for(let i=0;i<n;i++){const t=2*Math.PI*i/n;p[i*3]=r*Math.cos(t);p[i*3+1]=r*Math.sin(t);p[i*3+2]=z;}return p;};
test('form factor matches the analytic disk and vanishes behind the point',()=>{
    const poly=disk(1,2);
    assert.ok(Math.abs(polygonFormFactor(0,0,0,0,0,1,poly,[0,0,2])-.2)<2e-3);
    assert.equal(polygonFormFactor(0,0,0,0,0,-1,poly,[0,0,2]),0);
    assert.ok(polygonFormFactor(0,0,0,0,0,1,disk(100,1),[0,0,1])>.99);
});
test('aperture frame recovers plane, facing and area',()=>{
    const f=apertureFrame(disk(.01,.05));
    assert.ok(Math.abs(f.n[2]-1)<1e-6);assert.ok(Math.abs(f.area-Math.PI*1e-4)<2e-6);
    const e=apertureFrame(new Float64Array([-.02,-.005,0, .02,-.005,0, .02,.005,0, -.02,.005,0]));assert.ok(Math.abs(e.height-.01)<1e-9);
});
test('aperture shadow: lit through the opening, dark outside it, soft at the edge',()=>{
    const f=apertureFrame(disk(.01,.05)),up=[0,0,1];
    assert.ok(apertureShadow(0,0,0,up,f)>.99);
    assert.ok(apertureShadow(.03,0,0,up,f)<.01);
    const edge=apertureShadow(.01,0,0,up,f);assert.ok(edge>.2&&edge<.8);
    assert.equal(apertureShadow(0,0,0,[0,0,-1],f),0);
    assert.equal(apertureShadow(0,0,.06,up,f),1);
});
