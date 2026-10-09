import test from 'node:test';
import assert from 'node:assert/strict';
import {contactDeform} from '../../../web/vhuman_mobile_contact.js';
const native=Float64Array.from([-1,-1,0, 1,-1,0, 1,1,0, -1,1,0].map(v=>v*.02));
const spec=(active,extra={})=>({active:Int32Array.from(active),weight:Float64Array.from(active.map(()=>1)),candidates:Int32Array.from(active.flatMap(()=>[0,1,2,0,2,3])),K:2,
    clearance:5e-4,max_move:.008,alpha:.5,iterations:2,rows:new Int32Array(0),cols:new Int32Array(0),count:new Int32Array(8),...extra});
test('contact deformer matches the Python reference on an analytic plane',()=>{
    const p=Float64Array.from([0,0,.004, .005,0,-.002, 0,.005,.02]);contactDeform(p,native,spec([0,1]));
    assert.ok(Math.abs(p[2]-5e-4)<1e-9&&Math.abs(p[5]-5e-4)<1e-9);assert.equal(p[8],.02);
    const q=Float64Array.from([0,0,.05]);contactDeform(q,native,spec([0]));assert.ok(Math.abs(q[2]-(.05-.008))<1e-9);
});
