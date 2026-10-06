import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash,webcrypto} from 'node:crypto';
import {sha256} from '../../../web/vhuman_mobile_hash.js';

test('LAN SHA-256 matches reference across padding and yielding boundaries',async()=>{
    for(const length of [0,1,3,55,56,63,64,65,119,120,127,128,129,1000000,1048577]){
        const data=Uint8Array.from({length},(_,i)=>(i*73+19)&255);
        const expected=createHash('sha256').update(data).digest('hex');
        assert.equal(await sha256(data.buffer,null),expected,`fallback length ${length}`);
        assert.equal(await sha256(data.buffer,webcrypto.subtle),expected,`native length ${length}`);
    }
});

test('LAN SHA-256 detects changed asset bytes',async()=>{
    const data=new TextEncoder().encode('abc');
    assert.equal(await sha256(data.buffer,null),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    data[1]^=1;
    assert.notEqual(await sha256(data.buffer,null),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
});
