import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {AvatarStream} from '../../../web/vhuman_mobile_stream.js';

function processor(){
    let Processor;const messages=[];
    const context={sampleRate:48000,currentFrame:0,
        AudioWorkletProcessor:class{constructor(){this.port={postMessage:value=>messages.push(value)};}},
        registerProcessor:(name,type)=>{Processor=type;}};
    vm.runInNewContext(fs.readFileSync(new URL('../../../web/vhuman_mobile_audio.js',import.meta.url),'utf8'),context);
    const p=new Processor();
    return {p,messages,send:data=>p.port.onmessage({data}),block:()=>{
        const output=new Float32Array(128);p.process([],[ [output] ]);context.currentFrame+=128;return output;
    }};
}

test('audio resampling, underrun freeze and cancellation preserve sample positions',()=>{
    const {p,send,block}=processor();send({type:'begin',epoch:0});
    send({type:'audio',epoch:0,sequence:0,start:0,pcm:new Float32Array(3840).fill(.25)});
    for(let i=0;i<30;i++)assert.ok(block().every(v=>v===.25));
    assert.equal(p.read,1920);
    for(let i=0;i<40;i++)block();
    const frozen=p.read;block();assert.equal(p.read,frozen);assert.ok(p.underruns>0);
    send({type:'begin',epoch:1});send({type:'audio',epoch:0,sequence:1,start:3840,pcm:new Float32Array(1920)});
    assert.equal(p.written,0);assert.equal(p.read,0);assert.ok(block().every(v=>v===0));
});

test('audio gap fails before appending data',()=>{
    const {p,send,messages}=processor();send({type:'begin',epoch:0});
    send({type:'audio',epoch:0,sequence:1,start:0,pcm:new Float32Array(1920)});
    assert.equal(p.written,0);assert.equal(messages.at(-1).type,'error');
});

test('cancel ignores an older begin still in flight',()=>{
    const stream=new AvatarStream('package','identity',new Array(383).fill(0),()=>{});
    stream.ready=true;stream.socket={send:()=>{}};stream.node={port:{postMessage:()=>{}}};
    stream.command('first');stream.command();
    stream.receive(JSON.stringify({type:'begin',epoch:0}));assert.equal(stream.waiting,true);assert.equal(stream.epoch,-1);
    stream.receive(JSON.stringify({type:'begin',epoch:1}));assert.equal(stream.epoch,1);assert.equal(stream.accepted,0);
});

test('render clock interpolates device-time markers including underrun plateaus',()=>{
    const stream=new AvatarStream('package','identity',new Array(383).fill(0),()=>{});
    stream.waiting=false;stream.context={getOutputTimestamp:()=>({contextTime:.015})};
    stream.markers=[{time:0,position:0},{time:.01,position:240},{time:.02,position:240}];
    stream.sample();assert.equal(stream.position,240);
    stream.context.getOutputTimestamp=()=>({contextTime:.025});stream.markers.push({time:.03,position:480});
    stream.sample();assert.ok(Math.abs(stream.position-360)<1e-8);
});
