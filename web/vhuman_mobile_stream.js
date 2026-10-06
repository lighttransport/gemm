export class AvatarStream {
    constructor(packageHash,identity,reference,onStatus){
        this.packageHash=packageHash;this.identity=identity;this.reference=reference;this.onStatus=onStatus;
        this.socket=null;this.context=null;this.node=null;this.epoch=-1;this.expectedEpoch=-1;this.frames=[];this.markers=[];this.ready=false;this.waiting=true;this.position=0;this.underruns=0;
    }
    async connect(address){
        if(globalThis.isSecureContext===false)throw Error('Speech audio requires HTTPS or localhost');
        this.close();const url=new URL(address,location.href);
        if(!['ws:','wss:'].includes(url.protocol))throw Error('Use a WebSocket server address');
        this.context=new AudioContext();await this.context.audioWorklet.addModule('./vhuman_mobile_audio.js');
        this.node=new AudioWorkletNode(this.context,'avatar-audio',{numberOfInputs:0,numberOfOutputs:1,outputChannelCount:[1]});
        this.node.connect(this.context.destination);await this.context.resume();
        this.node.port.onmessage=({data})=>{
            if(data.type==='error'){this.error(data.message);return;}
            if(data.type==='clock'&&data.epoch===this.epoch){
                this.markers.push({time:data.frame/this.context.sampleRate,position:data.position});
                if(this.markers.length>128)this.markers.shift();this.underruns=data.underruns;
            }
        };
        const socket=new WebSocket(url);this.socket=socket;
        socket.onopen=()=>socket.send(JSON.stringify({type:'hello',schema:'vhuman.mobile_stream.v1',package_sha256:this.packageHash}));
        socket.onmessage=({data})=>{if(socket!==this.socket)return;try{this.receive(data);}catch(error){this.error(error.message);}};
        socket.onerror=()=>{if(socket===this.socket)this.error('Speech server connection failed');};
        socket.onclose=()=>{if(socket===this.socket){this.close();this.onStatus('Disconnected');}};
    }
    close(){
        const socket=this.socket;this.socket=null;socket?.close();
        this.node?.disconnect();this.context?.close();this.node=this.context=null;
        this.ready=false;this.waiting=true;this.epoch=this.expectedEpoch=-1;this.frames=[];this.markers=[];this.position=0;
    }
    error(message){this.close();this.onStatus(message);}
    command(text=null,language='en'){
        if(!this.ready)throw Error('Connect to the speech server first');
        if(text!==null&&(!text.length||text.length>4096))throw Error('Enter 1–4096 characters');
        this.waiting=true;this.expectedEpoch++;this.frames=[];this.markers=[];this.position=0;
        this.node.port.postMessage({type:'begin',epoch:-1});
        this.socket.send(JSON.stringify(text===null?{type:'cancel'}:{type:'speak',text,language}));
    }
    receive(raw){
        if(typeof raw!=='string'||raw.length>128*1024)throw Error('Invalid speech packet');
        const p=JSON.parse(raw),integer=x=>Number.isSafeInteger(x)&&x>=0;
        if(p.type==='ready'){
            if(this.ready||p.schema!=='vhuman.mobile_stream.v1'||p.geometry_sha256!==this.identity)throw Error('Avatar handshake mismatch');
            this.ready=true;this.onStatus('Connected');return;
        }
        if(!this.ready||!integer(p.epoch))throw Error('Missing speech handshake');
        if(p.type==='begin'){
            if(p.epoch<this.expectedEpoch)return;
            if(p.epoch!==this.expectedEpoch)throw Error('Unrequested speech epoch');
            if(p.epoch<=this.epoch)throw Error('Invalid speech epoch');
            this.epoch=p.epoch;this.sequence=this.accepted=0;this.lastMotion=-1;this.ended=false;this.waiting=false;
            this.frames=[];this.markers=[];this.position=0;this.node.port.postMessage({type:'begin',epoch:this.epoch});return;
        }
        if(p.epoch<this.epoch||this.waiting)return;
        if(p.epoch!==this.epoch)throw Error('Unexpected speech epoch');
        if(p.type==='audio'){
            if(this.ended||p.sequence!==this.sequence||p.sample_start!==this.accepted||p.sample_rate!==24000||typeof p.pcm!=='string')throw Error('Audio discontinuity');
            const raw=atob(p.pcm);if(!raw.length||raw.length%4||raw.length>4800*4)throw Error('Invalid PCM length');
            const data=Uint8Array.from(raw,c=>c.charCodeAt(0)),view=new DataView(data.buffer),pcm=new Float32Array(raw.length/4);
            for(let i=0;i<pcm.length;i++){pcm[i]=view.getFloat32(i*4,true);if(!Number.isFinite(pcm[i])||Math.abs(pcm[i])>1)throw Error('Invalid PCM amplitude');}
            this.node.port.postMessage({type:'audio',epoch:this.epoch,start:this.accepted,sequence:this.sequence,pcm},[pcm.buffer]);
            this.accepted+=raw.length/4;this.sequence++;
        }else if(p.type==='motion'){
            const valid=(a,n,b)=>Array.isArray(a)&&a.length===n&&a.every(v=>Number.isFinite(v)&&Math.abs(v)<=b);
            if(this.ended||!integer(p.sample_position)||p.sample_position<=this.lastMotion||this.frames.length>=256||
                !valid(p.expression,383,3)||!valid(p.rotations,12,3.15)||!valid(p.translation,3,10))throw Error('Invalid native motion');
            if(this.lastMotion<0&&p.sample_position!==0)throw Error('Motion must start at sample zero');
            this.frames.push(p);this.lastMotion=p.sample_position;
        }else if(p.type==='end'){
            if(this.ended||p.samples!==this.accepted)throw Error('Audio length mismatch');this.ended=true;
            this.node.port.postMessage({type:'end',epoch:this.epoch});
        }else if(p.type==='error')throw Error(p.message||'Speech server failed');
        else throw Error('Unknown speech packet');
    }
    sample(){
        if(this.waiting||!this.context)return null;
        // getOutputTimestamp maps AudioContext time to the actual output device.
        // Interpolate source-sample markers, including flat underrun intervals.
        const stamp=this.context.getOutputTimestamp?.();
        const audible=stamp?.contextTime??Math.max(0,this.context.currentTime-(this.context.outputLatency||0)-(this.context.baseLatency||0));
        let position=0;
        for(let i=0;i<this.markers.length;i++){
            const a=this.markers[i];if(a.time>audible)break;position=a.position;
            const b=this.markers[i+1];if(b&&b.time>audible){position=a.position+(b.position-a.position)*(audible-a.time)/(b.time-a.time);break;}
        }
        this.position=Math.max(this.position,position);
        if(this.ended&&this.position>=this.accepted)return {expression:this.reference,rotations:new Array(12).fill(0),translation:[0,0,0]};
        while(this.frames.length>2&&this.frames[1].sample_position<=this.position)this.frames.shift();
        const a=this.frames[0];if(!a)return null;
        const b=this.frames[1];if(!b)return a;
        const t=Math.max(0,Math.min(1,(this.position-a.sample_position)/(b.sample_position-a.sample_position)));
        const mix=name=>a[name].map((v,i)=>v+t*(b[name][i]-v));
        return {expression:mix('expression'),rotations:mix('rotations'),translation:mix('translation')};
    }
}
