/* 24kHz source PCM -> device-rate output. The source timeline pauses on silence
 * caused by underruns; renderer time is never used to advance speech. */
class AvatarAudio extends AudioWorkletProcessor {
    constructor(){
        super();this.pcm=new Float32Array(96000);this.epoch=-1;this.read=0;this.written=0;this.sequence=0;this.started=false;this.ended=false;this.blocks=0;this.underruns=0;
        this.port.onmessage=({data})=>{
            if(data.type==='begin'){
                this.epoch=data.epoch;this.read=this.written=this.sequence=0;this.started=this.ended=false;this.underruns=0;
                this.port.postMessage({type:'clock',epoch:this.epoch,frame:currentFrame,position:0,underruns:0});
            }else if(data.epoch===this.epoch&&data.type==='audio'){
                if(data.start!==this.written||data.sequence!==this.sequence||this.written-Math.floor(this.read)+data.pcm.length>this.pcm.length){this.port.postMessage({type:'error',message:'Audio gap or buffer overflow'});return;}
                for(const sample of data.pcm)this.pcm[(this.written++)%this.pcm.length]=sample;
                this.sequence++;
            }else if(data.epoch===this.epoch&&data.type==='end')this.ended=true;
        };
    }
    process(inputs,outputs){
        const output=outputs[0][0],step=24000/sampleRate;
        if(!this.started&&(this.written>=3840||(this.ended&&this.written)))this.started=true;
        for(let i=0;i<output.length;i++){
            const index=Math.floor(this.read),fraction=this.read-index;
            if(this.started&&index<this.written&&(index+1<this.written||this.ended)){
                const a=this.pcm[index%this.pcm.length],b=index+1<this.written?this.pcm[(index+1)%this.pcm.length]:a;
                output[i]=a+(b-a)*fraction;this.read=Math.min(this.written,this.read+step);
            }else{output[i]=0;if(this.started&&!this.ended)this.underruns++;}
        }
        if(++this.blocks%4===0)this.port.postMessage({type:'clock',epoch:this.epoch,frame:currentFrame+output.length,position:this.read,underruns:this.underruns});
        return true;
    }
}
registerProcessor('avatar-audio',AvatarAudio);
