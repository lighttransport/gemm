import createNative from './native.js';

let module, model=0, input=0, output=0, joint=0, vertices=0;
self.onmessage=async ({data})=>{
    try {
        if(data.type==='init') {
            if(model)throw Error('Already initialized');
            module=await createNative();
            module.FS.writeFile('/avatar.bin',new Uint8Array(data.weights));
            const name=new TextEncoder().encode('/avatar.bin\0'), pointer=module._malloc(name.length);
            module.HEAPU8.set(name,pointer);model=module._vh_mobile_load(pointer);module._free(pointer);
            module.FS.unlink('/avatar.bin');
            if(!model)throw Error('Native avatar rejected');
            vertices=module._vh_mobile_vertices(model);
            input=module._malloc(398*4);output=module._malloc(vertices*3*4);joint=module._malloc(12*4);
            if(!input||!output||!joint)throw Error('WASM allocation failed');
            self.postMessage({type:'ready',vertices});
        } else if(data.type==='pose') {
            if(!model||data.pose.length!==398)throw Error('Invalid pose');
            const start=performance.now();
            module.HEAPF32.set(data.pose,input/4);
            if(module._vh_mobile_eval(model,input,input+383*4,input+395*4,output))throw Error('Native pose rejected');
            const positions=module.HEAPF32.slice(output/4,output/4+vertices*3),joints=new Float32Array(48);
            for(let i=0;i<4;i++) {
                if(module._vh_mobile_joint_transform(model,i,joint))throw Error('Native joint rejected');
                joints.set(module.HEAPF32.subarray(joint/4,joint/4+12),i*12);
            }
            self.postMessage({type:'pose',positions,joints,id:data.id,milliseconds:performance.now()-start},[positions.buffer,joints.buffer]);
        }
    } catch(error) { self.postMessage({type:'error',message:String(error.stack||error)}); }
};
