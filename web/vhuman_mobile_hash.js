// SHA-256 fallback for LAN HTTP, where browsers do not expose SubtleCrypto.
// Hash one block at a time to avoid copying large avatar weight buffers.
const K=new Uint32Array([
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
]);
const rotate=(x,n)=>(x>>>n)|(x<<(32-n));

export async function sha256(data,subtle=globalThis.crypto?.subtle){
    if(subtle)return [...new Uint8Array(await subtle.digest('SHA-256',data))].map(v=>v.toString(16).padStart(2,'0')).join('');
    const input=new Uint8Array(data),view=new DataView(data),length=input.length;
    const h=new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);
    const w=new Uint32Array(64),tail=new Uint8Array(128),end=length-length%64;
    tail.set(input.subarray(end));tail[length-end]=0x80;
    const tailLength=length-end<56?64:128,tailView=new DataView(tail.buffer);
    tailView.setUint32(tailLength-8,Math.floor(length/0x20000000));
    tailView.setUint32(tailLength-4,(length*8)>>>0);
    for(let offset=0;offset<end+tailLength;offset+=64){
        const source=offset<end?view:tailView,base=offset<end?offset:offset-end;
        for(let i=0;i<16;i++)w[i]=source.getUint32(base+i*4);
        for(let i=16;i<64;i++){
            const x=w[i-15],y=w[i-2];
            w[i]=(w[i-16]+(rotate(x,7)^rotate(x,18)^(x>>>3))+w[i-7]+(rotate(y,17)^rotate(y,19)^(y>>>10)))>>>0;
        }
        let [a,b,c,d,e,f,g,j]=h;
        for(let i=0;i<64;i++){
            const t1=(j+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+K[i]+w[i])>>>0;
            const t2=((rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c)))>>>0;
            j=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0;
        }
        const values=[a,b,c,d,e,f,g,j];for(let i=0;i<8;i++)h[i]=(h[i]+values[i])>>>0;
        // Keep loading status and controls responsive during large weight checks.
        if(offset>0&&offset%1048576===0)await new Promise(resolve=>setTimeout(resolve,0));
    }
    return [...h].map(v=>v.toString(16).padStart(8,'0')).join('');
}
