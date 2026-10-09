// Per-pose lip/arch contact deformer for the refined mouth lining; a direct port of
// server/vhuman/reconstruction/contact_runtime.py (same candidates, weights, smoothing).
// Lining vertices are pressed onto the labial teeth/gum surface at a small clearance.

// Exact closest point on triangle abc; writes point and unit normal to out[0..5], returns distance.
function edgeClosest(px,py,pz,x0,y0,z0,x1,y1,z1,out,best){
    const ex=x1-x0,ey=y1-y0,ez=z1-z0;let t=((px-x0)*ex+(py-y0)*ey+(pz-z0)*ez)/Math.max(ex*ex+ey*ey+ez*ez,1e-15);
    t=t<0?0:t>1?1:t;const rx=x0+t*ex,ry=y0+t*ey,rz=z0+t*ez,d=Math.sqrt((px-rx)*(px-rx)+(py-ry)*(py-ry)+(pz-rz)*(pz-rz));
    if(d<best){out[0]=rx;out[1]=ry;out[2]=rz;return d;}return best;
}
function closest(px,py,pz,ax,ay,az,bx,by,bz,cx,cy,cz,out){
    const e1x=bx-ax,e1y=by-ay,e1z=bz-az,e2x=cx-ax,e2y=cy-ay,e2z=cz-az;
    const nx=e1y*e2z-e1z*e2y,ny=e1z*e2x-e1x*e2z,nz=e1x*e2y-e1y*e2x,nl=Math.sqrt(nx*nx+ny*ny+nz*nz)||1e-15;
    const ux=nx/nl,uy=ny/nl,uz=nz/nl,d=(px-ax)*ux+(py-ay)*uy+(pz-az)*uz,qx=px-d*ux,qy=py-d*uy,qz=pz-d*uz;
    out[3]=ux;out[4]=uy;out[5]=uz;
    const s0=((by-ay)*(qz-az)-(bz-az)*(qy-ay))*nx+((bz-az)*(qx-ax)-(bx-ax)*(qz-az))*ny+((bx-ax)*(qy-ay)-(by-ay)*(qx-ax))*nz;
    const s1=((cy-by)*(qz-bz)-(cz-bz)*(qy-by))*nx+((cz-bz)*(qx-bx)-(cx-bx)*(qz-bz))*ny+((cx-bx)*(qy-by)-(cy-by)*(qx-bx))*nz;
    const s2=((ay-cy)*(qz-cz)-(az-cz)*(qy-cy))*nx+((az-cz)*(qx-cx)-(ax-cx)*(qz-cz))*ny+((ax-cx)*(qy-cy)-(ay-cy)*(qx-cx))*nz;
    if(s0>=0&&s1>=0&&s2>=0){out[0]=qx;out[1]=qy;out[2]=qz;return Math.abs(d);}
    let best=Infinity;
    best=edgeClosest(px,py,pz,ax,ay,az,bx,by,bz,out,best);
    best=edgeClosest(px,py,pz,bx,by,bz,cx,cy,cz,out,best);
    best=edgeClosest(px,py,pz,cx,cy,cz,ax,ay,az,out,best);
    return best;
}

const TIE_SCALE=2e-4;

// spec: {active:Int32 (part vertices), weight, candidates (active*K*3 native ids), K, clearance, max_move,
//        alpha, iterations, rows, cols (part-vertex neighbour pairs)}; p: Float64Array part positions (modified).
export function contactDeform(p,native,spec){
    const n=p.length/3,act=spec.active,K=spec.K,disp=new Float64Array(n*3),tmp=new Float64Array(6),hit=new Float64Array(act.length*6),
        dk=new Float64Array(K),nk=new Float64Array(K*3);
    // Smooth pseudo-normal across shared-edge ties (see contact_runtime.signed_contact).
    const signed=(i,v)=>{let best=Infinity;const px=p[v*3],py=p[v*3+1],pz=p[v*3+2];
        for(let k=0;k<K;k++){const c=(i*K+k)*3,a=spec.candidates[c]*3,b=spec.candidates[c+1]*3,cc=spec.candidates[c+2]*3;
            const d=closest(px,py,pz,native[a],native[a+1],native[a+2],native[b],native[b+1],native[b+2],native[cc],native[cc+1],native[cc+2],tmp);
            dk[k]=d;nk[k*3]=tmp[3];nk[k*3+1]=tmp[4];nk[k*3+2]=tmp[5];
            if(d<best){best=d;hit[i*6]=tmp[0];hit[i*6+1]=tmp[1];hit[i*6+2]=tmp[2];}}
        let mx=0,my=0,mz=0;
        for(let k=0;k<K;k++){const w=Math.exp(-(dk[k]-best)/TIE_SCALE);mx+=w*nk[k*3];my+=w*nk[k*3+1];mz+=w*nk[k*3+2];}
        const ml=Math.sqrt(mx*mx+my*my+mz*mz)||1e-15;hit[i*6+3]=mx/ml;hit[i*6+4]=my/ml;hit[i*6+5]=mz/ml;
        return (px-hit[i*6])*hit[i*6+3]+(py-hit[i*6+1])*hit[i*6+4]+(pz-hit[i*6+2])*hit[i*6+5];};
    for(let i=0;i<act.length;i++){
        const v=act[i],sd=signed(i,v),s=-spec.weight[i]*(sd-spec.clearance);
        let dx=s*hit[i*6+3],dy=s*hit[i*6+4],dz=s*hit[i*6+5];const l=Math.hypot(dx,dy,dz),f=Math.min(1,spec.max_move/Math.max(l,1e-12));
        disp[v*3]=dx*f;disp[v*3+1]=dy*f;disp[v*3+2]=dz*f;
    }
    if(spec.rows&&spec.rows.length){
        const count=spec.count,avg=new Float64Array(n*3);
        for(let it=0;it<spec.iterations;it++){
            avg.fill(0);for(let e=0;e<spec.rows.length;e++){const r=spec.rows[e]*3,c=spec.cols[e]*3;avg[r]+=disp[c];avg[r+1]+=disp[c+1];avg[r+2]+=disp[c+2];}
            for(let v=0;v<n;v++)if(count[v]>0)for(let k=0;k<3;k++)disp[v*3+k]=(1-spec.alpha)*disp[v*3+k]+spec.alpha*avg[v*3+k]/count[v];
        }
    }
    for(let i=0;i<n*3;i++)p[i]+=disp[i];
    // Final one-sided pass: no visible lining left inside the arch.
    for(let i=0;i<act.length;i++){
        const v=act[i],sd=signed(i,v),push=Math.min(Math.max(spec.clearance-sd,0),spec.max_move)*spec.weight[i];
        p[v*3]+=push*hit[i*6+3];p[v*3+1]+=push*hit[i*6+4];p[v*3+2]+=push*hit[i*6+5];
    }
    return p;
}

// Load the exported spec for a part and return a per-pose updater on the render mesh.
export function partContact(part,json){
    const K=json.candidates[0].length,rows=[],cols=[];
    for(const [a,b] of json.edges){rows.push(a,b);cols.push(b,a);}
    const parts=json.part_vertices,count=new Int32Array(parts);for(const r of rows)count[r]++;
    const spec={active:Int32Array.from(json.active),weight:Float64Array.from(json.weight),candidates:Int32Array.from(json.candidates.flat(2)),K,
        clearance:json.params[0],max_move:json.params[1],alpha:json.params[2],iterations:json.params[3],rows:Int32Array.from(rows),cols:Int32Array.from(cols),count};
    const map=Int32Array.from(json.render_to_part),first=new Int32Array(parts).fill(-1);
    map.forEach((pv,r)=>{if(first[pv]<0)first[pv]=r;});
    const p=new Float64Array(parts*3);
    return native=>{
        const pos=part.mesh.geometry.attributes.position.array;
        for(let v=0;v<parts;v++){const r=first[v];if(r>=0){p[v*3]=pos[r*3];p[v*3+1]=pos[r*3+1];p[v*3+2]=pos[r*3+2];}}
        contactDeform(p,native,spec);
        for(let r=0;r<map.length;r++){const v=map[r];pos[r*3]=p[v*3];pos[r*3+1]=p[v*3+1];pos[r*3+2]=p[v*3+2];}
        part.mesh.geometry.attributes.position.needsUpdate=true;
    };
}
