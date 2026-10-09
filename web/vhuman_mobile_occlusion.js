// Dynamic mouth-interior occlusion driven by the lip aperture (see
// server/vhuman/reconstruction/oral_occlusion.py). Per oral vertex:
//   indirect: per-vertex visibility, affine in the lip opening area and height,
//             fitted to ray-traced escape visibility ("morphable" AO);
//   direct:   soft analytic "aperture shadow" for each directional light, gated by
//             min(1, indirect/direct_gate) as a self-shadowing proxy.
// Approximations of escape visibility, not path-traced lighting.

const TWO_PI=2*Math.PI;

// Cosine-weighted visibility of a closed polygon (flat xyz array) from a point,
// fan-triangulated from its centroid; triangles straddling the tangent plane are
// scaled by the fraction of their corners above it (matches the Python reference).
// Allocation-free: edge terms are computed once per point and shared by fan triangles.
let scratch=new Float64Array(0);
function edgeTerm(ax,ay,az,bx,by,bz,nx,ny,nz){
    const cx=ay*bz-az*by,cy=az*bx-ax*bz,cz=ax*by-ay*bx,len=Math.sqrt(cx*cx+cy*cy+cz*cz);
    const d=ax*bx+ay*by+az*bz,theta=Math.acos(d>1?1:d<-1?-1:d);
    return theta*(cx*nx+cy*ny+cz*nz)/(len>1e-9?len:1e-9);
}
export function polygonFormFactor(px,py,pz,nx,ny,nz,poly,centroid){
    const m=poly.length/3;
    if(scratch.length<m*6)scratch=new Float64Array(m*6);
    let x=centroid[0]-px,y=centroid[1]-py,z=centroid[2]-pz,l=Math.sqrt(x*x+y*y+z*z)||1e-9;
    const cx=x/l,cy=y/l,cz=z/l,cAbove=x*nx+y*ny+z*nz>0?1:0;
    for(let i=0;i<m;i++){
        x=poly[i*3]-px;y=poly[i*3+1]-py;z=poly[i*3+2]-pz;l=Math.sqrt(x*x+y*y+z*z)||1e-9;
        scratch[i*4]=x/l;scratch[i*4+1]=y/l;scratch[i*4+2]=z/l;scratch[i*4+3]=x*nx+y*ny+z*nz>0?1:0;
    }
    const centre=m*4;   // centre-edge terms stored after the unit vectors
    for(let i=0;i<m;i++)scratch[centre+i]=edgeTerm(cx,cy,cz,scratch[i*4],scratch[i*4+1],scratch[i*4+2],nx,ny,nz);
    let total=0;
    for(let i=0;i<m;i++){
        const j=(i+1)%m,above=cAbove+scratch[i*4+3]+scratch[j*4+3];
        if(!above)continue;
        const ff=scratch[centre+i]+edgeTerm(scratch[i*4],scratch[i*4+1],scratch[i*4+2],scratch[j*4],scratch[j*4+1],scratch[j*4+2],nx,ny,nz)-scratch[centre+j];
        total+=(ff<0?-ff:ff)/TWO_PI*above/3;
    }
    return total<0?0:total>1?1:total;
}

// Best-fit plane frame of the rim polygon (centroid, two in-plane axes, normal) and 2D outline.
export function apertureFrame(poly){
    const m=poly.length/3,c=[0,0,0];
    for(let i=0;i<m;i++)for(let k=0;k<3;k++)c[k]+=poly[i*3+k]/m;
    // Covariance power iteration for the smallest-variance axis (plane normal).
    const C=[0,0,0,0,0,0,0,0,0];
    for(let i=0;i<m;i++){const d=[poly[i*3]-c[0],poly[i*3+1]-c[1],poly[i*3+2]-c[2]];for(let a=0;a<3;a++)for(let b=0;b<3;b++)C[a*3+b]+=d[a]*d[b];}
    const trace=C[0]+C[4]+C[8];const S=C.map((v,i)=>(i%4===0?trace:0)-v);   // largest eigvec of S = smallest of C
    let n=[0,0,1];
    for(let it=0;it<40;it++){const t=[S[0]*n[0]+S[1]*n[1]+S[2]*n[2],S[3]*n[0]+S[4]*n[1]+S[5]*n[2],S[6]*n[0]+S[7]*n[1]+S[8]*n[2]];const l=Math.hypot(...t)||1;n=t.map(v=>v/l);}
    if(n[2]<0)n=n.map(v=>-v);     // face the front of the head (+Z in GNM)
    const ref=Math.abs(n[0])<.9?[1,0,0]:[0,1,0];
    let ux=[n[1]*ref[2]-n[2]*ref[1],n[2]*ref[0]-n[0]*ref[2],n[0]*ref[1]-n[1]*ref[0]];const lu=Math.hypot(...ux);ux=ux.map(v=>v/lu);
    const uy=[n[1]*ux[2]-n[2]*ux[1],n[2]*ux[0]-n[0]*ux[2],n[0]*ux[1]-n[1]*ux[0]];
    const outline=new Float64Array(m*2);let area=0;
    for(let i=0;i<m;i++){const d=[poly[i*3]-c[0],poly[i*3+1]-c[1],poly[i*3+2]-c[2]];outline[i*2]=d[0]*ux[0]+d[1]*ux[1]+d[2]*ux[2];outline[i*2+1]=d[0]*uy[0]+d[1]*uy[1]+d[2]*uy[2];}
    for(let i=0;i<m;i++){const j=(i+1)%m;area+=outline[i*2]*outline[j*2+1]-outline[j*2]*outline[i*2+1];}
    // Opening height: extent along the minor principal axis of the outline (2D PCA).
    let mx=0,my=0;for(let i=0;i<m;i++){mx+=outline[i*2]/m;my+=outline[i*2+1]/m;}
    let sxx=0,sxy=0,syy=0;for(let i=0;i<m;i++){const x=outline[i*2]-mx,y=outline[i*2+1]-my;sxx+=x*x;sxy+=x*y;syy+=y*y;}
    const angle=.5*Math.atan2(2*sxy,sxx-syy)+Math.PI/2,ax=Math.cos(angle),ay=Math.sin(angle);
    let lo=Infinity,hi=-Infinity;for(let i=0;i<m;i++){const d=(outline[i*2]-mx)*ax+(outline[i*2+1]-my)*ay;lo=Math.min(lo,d);hi=Math.max(hi,d);}
    return {c,n,ux,uy,outline,area:Math.abs(area)/2,height:hi-lo};
}

// Signed distance (inside positive) from a 2D point to a closed outline.
function signedDistance(x,y,outline){
    const m=outline.length/2;let inside=false,best=Infinity;
    for(let i=0,j=m-1;i<m;j=i++){
        const xi=outline[i*2],yi=outline[i*2+1],xj=outline[j*2],yj=outline[j*2+1];
        if(((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi||1e-12)+xi))inside=!inside;
        const dx=xj-xi,dy=yj-yi,t=Math.max(0,Math.min(1,((x-xi)*dx+(y-yi)*dy)/(dx*dx+dy*dy||1e-12)));
        best=Math.min(best,Math.hypot(x-xi-t*dx,y-yi-t*dy));
    }
    return inside?best:-best;
}

// Soft visibility of a directional light (unit l, pointing toward the light) through the aperture.
export function apertureShadow(px,py,pz,l,frame,{angularRadius=.12,minPenumbra=.0008}={}){
    const {c,n,ux,uy,outline}=frame;
    const denom=l[0]*n[0]+l[1]*n[1]+l[2]*n[2];
    const d0=(c[0]-px)*n[0]+(c[1]-py)*n[1]+(c[2]-pz)*n[2];       // signed distance point->plane along n
    if(denom<=1e-4)return 0;                                       // light grazes or comes from behind
    const t=d0/denom;
    if(t<=0)return 1;                                              // point already outside the aperture plane
    const hx=px+l[0]*t-c[0],hy=py+l[1]*t-c[1],hz=pz+l[2]*t-c[2];
    const sd=signedDistance(hx*ux[0]+hy*ux[1]+hz*ux[2],hx*uy[0]+hy*uy[1]+hz*uy[2],outline);
    const w=minPenumbra+t*Math.tan(angularRadius);
    const s=Math.max(0,Math.min(1,(sd+w)/(2*w)));
    return s*s*(3-2*s);
}

export const MAX_RIM=32;

function frameOf(a,b,c){
    const x=[b[0]-a[0],b[1]-a[1],b[2]-a[2]],lx=Math.hypot(...x);x.forEach((v,i)=>x[i]=v/lx);
    const w=[c[0]-a[0],c[1]-a[1],c[2]-a[2]],z=[x[1]*w[2]-x[2]*w[1],x[2]*w[0]-x[0]*w[2],x[0]*w[1]-x[1]*w[0]],lz=Math.hypot(...z);z.forEach((v,i)=>z[i]=v/lz);
    const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];return [x,y,z];
}
// Rotation R (row-major 3x3) mapping rest head frame to the current one.
export function headRotation(native,ref){
    const cur=ref.native_ids.map(id=>[native[id*3],native[id*3+1],native[id*3+2]]);
    const F=frameOf(...cur),G=frameOf(...ref.rest),R=new Float64Array(9);
    for(let i=0;i<3;i++)for(let j=0;j<3;j++)R[i*3+j]=F[0][i]*G[0][j]+F[1][i]*G[1][j]+F[2][i]*G[2][j];
    return R;
}

// CPU reference of the shader (verification/tests). part: {mesh, vertices, occlusion:{weights}}.
export function cpuOralOcclusion(part,native,spec,lights){
    const {rim,frame}=rimFrame(native,spec),area=frame.area*1e4,total=lights.reduce((s,x)=>s+x.intensity,0)||1;
    const p=part.mesh.geometry.attributes.position.array,nrm=part.mesh.geometry.attributes.normal.array,w=part.occlusion.weights,out=new Float32Array(part.vertices*2);
    for(let v=0;v<part.vertices;v++){
        const x=p[v*3],y=p[v*3+1],z=p[v*3+2];
        out[v*2]=Math.max(0,Math.min(1,w[v*3]+w[v*3+1]*area+w[v*3+2]*frame.height*100));
        let direct=0;for(const light of lights)direct+=light.intensity*apertureShadow(x,y,z,light.direction,frame);
        const aperture=spec.direct_mode==='gate'?1:direct/total;out[v*2+1]=aperture*Math.min(1,out[v*2]/(spec.direct_gate??.5));
    }
    return out;
}

function rimFrame(native,spec){
    const rim=new Float64Array(spec.rim.length*3);
    for(let i=0;i<spec.rim.length;i++)for(let k=0;k<3;k++)rim[i*3+k]=native[spec.rim[i]*3+k];
    return {rim,frame:apertureFrame(rim)};
}

export function oralUniforms(){
    // Flat typed arrays: three.js uploads number arrays directly for vec3[]/vec2[]/float[] uniforms.
    return {oralEnabled:{value:1},oralRim:{value:new Float32Array(MAX_RIM*3)},oralRimCount:{value:0},
        oralOutline:{value:new Float32Array(MAX_RIM*2)},oralCentroid:{value:new Float32Array(3)},
        oralAxisX:{value:new Float32Array(3)},oralAxisY:{value:new Float32Array(3)},oralNormal:{value:new Float32Array(3)},
        oralArea:{value:0},oralHeight:{value:0},oralGate:{value:.5},oralApertureMix:{value:1},oralLightColor:{value:new Float32Array(6)},oralBlend:{value:new Float32Array(2)},oralTransfer:{value:0},oralLightDir:{value:new Float32Array(6)},oralLightWeight:{value:new Float32Array(2)}};
}

// Per pose: upload the lip-rim polygon, its plane frame/outline/area and the light directions.
export function setOralUniforms(u,native,spec,lights){
    if(spec.rim.length>MAX_RIM||lights.length!==2)throw Error('Unsupported oral occlusion configuration');
    const {rim,frame}=rimFrame(native,spec),total=lights.reduce((s,x)=>s+x.intensity,0)||1;
    u.oralRim.value.fill(0);u.oralRim.value.set(rim);u.oralOutline.value.fill(0);u.oralOutline.value.set(frame.outline);
    u.oralRimCount.value=spec.rim.length;u.oralCentroid.value.set(frame.c);u.oralAxisX.value.set(frame.ux);u.oralAxisY.value.set(frame.uy);
    u.oralNormal.value.set(frame.n);u.oralArea.value=frame.area*1e4;u.oralHeight.value=frame.height*100;u.oralGate.value=spec.direct_gate??.5;u.oralApertureMix.value=spec.direct_mode==='gate'?0:1;
    lights.forEach((light,i)=>{u.oralLightDir.value.set(light.direction,i*3);u.oralLightWeight.value[i]=light.intensity/total;
        if(light.color)u.oralLightColor.value.set(light.color,i*3);});
    u.oralTransfer.value=spec.transfer?1:0;
    if(spec.transfer&&spec.head_reference){
        // Head rotation from three rigid upper-teeth vertices; blend learned transfer toward the analytic
        // aperture term as each light leaves its trained head-frame direction.
        const R=headRotation(native,spec.head_reference);
        lights.forEach((light,i)=>{const d=light.direction,h=[R[0]*d[0]+R[3]*d[1]+R[6]*d[2],R[1]*d[0]+R[4]*d[1]+R[7]*d[2],R[2]*d[0]+R[5]*d[1]+R[8]*d[2]];
            const t=spec.transfer.lights[light.name]||[0,0,1],c=Math.max(-1,Math.min(1,(h[0]*t[0]+h[1]*t[1]+h[2]*t[2])/Math.hypot(...t)));
            const a=Math.acos(c),x=Math.max(0,Math.min(1,(a-.15)/.35));u.oralBlend.value[i]=x*x*(3-2*x);});
    }
    return frame;
}

const GLSL=`
#define ORAL_MAX_RIM ${MAX_RIM}
uniform vec3 oralRim[ORAL_MAX_RIM];uniform vec2 oralOutline[ORAL_MAX_RIM];uniform int oralRimCount;
uniform vec3 oralCentroid,oralAxisX,oralAxisY,oralNormal;uniform float oralArea,oralHeight,oralGate,oralApertureMix;
uniform vec3 oralLightDir[2];uniform float oralLightWeight[2];uniform float oralBlend[2];uniform float oralTransfer;
attribute vec3 oralWeights,oralKey,oralFill;varying vec2 vOralOcclusion,vOralTransfer;
// Form factor kept for reference; the shipped model uses opening area/height.
float oralEdge(vec3 a,vec3 b,vec3 n){vec3 c=cross(a,b);return acos(clamp(dot(a,b),-1.0,1.0))*dot(c,n)/max(length(c),1e-9);}
float oralFormFactor(vec3 p,vec3 n){
    vec3 cu=normalize(oralCentroid-p);float cAbove=dot(oralCentroid-p,n)>0.0?1.0:0.0,total=0.0;
    for(int i=0;i<ORAL_MAX_RIM;i++){
        if(i>=oralRimCount)break;
        int j=i+1==oralRimCount?0:i+1;
        vec3 ra=oralRim[i]-p,rb=oralRim[j]-p;
        float above=cAbove+(dot(ra,n)>0.0?1.0:0.0)+(dot(rb,n)>0.0?1.0:0.0);
        if(above==0.0)continue;
        vec3 a=normalize(ra),b=normalize(rb);
        float ff=oralEdge(cu,a,n)+oralEdge(a,b,n)+oralEdge(b,cu,n);
        total+=abs(ff)/6.283185307*above/3.0;
    }
    return clamp(total,0.0,1.0);
}
float oralSignedDistance(vec2 q){
    bool inside=false;float best=1e9;
    for(int i=0;i<ORAL_MAX_RIM;i++){
        if(i>=oralRimCount)break;
        int j=i==0?oralRimCount-1:i-1;vec2 a=oralOutline[i],b=oralOutline[j];
        if(((a.y>q.y)!=(b.y>q.y))&&(q.x<(b.x-a.x)*(q.y-a.y)/(b.y-a.y+1e-12)+a.x))inside=!inside;
        vec2 e=b-a;float t=clamp(dot(q-a,e)/max(dot(e,e),1e-12),0.0,1.0);best=min(best,length(q-a-t*e));
    }
    return inside?best:-best;
}
float oralShadow(vec3 p,vec3 l){
    float denom=dot(l,oralNormal),d0=dot(oralCentroid-p,oralNormal);
    if(denom<=1e-4)return 0.0;
    float t=d0/denom;if(t<=0.0)return 1.0;
    vec3 h=p+l*t-oralCentroid;float sd=oralSignedDistance(vec2(dot(h,oralAxisX),dot(h,oralAxisY)));
    float w=0.0008+t*0.12058;   // tan(0.12 rad): soft-light penumbra growing with distance
    float s=clamp((sd+w)/(2.0*w),0.0,1.0);return s*s*(3.0-2.0*s);
}
`;

// Material patch: per-vertex visibility evaluated in the vertex shader from the
// uploaded lip rim; indirect light scaled by it (specular by its square, a
// specular-occlusion approximation), direct light by the aperture shadow.
export function patchOralMaterial(material,u){
    material.onBeforeCompile=shader=>{
        Object.assign(shader.uniforms,u);
        shader.vertexShader=GLSL+shader.vertexShader.replace('#include <begin_vertex>',`#include <begin_vertex>
            {// Indirect: per-vertex visibility as an affine function of the lip opening (area cm^2, height cm).
             float oralIndirect=clamp(oralWeights.x+oralWeights.y*oralArea+oralWeights.z*oralHeight,0.0,1.0);
             // Direct: aperture shadow, gated by the vertex's own visibility (teeth/tongue self-shadowing proxy).
             float s0=oralShadow(transformed,oralLightDir[0]),s1=oralShadow(transformed,oralLightDir[1]);
             float oralAperture=oralLightWeight[0]*s0+oralLightWeight[1]*s1;
             float oralGateTerm=min(1.0,oralIndirect/oralGate);
             float oralDirect=mix(1.0,oralAperture,oralApertureMix)*oralGateTerm;
             // Learned per-light transfer (direct+bounce, relative to an unoccluded surface facing the light),
             // blended toward the analytic aperture estimate when the light leaves its trained direction.
             vec3 oralN=normalize(objectNormal);
             float a0=max(dot(oralN,oralLightDir[0]),0.0)*s0*oralGateTerm,a1=max(dot(oralN,oralLightDir[1]),0.0)*s1*oralGateTerm;
             float t0=clamp(oralKey.x+oralKey.y*oralArea+oralKey.z*oralHeight,0.0,1.2),t1=clamp(oralFill.x+oralFill.y*oralArea+oralFill.z*oralHeight,0.0,1.2);
             vOralTransfer=vec2(mix(t0,a0,oralBlend[0]),mix(t1,a1,oralBlend[1]));
             vOralOcclusion=vec2(oralIndirect,oralDirect);}`);
        shader.fragmentShader='uniform float oralEnabled,oralTransfer;\nuniform vec3 oralLightColor[2];\nvarying vec2 vOralOcclusion,vOralTransfer;\n'+shader.fragmentShader.replace('#include <lights_fragment_end>',`#include <lights_fragment_end>
            float occIndirect=mix(1.0,vOralOcclusion.x,oralEnabled),occDirect=mix(1.0,vOralOcclusion.y,oralEnabled);
            // Visibility is fitted to Cycles-baked irradiance that already includes interreflection,
            // so no separate multi-bounce lift is applied.
            vec3 oralTransferDiffuse=BRDF_Lambert(diffuseColor.rgb)*(oralLightColor[0]*vOralTransfer.x+oralLightColor[1]*vOralTransfer.y);
            reflectedLight.directDiffuse=mix(reflectedLight.directDiffuse,mix(reflectedLight.directDiffuse*occDirect,oralTransferDiffuse,oralTransfer),oralEnabled);
            reflectedLight.directSpecular*=occDirect;
            reflectedLight.indirectDiffuse*=occIndirect;reflectedLight.indirectSpecular*=occIndirect*occIndirect;`);
    };
    material.customProgramCacheKey=()=> 'vhuman-oral-occlusion-v8';material.needsUpdate=true;
}
