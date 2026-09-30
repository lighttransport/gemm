/* Original mesh-bound anisotropic Gaussian preview. Static RGB radiance.
 * CPU triangle transport and stable depth sort; WebGL2 alpha quads, opaque depth.
 */
import * as THREE from 'three';

export class BoundGaussians {
  constructor(binding, triangles) {
    if(binding.format!=='vhuman.gaussian_binding.v1'||binding.triangle.length>20000)throw Error('invalid binding');
    if(!triangles?.length||triangles.some(t=>t.length!==3||t.some(i=>!Number.isInteger(i)||i<0)))throw Error('invalid attachment topology');
    for(let i=0;i<binding.triangle.length;i++){
      const id=binding.triangle[i],b=binding.barycentric[i],c=binding.covariance_local[i];
      if(!Number.isInteger(id)||id<0||id>=triangles.length||b?.length!==3||b.some(x=>!Number.isFinite(x)||x<0)||Math.abs(b.reduce((a,x)=>a+x,0)-1)>1e-5)throw Error('invalid barycentrics');
      if(c?.length!==3||c.some(r=>r.length!==3||r.some(x=>!Number.isFinite(x)))||Math.abs(c[0][1]-c[1][0])>1e-8||Math.abs(c[0][2]-c[2][0])>1e-8||Math.abs(c[1][2]-c[2][1])>1e-8)throw Error('invalid covariance');
      const det=c[0][0]*(c[1][1]*c[2][2]-c[1][2]*c[2][1])-c[0][1]*(c[1][0]*c[2][2]-c[1][2]*c[2][0])+c[0][2]*(c[1][0]*c[2][1]-c[1][1]*c[2][0]);
      if(c[0][0]<=0||c[0][0]*c[1][1]-c[0][1]*c[1][0]<=0||det<=0||!Number.isFinite(binding.opacity[i])||binding.opacity[i]<0||binding.opacity[i]>1||!Number.isFinite(binding.normal_offset[i])||Math.abs(binding.normal_offset[i])>.005||binding.rgb[i]?.length!==3||binding.rgb[i].some(x=>!Number.isFinite(x)||x<0))throw Error('invalid Gaussian parameters');
    }
    this.binding=binding; this.triangles=triangles;
    const n=binding.triangle.length;
    const geometry=new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,0,1,-1,0,1,1,0,-1,1,0],3));
    geometry.setIndex([0,1,2,0,2,3]);
    for(const [name,size] of [['center',3],['axisA',3],['axisB',3],['radiance',3],['opacity',1]])
      geometry.setAttribute(name,new THREE.InstancedBufferAttribute(new Float32Array(n*size),size));
    geometry.instanceCount=n;
    this.mesh=new THREE.Mesh(geometry,new THREE.ShaderMaterial({transparent:true,depthWrite:false,depthTest:true,
      premultipliedAlpha:true,toneMapped:true,
      vertexShader:`attribute vec3 center,axisA,axisB,radiance; attribute float opacity;
        varying vec2 q; varying vec3 rgb; varying float alpha;
        void main(){q=position.xy*3.;rgb=radiance;alpha=opacity;
          vec4 c=modelViewMatrix*vec4(center,1.);
          vec3 a=mat3(modelViewMatrix)*axisA,b=mat3(modelViewMatrix)*axisB;
          gl_Position=projectionMatrix*(c+vec4(a*q.x+b*q.y,0.));}`,
      fragmentShader:`varying vec2 q; varying vec3 rgb; varying float alpha;
        void main(){float a=alpha*exp(-.5*dot(q,q));if(a<.003)discard;
          gl_FragColor=vec4(rgb,a);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
          gl_FragColor.rgb*=a;
        }`}));
    this.mesh.frustumCulled=false; this.mesh.renderOrder=10;
    this.order=Array.from({length:n},(_,i)=>i);
  }
  update(vertices,camera) {
    const b=this.binding, result=[];
    const vec=(i)=>new THREE.Vector3(vertices[i*3],vertices[i*3+1],vertices[i*3+2]);
    for(let i=0;i<b.triangle.length;i++){
      const ids=this.triangles[b.triangle[i]];
      if(!ids)throw Error('triangle attachment mismatch');
      const p=ids.map(vec), e1=p[1].clone().sub(p[0]),e2=p[2].clone().sub(p[0]);
      const normal=e1.clone().cross(e2), area=normal.length();normal.normalize();
      const center=new THREE.Vector3(); p.forEach((v,k)=>center.addScaledVector(v,b.barycentric[i][k]));
      center.addScaledVector(normal,b.normal_offset[i]);
      // Full SPD push-forward and projection to a 2D screen ellipse.
      const frame=[e1,e2,normal], c=b.covariance_local[i], world=new THREE.Matrix3();
      const rows=Array.from({length:3},(_,r)=>Array.from({length:3},(_,s)=>{
        let x=0;for(let j=0;j<3;j++)for(let k=0;k<3;k++)x+=frame[j].getComponent(r)*c[j][k]*frame[k].getComponent(s);return x;}));
      world.set(...rows.flat());
      const mv=new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse,this.mesh.matrixWorld);
      const pc=center.clone().applyMatrix4(mv), rot=new THREE.Matrix3().setFromMatrix4(mv);
      const cv=world.clone().premultiply(rot).multiply(rot.clone().transpose());
      const z=Math.max(-pc.z,.001), jx=new THREE.Vector3(1,0,pc.x/z),jy=new THREE.Vector3(0,1,pc.y/z);
      const xx=jx.dot(jx.clone().applyMatrix3(cv)),xy=jx.dot(jy.clone().applyMatrix3(cv)),yy=jy.dot(jy.clone().applyMatrix3(cv));
      const angle=.5*Math.atan2(2*xy,xx-yy),disc=Math.sqrt((xx-yy)**2+4*xy*xy);
      const l1=Math.sqrt(Math.max(1e-10,Math.min(.0001,(xx+yy+disc)/2)));
      const l2=Math.sqrt(Math.max(1e-10,Math.min(.0001,(xx+yy-disc)/2)));
      const inverse=rot.clone().invert();
      const a=new THREE.Vector3(Math.cos(angle)*l1,Math.sin(angle)*l1,0).applyMatrix3(inverse);
      const d=new THREE.Vector3(-Math.sin(angle)*l2,Math.cos(angle)*l2,0).applyMatrix3(inverse);
      result.push({center,a,d,z:pc.z,opacity:area>1e-10&&pc.z<0?b.opacity[i]:0});
    }
    this.order.sort((i,j)=>result[i].z-result[j].z||i-j);
    const g=this.mesh.geometry;
    this.order.forEach((i,k)=>{
      const v=result[i]; for(const [name,value] of [['center',v.center.toArray()],['axisA',v.a.toArray()],
        ['axisB',v.d.toArray()],['radiance',b.rgb[i]],['opacity',[v.opacity]]])g.attributes[name].array.set(value,k*g.attributes[name].itemSize);
    });
    Object.values(g.attributes).forEach(a=>{if(a.isInstancedBufferAttribute)a.needsUpdate=true;});
  }
  dispose(){this.mesh.geometry.dispose();this.mesh.material.dispose();}
}
