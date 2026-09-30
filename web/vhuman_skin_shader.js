/* Original diffuse-only skin diffusion for Three.js r163 / WebGL2.
 * Authored metric radii are not anatomy measurements. Off is ordinary PBR.
 * Shader hooks compose with morph/LBS/contact/wrinkle hooks on each material.
 */
import * as THREE from 'three';

export class SkinRenderer {
  constructor(renderer) {
    this.renderer = renderer;
    this.enabled = false;
    this.radii = new THREE.Vector3(.0012, .0006, .0003);
    this.mode = { value: 0 };
    this.unitsPerM = 1;
    this.whiteMask = new THREE.DataTexture(new Uint8Array([255,255,255,255]),1,1);this.whiteMask.needsUpdate=true;
    this.supported = renderer.capabilities.isWebGL2 && renderer.extensions.has('EXT_color_buffer_float');
    this.status = this.supported ? 'PBR; skin scattering off' : 'PBR fallback: float targets unavailable';
    this.ms = 0;
    this.targets = [];
    this.occluder = new THREE.MeshBasicMaterial({color:0,side:THREE.DoubleSide});
    this.occluder.onBeforeCompile=shader=>{
      shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>',
        '#include <opaque_fragment>\ngl_FragColor=vec4(0.);');
    };
    this.occluder.customProgramCacheKey=()=> 'vh-skin-occluder-v1';
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quad.frustumCulled = false;
    this.scene.add(this.quad);
    const vertexShader = 'varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}';
    this.blur = new THREE.ShaderMaterial({depthTest:false,depthWrite:false,toneMapped:false,vertexShader,
      uniforms:{source:{value:null},guide:{value:null},direction:{value:new THREE.Vector2()},radii:{value:this.radii.clone()},
                focal:{value:1},texel:{value:new THREE.Vector2()}},
      fragmentShader:`varying vec2 vUv; uniform sampler2D source,guide; uniform vec2 direction,texel;
        uniform vec3 radii; uniform float focal;
        void main(){vec4 g=texture2D(guide,vUv); vec3 center=texture2D(source,vUv).rgb;
          if(g.a<=0.){gl_FragColor=vec4(center,1.);return;}
          vec3 total=center,weights=vec3(1.); vec3 radius=clamp(radii*focal/g.a,vec3(.01),vec3(32.));
          float maxr=max(radius.r,max(radius.g,radius.b));
          for(int i=-6;i<=6;i++){if(i==0)continue; float offset=float(i)*maxr/3.;
            vec2 uv=vUv+direction*texel*offset; vec4 q=texture2D(guide,uv);
            float gate=step(.00001,q.a)*exp(-abs(g.a-q.a)/.0015)*pow(max(dot(g.rgb*2.-1.,q.rgb*2.-1.),0.),16.);
            vec3 weight=exp(-.5*vec3(offset*offset)/(radius*radius))*gate;
            total+=texture2D(source,uv).rgb*weight; weights+=weight;
          } gl_FragColor=vec4(total/weights,1.);}`});
    this.composite = new THREE.ShaderMaterial({depthTest:false,depthWrite:false,toneMapped:true,vertexShader,
      uniforms:{beauty:{value:null},diffuse:{value:null},scattered:{value:null}},
      fragmentShader:`varying vec2 vUv; uniform sampler2D beauty,diffuse,scattered;
        void main(){vec4 b=texture2D(beauty,vUv);
          gl_FragColor=vec4(max(b.rgb-texture2D(diffuse,vUv).rgb+texture2D(scattered,vUv).rgb,vec3(0.)),b.a);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`});
  }
  patch(scene) {
    scene.traverse(o => {
      if (!o.isMesh) return;
      for (const m of [].concat(o.material)) {
        if (!m.isMeshStandardMaterial || m.userData.skinDiffusionPatched) continue;
        const skin = m.name === 'skin' || m.name === 'head' || m.userData.vhuman_skin_material;
        const previous = m.onBeforeCompile, key = m.customProgramCacheKey.bind(m);
        m.onBeforeCompile = (shader, renderer) => {
          previous.call(m, shader, renderer);
          shader.uniforms.vhSkinPass = this.mode;
          shader.uniforms.vhSkinMask = {value:m.userData.skinMask||this.whiteMask};
          shader.uniforms.vhSkinUseMask = {value:m.userData.skinMask?1:0};
          shader.fragmentShader = 'uniform int vhSkinPass; uniform sampler2D vhSkinMask; uniform float vhSkinUseMask;\n' + shader.fragmentShader;
          const diffuse = skin ? 'totalDiffuse * vhWeight' : 'vec3(0.)';
          const guide = skin ? 'vec4(normal * .5 + .5, vViewPosition.z * step(.5,vhWeight))' : 'vec4(0.)';
          shader.fragmentShader = shader.fragmentShader.replace('#include <opaque_fragment>',
            `#include <opaque_fragment>\nfloat vhWeight=1.;\n#ifdef USE_MAP\nvhWeight=mix(1.,texture2D(vhSkinMask,vMapUv).r,vhSkinUseMask);\n#endif\nif(vhSkinPass==1)gl_FragColor=vec4(${diffuse},1.);\nif(vhSkinPass==2)gl_FragColor=${guide};`);
        };
        m.customProgramCacheKey = () => key() + '|vh-skin-v1|' + skin;
        m.userData.skinDiffusionPatched = true;
        m.needsUpdate = true;
      }
    });
  }
  resize() {
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    if (this.targets[0]?.width === size.x && this.targets[0]?.height === size.y) return;
    this.targets.forEach(t => t.dispose());
    this.targets = Array.from({length:5},()=>new THREE.WebGLRenderTarget(size.x,size.y,
      {type:THREE.HalfFloatType,format:THREE.RGBAFormat,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,
       depthBuffer:true,stencilBuffer:false}));
    this.blur.uniforms.texel.value.set(1/size.x,1/size.y);
  }
  render(scene,camera) {
    const start=performance.now(), r=this.renderer;
    if (!this.enabled || !this.supported) {
      r.render(scene,camera);
      this.ms=performance.now()-start;
      return;
    }
    this.patch(scene); this.resize();
    const [beauty,diffuse,guide,horizontal,vertical]=this.targets;
    const tone=r.toneMapping, space=r.outputColorSpace, target=r.getRenderTarget();
    const background=scene.background, clear=r.getClearColor(new THREE.Color()), clearAlpha=r.getClearAlpha();
    const replacements=[],rawUniforms=[];
    scene.traverse(o=>{if(!o.isMesh)return;
      if(o.userData.analyticEye){for(const m of [].concat(o.material)){rawUniforms.push([m.uniforms.uRaw,m.uniforms.uRaw.value]);m.uniforms.uSkinPass=this.mode;}}
      else if([].concat(o.material).some(m=>m.transparent||!m.isMeshStandardMaterial))replacements.push([o,o.material]);
    });
    try {
      r.toneMapping=THREE.NoToneMapping; r.outputColorSpace=THREE.LinearSRGBColorSpace;
      rawUniforms.forEach(([u])=>u.value=1);
      this.blur.uniforms.radii.value.copy(this.radii).multiplyScalar(this.unitsPerM);
      this.mode.value=0; r.setRenderTarget(beauty); r.clear(); r.render(scene,camera);
      scene.background=null; r.setClearColor(0,0);
      replacements.forEach(([o])=>o.material=this.occluder);
      this.mode.value=1; r.setRenderTarget(diffuse); r.clear(); r.render(scene,camera);
      this.mode.value=2; r.setRenderTarget(guide); r.clear(); r.render(scene,camera);
      this.mode.value=0;
      this.blur.uniforms.guide.value=guide.texture;
      this.blur.uniforms.focal.value=camera.projectionMatrix.elements[5]*guide.height*.5;
      this.quad.material=this.blur;
      this.blur.uniforms.source.value=diffuse.texture; this.blur.uniforms.direction.value.set(1,0);
      r.setRenderTarget(horizontal); r.clear(); r.render(this.scene,this.camera);
      this.blur.uniforms.source.value=horizontal.texture; this.blur.uniforms.direction.value.set(0,1);
      r.setRenderTarget(vertical); r.clear(); r.render(this.scene,this.camera);
      r.toneMapping=tone; r.outputColorSpace=space; r.setRenderTarget(target);
      this.quad.material=this.composite;
      this.composite.uniforms.beauty.value=beauty.texture;
      this.composite.uniforms.diffuse.value=diffuse.texture;
      this.composite.uniforms.scattered.value=vertical.texture;
      r.render(this.scene,this.camera);
      this.status='Diffuse scattering; specular remains sharp';
    } finally {
      this.mode.value=0; scene.background=background;
      rawUniforms.forEach(([u,v])=>u.value=v);
      replacements.forEach(([o,m])=>o.material=m);
      r.setClearColor(clear,clearAlpha);
      r.toneMapping=tone; r.outputColorSpace=space; r.setRenderTarget(target);
      this.ms=performance.now()-start;
    }
  }
  dispose(){this.targets.forEach(t=>t.dispose());this.blur.dispose();this.composite.dispose();this.quad.geometry.dispose();this.occluder.dispose();this.whiteMask.dispose();}
}
