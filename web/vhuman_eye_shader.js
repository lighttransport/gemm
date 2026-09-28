// The analytic eye shader (a ray tracer in the eye frame, line by line the
// math of server/vhuman/eye/render.py),
// shared by the eye page (web/vhuman_eye.html) and the head page
// (web/vhuman_head.html). Geometry and optics numbers come from the server
// (optics.uniforms via POST /v1/eye/textures): nothing is hard-coded here
// but the material constants of iris.py / sclera.py.
//
// Each eye is a proxy sphere (back faces, eye-local metres) whose fragments
// trace the view ray against the analytic eyeball and write the depth of the
// surface hit, so the eye composites with other geometry (a head's lids).
import * as THREE from 'three';

export const COMMON_GLSL = /* glsl */`
#define PI 3.141592653589793
uniform vec3 uBoxDir[2];
uniform vec3 uBoxU[2];
uniform vec3 uBoxV[2];
uniform vec2 uBoxSize[2];
uniform float uBoxRad[2];
uniform vec3 uSky;
uniform vec3 uGround;
uniform vec3 uHorizon;
uniform float uExposure;
uniform float uOrtho;
uniform vec3 uCamDir;
uniform float uRaw;                 // 1: linear radiance out (environment capture)
varying vec3 vWorld;

float sstep(float e0, float e1, float x) {           // numpy smoothstep: also for e0 > e1
  float t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
// optics.environment: sky/ground gradient + two softboxes, edges blurred by roughness
vec3 environment(vec3 d, float rough) {
  d = normalize(d);
  float y = d.y;
  vec3 c = y >= 0.0 ? uHorizon + (uSky - uHorizon) * sqrt(clamp(y, 0.0, 1.0))
                    : uHorizon + (uGround - uHorizon) * sqrt(clamp(-y, 0.0, 1.0));
  float blur = 0.01 + 0.6 * rough * rough;
  for (int k = 0; k < 2; k++) {
    float dw = dot(d, uBoxDir[k]);
    float x = atan(dot(d, uBoxU[k]), dw);
    float z = atan(dot(d, uBoxV[k]), dw);
    float mx = clamp((uBoxSize[k].x - abs(x)) / blur + 0.5, 0.0, 1.0);
    float mz = clamp((uBoxSize[k].y - abs(z)) / blur + 0.5, 0.0, 1.0);
    float spread = (uBoxSize[k].x * uBoxSize[k].y) / ((uBoxSize[k].x + blur) * (uBoxSize[k].y + blur));
    c += uBoxRad[k] * spread * mx * mz * (dw > 0.0 ? 1.0 : 0.0);
  }
  return c;
}
vec3 ambient(vec3 n) { return uGround + (uSky - uGround) * (0.5 + 0.5 * n.y); }
// Khronos PBR Neutral (Three.js adaptation; server/vhuman/THIRD_PARTY_NOTICES.md), then sRGB
vec3 neutral(vec3 c) {
  const float startC = 0.8 - 0.04;
  const float desat = 0.15;
  float x = min(c.r, min(c.g, c.b));
  float off = x < 0.08 ? x - 6.25 * x * x : 0.04;
  c -= off;
  float peak = max(c.r, max(c.g, c.b));
  if (peak < startC) return c;
  float d = 1.0 - startC;
  float np_ = 1.0 - d * d / (peak + d - startC);
  c *= np_ / peak;
  float g = 1.0 - 1.0 / (desat * (peak - np_) + 1.0);
  return mix(c, vec3(np_), g);
}
vec3 toSRGB(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
vec3 srgb2lin(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c)); }
vec3 viewRayOrigin() { return uOrtho > 0.5 ? vWorld - uCamDir : cameraPosition; }
vec3 viewRayDir() { return uOrtho > 0.5 ? uCamDir : normalize(vWorld - cameraPosition); }
`;

export const VERT = /* glsl */`
varying vec3 vWorld;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

export const BG_FRAG = /* glsl */`
precision highp float;
${COMMON_GLSL}
void main() {
  vec3 c = environment(viewRayDir(), 0.0);
  gl_FragColor = uRaw > 0.5 ? vec4(c, 1.0) : vec4(toSRGB(neutral(c * uExposure)), 1.0);
}`;

// The eye: a ray tracer in the eye frame, line by line the math of
// server/vhuman/eye/render.py shade() (+ iris.colorize / sclera.colorize).
export const EYE_FRAG = /* glsl */`
precision highp float;
precision highp int;
${COMMON_GLSL}
uniform sampler2D uIrisMasks;
uniform sampler2D uIrisNormal;
uniform sampler2D uScleraMasks;
uniform sampler2D uScleraNormal;
uniform sampler2D uChart;
uniform sampler2D uIrisPhoto;
uniform float uHasPhoto;
uniform float uIrisTexSize;
uniform float uScleraR, uLimbusR, uCorneaR, uCorneaZ, uApexZ, uAlphaLimbus, uIrisZ, uIrisR, uConvexity, uIor;
uniform float uPupil, uPupilScale, uPRef, uTexScale, uCSizeRef, uBackUv, uF0, uChamber;
uniform vec2 uPrimUV;
uniform vec2 uSecUV;
uniform float uBlend, uBlendSoft, uRadial, uShadow, uRingSize, uRingSoft, uSat, uFeather, uIrisRot, uPhotoMix;
uniform vec3 uRingColor;
uniform vec3 uTint;
uniform float uCSize, uLimbusSoft;
uniform vec3 uLimbusColor;
uniform float uScleraRot, uTransSpread, uVascInt, uVascCov;
uniform vec3 uScleraTint;
uniform vec3 uTransColor;
uniform float uCorneaRough, uScleraRough;
// The eye frame (metres, +Z gaze) and the world; uProj for the depth write.
uniform mat4 uEyeFromWorld;
uniform mat4 uWorldFromEye;
uniform mat4 uProj;

// Shared procedural albedo constants from iris.py / sclera.py.
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
const vec3 ANGLE_COLOR = vec3(0.035, 0.028, 0.026);
const vec3 SCLERA_ALBEDO = vec3(0.74, 0.70, 0.66);
float sphereEntry(vec3 o, vec3 d, vec3 c, float r) {
  vec3 oc = o - c;
  float b = dot(oc, d);
  float cc = dot(oc, oc) - r * r;
  float disc = b * b - cc;
  float t = -b - sqrt(max(disc, 0.0));
  return (disc >= 0.0 && t > 0.0) ? t : 1e20;
}
// Independent angular atlas mapping, identical to optics.uv_radius.
float uvRadius(float alpha) {
  if (alpha <= uAlphaLimbus) return uCSizeRef * alpha / uAlphaLimbus;
  return uCSizeRef + (uBackUv - uCSizeRef) * (alpha - uAlphaLimbus) / (PI - uAlphaLimbus);
}
vec3 uvTangent(float alpha, float phi) {       // render._uv_tangent
  float h = 1e-4;
  float dr = (uvRadius(alpha + h) - uvRadius(alpha - h)) / (2.0 * h);
  float r = uvRadius(alpha);
  vec3 aHat = vec3(cos(alpha) * cos(phi), cos(alpha) * sin(phi), -sin(alpha));
  vec3 pHat = vec3(-sin(phi), cos(phi), 0.0);
  vec3 t = aHat * (dr * cos(phi)) + pHat * (-r * sin(phi) / max(sin(alpha), 1e-3));
  return length(t) < 1e-9 ? vec3(1.0, 0.0, 0.0) : normalize(t);
}
vec2 rotateUv(vec2 uv, float rot) {           // sclera.rotate_uv
  float a = 2.0 * PI * rot;
  vec2 d = uv - 0.5;
  float c = cos(a), s = sin(a);
  return vec2(0.5 + c * d.x + s * d.y, 0.5 - s * d.x + c * d.y);
}
float fresnel(float c) { return uF0 + (1.0 - uF0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0); }
vec3 chart(vec2 uv) {                          // nearest-pixel lookup of the generated palette
  ivec2 p = ivec2(int(round(255.0 * clamp(uv.x, 0.0, 1.0))), int(round(255.0 * clamp(uv.y, 0.0, 1.0))));
  return srgb2lin(texelFetch(uChart, p, 0).rgb);
}
float circMask(float L, float size, float center, float soft) {   // optics.circ_mask
  float x = clamp(1.0 - (L - (size - soft * center)) / max(soft, 1e-6), 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}
vec3 desat(vec3 c, float f) { return c + (vec3(dot(c, LUMA)) - c) * f; }   // iris.desaturate
float corneaMask(float r) {
  return 1.0 - sstep(uCSize - .5 * uLimbusSoft, uCSize + .5 * uLimbusSoft, r);
}
float transmission(float r) { return circMask(r, uCSize, 0.0, uTransSpread); }   // sclera.transmission_amount
vec3 scleraColor(vec4 m, float r) {
  float coverage = sstep(uVascCov, uVascCov + .08, r);
  float density = .35 * m.g + .25 * m.b + uVascInt * m.r * coverage;
  vec3 base = SCLERA_ALBEDO * uScleraTint * (.92 + .16 * m.a);
  base *= exp(-density * vec3(.15, 1.8, 2.0));
  base *= 1.0 + (uTransColor - 1.0) * transmission(r);
  base *= 1.0 + (uLimbusColor - 1.0) * corneaMask(r);
  return max(base, 0.0);
}
vec3 customIris(vec4 m, vec3 photo) {
  float driver = uRadial > .5 ? m.b : m.r;
  float blend = sstep(uBlend - .5 * uBlendSoft, uBlend + .5 * uBlendSoft, driver);
  vec3 c = mix(chart(uPrimUV), chart(uSecUV), blend);
  c *= (.65 + .7 * m.r) * (1.0 - .6 * uShadow * (1.0 - m.g));
  if (uHasPhoto > .5) c = mix(c, photo, uPhotoMix);
  return desat(c, 1.0 - uSat);
}
float irisIrradiance(vec3 iN, vec3 L) {
  return clamp(dot(iN, L), 0.0, 1.0);
}
// Lights and environment are fixed in the world (render.py's convention).
vec3 lightEye(int k) { return normalize(mat3(uEyeFromWorld) * uBoxDir[k]); }
float lightE(int k) { return uBoxRad[k] * 4.0 * uBoxSize[k].x * uBoxSize[k].y; }
vec3 envEye(vec3 d, float r) { return environment(mat3(uWorldFromEye) * d, r); }
vec3 ambEye(vec3 n) { return ambient(normalize(mat3(uWorldFromEye) * n)); }

void main() {
  vec3 ro = (uEyeFromWorld * vec4(viewRayOrigin(), 1.0)).xyz;
  vec3 rd = normalize(mat3(uEyeFromWorld) * viewRayDir());
  float ts = sphereEntry(ro, rd, vec3(0.0), uScleraR);
  float tc = sphereEntry(ro, rd, vec3(0.0, 0.0, uCorneaZ), uCorneaR);
  float t = min(ts, tc);
  bool hit = t < 1e19;
  if (!hit) t = max(-dot(ro, rd), 0.0);     // keep everything finite for the derivatives; discarded below
  bool onC = hit && tc <= ts;
  vec3 x = ro + t * rd;
  vec3 n = onC ? normalize(x - vec3(0.0, 0.0, uCorneaZ)) : normalize(x);
  float ndv = clamp(dot(n, -rd), 1e-4, 1.0);
  vec3 dir = normalize(x);
  float alpha = acos(clamp(dir.z, -1.0, 1.0));
  float ruv = uvRadius(alpha);
  float phiS = atan(dir.y, dir.x);
  vec2 uv = vec2(0.5 + ruv * cos(phiS), 0.5 - ruv * sin(phiS));
  float cw = 1.0 - sstep(uCSizeRef - 0.004, uCSizeRef + 0.004, ruv);
  float fres = fresnel(ndv);

  // Sclera: tinted albedo, normal-mapped, wrapped diffuse near the limbus.
  vec2 ruvS = rotateUv(uv, uScleraRot);
  vec4 mS = texture(uScleraMasks, ruvS);
  vec3 albS = scleraColor(mS, ruv);
  vec3 nmS = texture(uScleraNormal, ruvS).xyz * 2.0 - 1.0;
  vec3 tang = uvTangent(alpha, phiS);
  vec3 bit = cross(n, tang);
  vec3 nS = normalize(tang * nmS.x + bit * nmS.y + n * nmS.z);
  float wrap = 0.2 + 0.6 * transmission(ruv);
  vec3 diffS = vec3(0.0);
  for (int k = 0; k < 2; k++) {
    float ndl = dot(nS, lightEye(k));
    diffS += lightE(k) * max((ndl + wrap) / (1.0 + wrap), 0.0) / PI;
  }
  diffS += ambEye(nS);
  vec3 colS = albS * diffS;

  // Cornea: refract into the chamber and land on the iris surface.
  vec3 td = refract(rd, n, 1.0 / uIor);
  float h = uConvexity, rl = uLimbusR;
  float tzd = min(td.z, -1e-6);
  float tt = (uIrisZ + h - x.z) / tzd;
  for (int i = 0; i < 2; i++) {
    vec3 q0 = x + tt * td;
    float zt = uIrisZ + h * (1.0 - min(length(q0.xy), rl) / rl);
    tt = (zt - x.z) / tzd;
  }
  vec3 q = x + tt * td;
  float rh = length(q.xy);
  float tI = rh / uIrisR;
  float phi = atan(q.y, q.x);
  float rotA = 2.0 * PI * uIrisRot;
  float phiR = phi - rotA;
  float tc1 = min(tI, 1.0);
  float tTex = min(tc1 <= uPupil ? tc1 * uPRef / uPupil : uPRef + (1.0 - uPRef) * (tc1 - uPupil) / (1.0 - uPupil), 1.0);
  vec2 iuv = vec2(0.5 + uTexScale * tTex * cos(phiR), 0.5 - uTexScale * tTex * sin(phiR));
  // Mip gradients of the refracted lookup, clamped: the cornea/sclera seam makes them jump.
  vec2 gx = dFdx(iuv), gy = dFdy(iuv);
  float gm = max(length(gx), length(gy)), glim = 3.0 / uIrisTexSize;
  if (gm > glim) { gx *= glim / gm; gy *= glim / gm; }
  vec4 mI = textureGrad(uIrisMasks, iuv, gx, gy);
  vec3 photo = srgb2lin(textureGrad(uIrisPhoto, iuv, gx, gy).rgb);
  // Independent iris albedo, ring, pupil and limbus blend.
  float rUv = tI * uCSize;
  float cm = corneaMask(rUv);
  vec3 irisC = customIris(mI, photo) * uTint;
  float ring = sstep(uRingSize - uRingSoft, uRingSize + uRingSoft, tTex);
  irisC *= 1.0 + (uRingColor - 1.0) * ring;
  float edge = .005 + .04 * uFeather;
  irisC *= sstep(uPRef - edge, uPRef + edge, tTex);
  vec2 uvP = vec2(0.5 + rUv * cos(phi), 0.5 - rUv * sin(phi));
  vec3 sclP = scleraColor(textureLod(uScleraMasks, rotateUv(uvP, uScleraRot), 0.0), rUv);
  vec3 albI = sclP + (irisC - sclP) * cm;
  if (rh > rl) albI = ANGLE_COLOR;                   // the chamber wall beyond the limbus
  vec3 nm = textureGrad(uIrisNormal, iuv, gx, gy).xyz * 2.0 - 1.0;
  float cr = cos(rotA), sr = sin(rotA);
  vec3 irisN = normalize(vec3((nm.x * cr - nm.y * sr) * 0.6, (nm.x * sr + nm.y * cr) * 0.6, nm.z));
  vec3 diffI = vec3(0.0);
  for (int k = 0; k < 2; k++) diffI += lightE(k) * irisIrradiance(irisN, lightEye(k)) / PI;
  diffI += ambEye(irisN) * (1.0 - 0.5 * (1.0 - mI.g) * uShadow);
  vec3 colI = albI * diffI * (1.0 - fresnel(clamp(dot(-td, n), 0.0, 1.0)));

  vec3 refl = reflect(rd, n);
  vec3 env = envEye(refl, uCorneaRough) * cw + envEye(refl, uScleraRough) * (1.0 - cw);
  vec3 col = (colI * cw + colS * (1.0 - cw)) * (1.0 - fres) + fres * env;
  if (!hit) discard;
  // the depth of the surface hit, not of the proxy: lids and eyeshells in front occlude it
  vec4 clip = uProj * viewMatrix * (uWorldFromEye * vec4(x, 1.0));
  gl_FragDepth = clamp(0.5 * clip.z / clip.w + 0.5, 0.0, 1.0);
  gl_FragColor = vec4(toSRGB(neutral(col * uExposure)), 1.0);
}`;

// ---------------------------------------------------------------- uniforms and materials
function blankTexture(rgba) {
  const t = new THREE.DataTexture(new Uint8Array(rgba), 1, 1, THREE.RGBAFormat);
  t.needsUpdate = true;
  return t;
}
export const BLANK = blankTexture([0, 0, 0, 255]);

// The studio (lights, sky), shared by the eye materials and the background.
export function makeStudioUniforms() {
  return {
    uBoxDir: { value: [new THREE.Vector3(), new THREE.Vector3()] },
    uBoxU: { value: [new THREE.Vector3(), new THREE.Vector3()] },
    uBoxV: { value: [new THREE.Vector3(), new THREE.Vector3()] },
    uBoxSize: { value: [new THREE.Vector2(), new THREE.Vector2()] },
    uBoxRad: { value: [0, 0] },
    uSky: { value: new THREE.Vector3() }, uGround: { value: new THREE.Vector3() }, uHorizon: { value: new THREE.Vector3() },
    uExposure: { value: 1 }, uOrtho: { value: 0 }, uCamDir: { value: new THREE.Vector3(0, 0, -1) }, uRaw: { value: 0 },
  };
}

// optics.uniforms' studio -> uniforms. `yaw` turns the studio about +Y (the
// eye page's eye gazes along +Z; a Pixal3D head faces -Z, so yaw = pi there).
export function applyStudio(studio, g, yaw = 0) {
  const up = new THREE.Vector3(0, 1, 0);
  g.softboxes.forEach((b, k) => {
    const w = new THREE.Vector3(...b.direction).normalize().applyAxisAngle(up, yaw);
    const uu = new THREE.Vector3().crossVectors(up, w).normalize();
    const vv = new THREE.Vector3().crossVectors(w, uu);
    studio.uBoxDir.value[k].copy(w); studio.uBoxU.value[k].copy(uu); studio.uBoxV.value[k].copy(vv);
    studio.uBoxSize.value[k].set(b.size[0], b.size[1]); studio.uBoxRad.value[k] = b.radiance;
  });
  studio.uSky.value.set(...g.sky); studio.uGround.value.set(...g.ground); studio.uHorizon.value.set(...g.horizon);
}

// The studio as a background (BackSide sphere) or, with uRaw, for an environment capture.
export function makeBackgroundMaterial(studio, { raw = false } = {}) {
  const u = Object.assign({}, studio, { uRaw: { value: raw ? 1 : 0 } });
  return new THREE.ShaderMaterial({ uniforms: u, vertexShader: VERT, fragmentShader: BG_FRAG, side: THREE.BackSide,
                                    depthWrite: false });
}

const SCALARS = ['uScleraR', 'uLimbusR', 'uCorneaR', 'uCorneaZ', 'uApexZ', 'uAlphaLimbus', 'uIrisZ', 'uIrisR', 'uConvexity',
                 'uIor', 'uPupil', 'uPupilScale', 'uPRef', 'uTexScale', 'uCSizeRef', 'uBackUv', 'uF0', 'uChamber', 'uBlend',
                 'uBlendSoft', 'uRadial', 'uShadow', 'uRingSize', 'uRingSoft', 'uSat', 'uFeather', 'uIrisRot', 'uPhotoMix',
                 'uCSize', 'uLimbusSoft', 'uScleraRot', 'uTransSpread', 'uVascInt', 'uVascCov', 'uCorneaRough',
                 'uScleraRough'];

export function makeEyeMaterial(studio) {
  const u = Object.assign({}, studio, {
    uIrisMasks: { value: BLANK }, uIrisNormal: { value: BLANK }, uScleraMasks: { value: BLANK }, uScleraNormal: { value: BLANK },
    uChart: { value: BLANK }, uIrisPhoto: { value: BLANK }, uHasPhoto: { value: 0 }, uIrisTexSize: { value: 1024 },
    uPrimUV: { value: new THREE.Vector2() }, uSecUV: { value: new THREE.Vector2() },
    uRingColor: { value: new THREE.Vector3() }, uTint: { value: new THREE.Vector3(1, 1, 1) },
    uLimbusColor: { value: new THREE.Vector3(1, 1, 1) }, uScleraTint: { value: new THREE.Vector3(1, 1, 1) },
    uTransColor: { value: new THREE.Vector3(1, 1, 1) },
    uEyeFromWorld: { value: new THREE.Matrix4() }, uWorldFromEye: { value: new THREE.Matrix4() }, uProj: { value: new THREE.Matrix4() },
  });
  for (const k of SCALARS) u[k] = { value: 0 };
  return new THREE.ShaderMaterial({ uniforms: u, vertexShader: VERT, fragmentShader: EYE_FRAG,
                                    side: THREE.BackSide });
}

// The proxy: a sphere around the eyeball (eye-local metres). With
// `followNode`, the eye frame is the mesh's world transform (a child of a
// GLB eye node: rotation, translation, units per metre); otherwise the
// caller sets uWorldFromEye / uEyeFromWorld.
export function makeEyeProxy(material, g, { followNode = false } = {}) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(g.apexZ * 1.08, 64, 48), material);
  mesh.onBeforeRender = (renderer, scene, camera) => {
    const u = mesh.material.uniforms;
    u.uProj.value.copy(camera.projectionMatrix);
    if (followNode) {
      u.uWorldFromEye.value.copy(mesh.matrixWorld);
      u.uEyeFromWorld.value.copy(mesh.matrixWorld).invert();
    }
  };
  return mesh;
}

// optics.uniforms' eye geometry -> uniforms.
export function applyEyeGeometry(material, g) {
  const u = material.uniforms;
  const set = { uScleraR: g.scleraRadius, uLimbusR: g.limbusRadius, uCorneaR: g.corneaRadius, uCorneaZ: g.corneaCenterZ,
                uApexZ: g.apexZ, uAlphaLimbus: g.alphaLimbus, uPRef: g.pRef, uTexScale: g.irisTexScale,
                uCSizeRef: g.corneaSizeRef, uBackUv: g.backUvRadius, uF0: g.f0 };
  for (const [k, v] of Object.entries(set)) u[k].value = v;
}

// Live parameters -> uniforms (params.py semantics). Returns the pupil ratio and scale.
export function pupilScale(p) { return p.pupil.dilation * p.pupil.scale; }                      // params.pupil_scale
export function pupilRatio(p, pRef) { return Math.min(Math.max(pRef * pupilScale(p), 0.05), 0.85); }
export function scleraTint(p) {
  if (p.sclera.use_custom_tint) return p.sclera.tint;
  return [0.9, 0.85, 0.8].map((c) => 1 + (c - 1) * p.sclera.skin_u);
}
export function applyEyeParams(material, p, g) {
  const u = material.uniforms, i = p.iris, o = p.optics;
  u.uIrisZ.value = g.apexZ - o.chamber_depth;
  u.uChamber.value = o.chamber_depth;
  u.uIrisR.value = g.limbusRadius * p.cornea.size / g.corneaSizeRef;
  u.uConvexity.value = o.iris_convexity; u.uIor.value = o.ior;
  u.uF0.value = ((p.optics.ior - 1) / (p.optics.ior + 1)) ** 2;
  u.uPupil.value = pupilRatio(p, g.pRef); u.uPupilScale.value = pupilScale(p);
  u.uPrimUV.value.set(i.primary_color_u, i.primary_color_v);
  u.uSecUV.value.set(i.secondary_color_u, i.secondary_color_v);
  u.uBlend.value = i.color_blend; u.uBlendSoft.value = i.color_blend_softness;
  u.uRadial.value = i.blend_method === 'Radial' ? 1 : 0;
  u.uShadow.value = i.shadow_details; u.uRingSize.value = i.limbal_ring_size; u.uRingSoft.value = i.limbal_ring_softness;
  u.uRingColor.value.set(...i.limbal_ring_color); u.uSat.value = i.global_saturation; u.uTint.value.set(...i.global_tint);
  u.uFeather.value = p.pupil.feather; u.uIrisRot.value = i.rotation; u.uPhotoMix.value = i.photo_mix;
  u.uCSize.value = p.cornea.size; u.uLimbusSoft.value = p.cornea.limbus_softness; u.uLimbusColor.value.set(...p.cornea.limbus_color);
  u.uScleraRot.value = p.sclera.rotation; u.uScleraTint.value.set(...scleraTint(p));
  u.uTransSpread.value = p.sclera.transmission_spread; u.uTransColor.value.set(...p.sclera.transmission_color);
  u.uVascInt.value = p.sclera.vascularity_intensity; u.uVascCov.value = p.sclera.vascularity_coverage;
  u.uCorneaRough.value = o.cornea_roughness; u.uScleraRough.value = o.sclera_roughness;
  return { pupilRatio: u.uPupil.value, pupilScale: u.uPupilScale.value };
}

// A server PNG as a data texture (masks, normals, and sRGB data the shader decodes).
export async function loadBitmapTexture(url, { mips = true, anisotropy = 1 } = {}) {
  const blob = await (await fetch(url)).blob();
  const bmp = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  const t = new THREE.Texture(bmp);
  t.colorSpace = THREE.NoColorSpace;
  t.flipY = false;                        // PNG row 0 = v 0 (glTF convention)
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  if (mips) { t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter; t.anisotropy = anisotropy; }
  else { t.generateMipmaps = false; t.minFilter = THREE.NearestFilter; t.magFilter = THREE.NearestFilter; }
  t.needsUpdate = true;
  return t;
}

// The texture URLs of a /v1/eye/textures response -> textures (loaded in
// parallel), then bound to the material; the old ones are disposed.
export async function loadEyeTextures(resp, { anisotropy = 1 } = {}) {
  const names = ['iris_masks', 'iris_normal', 'sclera_masks', 'sclera_normal'];
  const tex = await Promise.all([
    ...names.map((n) => loadBitmapTexture(resp.urls[n], { anisotropy })),
    loadBitmapTexture(resp.urls.iris_color_chart, { mips: false }),
    resp.urls.iris_photo ? loadBitmapTexture(resp.urls.iris_photo, { anisotropy }) : Promise.resolve(null),
  ]);
  return { irisMasks: tex[0], irisNormal: tex[1], scleraMasks: tex[2], scleraNormal: tex[3], chart: tex[4], photo: tex[5],
           res: resp.res, dispose: () => tex.forEach((t) => t && t.dispose()) };
}
export function bindEyeTextures(material, t) {
  const u = material.uniforms;
  const old = [u.uIrisMasks, u.uIrisNormal, u.uScleraMasks, u.uScleraNormal, u.uChart, u.uIrisPhoto].map((x) => x.value);
  u.uIrisMasks.value = t.irisMasks; u.uIrisNormal.value = t.irisNormal;
  u.uScleraMasks.value = t.scleraMasks; u.uScleraNormal.value = t.scleraNormal; u.uChart.value = t.chart;
  u.uIrisPhoto.value = t.photo || BLANK;
  u.uHasPhoto.value = t.photo ? 1 : 0;
  u.uIrisTexSize.value = t.res;
  const now = new Set([t.irisMasks, t.irisNormal, t.scleraMasks, t.scleraNormal, t.chart, t.photo]);
  old.forEach((x) => x && x !== BLANK && !now.has(x) && x.dispose && x.dispose());
}
