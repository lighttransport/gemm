// The runtime eye: a low-polygon eyeball mesh (server/vhuman/eye/geometry.py
// runtime_shell) shaded with the same analytic material as the proxy ray
// tracer (vhuman_eye_shader.js EYE_SHADE_GLSL, i.e. render.py shade_points).
// Standard depth, front faces: it composites like any mesh, so it also
// serves the rig viewer, the mobile player and other renderers.
import * as THREE from 'three';
import { COMMON_GLSL, EYE_SHADE_GLSL, VERT, makeEyeUniforms } from './vhuman_eye_shader.js';

export const EYE_MESH_FRAG = /* glsl */`
precision highp float;
precision highp int;
${COMMON_GLSL}
${EYE_SHADE_GLSL}
void main() {
  vec3 xf = (uEyeFromWorld * vec4(vWorld, 1.0)).xyz;
  vec3 rd = normalize(mat3(uEyeFromWorld) * viewRayDir());
  bool hit;
  vec3 x = projectToSurface(xf, rd, hit);              // render.project_to_surface
  float alpha = acos(clamp(normalize(x).z, -1.0, 1.0));
  vec3 col = shadeEye(x, rd, alpha <= uAlphaLimbus);   // render.surface_normal's classification
  gl_FragColor = uSkinPass > .5 ? vec4(0.) : (uRaw > .5 ? vec4(col, 1.) : vec4(toSRGB(neutral(col * uExposure)), 1.0));
}`;

export function makeEyeMeshMaterial(studio) {
  return new THREE.ShaderMaterial({ uniforms: makeEyeUniforms(studio), vertexShader: VERT, fragmentShader: EYE_MESH_FRAG,
                                    side: THREE.FrontSide });
}

// geometry.runtime_alphas: dense cornea, moderate visible sclera, an equator
// ring for the silhouette, coarse rings inside the socket.
export function runtimeAlphas(g, rings = 16) {
  const aL = g.alphaLimbus, deg80 = 80 * Math.PI / 180;
  const front = Math.max(Math.round(rings * 0.4), 3), mid = Math.max(Math.round(rings * 0.35), 3);
  const back = Math.max(rings - front - mid - 1, 1);
  const tail = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * (i + 1) / n);   // linspace(a, b, n+1)[1:]
  const hidden = Array.from({ length: back }, (_, i) => Math.PI / 2 + (Math.PI / 2) * (i + 1) / (back + 1));
  return [...tail(0, 1.35 * aL, front), ...tail(1.35 * aL, deg80, mid), Math.PI / 2, ...hidden];
}

// optics.surface_radius: smooth union of the sclera and cornea balls.
export function surfaceRadius(alpha, g) {
  const zc = g.corneaCenterZ, rc = g.corneaRadius, k = g.limbusBlend;
  const s = Math.sin(alpha);
  const cornea = zc * Math.cos(alpha) + Math.sqrt(Math.max(rc * rc - (zc * s) ** 2, 0));
  const a = g.scleraRadius, b = cornea;
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.max(a, b) + h * h * k * 0.25;
}

// geometry.runtime_shell as a BufferGeometry (eye-local metres, +Z gaze).
export function makeRuntimeShellGeometry(g, rings = 16, segments = 36) {
  const alphas = runtimeAlphas(g, rings);
  const pos = [0, 0, g.apexZ], nrm = [0, 0, 1];
  for (const a of alphas) {
    const rho = surfaceRadius(a, g);
    for (let s = 0; s < segments; s++) {
      const f = 2 * Math.PI * s / segments;
      const d = [Math.sin(a) * Math.cos(f), Math.sin(a) * Math.sin(f), Math.cos(a)];
      pos.push(d[0] * rho, d[1] * rho, d[2] * rho);
      nrm.push(...d);
    }
  }
  pos.push(0, 0, -g.scleraRadius); nrm.push(0, 0, -1);
  const n = alphas.length, first = 1, last = 1 + n * segments, idx = [];
  for (let s = 0; s < segments; s++) idx.push(0, first + s, first + (s + 1) % segments);
  for (let r = 0; r < n - 1; r++) for (let s = 0; s < segments; s++) {
    const a = first + r * segments + s, b = first + r * segments + (s + 1) % segments, c = a + segments, d = b + segments;
    idx.push(a, c, b, b, c, d);
  }
  const lastRing = first + (n - 1) * segments;
  for (let s = 0; s < segments; s++) idx.push(last, lastRing + (s + 1) % segments, lastRing + s);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  geo.setIndex(idx);
  return geo;
}

// A runtime eye mesh whose material tracks the mesh's world transform as the
// eye frame (a child of a GLB eye node, or placed by the caller).
export function makeRuntimeEye(material, g, { rings = 16, segments = 36 } = {}) {
  const mesh = new THREE.Mesh(makeRuntimeShellGeometry(g, rings, segments), material);
  attachEyeFrame(mesh);
  return mesh;
}

export function attachEyeFrame(mesh) {
  mesh.onBeforeRender = (renderer, scene, camera) => {
    const u = mesh.material.uniforms;
    u.uProj.value.copy(camera.projectionMatrix);
    u.uWorldFromEye.value.copy(mesh.matrixWorld);
    u.uEyeFromWorld.value.copy(mesh.matrixWorld).invert();
  };
}
