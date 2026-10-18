// The rig evaluator shared by the rig viewer and the mobile player (same
// equations as server/vhuman/rig/rigdef.py and rig/mldeformer.py).
import * as THREE from 'three';

// ---- the linear rig (same equations as server/vhuman/rig/rigdef.py) --------------------
const ATTRS = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'];
export class LinearRig {
  constructor(def) {
    this.def = def;
    this.controls = def.controls.map((c) => c.name);
    this.cidx = Object.fromEntries(this.controls.map((n, i) => [n, i]));
    this.inputs = this.controls.concat(def.correctives.map((c) => c.name));
    this.iidx = Object.fromEntries(this.inputs.map((n, i) => [n, i]));
    this.joints = def.joints;
    this.jidx = Object.fromEntries(this.joints.map((j, i) => [j.name, i]));
    this.M = def.joint_matrix.map((e) => [this.jidx[e.joint] * 6 + ATTRS.indexOf(e.attr), this.iidx[e.input], e.value]);
    this.shapes = def.blendshapes.map((b) => [b.name, this.iidx[b.input]]);
  }
  evaluate(values) {
    const x = this.def.controls.map((c) => Math.min(c.max, Math.max(c.min, values[c.name] || 0)));
    const clamp01 = (v) => Math.min(1, Math.max(0, v));
    const inp = x.concat(this.def.correctives.map((c) => Math.min(1, c.inputs.reduce((p, n) => p * clamp01(x[this.cidx[n]]), c.weight ?? 1))));
    const delta = new Float64Array(this.joints.length * 6);
    for (const [row, col, v] of this.M) delta[row] += v * inp[col];
    const weights = {};
    for (const [name, src] of this.shapes) weights[name] = inp[src];
    if (this.ml && this.mlOn) Object.assign(weights, this.ml.weights(Object.fromEntries(this.inputs.map((n, i) => [n, inp[i]]))));
    return { delta, weights };
  }
}
// ---- the ML corrective deformer (server/vhuman/rig/mldeformer.py) --------------------------
export async function loadSafetensors(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const n = Number(new DataView(buf).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const out = {};
  for (const [k, h] of Object.entries(header)) {
    if (k === '__metadata__') continue;
    const [a, b] = h.data_offsets;
    out[k] = { shape: h.shape, data: new Float32Array(buf.slice(8 + n + a, 8 + n + b)) };
  }
  return out;
}
export class MLDeformer {        // c = fc2(relu(fc1((x - mean) * scale))) * out.scale + out.mean; weight_k = c_k / out.scale_k
  constructor(t, def) {
    this.t = t; this.def = def;
    [this.H, this.I] = t['fc1.weight'].shape; this.K = t['fc2.weight'].shape[0];
    this.h = new Float32Array(this.H); this.x = new Float32Array(this.I);
    this.lo = def.inputs.map((n) => (def.ranges[n] || [0, 1])[0]); this.hi = def.inputs.map((n) => (def.ranges[n] || [0, 1])[1]);
  }
  weights(values) {
    const t = this.t, x = this.x, h = this.h, out = { ml_mean: 1 };
    this.def.inputs.forEach((n, i) => { x[i] = (Math.max(this.lo[i], Math.min(this.hi[i], values[n] || 0)) - t['input.mean'].data[i]) * t['input.scale'].data[i]; });
    const W1 = t['fc1.weight'].data, b1 = t['fc1.bias'].data, W2 = t['fc2.weight'].data, b2 = t['fc2.bias'].data;
    for (let j = 0; j < this.H; j++) { let s = b1[j]; for (let i = 0; i < this.I; i++) s += W1[j * this.I + i] * x[i]; h[j] = s > 0 ? s : 0; }
    const om = t['output.mean'].data, os = t['output.scale'].data;
    for (let k = 0; k < this.K; k++) {
      let s = b2[k]; for (let j = 0; j < this.H; j++) s += W2[k * this.H + j] * h[j];
      out[this.def.targets[k + 1]] = (s * os[k] + om[k]) / os[k];
    }
    return out;
  }
}


export function eulerZYX(rx, ry, rz) {   // Rz @ Ry @ Rx (radians)
  const cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry), cz = Math.cos(rz), sz = Math.sin(rz);
  const Rx = new THREE.Matrix4().set(1, 0, 0, 0, 0, cx, -sx, 0, 0, sx, cx, 0, 0, 0, 0, 1);
  const Ry = new THREE.Matrix4().set(cy, 0, sy, 0, 0, 1, 0, 0, -sy, 0, cy, 0, 0, 0, 0, 1);
  const Rz = new THREE.Matrix4().set(cz, -sz, 0, 0, sz, cz, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
  return Rz.multiply(Ry).multiply(Rx);
}
export function restMatrix(j) {       // row-major nested rest rotation -> Matrix4
  const R = j.rest_rotation;
  return new THREE.Matrix4().set(R[0][0], R[0][1], R[0][2], 0, R[1][0], R[1][1], R[1][2], 0, R[2][0], R[2][1], R[2][2], 0, 0, 0, 0, 1);
}
