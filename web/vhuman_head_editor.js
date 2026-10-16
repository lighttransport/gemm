import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { SkinRenderer } from '/vhuman_skin_shader.js';
import * as Eye from '/vhuman_eye_shader.js';

export function initializeHeadEditor() {
const $ = (id) => document.getElementById(id);
const state = { heads: [], current: null, job: null, model: null };
window.__headState = state;

async function api(path, body) {
  const opts = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

// ---- viewer
const viewer = $('viewer');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewer.prepend(renderer.domElement);
const skinRenderer = new SkinRenderer(renderer);
window.__skinRenderer = skinRenderer;
const scene = new THREE.Scene();
const skinControl = document.createElement('label');
skinControl.style.cssText='position:absolute;right:12px;bottom:12px;z-index:3;background:#171c24;padding:8px;border-radius:8px';
skinControl.innerHTML='<input type="checkbox" style="width:auto"> Skin scattering <span></span>';
const skinToggle=skinControl.querySelector('input'), skinTiming=skinControl.querySelector('span');
skinToggle.disabled=!skinRenderer.supported;
skinControl.title=skinRenderer.status;
renderer.domElement.parentElement.append(skinControl);
skinToggle.onchange=()=>{skinRenderer.enabled=skinToggle.checked; render();};
setInterval(()=>{skinTiming.textContent=skinRenderer.enabled?`${skinRenderer.ms.toFixed(1)} ms CPU`:'';},1000);

const pmrem = new THREE.PMREMGenerator(renderer);
const roomEnv = pmrem.fromScene(new RoomEnvironment(renderer), 0.03).texture;
const key = new THREE.DirectionalLight(0xffffff, 1.2);
key.position.set(-1.2, 1.4, -2.0);
scene.add(key);
// The eye shader's studio (softboxes + sky, optics.uniforms), turned to face
// the head (it looks along -Z); as an environment it lights the head too.
const studio = Eye.makeStudioUniforms();
let studioEnv = null, geometry = null;
function buildStudioEnv(g) {
  Eye.applyStudio(studio, g, Math.PI);
  const envScene = new THREE.Scene();
  const bg = new THREE.Mesh(new THREE.SphereGeometry(10, 64, 32), Eye.makeBackgroundMaterial(studio, { raw: true }));
  envScene.add(bg);
  if (studioEnv) studioEnv.dispose();
  studioEnv = pmrem.fromScene(envScene, 0.0).texture;
  bg.geometry.dispose(); bg.material.dispose();
}
function applyLighting() {
  const useStudio = $('studio').checked && studioEnv;
  scene.environment = useStudio ? studioEnv : roomEnv;
  key.visible = !useStudio;
  render();
}
const camera = new THREE.PerspectiveCamera(22, 1, 0.005, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.addEventListener('change', render);
Object.assign(state, { render: () => render(), camera, controls });   // for tests and debugging
let eyeCenter = new THREE.Vector3(), headCenter = new THREE.Vector3(), headSize = 1;
function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  render();
}
function render() { skinRenderer.render(scene, camera); }
window.addEventListener('resize', resize);

function setView(name) {
  // The face looks along -Z (Pixal3D's GLB frame).
  const d = headSize * 3.4;
  const dirs = { front: [0, 0, -1], three: [-0.6, 0.05, -0.8], profile: [-1, 0, -0.02] };
  if (name === 'eyes') {
    controls.target.copy(eyeCenter);
    camera.position.copy(eyeCenter).add(new THREE.Vector3(-0.08, 0.02, -1).normalize().multiplyScalar(headSize * 0.8));
  } else {
    controls.target.copy(headCenter);
    camera.position.copy(headCenter).add(new THREE.Vector3(...dirs[name]).normalize().multiplyScalar(d));
  }
  controls.update();
  render();
}
document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
$('exposure').addEventListener('input', () => {
  renderer.toneMappingExposure = +$('exposure').value;
  studio.uExposure.value = +$('exposure').value;      // the eye shader tone-maps itself (the same Neutral curve)
  render();
});
for (const id of ['show-head', 'show-eyes', 'analytic']) $(id).addEventListener('change', applyVisibility);
$('studio').addEventListener('change', applyLighting);
const shellMaterials = new WeakMap();
function applyLidShadow() {
  const strength = +$('lid-shadow').value;
  $('lid-shadow-value').value = strength.toFixed(2);
  state.model?.traverse((o) => {
    if (!o.isMesh || !/^eye_(left|right)_eyeshell$/.test(o.name)) return;
    for (const mat of (Array.isArray(o.material) ? o.material : [o.material])) {
      if (!shellMaterials.has(mat)) shellMaterials.set(mat, { opacity: mat.opacity, depthWrite: mat.depthWrite });
      const base = shellMaterials.get(mat);
      mat.opacity = base.opacity * strength;
      // A fully transparent shell must not occlude other translucent layers.
      mat.depthWrite = strength > 0 && base.depthWrite;
    }
  });
  render();
}
$('lid-shadow').addEventListener('input', applyLidShadow);
function applyVisibility() {
  if (!state.model) return;
  state.model.traverse((o) => {
    if (!o.isMesh) return;
    const isEye = /eye_/.test(o.name) || /eye_/.test(o.parent?.name || '');
    let on = isEye ? $('show-eyes').checked : $('show-head').checked;
    // analytic: the shader proxies replace the GLB's eyeball and iris meshes (the eyeshells stay)
    const analytic = $('analytic').checked && state.eyesAnalytic;
    if (/^eye_(left|right)_(shell|iris)$/.test(o.name)) on = on && !analytic;
    if (o.userData.analyticEye) on = on && analytic;
    o.visible = on;
  });
  $('vlabel').textContent = $('analytic').checked && state.eyesAnalytic
    ? 'GLB head · analytic eye shader (ray-traced refraction)' : 'GLB · head + portable eyes (KHR transmission cornea)';
  render();
}

const loader = new GLTFLoader();
let headLoadGeneration = 0;
async function loadHead(h) {
  const generation = ++headLoadGeneration;
  window.__headReady = false;
  state.current = h;
  $('apply-skin').disabled = false;
  $('video-go').disabled = !videoAvailable || !!videoJob;
  refreshVideos(h.id);
  document.querySelectorAll('.head').forEach((el) => el.classList.toggle('sel', el.dataset.id === h.id));
  $('empty').textContent = 'loading…';
  $('empty').style.display = 'flex';
  $('portrait').src = h.urls.portrait || '';
  $('landmarks').src = h.urls.landmarks || '';
  $('dl-glb').href = h.urls.head_eyes || '#';
  $('dl-pixal').href = h.urls.pixal3d || '#';
  $('dl-fit').href = h.urls.fit || '#';
  $('skin-downloads').replaceChildren(...['skin_basecolor', 'skin_normal', 'skin_orm', 'skin_mask', 'skin'].filter(n => h.urls[n]).map(n => {
    const a = document.createElement('a'); a.className = 'dl'; a.href = h.urls[n]; a.download = ''; a.textContent = n.replaceAll('_', ' '); return a;
  }));
  const f = h.fit || {};
  const rows = [
    ['person', h.subject || '—'], ['seed / quality', `${h.seed ?? '—'} / ${h.quality || '—'}`],
    ['scale (mesh units per m)', f.units_per_m?.toFixed(3)], ['head+bust height', f.head_height_m ? (f.head_height_m * 100).toFixed(1) + ' cm' : '—'],
    ['IPD vs iris-width scale', f.k_ipd ? `${f.k_ipd.toFixed(3)} vs ${f.k_iris.toFixed(3)}` : '—'],
    ['skin textures', h.skin ? `${h.skin.resolution.join(' × ')} · ${h.skin.seconds.toFixed(1)} s` : 'source texture'],
    ['iris plate', h.plate || 'procedural'], ['iris source', h.iris?.source || 'library'], ['total time', h.seconds ? h.seconds.toFixed(1) + ' s' : '—'],
  ];
  $('facts').replaceChildren(...rows.map(([k, v]) => { const tr = document.createElement('tr'); tr.innerHTML = '<td></td><td></td>'; tr.children[0].textContent = k; tr.children[1].textContent = v ?? '—'; return tr; }));
  if (state.model) { scene.remove(state.model); state.model = null; }
  if (!h.urls.head_eyes) { $('empty').textContent = 'no GLB'; return; }
  const gltf = await loader.loadAsync(h.urls.head_eyes);
  if (generation !== headLoadGeneration) return;
  state.model = gltf.scene;
  skinRenderer.unitsPerM=h.fit?.units_per_m||1;
  try {
    const mask=await new THREE.TextureLoader().loadAsync(`/v1/heads/${h.id}/skin_mask.png`);
    mask.flipY=false;mask.colorSpace=THREE.NoColorSpace;
    state.model.traverse(o=>{if(o.isMesh)for(const m of [].concat(o.material))if(m.name==='head')m.userData.skinMask=mask;});
  }catch(_){} // imported legacy heads can lack a authored skin mask
  scene.add(state.model);
  const box = new THREE.Box3().setFromObject(state.model);
  box.getCenter(headCenter);
  headSize = box.getSize(new THREE.Vector3()).length() * 0.5;
  const eyes = [];
  state.model.traverse((o) => { if (/^eye_(left|right)$/.test(o.name)) eyes.push(o.getWorldPosition(new THREE.Vector3())); });
  if (eyes.length) eyeCenter = eyes.reduce((a, b) => a.add(b), new THREE.Vector3()).multiplyScalar(1 / eyes.length);
  state.eyesAnalytic = false;
  try { await addAnalyticEyes(h, state.model); } catch (e) { console.error('analytic eyes:', e); }
  if (generation !== headLoadGeneration) return;
  $('empty').style.display = 'none';
  applyLidShadow();
  applyVisibility();
  setView('front');
  window.__headReady = true;
}

// The analytic eyes: per eye, the fit's parameters (its side) and iris plate
// -> /v1/eye/textures -> a shader proxy under the GLB's eye node.
async function addAnalyticEyes(h, model) {
  if (!h.urls.fit) return;
  const rec = await fetch(h.urls.fit).then((r) => r.json());
  if (state.model !== model) return;
  const detail = rec.plate ? { source: 'plate', id: rec.plate } : { source: 'procedural' };
  const made = [];
  for (const side of ['right', 'left']) {
    const node = model.getObjectByName(`eye_${side}`);
    if (!node || !rec.eye_params) continue;
    const params = JSON.parse(JSON.stringify(rec.eye_params));
    params.optics.side = side;
    const resp = await api('/v1/eye/textures', { params, res: 1024, detail });
    const g = resp.uniforms;
    if (!geometry) { geometry = g; buildStudioEnv(g); applyLighting(); }
    const mat = Eye.makeEyeMaterial(studio);
    Eye.applyEyeGeometry(mat, g);
    Eye.applyEyeParams(mat, params, g);
    Eye.bindEyeTextures(mat, await Eye.loadEyeTextures(resp, { anisotropy: renderer.capabilities.getMaxAnisotropy() }));
    const proxy = Eye.makeEyeProxy(mat, g, { followNode: true });
    proxy.name = `eye_${side}_analytic`;
    proxy.userData.analyticEye = true;
    made.push([node, proxy]);
  }
  if (state.model !== model) return;               // another head was picked meanwhile
  for (const [node, proxy] of made) node.add(proxy);
  state.eyesAnalytic = made.length > 0;
}

// ---- heads list and jobs
async function refreshHeads(selectId) {
  const { heads } = await api('/v1/heads');
  state.heads = heads;
  const box = $('heads');
  if (!heads.length) { box.innerHTML = '<span class="msg">none yet</span>'; return; }
  box.replaceChildren(...heads.map((h) => {
    const el = document.createElement('div');
    el.className = 'head';
    el.dataset.id = h.id;
    const img = document.createElement('img'); img.src = h.urls.portrait || ''; img.alt = '';
    const txt = document.createElement('div');
    const b = document.createElement('b'); b.textContent = h.subject || h.id;
    const s = document.createElement('span'); s.textContent = `${h.id} · ${h.quality || ''}${h.seconds ? ' · ' + Math.round(h.seconds) + ' s' : ''}`;
    txt.append(b, s);
    el.append(img, txt);
    el.addEventListener('click', () => loadHead(h));
    return el;
  }));
  const pick = heads.find((h) => h.id === selectId) || (!state.current && heads[0]);
  if (pick) loadHead(pick);
}

const skinRegions = Object.fromEntries(['scalp', 'forehead', 'nose', 'under_eye', 'cheeks', 'lips', 'chin', 'ears'].map(r => [r, {redness: 0, saturation: 0, lightness: 0}]));
$('skin-region').replaceChildren(...Object.keys(skinRegions).map(r => { const o = document.createElement('option'); o.value = r; o.textContent = r.replace('_', ' '); return o; }));
$('skin-region').value = 'cheeks';
$('skin-region').addEventListener('change', () => { for (const k of ['redness', 'saturation', 'lightness']) $('skin-' + k).value = skinRegions[$('skin-region').value][k]; });
for (const k of ['redness', 'saturation', 'lightness']) $('skin-' + k).addEventListener('input', () => { skinRegions[$('skin-region').value][k] = +$('skin-' + k).value; });
function skinParams() {
  return {enabled: $('skin-enabled').checked, tone_u: +$('skin-tone-u').value, tone_v: +$('skin-tone-v').value,
    tone_strength: +$('skin-tone-strength').value, delight_strength: +$('skin-delight').value,
    roughness: +$('skin-roughness').value, pore_strength: +$('skin-pores').value,
    seed: +$('seed').value, freckles: {density: +$('skin-freckles').value, strength: +$('skin-freckle-strength').value,
      saturation: +$('skin-freckle-saturation').value, tone_shift: +$('skin-freckle-tone').value, mask: $('skin-freckle-mask').value}, regions: skinRegions};
}
$('apply-skin').addEventListener('click', async () => {
  if (!state.current) return;
  $('apply-skin').disabled = true; $('go').disabled = true; $('msg').className = 'msg';
  try {
    const job = await api('/v1/jobs', {kind: 'head_skin', head_id: state.current.id, skin: skinParams()});
    state.job = job.id; poll(job.id);
  } catch (e) { $('msg').className = 'msg bad'; $('msg').textContent = e.message; $('apply-skin').disabled = false; $('go').disabled = false; }
});
$('go').addEventListener('click', async () => {
  $('go').disabled = true; $('apply-skin').disabled = true;
  $('msg').className = 'msg';
  try {
    const job = await api('/v1/jobs', { kind: 'head', subject: $('subject').value, seed: +$('seed').value,
                                        quality: $('quality').value, fov: +$('fov').value, iris_source: $('iris-source').value, qwen_preset: $('qwen-preset').value, skin: skinParams() });
    state.job = job.id;
    poll(job.id);
  } catch (e) { $('msg').className = 'msg bad'; $('msg').textContent = e.message; $('go').disabled = false; $('apply-skin').disabled = !state.current; }
});
async function poll(id) {
  const j = await api(`/v1/jobs/${id}`);
  $('bar').style.width = `${Math.round((j.progress || 0) * 100)}%`;
  $('msg').textContent = j.state === 'failed' ? j.error : `${j.state} · ${j.message || ''}`;
  if (j.state === 'failed') $('msg').className = 'msg bad';
  if (['done', 'failed', 'cancelled'].includes(j.state)) {
    $('go').disabled = false; $('apply-skin').disabled = !state.current;
    if (j.state === 'done') refreshHeads(j.result.id);
    return;
  }
  setTimeout(() => poll(id), 1500);
}

let videoAvailable = false, videoJob = null;
async function refreshVideos(headId) {
  const result = await api(`/v1/heads/${headId}/videos`);
  if (state.current?.id !== headId) return;
  $('video-clips').replaceChildren();
  for (const clip of result.videos) {
    const label = document.createElement('p');
    label.textContent = `${clip.request.expression || 'video'} · ${clip.backend}`;
    const player = document.createElement('video');
    player.controls = true; player.playsInline = true; player.preload = 'metadata';
    player.src = clip.url; player.poster = clip.poster; player.style.width = '100%';
    $('video-clips').append(label, player);
  }
}
async function pollVideo(id, headId) {
  if (videoJob !== id) return;
  try {
    const job = await api(`/v1/jobs/${id}`);
    $('video-msg').textContent = job.state === 'failed' ? job.error : `${job.state} · ${job.message}`;
    if (['done', 'failed', 'cancelled'].includes(job.state)) {
      videoJob = null; $('video-cancel').disabled = true;
      $('video-go').disabled = !videoAvailable || !state.current;
      if (job.state === 'done') {
        try { await refreshVideos(headId); }
        catch (error) { $('video-msg').textContent = error.message; }
      }
      return;
    }
    setTimeout(() => pollVideo(id, headId), 1500);
  } catch (error) {
    $('video-msg').textContent = error.message;
    setTimeout(() => pollVideo(id, headId), 3000);
  }
}
$('video-go').addEventListener('click', async () => {
  if (!state.current || videoJob) return;
  const headId = state.current.id;
  videoJob = 'submitting';
  $('video-go').disabled = true;
  try {
    const job = await api('/v1/jobs', {kind: 'video_generate',
      head_id: headId, expression: $('video-expression').value, preset: $('video-preset').value,
      frames: Number($('video-frames').value), seed: Number($('seed').value), prompt: $('video-prompt').value});
    videoJob = job.id; $('video-cancel').disabled = false;
    pollVideo(job.id, headId);
  } catch (error) {
    videoJob = null;
    $('video-msg').textContent = error.message;
    $('video-go').disabled = !videoAvailable || !state.current;
  }
});
$('video-cancel').addEventListener('click', async () => {
  if (videoJob) {
    try { await api(`/v1/jobs/${videoJob}/cancel`, {}); }
    catch (error) { $('video-msg').textContent = error.message; }
  }
});

api('/health').then((h) => {
  videoAvailable = !!h.video?.available;
  const previousPreset = $('video-preset').value, previousFrames = Number($('video-frames').value);
  $('video-preset').replaceChildren(...(h.video?.presets || []).map(preset =>
    new Option(({quality: 'Quality', fast12: 'Fast · 12 steps', fast5: 'Fast · 5 steps'})[preset] || preset, preset)));
  if (h.video?.presets?.includes(previousPreset)) $('video-preset').value = previousPreset;
  $('video-frames').replaceChildren(...(h.video?.frames || []).map(frames =>
    new Option(`${frames} frames · ${(frames / (h.video?.fps || 24)).toFixed(2)} seconds`, String(frames))));
  $('video-frames').value = String(h.video?.frames?.includes(previousFrames) ? previousFrames : h.video?.default_frames);
  $('video-go').disabled = !videoAvailable || !state.current;
  if (!videoAvailable) $('video-msg').textContent = 'Video backend needs model setup and experimental enablement.';
  else if (h.video.mock) $('video-msg').textContent = 'Mock mode: static portrait clip.';
  else if (h.video.identity_conditioned === false) $('video-msg').textContent = 'H3 generates from text. The portrait does not condition the video; identity needs review.';
  else $('video-msg').textContent = 'Experimental video backend; quality and memory fit are unverified.';
  const g = h.gpu ? `${h.gpu.name} · ${h.gpu.free_mib} MiB free` : 'no GPU';
  $('health').textContent = `${g} · Qwen ${h.qwen.available ? 'ready' : 'n/a'} · Pixal3D ${h.pixal3d.available ? 'ready' : 'n/a'}`;
}).catch(() => {});
resize();
refreshHeads();
window.addEventListener('vhuman-head-created', event => refreshHeads(event.detail));

}
