import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';

const width = 1280, height = 720;
const renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: 'high-performance'});
renderer.setPixelRatio(1);
renderer.setSize(width, height);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);
const gl = renderer.getContext();
const debug = gl.getExtension('WEBGL_debug_renderer_info');
const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x242830);
const pmrem = new THREE.PMREMGenerator(renderer);
const room = new RoomEnvironment();
const environment = pmrem.fromScene(room, .04);
room.dispose();
pmrem.dispose();
scene.environment = environment.texture;
scene.add(new THREE.HemisphereLight(0xffffff, 0x303040, 1));
const key = new THREE.DirectionalLight(0xffffff, 3);
key.position.set(1, 2, 3);
scene.add(key);
const camera = new THREE.PerspectiveCamera(35, width/height, .001, 100);
let model, mixer, clip, turntable = false, lastTime, active, queries = [], frame = 0;
const errors = [], results = [];
window.addEventListener('error', e => errors.push(String(e.error || e.message)));
window.addEventListener('unhandledrejection', e => errors.push(String(e.reason)));
renderer.domElement.addEventListener('webglcontextlost', e => {
    e.preventDefault(); errors.push(`WebGL context lost at ${performance.now().toFixed(1)} ms: ${e.statusMessage}`);
});
document.addEventListener('visibilitychange', () => {
    if (active && document.hidden) errors.push('Page hidden during measurement');
});

function stats(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a-b);
    const quantile = p => {
        const at = (sorted.length-1)*p, low = Math.floor(at), f = at-low;
        return sorted[low]*(1-f) + sorted[Math.ceil(at)]*f;
    };
    return {samples: values.length, mean: values.reduce((a,b) => a+b, 0)/values.length,
        p50: quantile(.5), p95: quantile(.95), p99: quantile(.99), max: sorted.at(-1)};
}

function memory() {
    return {geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures,
        programs: renderer.info.programs.length, heap_bytes: performance.memory?.usedJSHeapSize ?? null};
}

function drainQueries() {
    if (!timer) return;
    if (gl.getParameter(timer.GPU_DISJOINT_EXT)) {
        for (const item of queries) gl.deleteQuery(item.query);
        queries = [];
        if (active) active.disjoint++;
        return;
    }
    while (queries.length && gl.getQueryParameter(queries[0].query, gl.QUERY_RESULT_AVAILABLE)) {
        const item = queries.shift();
        const ms = gl.getQueryParameter(item.query, gl.QUERY_RESULT)/1e6;
        gl.deleteQuery(item.query);
        item.measurement?.gpu_ms.push(ms);
    }
}

function unload() {
    if (!model) return;
    mixer?.stopAllAction();
    mixer?.uncacheRoot(model);
    scene.remove(model);
    const geometries = new Set(), materials = new Set(), textures = new Set(), skeletons = new Set();
    model.traverse(obj => {
        if (obj.geometry) geometries.add(obj.geometry);
        if (obj.skeleton) skeletons.add(obj.skeleton);
        for (const mat of (Array.isArray(obj.material) ? obj.material : [obj.material])) {
            if (!mat) continue;
            materials.add(mat);
            for (const value of Object.values(mat)) if (value?.isTexture) textures.add(value);
        }
    });
    for (const texture of textures) {texture.dispose(); texture.source?.data?.close?.();}
    for (const material of materials) material.dispose();
    for (const geometry of geometries) geometry.dispose();
    for (const skeleton of skeletons) skeleton.dispose();
    renderer.renderLists.dispose();
    model = mixer = clip = null;
    return memory();
}

async function load(url, rotate = false) {
    if (active) throw new Error('Cannot change assets during measurement');
    unload();
    const start = performance.now();
    const gltf = await new GLTFLoader().loadAsync(url);
    model = gltf.scene;
    scene.add(model);
    turntable = rotate;
    mixer = new THREE.AnimationMixer(model);
    clip = gltf.animations[0];
    if (clip) mixer.clipAction(clip).play();
    const bounds = new THREE.Box3();
    const skinned = [];
    model.traverse(obj => {if (obj.isSkinnedMesh) skinned.push(obj);});
    const framing = skinned.length ? skinned : [model];
    // Include the whole walking path in the fixed framing, rather than losing the avatar offscreen.
    // Use animated surfaces so a large static terrain does not dwarf the avatar.
    for (let i = 0; i < (clip ? 32 : 1); i++) {
        if (clip) mixer.setTime(clip.duration*i/32);
        model.updateMatrixWorld(true);
        for (const object of framing) bounds.union(new THREE.Box3().setFromObject(object, true));
    }
    mixer.setTime(0);
    model.updateMatrixWorld(true);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const extent = Math.max(size.y, size.x/camera.aspect);
    const distance = (extent/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2)))+size.z/2)*1.2;
    camera.position.copy(center).add(new THREE.Vector3(distance*.15, distance*.05, distance));
    camera.near = distance/100;
    camera.far = distance*20;
    camera.lookAt(center);
    camera.updateProjectionMatrix();
    await renderer.compileAsync(scene, camera);
    renderer.render(scene, camera);
    let vertices = 0, triangles = 0, meshes = 0, morphTargets = 0;
    model.traverse(obj => {
        if (!obj.isMesh) return;
        meshes++;
        vertices += obj.geometry.attributes.position.count;
        triangles += (obj.geometry.index?.count || obj.geometry.attributes.position.count)/3;
        morphTargets += obj.morphTargetInfluences?.length || 0;
    });
    return {load_compile_ms: performance.now()-start, context_lost: gl.isContextLost(),
        framing: skinned.length ? 'skinned surfaces over full clip, static terrain excluded' : 'whole static scene',
        clip: clip?.name ?? null,
        clip_duration: clip?.duration ?? null, vertices, triangles, meshes, morph_targets: morphTargets,
        camera: {position: camera.position.toArray(), center: center.toArray(), near: camera.near, far: camera.far},
        memory: memory()};
}

function begin(seconds) {
    if (active || !model) throw new Error('Model required and no active measurement');
    mixer.setTime(0);
    active = {start: performance.now(), seconds, previous: null, frames: 0,
        intervals_ms: [], cpu_submit_ms: [], gpu_ms: [], disjoint: 0, windows: [], memory_start: memory(),
        animation_time_range: [Infinity, -Infinity]};
}

function animate(now) {
    requestAnimationFrame(animate);
    const dt = lastTime === undefined ? 0 : (now-lastTime)/1000;
    lastTime = now;
    drainQueries();
    if (mixer) mixer.update(dt);
    if (model && turntable) model.rotation.y = Math.sin(now/5000)*.8;
    let query;
    // One GPU timer per 30 display frames, with asynchronous retrieval.
    if (active && timer && frame%30 === 0 && queries.length < 8) {
        query = gl.createQuery();
        gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
    }
    const start = performance.now();
    renderer.render(scene, camera);
    const cpu = performance.now()-start;
    if (query) {
        gl.endQuery(timer.TIME_ELAPSED_EXT);
        queries.push({query, measurement: active});
    }
    frame++;
    if (!active) return;
    const elapsed = performance.now()-active.start;
    if (active.previous !== null) active.intervals_ms.push(now-active.previous);
    active.previous = now;
    active.frames++;
    active.animation_time_range[0] = Math.min(active.animation_time_range[0], mixer.time);
    active.animation_time_range[1] = Math.max(active.animation_time_range[1], mixer.time);
    active.cpu_submit_ms.push(cpu);
    if (elapsed >= (active.windows.length+1)*5000) {
        active.windows.push({elapsed_ms: elapsed, frames: active.frames, memory: memory()});
    }
    if (elapsed >= active.seconds*1000) {
        // Pending queries are tagged with this object and continue draining after completion.
        active.elapsed_ms = elapsed;
        active.fps = (active.frames-1)*1000/active.intervals_ms.reduce((a,b) => a+b, 0);
        active.memory_end = memory();
        active.render_info = {calls: renderer.info.render.calls, triangles: renderer.info.render.triangles};
        results.push(active);
        active = null;
    }
}

window.bench = {load, begin, unload, memory, errors, results,
    get contextLost() {return gl.isContextLost();},
    get active() {return !!active;},
    summary() {
        return results.map(r => ({...r, frame_interval_ms: stats(r.intervals_ms),
            cpu_submit_ms: stats(r.cpu_submit_ms), gpu_ms: stats(r.gpu_ms),
            intervals_ms: r.intervals_ms}));
    },
    driver: {renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
        vendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
        version: gl.getParameter(gl.VERSION), three_revision: THREE.REVISION, width, height,
        pixel_ratio: renderer.getPixelRatio(), antialias: gl.getContextAttributes().antialias,
        gpu_timer: !!timer, scheduling: 'requestAnimationFrame, browser/display limited; no imposed FPS cap'}};
requestAnimationFrame(animate);
