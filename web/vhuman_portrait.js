// Portrait review is independent of the optional Three.js head viewer.
const $ = id => document.getElementById(id);
const angles = [[45, 'Left three-quarter'], [-45, 'Right three-quarter'], [90, 'Left profile'], [-90, 'Right profile']];
const state = {session: null, sessions: [], job: null, starting: false, saving: false, health: null, candidates: {}, model: 'qwen-image21', steps: {'qwen-image21': 24, klein3: 16}};
const pendingKey = 'vhuman.portrait.pending.v1', sessionKey = 'vhuman.portrait.session.v1';
const busy = () => !!(state.job || state.starting || state.saving);
const el = (tag, text, cls) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) {
    const error = Error(result.error || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return result;
}
function message(text, bad = false) {
  $('portrait-message').textContent = text;
  $('portrait-message').className = bad ? 'msg bad' : 'msg';
}
function remember(key, value) {
  try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* private browsing */ }
}
function recall(key) { try { return localStorage.getItem(key); } catch { return null; } }
function generator(views = false) {
  const h = state.health;
  const model = views ? 'qwen-image21' : $('portrait-model').value;
  return h?.portrait_generators?.[model] || (model === 'qwen-image21' ? h?.qwen : null);
}
function minimumMemory(views = false) {
  return (views || $('portrait-model').value === 'qwen-image21') ?
    ($('portrait-memory').value === 'low8' ? 7680 : 12288) : (generator()?.min_free_mib || 9216);
}
function generationReady(views = false) {
  const g = generator(views);
  return !!g?.available && (g.backend === 'mock' || (state.health?.gpu?.free_mib || 0) >= minimumMemory(views));
}
function restoreParameters(p) {
  for (const row of state.session?.portraits || []) state.steps[row.portrait_model || 'qwen-image21'] = row.steps;
  const model = p?.portrait_model || 'qwen-image21';
  $('portrait-model').value = model;
  state.model = model;
  $('portrait-seed').value = p?.seed ?? 11;
  $('portrait-steps').value = p?.steps ?? state.steps[model];
  if (p?.preset === 'low8' || p?.preset === 'fast12') $('portrait-memory').value = p.preset;
}
function compare(url, label) {
  const front = state.session?.portraits.find(p => p.id === state.session.active_revision);
  $('portrait-compare-front').src = front?.url || url;
  $('portrait-compare-view').src = url;
  $('portrait-compare-label').textContent = label;
  $('portrait-lightbox').showModal();
}
function parameters(views = false) {
  const stepId = views ? 'portrait-view-steps' : 'portrait-steps';
  for (const id of ['portrait-subject', 'portrait-seed', stepId]) {
    if (!$(id).reportValidity()) throw Error('Check the portrait settings.');
  }
  const result = {seed: Number($('portrait-seed').value), steps: Number($(stepId).value)};
  if (views || $('portrait-model').value === 'qwen-image21') result.qwen_preset = $('portrait-memory').value;
  if (!views) result.portrait_model = $('portrait-model').value;
  return result;
}
function controls() {
  const doc = state.session, approved = !!doc?.selection.portrait_reviewed, locked = busy();
  $('portrait-generate').disabled = locked || !generationReady();
  $('portrait-generate-views').disabled = locked || !approved || !generationReady(true);
  $('portrait-approved').disabled = locked || !doc;
  $('portrait-reconstruct-front').disabled = locked || !approved;
  $('portrait-reconstruct-views').disabled = locked || !approved || !doc?.selection.view_ids.length;
  $('portrait-new').disabled = locked;
  $('portrait-sessions').disabled = locked;
  $('portrait-memory').disabled = locked;
  $('portrait-model').disabled = locked;
  $('portrait-view-steps').disabled = locked;
  for (const node of $('portrait-views').querySelectorAll('input,select,button')) {
    node.disabled = locked || !approved || (node.tagName === 'BUTTON' && !generationReady(true));
  }
  $('portrait-cancel').hidden = !state.job;
  for (let i = 1; i <= 3; i++) $('portrait-step-'+i).classList.toggle('active', i === (!approved ? 1 : doc.selection.view_ids.length ? 3 : 2));
}
function render() {
  const doc = state.session;
  $('portrait-sessions').replaceChildren(new Option('New portrait session', ''), ...state.sessions.map(d => new Option(d.subject, d.id)));
  $('portrait-sessions').value = doc?.id || '';
  $('portrait-generate').textContent = doc ? 'Regenerate front portrait' : 'Generate portrait';
  $('portrait-approved').checked = !!doc?.selection.portrait_reviewed;
  const front = doc?.portraits.find(p => p.id === doc.active_revision);
  $('portrait-image').hidden = !front;
  $('portrait-empty').hidden = !!front;
  $('portrait-caption').hidden = !front;
  if (front) {
    $('portrait-image').src = front.url;
    $('portrait-caption').textContent = `${front.model || 'Qwen-Image 2.1'} · ${front.size.join(' × ')} · seed ${front.seed}`;
    $('portrait-image').onclick = () => compare(front.url, 'Current front portrait');
  } else $('portrait-image').removeAttribute('src');
  $('portrait-lineage').textContent = doc ? `${doc.portraits.length} portrait revision${doc.portraits.length === 1 ? '' : 's'} · ${doc.view_runs.filter(r => !r.stale).length} current view runs` : 'Your portrait and views will appear here.';
  $('portrait-bundle').hidden = !doc;
  if (doc) $('portrait-bundle').href = doc.bundle_url;
  $('portrait-history').replaceChildren(...(doc?.portraits || []).slice().reverse().map(p => {
    const button = el('button');
    button.title = `${p.model || 'Qwen-Image 2.1'} · seed ${p.seed} · click to inspect`;
    const img = el('img'); img.src = p.url; img.alt = `${p.model || 'Qwen-Image 2.1'} portrait seed ${p.seed}`;
    button.append(img); button.onclick = () => compare(p.url, `Portrait history · seed ${p.seed}`);
    return button;
  }));
  const currentRuns = doc?.view_runs.filter(r => !r.stale) || [];
  $('portrait-views').replaceChildren(...angles.map(([angle, label]) => {
    const versions = currentRuns.flatMap(r => r.views.filter(v => v.yaw_deg === angle).map(v => ({...v, method: r.method})));
    const chosen = versions.find(v => v.id === state.candidates[angle]) || versions.find(v => doc.selection.view_ids.includes(v.id)) || versions.at(-1);
    if (chosen) state.candidates[angle] = chosen.id;
    const approved = chosen && doc.selection.view_ids.includes(chosen.id);
    const card = el('article', undefined, 'portrait-view'+(approved ? ' approved' : ''));
    card.dataset.angle = angle;
    if (chosen) {
      const img = el('img'); img.src = chosen.url; img.alt = label;
      img.onclick = () => compare(chosen.url, `${label} · ${chosen.size.join(' × ')} · requested angle ${angle}°`);
      card.append(img);
    } else card.append(el('div', 'No view yet', 'portrait-placeholder'));
    card.append(el('h4', label), el('span', `${angle > 0 ? '+' : ''}${angle}° requested`, 'msg'));
    if (chosen) {
      const select = el('select'); select.setAttribute('aria-label', `${label} version`);
      select.append(...versions.map(v => new Option(`${v.method === 'sheet' ? 'Joint sheet' : 'Individual'} · ${v.size[0]} px · seed ${v.seed}`, v.id)));
      select.value = chosen.id;
      select.onchange = async () => {
        state.candidates[angle] = select.value;
        const ids = doc.selection.view_ids.filter(id => !versions.some(v => v.id === id));
        await saveReview({view_ids: ids});
      };
      const check = el('input'); check.type = 'checkbox'; check.checked = !!approved;
      check.setAttribute('aria-label', `Approve ${label.toLowerCase()}`);
      check.onchange = () => saveReview({view_ids: check.checked ? [...doc.selection.view_ids.filter(id => !versions.some(v => v.id === id)), chosen.id] : doc.selection.view_ids.filter(id => id !== chosen.id)});
      const review = el('label', undefined, 'portrait-check'); review.append(check, el('span', 'Reviewed · use this view'));
      const download = el('a', 'Download PNG ↓'); download.href = chosen.url; download.download = `${angle}.png`;
      card.append(select, review, download);
    }
    const regenerate = el('button', chosen ? 'Regenerate this view' : 'Generate this view');
    regenerate.onclick = () => generateViews('individual', [angle]);
    card.append(regenerate);
    return card;
  }));
  $('portrait-sheets').replaceChildren(...currentRuns.filter(r => r.sheet_url).map((r, i) => {
    const button = el('button', `Inspect joint sheet ${i+1}`);
    button.onclick = () => compare(r.sheet_url, 'Joint sheet · four 512 × 512 panels');
    return button;
  }));
  const count = doc?.selection.view_ids.length || 0;
  $('portrait-selection').textContent = !doc?.selection.portrait_reviewed ? 'Approve the front portrait to continue.' : `Front portrait + ${count} approved side view${count === 1 ? '' : 's'}. Side cameras will be estimated; views that fail checks are reported.`;
  $('portrait-reconstruct-views').textContent = `Reconstruct ${count} selected view${count === 1 ? '' : 's'} + front`;
  $('portrait-results').replaceChildren(...(doc?.reconstructions || []).filter(r => r.revision === doc.active_revision).slice().reverse().map(r => {
    const box = el('div', undefined, 'portrait-result'), status = r.multiview?.status;
    box.append(el('p', status === 'accepted' ? 'Multiview candidate and front baseline ready for comparison.' : status === 'rejected' || status === 'failed' ? `Front baseline ready. Multiview ${status}: ${r.multiview.reason}` : 'Front reconstruction ready.'));
    if (r.baseline_viewer_url) {
      const link = el('a', 'Inspect front baseline →'); link.href = r.baseline_viewer_url; box.append(link);
    }
    if (r.viewer_url && (!r.baseline_viewer_url || status === 'accepted')) {
      const link = el('a', status === 'accepted' ? 'Inspect multiview candidate →' : 'Open face editor →'); link.href = r.viewer_url; box.append(link);
    }
    for (const v of r.multiview?.views || []) {
      box.append(el('p', `${v.requested_yaw_deg}°: ${v.accepted ? `camera estimated at ${(v.fitted_yaw_deg ?? v.estimated_yaw_deg).toFixed(1)}° · ${v.inliers} initial inliers` : v.reason}`, 'msg'));
    }
    return box;
  }));
  controls();
}
async function saveReview(changes) {
  if (busy() || !state.session) return;
  state.saving = true; controls();
  try {
    const d = state.session;
    state.session = await api(`/v1/portrait-sessions/${d.id}/review`, {...d.selection, ...changes});
    message('Review saved.');
  } catch (error) { message(error.message, true); }
  finally { state.saving = false; render(); }
}
async function refreshSessions(selectId = state.session?.id) {
  state.sessions = (await api('/v1/portrait-sessions')).sessions;
  state.session = state.sessions.find(d => d.id === selectId) || null;
  if (state.session) remember(sessionKey, state.session.id);
  render();
}
async function refreshHealth() {
  try {
    state.health = await api('/health');
    const h = state.health, g = generator(), mock = g?.backend === 'mock';
    $('health').textContent = mock ? 'Mock preview' : !g?.available ? 'Portrait generator needs setup' : generationReady() ? 'Portrait generator ready' : 'GPU busy';
    $('portrait-memory').querySelector('[value="fast12"]').disabled = !mock && (h.gpu?.free_mib || 0) < 12288;
    for (const option of $('portrait-model').options) {
      const ready = h.portrait_generators?.[option.value];
      option.disabled = !!ready && !ready.available;
    }
    $('portrait-gpu').textContent = mock ? 'Mock mode · synthetic test images' : !g?.available ? g?.reason || 'Portrait generator needs setup.' : `${h.gpu?.name || 'GPU'} · ${((h.gpu?.free_mib || 0)/1024).toFixed(1)} GiB free. ${generationReady() ? 'Ready to generate.' : 'Generation needs '+(minimumMemory()/1024).toFixed(1)+' GiB free. Free the GPU, then retry.'}`;
    const views = generator(true);
    $('portrait-views-status').textContent = !views?.available ? `Side views unavailable: ${views?.reason || 'Qwen needs setup.'}` : generationReady(true) ? 'Side-view generator ready.' : `Side views need ${(minimumMemory(true)/1024).toFixed(1)} GiB free.`;
    controls();
  } catch (error) { $('portrait-gpu').textContent = error.message; }
}
async function startJob(body) {
  if (busy()) return;
  state.starting = true; controls(); message('Submitting job…');
  try {
    const job = await api('/v1/jobs', body);
    state.job = job.id;
    remember(pendingKey, job.id);
    poll(job.id);
  } catch (error) { message(error.message, true); }
  finally { state.starting = false; controls(); }
}
async function poll(id) {
  try {
    const j = await api(`/v1/jobs/${id}`);
    const fraction = Math.round((j.progress || 0)*100);
    $('portrait-bar').style.width = fraction+'%';
    $('portrait-progress').setAttribute('aria-valuenow', fraction);
    message(j.state === 'failed' ? j.error : `${j.state} · ${j.message || ''}`, j.state === 'failed');
    if (['done', 'failed', 'cancelled'].includes(j.state)) {
      state.job = null; remember(pendingKey, null);
      if (j.state === 'done') {
        state.candidates = {};
        await refreshSessions(j.result.session_id || state.session?.id || j.request.session_id);
        if (j.kind === 'portrait_reconstruct') window.dispatchEvent(new CustomEvent('vhuman-head-created', {detail: j.result.id}));
      }
      await refreshHealth(); controls();
      return;
    }
  } catch (error) {
    if (error.status === 404) {
      state.job = null; remember(pendingKey, null); controls();
      message('This job has left the server history. Your saved portrait session is still available.', true);
      return;
    }
    message(`Connection interrupted: ${error.message}. Retrying…`, true);
  }
  if (state.job === id) setTimeout(() => poll(id), 1200);
}
async function generateViews(method, selectedAngles) {
  try {
    await startJob({kind: 'portrait_views', ...parameters(true), session_id: state.session.id,
      revision: state.session.active_revision, method, ...(selectedAngles ? {angles: selectedAngles} : {})});
  } catch (error) { message(error.message, true); }
}
async function reconstruct(multiview) {
  const d = state.session;
  await startJob({kind: 'portrait_reconstruct', session_id: d.id, revision: d.active_revision,
    subject: d.subject, view_ids: multiview ? d.selection.view_ids : [], face_model: 'gnm_v3',
    profile: 'full', res: Number($('portrait-fit-res').value), texture_res: Number($('portrait-texture-res').value),
    iterations: Number($('portrait-fit-iters').value), build_rig: true});
}
$('portrait-generate').onclick = async () => {
  try {
    await startJob({kind: 'portrait_generate', ...parameters(), subject: $('portrait-subject').value,
      ...(state.session ? {session_id: state.session.id} : {})});
  } catch (error) { message(error.message, true); }
};
$('portrait-approved').onchange = () => saveReview({portrait_reviewed: $('portrait-approved').checked});
$('portrait-generate-views').onclick = () => generateViews(document.querySelector('[name="portrait-method"]:checked').value);
$('portrait-reconstruct-front').onclick = () => reconstruct(false);
$('portrait-reconstruct-views').onclick = () => reconstruct(true);
$('portrait-refresh').onclick = refreshHealth;
$('portrait-memory').onchange = refreshHealth;
$('portrait-model').onchange = () => {
  state.steps[state.model] = Number($('portrait-steps').value);
  state.model = $('portrait-model').value;
  $('portrait-steps').value = state.steps[state.model];
  refreshHealth();
};
$('portrait-cancel').onclick = async () => {
  try { if (state.job) await api(`/v1/jobs/${state.job}/cancel`, {}); }
  catch (error) { message(error.message, true); }
};
$('portrait-new').onclick = () => {
  state.session = null; state.candidates = {}; restoreParameters(null); remember(sessionKey, null); render(); message('Ready for a new face.');
};
$('portrait-sessions').onchange = async () => {
  state.candidates = {};
  await refreshSessions($('portrait-sessions').value);
  const d = state.session, p = d?.portraits.find(p => p.id === d.active_revision);
  if (d) $('portrait-subject').value = d.subject;
  restoreParameters(p);
  await refreshHealth();
};
$('portrait-close-compare').onclick = () => $('portrait-lightbox').close();
render();
Promise.all([refreshSessions(recall(sessionKey)), refreshHealth()]).then(async () => {
  const d = state.session;
  if (d) {
    $('portrait-subject').value = d.subject;
    const p = d.portraits.find(p => p.id === d.active_revision);
    restoreParameters(p);
    await refreshHealth();
  }
  const pending = recall(pendingKey);
  if (pending) { state.job = pending; controls(); poll(pending); }
  window.__portraitReady = true;
}).catch(error => message(error.message, true));
