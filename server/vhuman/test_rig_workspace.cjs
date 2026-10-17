/* Run actual page handlers with delayed requests and DOM/renderer doubles. */
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync('web/vhuman_rig.html', 'utf8');
const source = (start, end) => html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));
const deferred = () => {let resolve; const promise = new Promise(r=>{resolve=r;}); return {promise, resolve};};
function context(extra={}) {
  const nodes = new Map(), handlers = new Map();
  const $ = id => {
    if (!nodes.has(id)) nodes.set(id, {value:'',style:{},files:[],checked:false,
      replaceChildren(...items){this.items=items;},addEventListener(event,cb){handlers.set(`${id}:${event}`,cb);}});
    return nodes.get(id);
  };
  const c = vm.createContext({$,handlers,state:{speechAvailable:true,bodyAvailable:true},
    loadGen:0,selectionGen:0,speechLoadGen:0,emotionKeys:[],URLSearchParams,location:{search:''},
    Option:function(text,value){this.text=text;this.value=value;},stopAnim(){},renderHeads(){},showFacts(){},
    async loadTakes(){},async loadMotions(){},...extra});
  vm.runInContext(source('function acceptedTarget()', 'function renderEmotionKeys()'), c);
  return c;
}
async function headSelectionRace() {
  const pending={}, rigs=[], facts=[];
  const c=context({api:url=>{const d=deferred();pending[url]=d;return d.promise;},
    async loadRig(h){rigs.push(h.id);},showFacts(h){facts.push(h.id);}});
  vm.runInContext(source('async function select(', 'async function refresh('), c);
  const a=c.select({id:'A',rig:{}}), b=c.select({id:'B',rig:{}});
  pending['/v1/heads/B/reconstruction'].resolve({candidates:[{id:'B-run',rig:{}}]}); await b;
  pending['/v1/heads/A/reconstruction'].resolve({candidates:[{id:'A-run',rig:{}}]}); await a;
  assert.equal(c.state.current.id,'B'); assert.deepEqual(rigs,['B']); assert.deepEqual(facts,['B']);
  assert.equal(c.state.candidates[0].id,'B-run');
}
async function candidateActions() {
  const calls=[], refreshes=[];
  const c=context({setTimeout:fn=>fn(),async refresh(...args){refreshes.push(args);},
    async api(url,body){if(body){calls.push(body);return {id:'job'};}return {state:'done',progress:1,result:{id:'take'}};}});
  Object.assign(c.state,{current:{id:'head',rig:{},body:{}},candidate:'candidate',take:{},
    takes:[{urls:{soft_tissue_report:'report'}}],tissueAvailable:true});
  c.$('face-video').files=[{}]; c.$('speech-text').value='test'; c.$('res').value='1024';
  vm.runInContext(source('async function runSpeechJob(', "$('speech-generate').addEventListener")+
    source("$('build').addEventListener('click'", "$('avatar-mode').addEventListener"),c);
  c.updateTargetActions();
  for(const id of ['speech-generate','speech-rebuild','fit-face-video','train-soft','soft-tissue','build-body','avatar-mode'])
    assert.equal(c.$(id).disabled,true,id);
  await c.runSpeechJob(false); assert.equal(calls.length,0);
  await c.handlers.get('build:click')(); assert.equal(calls[0].reconstruction_run,'candidate');
  assert.deepEqual(refreshes[0],['head','candidate']);
  c.state.candidate=null; c.updateTargetActions(); assert.equal(c.$('speech-generate').disabled,false);
  await c.runSpeechJob(false); assert.equal(calls[1].kind,'rig_speech');
}
async function staleRigResource(stage) {
  const pending=deferred(),reached=deferred();
  const def=id=>({id,joints:[],controls:[],blendshapes:[],...(stage==='model'?{ml_deformer:{model:'ml'}}:{})});
  const c=context({scene:{add(){},remove(){}},viz:{labels:[]},skelHelper:null,
    THREE:{SkeletonHelper:function(){},DoubleSide:2},loader:{async loadAsync(){return {scene:{traverse(){}}};}},
    async api(url){return def(url.includes('/A/')?'A':'B');},
    async fetch(url){if(stage==='viz'&&url.includes('/A/')){reached.resolve();await pending.promise;}return {ok:true,json:async()=>({url})};},
    async loadSafetensors(url){if(stage==='model'&&url.includes('/A/')){reached.resolve();await pending.promise;}return {};},
    LinearRig:function(d){this.def=d;},MLDeformer:function(){},restMatrix3(){},buildControls(){},applyRig(){},
    setupContactOffsets(){},setupSoftOffsets(){},skinRenderer:{patch(){}},updateHeat(){},setView(){},updateTakeFrame(){},patchWrinkles(){}});
  vm.runInContext(source('async function loadRig(', 'function showFacts('),c);
  c.state.current={id:'A'}; c.state.candidate=null;
  const a=c.loadRig(c.state.current); await reached.promise;
  c.state.current={id:'B'}; await c.loadRig(c.state.current); pending.resolve(); await a;
  assert.equal(c.state.rig.def.id,'B',stage); assert.ok(c.state.vizData.url.includes('/B/'),stage);
}
(async()=>{
  const script=html.slice(html.indexOf('<script type="module">')+22,html.lastIndexOf('</script>'));
  new Function(script.replace(/^import .*$/gm,''));
  await headSelectionRace(); await candidateActions(); await staleRigResource('viz'); await staleRigResource('model');
  console.log('rig workspace: 4 regression scenarios passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
