/* Pair browsing, filters, scrubbing and trial workflow follow demo_dynamic. */
'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f4 = n => n == null ? '·' : Number(n).toFixed(4);
const api = async (url, options) => {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
};
const fail = error => { $('error').hidden = false; $('error-text').textContent = error.message || String(error); };
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
let CFG, PAIRS = [], TRIALS = [], ATTR = '', SEL = null, BUSY = false, WATCHING = false;
// The settings form as last left for each IntPhys2 condition, and the condition the form now holds.
const FORMS = new Map(); let FORM_ATTR = null;
let SORT = {key:'scene', direction:1}, SCRUBS = {}, EPOCH = 0;
const labels = {good:'Possible', bad:'Impossible'};
let GRAPH_RESULTS = {}, LATEST = new Map();

// The caption a window adopted: the first argmax of its scores, since the window's score is their max.
// A single caption is trivially adopted and is not marked.
function adoptedIndex(scores) {
  if(!(scores?.length>1))return -1;
  return scores.reduce((best,s,i)=>s!=null&&(scores[best]==null||s>scores[best])?i:best,0);
}
// Numbered captions; with scores, each caption carries its own CLIP cosine.
function captions(list, cls, scores) {
  if(!list?.length)return '';
  const adopted=adoptedIndex(scores);
  return `<ol class="captions ${cls}">${list.map((c,i)=>`<li${i===adopted?` class="adopted" title="Adopted: highest per-caption score"`:''}>${esc(c)}${scores?`<span class="meta">${f4(scores[i])}</span>`:''}</li>`).join('')}</ol>`;
}
// The segment table: one row per caption, with the window's own cells spanning them.
function segmentRows(r, list) {
  const n=Math.max(list.length,1), adopted=adoptedIndex(r.prediction_scores);
  // No window-score cell: it always equals the adopted (★) caption's score in the row below.
  const head=`<td rowspan="${n}">${r.start_sec.toFixed(2)}–${r.end_sec.toFixed(2)}</td><td rowspan="${n}" class="${r.trusted?'pos-val':'neg-val'}">${r.trusted?'✓':'✗'}</td>`;
  return Array.from({length:n},(_,i)=>{
    const cls=['segment-row',r.is_anomaly&&'anomaly',i===0&&'seg-start',i===n-1&&'seg-end',i===adopted&&'adopted'].filter(Boolean).join(' ');
    return `<tr data-segment="${r.index}" class="${cls}">${i===0?head:''}<td class="num caption-no">${list.length?(i===adopted?'★':'')+(i+1):''}</td><td class="num">${f4(r.prediction_scores?.[i])}</td><td class="caption-cell">${list.length?esc(list[i]):'—'}</td></tr>`;
  }).join('');
}
const GT_COLOR = '#ff2d2d';
// Only the Impossible side has a violation to mark; the GT comes from gt/intphys2_gt.csv.
function gtOf(side) { return side==='bad' ? PAIRS.find(p=>p.group===SEL)?.gt : null; }
// `spacer` keeps the same note invisible on the Possible side, so both charts start at the same height
// however many lines the Impossible note wraps to.
function gtNote(gt, spacer=false) {
  if(!gt)return '';
  const range=gt.end_sec>gt.onset_sec?`${gt.onset_sec}–${gt.end_sec}s`:`${gt.onset_sec}s`;
  return `<p class="meta"${spacer?' style="visibility:hidden" aria-hidden="true"':''}><b style="color:${GT_COLOR}">GT ${range}</b> · frames ${gt.onset_frame}–${gt.end_frame} · ${gt.confirmed?'user-confirmed':'unconfirmed candidate'}${gt.description?' · '+esc(gt.description):''}</p>`;
}
// Raw text of every attempt; the reasoning survives only here.
function rawOutputs(r) {
  const raws=r.raw_outputs;
  if(!raws?.length)return '';
  const think=raws[raws.length-1].includes('</think>');
  return `<details><summary>MLLM raw output · ${raws.length} attempt(s) · ${think?'&lt;think&gt; present':'&lt;think&gt; absent'}</summary>${raws.map((raw,i)=>`<p class="hint">attempt ${i+1}${i===raws.length-1?' (parsed)':' (format error → asked again)'}</p><pre class="input-prompt">${esc(raw)}</pre>`).join('')}</details>`;
}
// The prompt of every attempt as the chat template rendered it, after the run's one system turn
// (`head`); runs before 09-30 kept only the request.
function modelInputs(r,head) {
  const inputs=r.model_inputs;
  if(!inputs?.length)return `<details><summary>MLLM input prompt of this window</summary><pre class="input-prompt">${esc(r.input_prompt)}</pre>
    <p class="hint">The system prompt, the trusted window history and the current window video are passed along with this input. A correction request may be added after a format error.</p></details>`;
  const frames=r.sampled_frames?.mllm?.length;
  return `<details><summary>MLLM actual input of this window · ${inputs.length} attempt(s)</summary>
    <p class="hint">Text passed to the engine as is, after applying the chat template. &lt;|video_pad|&gt; is replaced by this window's video${frames?` (${frames} frames)`:''}.</p>
    ${inputs.map((input,i)=>`${inputs.length>1?`<p class="hint">attempt ${i+1}${i?' (correction request added)':''}</p>`:''}<pre class="input-prompt">${esc(head+input)}</pre>`).join('')}</details>`;
}
// Python's round(), which PhysTrace's frames_for uses: halves go to the even neighbour.
function roundHalfEven(value) {
  const floor=Math.floor(value), diff=value-floor;
  if(Math.abs(diff-0.5)>1e-9)return Math.round(value);
  return floor%2===0?floor:floor+1;
}
// What an empty "MLLM input frames" becomes on the server (VideoPredictor.frames_per_window).
function autoMllmFrames() {
  if(CFG.mllm_frames)return CFG.mllm_frames;
  const frames=roundHalfEven(Number($('segment-seconds').value)*CFG.mllm_frames_per_second/2)*2;
  return frames>=2?frames:null;
}
function mllmFrames() { const value=$('mllm-frames').value; return value===''?null:Number(value); }
function syncMllmFrames() {
  const auto=autoMllmFrames();
  $('mllm-frames').placeholder=auto?`auto ${auto}`:'auto';
  $('mllm-frames-auto').textContent=auto?`(${auto} if empty)`:'';
}
function clipFrames() { return Number($('clip-frames').value); }
function setClipFrames(count) {
  // Older trials kept the frame count only in the result's model block; 0 is PE-AV's every frame.
  if(count!=null)$('clip-frames').value=count;
}
const clipFramesLabel = count => count===0 ? 'all' : (count ?? '–');
// score/batch_infer stores the CLIP model in a trial's params; a run started here does not.
const isBatch = trial => trial.params?.clip_model != null;
// The server's range (demo.app clip_frames_range): PE-AV 0 = every frame, InternVideo2 2..32.
function clipFramesRange() { return CFG.clip_frames_range ?? (CFG.clip_reads_all_frames ? [0, 32] : [2, 32]); }
// Frames in the longest window of one video, by split_windows' own rounding (PhysTrace/video_windows.py).
function longestWindow(info, seconds, hop) {
  let longest=0;
  for(let index=0, end=0; end<info.n_frames; index++){
    const start=Math.floor(index*hop*info.fps+0.5);
    end=Math.min(info.n_frames, Math.floor((index*hop+seconds)*info.fps+0.5));
    longest=Math.max(longest, end-start);
  }
  return longest;
}
// PE-AV reads a window whole once the count reaches its frames (pe_av.sample_indices), so the useful
// maximum is the selected pair's fps × window; a larger value is pulled down to it.
function syncClipFrames() {
  if(!CFG.clip_reads_all_frames)return null;
  const [fewest,most]=clipFramesRange();
  const seconds=Number($('segment-seconds').value), hop=Number($('hop-seconds').value);
  const infos=Object.values(SCRUBS).filter(s=>s.fps);
  const usable=infos.length&&seconds>0&&hop>0&&hop<=seconds;
  const frames=usable?Math.max(...infos.map(info=>longestWindow(info,seconds,hop))):null;
  const limit=frames?Math.min(most,frames):most;
  $('clip-frames').max=limit;
  const fps=infos.length?[...new Set(infos.map(info=>info.fps.toFixed(0)))].join('/'):null;
  // A window longer than the video is cut at its end, so fps × seconds can overstate it.
  const nominal=infos.length?Math.max(...infos.map(info=>Math.floor(seconds*info.fps+0.5))):0;
  const basis=frames&&frames<nominal?`up to the video length: ${frames}`:`${fps}fps × ${seconds}s = ${frames}`;
  $('clip-frames-range').textContent=frames
    ? `(${fewest}~${limit}, 0 = every frame · ${basis})`
    : `(${fewest}~${most}, 0 = every frame)`;
  const value=Number($('clip-frames').value);
  if(frames&&Number.isInteger(value)&&value>limit){
    $('clip-frames').value=limit;
    return `CLIP input frames ${value} → ${limit}: it exceeds the frames of one window (${basis}), so every frame is used.`;
  }
  return null;
}
function latest(group) { return LATEST.get(group); }
function attributePairs() { return PAIRS.filter(p => !ATTR || p.attribute === ATTR); }
function visiblePairs() {
  const text = $('f-text').value.toLowerCase().trim();
  const filters = [...document.querySelectorAll('.f-meta')].filter(s => s.value);
  return attributePairs().filter(p =>
    filters.every(f => p[f.dataset.col] === f.value) &&
    (!text || [p.scene,p.category,p.good_name,p.bad_name,p.group].join(' ').toLowerCase().includes(text))
  ).sort((a,b) => String(a[SORT.key]).localeCompare(String(b[SORT.key]), undefined, {numeric:true}) * SORT.direction);
}
// The tabs wrap (flex-wrap), so the last item takes the leftover row alone and looks the biggest.
// 'All' takes that place by moving to the end, and the attribute that was last moves to the first slot.
function attributeTabs() {
  const attrs = [...CFG.attributes];
  return attrs.length ? [attrs.pop(), ...attrs, ''] : [''];
}
// A record holds the checkpoint's full path, but the folder name is enough for the table.
function modelName(models) {
  const path = models && models.mllm;
  return path ? String(path).replace(/\/+$/, '').split('/').pop() : 'earlier record';
}
function buildFilters() {
  $('attr-tabs').innerHTML = attributeTabs().map(a => `<button class="tab${a === ATTR ? ' on' : ''}" data-attr="${esc(a)}">${esc(a || 'All')}<span>${PAIRS.filter(p=>!a || p.attribute===a).length} pairs</span></button>`).join('');
  $('attr-tabs').querySelectorAll('button').forEach(b => b.onclick = () => { ATTR=b.dataset.attr; buildFilters(); renderList(); });
  $('filters').innerHTML = CFG.filter_cols.map(col => `<label>${esc(col)}<select class="f-meta" data-col="${esc(col)}"><option value="">All</option>${[...new Set(attributePairs().map(p=>p[col]))].sort().map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('')}</select></label>`).join('');
  document.querySelectorAll('.f-meta').forEach(s=>s.onchange=renderList);
}
function renderList() {
  const pairs=visiblePairs();
  $('list-count').textContent=`${pairs.length} pairs / ${PAIRS.length} total`;
  $('pair-head').innerHTML=CFG.meta_cols.map(k=>`<th data-key="${esc(k)}">${esc(k)}${SORT.key===k ? (SORT.direction===1?' ▲':' ▼'):''}</th>`).join('')+'<th>Lowest CLIP P</th><th>Lowest CLIP I</th>';
  $('pair-head').querySelectorAll('[data-key]').forEach(th=>th.onclick=()=>{ SORT={key:th.dataset.key,direction:SORT.key===th.dataset.key?-SORT.direction:1};renderList(); });
  $('pair-table').querySelector('tbody').innerHTML=pairs.map(p=>{
    const s=latest(p.group)?.summary;
    return `<tr tabindex="0" class="pair-row${SEL===p.group?' selected':''}" data-group="${esc(p.group)}">${CFG.meta_cols.map(k=>`<td>${esc(p[k])}</td>`).join('')}<td class="num pos-val">${f4(s?.good?.min_score)}</td><td class="num neg-val">${f4(s?.bad?.min_score)}</td></tr>`;
  }).join('');
  $('pair-table').querySelectorAll('[data-group]').forEach(tr=>{
    tr.onclick=()=>selectPair(tr.dataset.group).catch(fail);
    tr.onkeydown=e=>{if(e.key==='Enter')tr.click();};
  });
}
function renderTrials() {
  $('trials-table').querySelector('tbody').innerHTML=TRIALS.filter(t=>t.group===SEL).map(t=>
    `<tr class="trial-row" data-id="${esc(t.id)}"><td class="check"><input type="checkbox" class="trial-check" value="${esc(t.id)}" aria-label="Select trial to delete"></td><td>${esc(new Date(t.created_at).toLocaleString())}${isBatch(t)?' <span class="meta">(batch)</span>':''}</td><td>${esc(modelName(t.models))}</td><td>${esc(t.models?.clip ?? '–')}</td><td>${t.params.window_seconds}</td><td>${t.params.hop_seconds}</td><td class="num">${clipFramesLabel(t.params.clip_frames ?? t.models?.clip_frames)}</td>${['good','bad'].map(side=>{const s=t.summary;return `<td>${s==null?'…':s[side]?f4(s[side].min_score):'–'}</td>`;}).join('')}</tr>`).join('');
  $('trials-table').querySelectorAll('[data-id]').forEach(tr=>tr.onclick=()=>loadTrial(tr.dataset.id).catch(fail));
  // A checkbox only marks the row for deletion; it must not also load the trial.
  $('trials-table').querySelectorAll('.trial-check').forEach(box=>{box.onclick=e=>e.stopPropagation();box.onchange=syncTrialChecks;});
  syncTrialChecks();
}
function checkedTrials() { return [...document.querySelectorAll('.trial-check:checked')].map(box=>box.value); }
function syncTrialChecks() {
  const boxes=document.querySelectorAll('.trial-check'), checked=checkedTrials().length;
  $('trials-check-all').checked=boxes.length>0&&checked===boxes.length;
  $('trials-check-all').indeterminate=checked>0&&checked<boxes.length;
  $('delete-trials').disabled=!checked;
  $('delete-trials').textContent=checked?`Delete selected (${checked})`:'Delete selected';
}
async function deleteTrials() {
  const ids=checkedTrials();
  if(!ids.length||!confirm(`Delete ${ids.length} selected trial(s)?\nThey are also removed from the stored CSVs (trials, results) and cannot be restored.`))return;
  await api('/api/delete_trials',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids})});
  await refreshTrials();
}
async function refreshTrials() {
  TRIALS=(await api('/api/trials')).trials;
  LATEST=new Map();
  for(const t of TRIALS)if(!LATEST.has(t.group))LATEST.set(t.group,t);  // /api/trials is newest first
  renderList(); renderTrials();
}
function promptValues() {
  return {system_prompt:$('system-prompt').value,segment_template:$('segment-template').value,
    num_prompts:Number($('num-prompts').value),history_windows:Number($('history-windows').value),
    lowest_windows:Number($('lowest-windows').value),
    max_new_tokens:Number($('max-new-tokens').value),thinking_budget:Number($('thinking-budget').value),
    mllm_frames:mllmFrames()};
}
// Each IntPhys2 condition has its own default wording (server: prompts_for).
function defaultPrompts(group) { return CFG.default_prompts[PAIRS.find(p=>p.group===group).attribute]; }
function setPrompts(prompts) {
  $('system-prompt').value=prompts.system_prompt;
  $('segment-template').value=prompts.segment_template;
  $('num-prompts').value=prompts.num_prompts ?? 1;
  $('history-windows').value=prompts.history_windows ?? 0;
  $('lowest-windows').value=prompts.lowest_windows ?? 1;
  // Stored trials from before 09-29 carry no limits; the server's own defaults stand in.
  $('max-new-tokens').value=prompts.max_new_tokens ?? CFG.defaults.max_new_tokens;
  $('thinking-budget').value=prompts.thinking_budget ?? CFG.defaults.thinking_budget;
  $('mllm-frames').value=prompts.mllm_frames ?? '';
}
function formValues() {
  return {prompts:promptValues(), window:$('segment-seconds').value, hop:$('hop-seconds').value, clip:$('clip-frames').value};
}
function setForm(form) {
  setPrompts(form.prompts);
  $('segment-seconds').value=form.window; $('hop-seconds').value=form.hop; $('clip-frames').value=form.clip;
  syncMllmFrames();
}
// Pairs of one condition share the form: moving between them keeps edits and loaded trials. Moving to
// another condition saves the form under the old one and brings back the new one's, or its defaults the
// first time; the reset button brings the defaults back at any time.
function switchForm(group) {
  const attribute=PAIRS.find(p=>p.group===group).attribute;
  if(attribute===FORM_ATTR)return;
  if(FORM_ATTR!==null)FORMS.set(FORM_ATTR,formValues());
  if(FORMS.has(attribute))setForm(FORMS.get(attribute));
  else setDefaults(group);
  FORM_ATTR=attribute;
}
function setDefaults(group) {
  setPrompts(defaultPrompts(group));
  $('segment-seconds').value=CFG.defaults.segment_seconds;
  $('hop-seconds').value=CFG.defaults.hop_seconds;
  setClipFrames(CFG.clip_frames);
  syncMllmFrames();
}
function runSides() { return $('run-sides').value.split(','); }
function syncRun() {
  const sides=runSides();
  $('run-btn').disabled=BUSY || !SEL;
  $('run-btn').textContent=sides.length===2?'Analyze pair':`Analyze ${labels[sides[0]]} only`;
  const partial=sides.length===1, skipped=partial?labels[sides[0]==='good'?'bad':'good']:'';
  document.querySelector('.run-target').classList.toggle('partial',partial);
  $('run-target-note').classList.toggle('partial',partial);
  $('run-target-note').textContent=partial
    ? `${skipped} is not run and is recorded as not run. ${skipped==='Possible'?'Possible is the false-positive control, so run both sides at least once before fixing the prompt.':''}`
    : 'Both videos are analyzed. Running one side takes half the time.';
  for(const id of ['system-prompt','segment-template','reset-prompts','run-sides',
                   'num-prompts','history-windows','lowest-windows','max-new-tokens','thinking-budget','clip-frames','mllm-frames'])$(id).disabled=BUSY;
}
function scrub(side, index) {
  const s=SCRUBS[side]; if(!s)return;
  index=Math.max(0,Math.min(s.n_frames-1,Math.round(index)));
  s.index=index;
  $(`range-${side}`).value=index;
  $(`frame-${side}`).src=`/api/frame?name=${encodeURIComponent(s.name)}&i=${index}`;
  $(`caption-${side}`).innerHTML=`<b>f${index}</b> / ${s.n_frames-1} · ${(index/s.fps).toFixed(2)}s`;
}
function seek(side, seconds) {
  const s=SCRUBS[side];if(!s)return;
  scrub(side,seconds*s.fps);
}
async function selectPair(group) {
  const epoch=++EPOCH;
  SEL=group; SCRUBS={}; GRAPH_RESULTS={};
  const pair=PAIRS.find(p=>p.group===group);
  $('empty').hidden=true; $('detail').hidden=false; $('error').hidden=true;
  $('pair-title').textContent=group;
  $('pair-trial-meta').textContent='';
  $('pair-chips').innerHTML=['attribute',...CFG.meta_cols].map(k=>`<span class="chip"><span class="chip-k">${esc(k)}</span>${esc(pair[k])}</span>`).join('');
  // Match demo_dynamic: both pane headers first, then the two frame cells.
  $('scrub-panes').innerHTML=['good','bad'].map(side=>`
    <div class="pane-head" data-pane="${side}">
      <span class="${side==='good'?'k-pos':'k-imp'}">${labels[side].toUpperCase()}</span>
      <span class="pane-name" title="${esc(pair[side+'_name'])}">${esc(pair[side+'_name'])}</span>
    </div>`).join('') + ['good','bad'].map(side=>`
    <div class="cell"><div class="scrub" id="scrub-${side}" tabindex="0">
      <img id="frame-${side}" class="scrub-img" alt="${labels[side]} frame">
      <div class="scrub-ctl">
        <button class="step" data-pane="${side}" data-step="-10" type="button">◀◀</button>
        <button class="step" id="prev-${side}" data-pane="${side}" data-step="-1" type="button">◀</button>
        <div class="scrub-track"><input aria-label="${labels[side]} frame" id="range-${side}" type="range" min="0" max="0" value="0" disabled></div>
        <button class="step" id="next-${side}" data-pane="${side}" data-step="1" type="button">▶</button>
        <button class="step" data-pane="${side}" data-step="10" type="button">▶▶</button>
      </div>
      <div class="scrub-foot"><span class="scrub-cap" id="caption-${side}"></span>
        <span class="hint scrub-state" id="scrub-state-${side}">Preparing frames…</span></div>
    </div></div>`).join('');
  $('panes').innerHTML=['good','bad'].map(side=>`<div class="card result-panel" id="result-${side}"><h2>${labels[side]}</h2><p class="meta">Run an analysis to see the per-window predictions and scores. Click a row under Trials below to load an earlier result.</p></div>`).join('');
  switchForm(group);
  renderList();renderTrials();syncRun();
  await Promise.all(['good','bad'].map(async side=>{
    const name=pair[side+'_name'];
    const info=await api('/api/video_info?name='+encodeURIComponent(name));
    if(epoch!==EPOCH)return;
    SCRUBS[side]={name,...info,index:0};
    const range=$(`range-${side}`);range.max=info.n_frames-1;range.disabled=false;
    range.oninput=()=>scrub(side,Number(range.value));
    document.querySelectorAll(`.step[data-pane="${side}"]`).forEach(button=>button.onclick=()=>scrub(side,SCRUBS[side].index+Number(button.dataset.step)));
    $(`scrub-state-${side}`).textContent=`${info.n_frames} frames · ${info.fps.toFixed(0)}fps`;
    $(`scrub-${side}`).onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();scrub(side,SCRUBS[side].index+(e.key==='ArrowLeft'?-1:1));}};
    scrub(side,0);
  }));
  if(epoch===EPOCH)syncClipFrames();
}
function sharedScoreRange(results) {
  const values=Object.values(results).flatMap(r=>r.segments.map(w=>w.prediction_score)).filter(Number.isFinite);
  if(!values.length)return [-0.01,0.01];
  const lo=Math.min(...values), hi=Math.max(...values);
  const pad=Math.max((hi-lo)*0.1,0.001);
  // Positive scores start at the actual shared minimum, without lower padding.
  const lower=lo>=0?lo:Math.max(-1,lo-pad);
  return [lower,Math.max(lower+0.001,hi+pad)];
}
// The chosen window, marked in every view of it: the chart's point, the table rows and the card.
function markSelected(panel) {
  const index=panel.selectedSegment;
  panel.querySelectorAll('.segment-row,circle[data-segment]').forEach(el=>el.classList.toggle('selected',Number(el.dataset.segment)===index));
  panel.querySelectorAll('.segment-card').forEach(el=>el.classList.toggle('selected',Number(el.dataset.card)===index));
}
function bindSegments(root, select) {
  root.querySelectorAll('[data-segment]').forEach(el=>{
    el.onclick=()=>select(Number(el.dataset.segment));
    el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();el.click();}};
  });
}
// The only place charts are drawn: both sides share one y-range, so both redraw together.
function refreshScoreCharts() {
  const range=sharedScoreRange(GRAPH_RESULTS);
  for(const [side,result] of Object.entries(GRAPH_RESULTS)) {
    const panel=$(`result-${side}`), previous=panel?.querySelector('.score-plot-scroll');
    if(!previous)continue;
    previous.outerHTML=chart(result,range,gtOf(side));
    bindSegments(panel.querySelector('.score-plot-scroll'), panel.selectWindow);
    markSelected(panel);
  }
}
function chart(result,range,gt) {
  const width=680;
  const [lo,hi]=range;
  const x=t=>55+(width-95)*t/result.duration_sec, y=s=>20+200*(hi-s)/(hi-lo);
  let out=`<div class="score-plot-scroll"><svg viewBox="0 0 ${width} 280" role="img" data-y-min="${lo}" data-y-max="${hi}" aria-label="CLIP cosine of the prediction captions by window end time">`;
  for(const score of Array.from({length:5},(_,i)=>lo+(hi-lo)*i/4))out+=`<path d="M40 ${y(score)} H${width-40}" stroke="#2a2f3a"/><text x="8" y="${y(score)+4}" fill="#9aa3b2" font-size="11">${score.toFixed(4)}</text>`;
  // Invisible per-window strips keep every window clickable; the red bands are the violation intervals, which start
  // where the window before a flagged one ends (its overlap was already seen as normal).
  for(const r of result.segments)out+=`<rect data-segment="${r.index}" x="${x(r.start_sec)}" y="20" width="${x(r.end_sec)-x(r.start_sec)}" height="200" fill="transparent"/>`;
  for(const a of result.anomaly_intervals)out+=`<rect x="${x(a.start_sec)}" y="20" width="${Math.max(x(a.end_sec)-x(a.start_sec),1)}" height="200" fill="#ff6b6b" opacity=".2" pointer-events="none"><title>candidate ${a.start_sec.toFixed(2)}–${a.end_sec.toFixed(2)}s</title></rect>`;
  if(gt){
    const band=Math.max(x(gt.end_sec)-x(gt.onset_sec),0);
    if(band>0)out+=`<rect x="${x(gt.onset_sec)}" y="20" width="${band}" height="200" fill="${GT_COLOR}" opacity=".12"/>`;
    for(const t of gt.end_sec>gt.onset_sec?[gt.onset_sec,gt.end_sec]:[gt.onset_sec])out+=`<path d="M${x(t)} 18 V220" stroke="${GT_COLOR}" stroke-width="2"><title>GT ${t}s</title></path>`;
    out+=`<text x="${x(gt.onset_sec)+3}" y="32" fill="${GT_COLOR}" font-size="11" font-weight="700">GT ${gt.onset_sec}${gt.end_sec>gt.onset_sec?'–'+gt.end_sec:''}s</text>`;
  }
  // Schema-3 records carry no score for their first window.
  const scored=result.segments.filter(r=>r.prediction_score!=null);
  out+=`<polyline fill="none" stroke="#5b9dff" stroke-width="2" points="${scored.map(r=>`${x(r.end_sec)},${y(r.prediction_score)}`).join(' ')}"/>`;
  out+=scored.map(r=>`<circle data-segment="${r.index}" tabindex="0" role="button" aria-label="${r.start_sec.toFixed(2)}–${r.end_sec.toFixed(2)}s prediction score ${f4(r.prediction_score)}" cx="${x(r.end_sec)}" cy="${y(r.prediction_score)}" r="5" fill="${r.is_anomaly?'#ff6b6b':'#5b9dff'}"><title>${r.start_sec.toFixed(2)}–${r.end_sec.toFixed(2)}s · ${f4(r.prediction_score)} · ${r.trusted?'trusted':'untrusted'}</title></circle>`).join('');
  out+='<text x="55" y="244" fill="#9aa3b2" font-size="11">0</text>';
  for(const r of result.segments)out+=`<g class="time-tick" data-segment="${r.index}" tabindex="0" role="button" aria-label="Select window ending at ${r.end_sec.toFixed(2)}s"><path d="M${x(r.end_sec)} 220 v6" stroke="#9aa3b2"/><rect x="${x(r.end_sec)-24}" y="227" width="48" height="24" fill="transparent"/><text x="${x(r.end_sec)}" y="244" text-anchor="middle" fill="#9aa3b2" font-size="12">${Number(r.end_sec.toFixed(2))}</text></g>`;
  return out+`<text x="${width/2}" y="271" text-anchor="middle" fill="#9aa3b2" font-size="12">Window end time (s) · click a dot or time → input frames</text></svg></div>`;
}
function showWindowFrames(side,result,row) {
  const pair=PAIRS.find(p=>p.group===SEL), name=pair?.[side+'_name'];
  const sampled=row.sampled_frames || {};
  const renderStrip=(kind,label)=>{
    const indices=sampled[kind];
    if(!indices?.length)return `<h3>${label}</h3><p class="meta">This record has no input-frame information.</p>`;
    return `<h3>${label} · ${indices.length} frames</h3><div class="model-filmstrip">${indices.map(index=>`<button type="button" class="model-frame" data-frame="${index}" title="Go to original frame ${index}"><img loading="lazy" src="/api/frame?name=${encodeURIComponent(name)}&i=${index}" alt="${label} frame ${index}"><span>f${index} · ${(index/result.fps).toFixed(2)}s</span></button>`).join('')}</div>`;
  };
  $(`model-frames-${side}`).innerHTML=`<h3>Selected window ${row.start_sec.toFixed(2)}–${row.end_sec.toFixed(2)}s · CLIP input frames</h3>
    <p class="meta">${row.sampled_frames_reconstructed?'Frames reconstructed from the stored settings and the sampler.':'Original frame indices used in the run.'} Shown before resize and normalization. Each CLIP score is for the whole window, not per frame.</p>
    ${renderStrip('clip','InternVideo2-CLIP')}
    <p class="meta">${row.history_source_segment==null?'No trusted window history: no earlier window was used for the prediction.':`This window's prediction used the state history up to trusted window ${row.history_source_segment+1}.`}</p>`;
  $(`model-frames-${side}`).querySelectorAll('[data-frame]').forEach(button=>button.onclick=()=>scrub(side,Number(button.dataset.frame)));
}
function showResult(side,result,files=[]) {
  GRAPH_RESULTS[side]=result;
  const panel=$(`result-${side}`);
  const rule=result.anomaly_rule;
  const status=result.has_anomaly?(rule?`Physics-error candidates: lowest ${rule.lowest_windows} CLIP window(s)`:'Physics-error candidate found'):'No candidates';
  // Runs from before 09-30 flagged what the model marked untrusted, not the lowest CLIP scores.
  const oldRule=!rule?'<p class="analysis-fresh">The error candidates of this record were marked by the earlier rule (MLLM trusted=false).</p>':'';
  const intervals=result.anomaly_intervals.map(r=>`${r.start_sec.toFixed(2)}–${r.end_sec.toFixed(2)}s`).join(', ');
  const name='CLIP';
  const cards=result.segments.map((r,i)=>{
    const scoreLabel=`This window's prediction ↔ this window's video · ${name} ${f4(r.prediction_score)}`;
    return `<article class="segment-card${r.is_anomaly?' anomaly':''}" data-card="${r.index}">
      <h3><button data-segment="${r.index}" type="button">Window ${i+1} · ${r.start_sec.toFixed(2)}–${r.end_sec.toFixed(2)}s</button><span class="verdict ${r.trusted?'pos-val':'neg-val'}">${r.trusted?'trusted':'untrusted'}</span>${r.is_anomaly?`<span class="verdict neg-val">· physics-error candidate${rule?' (lowest CLIP)':''}</span>`:''}</h3>
      <div class="score-label score-source">${scoreLabel}${r.prediction_scores?.length?' <span class="meta">(highest per-caption score)</span>':''}</div>
      ${captions(r.prediction,'source-captions',r.prediction_scores)}
      <p class="meta">${r.history_source_segment==null?(i===0?'first window: excluded from candidates':'no history used'):`History used: states up to trusted window ${r.history_source_segment+1}`}</p>
      <div class="score-label score-state">Observed state <span class="meta">(history not used)</span></div>
      ${captions(r.state,'state-captions')}
      ${rawOutputs(r)}
      ${modelInputs(r,result.model_input_system||'')}
    </article>`;
  }).join('');
  panel.innerHTML=`<h2 class="${side==='good'?'k-pos':'k-imp'}">${labels[side]} · ${status}</h2>${oldRule}<div class="result-summary">${intervals?'candidates: '+intervals:''}</div>${gtNote(gtOf('bad'),side!=='bad')}<div class="score-plot-scroll"></div>
    <p class="plot-legend"><span class="prediction">● ${name} score: this window's prediction ↔ this window's video</span> · red dots/windows: physics-error candidates${rule?'(lowest CLIP windows)':'(earlier rule)'}${gtOf(side)?' · red vertical line: GT':''}</p>
    <p class="meta">Click a window or the graph to see its frames.</p>
    <div class="downloads">${files.map(url=>`<a class="ghost" download href="${esc(url)}">${esc(url.split('.').pop().toUpperCase())}</a>`).join('')}</div>
    <div class="model-input-frames" id="model-frames-${side}"></div>
    <div class="table-wrap"><table class="tbl segment-table"><thead><tr><th>Window (s)</th><th>trusted</th><th class="num">#</th><th class="num">Caption score</th><th>prediction captions <span class="adopted-key">★ adopted</span></th></tr></thead><tbody>
    ${result.segments.map(r=>segmentRows(r,r.prediction||[])).join('')}</tbody></table></div>
    ${cards}<div id="state-detail-${side}" class="meta"></div>
    <details><summary>System prompt used in the run</summary><pre>${esc(result.trusted_history?.[0]?.content || '')}</pre></details>
    ${result.trusted_history_text!=null
      ?`<details><summary>Trusted history shown to the MLLM (text actually given to the model)</summary><pre class="history">${esc(result.trusted_history_text)}</pre>
        <p class="hint">The system prompt and the "Earlier clips" list of the trusted windows. For each window the current window video and the request follow this list directly (in the same user turn). If a history window count was set, only the latest windows are included.</p></details>`
      :`<details><summary>Trusted history shown to the MLLM (stored structure, earlier record)</summary><pre class="history">${esc(JSON.stringify(result.trusted_history,null,2))}</pre></details>`}`;
  const selectWindow=index=>{
    const row=result.segments[index];seek(side,row.start_sec);
    showWindowFrames(side,result,row);
    panel.selectedSegment=index;
    markSelected(panel);
    $(`state-detail-${side}`).innerHTML=`<h3>${row.start_sec.toFixed(2)}–${row.end_sec.toFixed(2)}s · ${row.trusted?'trusted':'untrusted'}</h3><b>Observed state</b>${captions(row.state,'state-captions')}<b>prediction (uses history)</b>${captions(row.prediction,'next-captions',row.prediction_scores,row)||'<p>last window</p>'}`;
  };
  panel.selectWindow=selectWindow;
  bindSegments(panel,selectWindow);
  showWindowFrames(side,result,result.segments[Math.min(1,result.segments.length-1)]);
  refreshScoreCharts();
}
async function loadTrial(id,epoch=EPOCH) {
  const trial=await api('/api/trial?id='+encodeURIComponent(id));
  if(epoch!==EPOCH || trial.group!==SEL)return;
  setClipFrames(trial.params.clip_frames ?? Object.values(trial.results)[0]?.models?.clip_frames);
  $('hop-seconds').value=trial.params.hop_seconds;
  $('segment-seconds').value=trial.params.window_seconds;
  syncClipFrames();
  setPrompts(trial.params.prompts);
  // Trials from before the per-run count ran at the server's fixed count, whatever their window length.
  const usedFrames=Object.values(trial.results)[0]?.models?.mllm_frames;
  if(trial.params.prompts?.mllm_frames==null&&usedFrames!=null&&usedFrames!==autoMllmFrames())$('mllm-frames').value=usedFrames;
  syncMllmFrames();
  const sides=trial.params.sides || ['good','bad'];
  $('pair-trial-meta').textContent=`Analyzed at: ${new Date(trial.created_at).toLocaleString()}`
    +(sides.length===2?'':` · run on: ${labels[sides[0]]} only`)
    +(isBatch(trial)?' · batch run record':'');
  GRAPH_RESULTS={...trial.results};
  for(const side of ['good','bad'])
    if(trial.results[side])showResult(side,trial.results[side],trial.files[side]);
    else showSkipped(side,'Not run in this record');
}
function showSkipped(side, reason) {
  const panel=$(`result-${side}`);
  if(panel)panel.innerHTML=`<h2 class="${side==='good'?'k-pos':'k-imp'}">${labels[side]} · not run</h2>`
    +`<p class="meta">${esc(reason)}</p>`;
}
function clearAnalysisPanels(message, sides=['good','bad']) {
  GRAPH_RESULTS={};
  $('pair-trial-meta').textContent=message;
  for(const side of ['good','bad']) {
    const panel=$(`result-${side}`);
    if(!panel)continue;
    if(sides.includes(side))panel.innerHTML=`<h2>${labels[side]} · analyzing</h2><p class="analysis-pending" role="status">${esc(message)} · waiting for new result</p>`;
    else showSkipped(side,'Not part of this run');
  }
}
async function watchJob(id) {
  if(WATCHING)return;
  WATCHING=true;BUSY=true;syncRun();$('progress').hidden=false;
  const shown=new Set(), cleared=new Set();
  try {
    while(true){
      const job=await api('/api/job?id='+encodeURIComponent(id));
      $('progress-text').textContent=`${job.group} · ${job.progress}`;
      if(job.group===SEL && !cleared.has(EPOCH)){clearAnalysisPanels('Analysis in progress');cleared.add(EPOCH);}
      if(job.group===SEL)for(const side of job.completed||[]){
        const key=`${EPOCH}-${side}`, epoch=EPOCH;
        if(shown.has(key))continue;
        const result=await api(`/api/job_result?id=${encodeURIComponent(id)}&side=${side}`);
        if(epoch!==EPOCH)continue;  // pair switched while fetching; the next poll renders it
        shown.add(key);
        showResult(side,result);
        $(`result-${side}`).insertAdjacentHTML('afterbegin','<p class="analysis-fresh">Result of this analysis · the other video may still be in progress.</p>');
      }
      if(job.state==='error')throw new Error(job.error);
      if(job.state==='done'){
        await refreshTrials();if(job.group===SEL)await loadTrial(job.trial_id);
        $('run-status').textContent='Analysis done · updated with this run';if(job.group===SEL)$('pair-trial-meta').textContent+=' · latest run done';break;
      }
      await delay(1500);
    }
  }catch(e){$('run-status').textContent='Analysis failed · only finished videos are shown';if(SEL)$('pair-trial-meta').textContent='Analysis failed · see the error below';document.querySelectorAll('.analysis-pending').forEach(el=>el.textContent='Analysis failed · no new result for this video.');fail(e);}finally{WATCHING=false;BUSY=false;syncRun();$('progress').hidden=true;}
}
async function runPair() {
  if(BUSY||!SEL)return;
  const seconds=Number($('segment-seconds').value), hop=Number($('hop-seconds').value);
  if(!Number.isFinite(seconds)||seconds<=0||!Number.isFinite(hop)||hop<=0||hop>seconds)return fail(new Error('Check the window length and the condition 0 < hop ≤ window.'));
  const clamped=syncClipFrames();
  const clip_frames=clipFrames();
  const [lowest,highest]=clipFramesRange();
  if(!Number.isInteger(clip_frames)||clip_frames<lowest||clip_frames>highest)return fail(new Error(`CLIP input frames must be an integer from ${lowest} to ${highest}${CFG.clip_reads_all_frames?' (0 = every frame of the window)':''}.`));
  const prompts=promptValues();
  if(!Number.isInteger(prompts.num_prompts)||prompts.num_prompts<1)return fail(new Error('Captions per field n must be an integer >= 1.'));
  const limit=CFG.defaults.max_new_tokens_limit;
  if(!Number.isInteger(prompts.max_new_tokens)||prompts.max_new_tokens<1||(limit&&prompts.max_new_tokens>limit))return fail(new Error(`Output limit must be an integer from 1${limit?` to ${limit}`:''} (max_model_len − 8192 reserved for the input).`));
  if(!Number.isInteger(prompts.thinking_budget)||prompts.thinking_budget<0||(prompts.thinking_budget&&prompts.thinking_budget>=prompts.max_new_tokens))return fail(new Error('Thinking budget must be an integer >= 0 and below the output limit.'));
  if(!Number.isInteger(prompts.lowest_windows)||prompts.lowest_windows<0)return fail(new Error('Physics-error candidate windows k must be an integer >= 0.'));
  if(!Number.isInteger(prompts.history_windows)||prompts.history_windows<-1)return fail(new Error('History windows must be an integer >= -1.'));
  if(prompts.mllm_frames!==null&&(!Number.isInteger(prompts.mllm_frames)||prompts.mllm_frames<2||prompts.mllm_frames%2))return fail(new Error('MLLM input frames must be empty (auto) or an even number >= 2. The vision encoder patches frames in pairs.'));
  if(prompts.mllm_frames===null&&autoMllmFrames()===null)return fail(new Error(`A window length of ${seconds}s gives fewer than 2 automatic MLLM frames. Lengthen the window or enter the MLLM input frames yourself.`));
  const sides=runSides();
  BUSY=true;syncRun();$('error').hidden=true;$('run-status').textContent=clamped?`Requesting new analysis · ${clamped}`:'Requesting new analysis';clearAnalysisPanels('Requesting new analysis',sides);
  try{
    const job=await api('/api/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({group:SEL,window_seconds:seconds,hop_seconds:hop,sides,prompts,clip_frames})});
    await watchJob(job.id);
  }catch(e){BUSY=false;syncRun();$('run-status').textContent='Analysis request failed';document.querySelectorAll('.analysis-pending').forEach(el=>el.textContent='Analysis request failed · see the error below');fail(e);}
}
async function init() {
  CFG=await api('/api/config');PAIRS=(await api('/api/pairs')).pairs;
  $('head-sub').textContent=`IntPhys2 · ${CFG.model}`;
  [$('clip-frames').min,$('clip-frames').max]=clipFramesRange();
  $('segment-seconds').addEventListener('input',syncMllmFrames);
  for(const id of ['segment-seconds','hop-seconds','clip-frames'])$(id).addEventListener('change',()=>{
    // Only a clamp note is replaced; a run's own status stays.
    const note=syncClipFrames(), status=$('run-status');
    if(note)status.textContent=note;
    else if(status.textContent.startsWith('CLIP input frames'))status.textContent='';
  });
  const [fewest,most]=clipFramesRange();
  $('clip-frames-range').textContent=CFG.clip_reads_all_frames?`(${fewest}~${most}, 0 = every frame)`:`(${fewest}~${most})`;
  $('hop-seconds').value=CFG.defaults.hop_seconds;
  setClipFrames(CFG.clip_frames);
  $('segment-seconds').value=CFG.defaults.segment_seconds;
  syncMllmFrames();
  // The engine's context sets how far the two limits can go.
  if(CFG.defaults.max_new_tokens_limit)for(const id of ['max-new-tokens','thinking-budget'])$(id).max=CFG.defaults.max_new_tokens_limit;
  $('max-new-tokens-limit').textContent=CFG.defaults.max_new_tokens_limit ?? 'shown after the model loads';
  // Everything the form holds goes back to the server's defaults, window geometry and CLIP frames included;
  // the frame limit follows the restored window length.
  $('reset-prompts').onclick=()=>{
    if(!SEL)return;
    setDefaults(SEL);
    syncClipFrames();
    if($('run-status').textContent.startsWith('CLIP input frames'))$('run-status').textContent='';
  };
  buildFilters();await refreshTrials();syncRun();
  $('f-text').addEventListener('input',renderList);
  $('reset-filters').onclick=()=>{ATTR='';$('f-text').value='';buildFilters();renderList();};
  $('video-search').onsubmit=e=>{
    e.preventDefault();const name=$('video-search-input').value.trim().replace(/\.mp4$/i,'').toLowerCase();
    if(!name)return;
    const matches=PAIRS.filter(p=>[p.good_name,p.bad_name].some(n=>n.toLowerCase().includes(name)));
    $('video-search-status').textContent=matches.length?`${matches.length} pair(s) found`:'No matching video';
    if(matches.length){$('reset-filters').click();$('f-text').value=name;renderList();selectPair(matches[0].group).catch(fail);}
  };
  $('run-btn').onclick=runPair;
  $('run-sides').onchange=syncRun;
  $('error-close').onclick=()=>{$('error').hidden=true;};
  $('clear-trials').onclick=()=>{$('trials-table').querySelector('tbody').innerHTML='';syncTrialChecks();};
  $('trials-check-all').onchange=e=>{document.querySelectorAll('.trial-check').forEach(box=>{box.checked=e.target.checked;});syncTrialChecks();};
  $('delete-trials').onclick=()=>deleteTrials().catch(fail);
  document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter'){e.preventDefault();runPair();}});
  async function status(){
    try{
      const s=await api('/api/status');
      $('model-badge').className='badge '+(s.active_job?'loading':s.model==='ready'?'ready':'');
      $('model-badge').textContent=s.active_job?'Analyzing…':s.model==='ready'?'Model ready':'Model loads on first analysis';
      if(s.active_job&&!WATCHING)watchJob(s.active_job);
    }catch(e){$('model-badge').textContent='Check server connection';}
  }
  await status();setInterval(status,3000);
}
init().catch(fail);
