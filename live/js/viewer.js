import { SERIES_LABELS, loadCandidates, loadChain, loadIndex } from './data.js';
import { Mlflow, forgetLogin } from './mlflow.js';

const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const CLUSTER={task_verbatim:'task',concept_name:'task',solution_correct:'sol',
 solution_incorrect:'sol',solution_incorrect_1:'sol',solution_incorrect_2:'sol',
 solution_incorrect_3:'sol',correctness_keyword:'sol',solution_keyword:'sol',solution_word:'sol',
 answer_word:'sol',problem_setup:'beh',plan_generation:'beh',fact_retrieval:'beh',
 active_computation:'beh',uncertainty_management:'beh',result_consolidation:'beh',
 self_checking:'beh',final_answer_emission:'beh',close_to_solution:'self',
 heading_toward_correct:'self',backtracking:'inf',realization:'inf',reconsideration:'inf'};
const CNAME={task:'task/concept',sol:'solutions & keywords',beh:'reasoning behaviour',
 self:'self-assessment',inf:'inflection points'};
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.body).getPropertyValue(n).trim();
// Past eight queries no ordering keeps adjacent hues apart, so colour falls back to the cluster.
const QSLOTS=8;
function slotOf(q){const i=QS().indexOf(q);return QS().length<=QSLOTS&&i>=0?i+1:0;}
const colOf=q=>{const s=slotOf(q);return css(s?'--q'+s:'--c-'+(CLUSTER[q]||'task'));};
const num=n=>n.toLocaleString();

let client=null,experiments=[],SETS=[],VIEW=null,ticket=0;
let run=0,cur=0,sel=null,pos=0,lo=0,hi=0,filter='',gran='token',yfit='shared';
const CHAINS={};
const R=()=>SETS[run];
const C=()=>VIEW;
const QS=()=>Object.keys(C().alpha);

function status(text,bad){const s=$('status');s.textContent=text;s.classList.toggle('bad',!!bad);}

// Only the teacher-forced s telescopes, so only sums of it across a sentence mean something.
const BUCK={};
function bucketsOf(c,g,sk){
  if(g==='token')return null;
  const key=c.runId+':'+g+':'+sk;
  if(BUCK[key])return BUCK[key];
  const starts=c.buckets[g]||[0],sums={},edges=[];
  for(let b=0;b<starts.length;b++)edges.push([starts[b],b+1<starts.length?starts[b+1]:c.n]);
  let mx=1e-6;
  for(const q in c[sk]){
    const a=c[sk][q],out=new Float32Array(edges.length);
    for(let b=0;b<edges.length;b++){
      let acc=0;
      for(let i=edges[b][0];i<edges[b][1];i++)acc+=a[i]===a[i]?a[i]:0;
      out[b]=acc;
      const v=Math.abs(acc);if(v>mx)mx=v;
    }
    sums[q]=out;
  }
  return BUCK[key]={edges:edges,sums:sums,max:mx};
}

// The per-query fit trades away comparability when one query's s is far flatter than the loudest.
const QCAP={};
function qCapOf(c,q,sk){
  const key=c.runId+':'+q+':'+sk;
  if(QCAP[key]!==undefined)return QCAP[key];
  const a=c[sk][q],t=new Float64Array(a.length);
  for(let i=0;i<a.length;i++)t[i]=a[i]===a[i]?Math.abs(a[i]):0;
  t.sort();
  return QCAP[key]=t[Math.min(t.length-1,Math.floor(t.length*0.999))]||1e-3;
}
function absMax(arr){
  let m=1e-6;
  for(let i=0;i<arr.length;i++){const v=Math.abs(arr[i]);if(v>m)m=v;}
  return m;
}

function seg(box,items,get,set){
  box.textContent='';
  items.forEach(([val,label])=>{
    const b=document.createElement('button');
    b.className='gb';b.setAttribute('role','tab');b.setAttribute('aria-selected',get()===val);
    b.textContent=label;
    b.onclick=()=>{set(val);[...box.children].forEach(x=>x.setAttribute('aria-selected',false));
      b.setAttribute('aria-selected',true);render();};
    box.appendChild(b);
  });
}

function GRAN_INIT(){
  seg($('gran'),[['token','per token'],['sent','per sentence'],['para','per paragraph']],
      ()=>gran,v=>{gran=v;});
  seg($('yfit'),[['shared','all queries'],['query','fit to query']],
      ()=>yfit,v=>{yfit=v;});
}

function RUNS_INIT(){
  const box=$('runs');box.textContent='';
  SETS.forEach((r,i)=>{
    const b=document.createElement('button');
    b.className='rb';b.setAttribute('role','tab');b.setAttribute('aria-selected',i===run);
    b.innerHTML=`<span class="rn">${esc(r.tag)}</span>`+
      `<span class="rm">${r.cases.length} CoTs${r.complete?'':' · partial'}</span>`;
    b.onclick=()=>{if(run===i)return;run=i;cur=0;sel=null;
      [...box.children].forEach((x,j)=>x.setAttribute('aria-selected',j===i));
      CASES_INIT();drawRunWarn();openChain();};
    box.appendChild(b);
  });
}

function drawRunWarn(){
  const r=R();
  if(!r||r.complete){$('runwarn').innerHTML='';return;}
  $('runwarn').innerHTML=`<div class="warn"><b>Partial set.</b> ${r.cases.length} of the `+
    `${r.expected} chains in this set finished uploading to MLflow, so only those are listed.</div>`;
}

function matches(c,f){
  if(!f)return true;
  return (c.id+' '+c.question).toLowerCase().includes(f);
}

function CASES_INIT(){
  const box=$('cases');box.textContent='';
  const f=filter.trim().toLowerCase(),all=R()?R().cases:[];
  let shown=0;
  all.forEach((c,i)=>{
    if(!matches(c,f))return;
    shown++;
    const b=document.createElement('button');
    b.className='cr';b.setAttribute('role','option');b.setAttribute('aria-selected',i===cur);
    const tone=c.correct===true?'--ok-ink':c.correct===false?'--bad-ink':'--neu-ink';
    const label=c.correct===true?'correct':c.correct===false?'wrong':'ungraded';
    b.innerHTML=`<span class="ci">${esc(c.id)}</span>`+
      `<span class="cq">${esc(c.question.split('\n')[0])}</span>`+
      `<span class="cn">${num(c.n)} tok</span>`+
      `<span class="cd" style="background:var(${tone})" title="${label}"></span>`;
    b.onclick=()=>{cur=i;sel=null;
      [...box.children].forEach(x=>x.setAttribute('aria-selected',false));
      b.setAttribute('aria-selected',true);openChain();};
    box.appendChild(b);
  });
  if(!shown){
    const e=document.createElement('div');e.className='empty';
    e.textContent=all.length?'no chain of thought matches that filter':'no chains in this set';
    box.appendChild(e);
  }
  $('pcount').textContent=shown===all.length?`${all.length} CoTs`:`${shown} of ${all.length}`;
}

// A later click wins: a chain that finishes loading after another was picked is kept, not drawn.
async function openChain(){
  const meta=R()&&R().cases[cur];
  if(!meta)return;
  const mine=++ticket;
  status(`Loading ${meta.id}…`);
  try{
    CHAINS[meta.runId]??=loadChain(client,meta,R().queries);
    const c=await CHAINS[meta.runId];
    if(mine!==ticket)return;
    VIEW=c;status('');resetView();render();
  }catch(err){
    delete CHAINS[meta.runId];
    if(mine===ticket)status(`Could not load ${meta.id}. ${err.message}`,true);
  }
}

function resetView(){const n=C().n;lo=0;hi=n-1;pos=Math.floor(hi/2);}

// One width is measured and used, keeping the last good one when a mid-relayout measure reads zero.
function fit(cv,h){
  const r=cv.parentElement.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  const cssw=Math.max(320,r.width||cv._w||320);
  cv._w=cssw;
  cv.width=Math.round(cssw*dpr);cv.height=Math.round(h*dpr);
  cv.style.height=h+'px';
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);
  return [g,cssw,h];
}

// The x axis is character position, so the panels line up with the chain of thought above them.
function charEdges(c){
  if(c._edges)return c._edges;
  const e=new Float64Array(c.toks.length+1);
  for(let i=0;i<c.toks.length;i++)e[i+1]=e[i]+Math.max(1,c.toks[i].length);
  return c._edges=e;
}
function tokSpan(c,i,x0,w){
  const e=charEdges(c),a=e[lo],b=e[hi+1],k=w/Math.max(1e-9,b-a);
  return [x0+(e[i]-a)*k,x0+(e[i+1]-a)*k];
}
const tokMid=(c,i,x0,w)=>{const[a,b]=tokSpan(c,i,x0,w);return (a+b)/2;};

// A min and max per pixel column, with no averaging, so spikes survive.
function series(g,c,arr,x0,w,y,wide){
  const mn=new Float64Array(w).fill(Infinity),mx=new Float64Array(w).fill(-Infinity);
  for(let i=lo;i<=hi;i++){
    const v=arr[i];if(!(v===v))continue;
    const [sa,sb]=tokSpan(c,i,x0,w);
    let p0=Math.floor(sa-x0),p1=Math.ceil(sb-x0)-1;
    if(p1<p0)p1=p0;
    for(let px=Math.max(0,p0);px<=Math.min(w-1,p1);px++){
      if(v<mn[px])mn[px]=v;
      if(v>mx[px])mx[px]=v;
    }
  }
  g.beginPath();
  for(let px=0;px<w;px++){
    if(mn[px]===Infinity)continue;
    const X=x0+px+.5;
    if(mx[px]-mn[px]<1e-9){g.moveTo(X,y(mn[px]));g.lineTo(X,y(mn[px])+.6);}
    else{g.moveTo(X,y(mx[px]));g.lineTo(X,y(mn[px]));}
  }
  g.lineWidth=wide?1.6:1;g.stroke();
}

function drawChart(){
  const c=C(),[g,W,H]=fit($('chart'),300);
  const PL=42,PR=8,PT=10,PB=24,w=Math.max(1,Math.round(W-PL-PR));
  g.clearRect(0,0,W,H);
  const y=a=>PT+(1-a)*(H-PT-PB);
  g.strokeStyle=css('--rule-2');g.lineWidth=1;
  g.fillStyle=css('--ink-3');g.font='9px ui-monospace,monospace';g.textAlign='right';
  for(let v=0;v<=1;v+=0.25){
    g.beginPath();g.moveTo(PL,y(v));g.lineTo(W-PR,y(v));g.stroke();
    g.fillText(v.toFixed(2),PL-5,y(v)+3);
  }
  QS().forEach(q=>{
    if(sel&&q===sel)return;
    g.strokeStyle=colOf(q);g.globalAlpha=sel?0.08:0.34;
    series(g,c,c.alpha[q],PL,w,y,false);
  });
  g.globalAlpha=1;
  if(sel){g.strokeStyle=colOf(sel);series(g,c,c.alpha[sel],PL,w,y,true);}
  const cx=tokMid(c,pos,PL,w);
  g.strokeStyle=css('--ink-2');g.setLineDash([3,3]);g.beginPath();
  g.moveTo(cx,PT);g.lineTo(cx,H-PB);g.stroke();g.setLineDash([]);
  g.textAlign='left';g.fillText('token '+num(lo),PL,H-8);
  g.textAlign='right';g.fillText('token '+num(hi),W-PR,H-8);
}

// One scale per chain, its 99.9th percentile of |s|, so zooming never changes what a height means.
const S_PANELS=[['schart','sscale','sAlpha'],['schart2','sscale2','s'],
                ['schart3','sscale3','sReweight']];
const D_PANELS=[['dchart','dscale','d'],['dchart2','dscale2','dEq12']];

function drawS(){
  S_PANELS.forEach(([cv,note,key])=>drawSPanel(cv,note,key));
  D_PANELS.forEach(([cv,note,key])=>drawDPanel(cv,note,key));
}

// delta is a probability, so its panels share the belief's fixed 0 to 1 axis and are read per token.
function drawDPanel(canvasId,noteId,key){
  const c=C();
  const [g,W,H]=fit($(canvasId),130);
  const PL=42,PR=8,PT=10,PB=18,w=Math.max(1,Math.round(W-PL-PR));
  g.clearRect(0,0,W,H);
  const y=v=>PT+(1-Math.max(0,Math.min(1,v)))*(H-PT-PB);
  g.fillStyle=css('--ink-3');g.font='9px ui-monospace,monospace';g.textAlign='right';
  [1,0.75,0.5,0.25,0].forEach(v=>{
    g.strokeStyle=v===0?css('--rule'):css('--rule-2');
    g.beginPath();g.moveTo(PL,y(v));g.lineTo(W-PR,y(v));g.stroke();
    g.fillText(v.toFixed(2),PL-5,y(v)+3);
  });
  (sel?[sel]:QS()).forEach(q=>{
    g.strokeStyle=colOf(q);g.globalAlpha=sel?1:0.28;series(g,c,c[key][q],PL,w,y,!!sel);
  });
  g.globalAlpha=1;
  const cx=tokMid(c,pos,PL,w);
  g.strokeStyle=css('--ink-2');g.setLineDash([3,3]);g.beginPath();
  g.moveTo(cx,PT);g.lineTo(cx,H-PB);g.stroke();g.setLineDash([]);
  $(noteId).innerHTML=`<span class="eq">${seriesLabel(key)}</span>`;
}

const seriesLabel=key=>SERIES_LABELS[key]||key;

function drawSPanel(canvasId,noteId,key){
  const c=C();
  const [g,W,H]=fit($(canvasId),130);
  const PL=42,PR=8,PT=10,PB=18,w=Math.max(1,Math.round(W-PL-PR));
  g.clearRect(0,0,W,H);
  const B=bucketsOf(c,gran,key);
  const fitted=yfit==='query'&&sel;
  const cap=(c.sCaps&&c.sCaps[key])||c.sCap;
  const mx=B?(fitted?absMax(B.sums[sel]):B.max):(fitted?qCapOf(c,sel,key):cap);
  const y=v=>PT+(1-(Math.max(-mx,Math.min(mx,v))/mx+1)/2)*(H-PT-PB);
  g.fillStyle=css('--ink-3');g.font='9px ui-monospace,monospace';g.textAlign='right';
  const ticks=[mx,mx/2,0,-mx/2,-mx];
  ticks.forEach(v=>{
    g.strokeStyle=v===0?css('--rule'):css('--rule-2');
    g.beginPath();g.moveTo(PL,y(v));g.lineTo(W-PR,y(v));g.stroke();
    g.fillText((v>0?'+':'')+v.toFixed(1),PL-5,y(v)+3);
  });
  const qs=sel?[sel]:QS();
  if(!B){
    qs.forEach(q=>{g.strokeStyle=colOf(q);g.globalAlpha=sel?1:0.28;series(g,c,c[key][q],PL,w,y,!!sel);});
  }else{
    const X=(i,end)=>tokSpan(c,Math.min(i,c.n-1),PL,w)[end&&i>=c.n?1:0];
    qs.forEach(q=>{
      const sums=B.sums[q];g.fillStyle=colOf(q);g.globalAlpha=sel?0.75:0.16;
      for(let b=0;b<B.edges.length;b++){
        const [a,e]=B.edges[b];
        if(e<=lo||a>hi)continue;
        const x0=X(Math.max(a,lo)),x1=X(Math.min(e,hi+1),true);
        g.fillRect(x0,Math.min(y(0),y(sums[b])),Math.max(1,x1-x0-1),Math.abs(y(sums[b])-y(0)));
      }
    });
  }
  g.globalAlpha=1;
  const cx=tokMid(c,pos,PL,w);
  g.strokeStyle=css('--ink-2');g.setLineDash([3,3]);g.beginPath();
  g.moveTo(cx,PT);g.lineTo(cx,H-PB);g.stroke();g.setLineDash([]);
  $(noteId).innerHTML=`<span class="eq">${seriesLabel(key)}</span>`;
}

// A chain's candidates are fetched once, the first time a query is inspected on it.
function requestCandidates(c){
  if(c._candidates)return;
  c._candidates=loadCandidates(client,c)
    .then(found=>{c.cand=found;if(C()===c)drawCand();})
    .catch(err=>{
      c._candidates=null;
      if(C()===c)$('cand').innerHTML=`<p class="hint" style="margin:0">Could not load the candidates. ${esc(err.message)}</p>`;
    });
}

// delta(v) is normalised over one position's candidates, so these numbers are never summed across tokens.
function drawCand(){
  const c=C(),box=$('cand');
  const note=t=>{box.innerHTML=`<p class="hint" style="margin:0">${t}</p>`;};
  if(!sel)return note('Pick a query on the right to inspect candidates at this token.');
  if(!c.cand){requestCandidates(c);return note('Loading the candidates of this chain…');}
  const at=c.cand.index[sel];
  if(!at||at.start[pos]<0)return note(`No candidates were read for ${esc(sel)} at this token.`);
  const T=c.cand.table,rows=[];
  for(let i=at.start[pos];i<at.end[pos];i++){
    rows.push({tok:T.piece[i]??'',A:Math.exp(T.log_a[i]??NaN),al:T.alpha[i]??NaN,
      d:Math.exp(T.log_delta_reweighted[i]??NaN),real:T.is_realized[i]});
  }
  // A(v) is the same under every query, so ordering on it keeps rows still when the query changes.
  rows.sort((x,y)=>y.A-x.A||y.al-x.al);
  box.innerHTML=`<table class="ctab"><thead><tr><th>token</th><th>A(v)</th>`+
    `<th>&alpha;(v)</th><th>&delta;(v)</th></tr></thead><tbody>`+
    rows.map(r=>`<tr class="${r.real?'real':''}"><td class="tk">${esc(JSON.stringify(r.tok).slice(1,-1))}</td>`+
      `<td>${r.A.toFixed(3)}</td><td>${r.al.toFixed(3)}</td>`+
      `<td>${r.d.toFixed(3)}</td></tr>`).join('')+
    `</tbody></table>`;
}

function drawQList(){
  const c=C(),box=$('qlist');box.textContent='';
  QS().map(q=>({q,a:c.alpha[q][pos]}))
    .sort((u,v)=>v.a-u.a).forEach(r=>{
    const b=document.createElement('button');
    b.className='qr';b.setAttribute('aria-selected',sel===r.q);
    b.innerHTML=`<span class="sw" style="background:${colOf(r.q)}"></span>`+
      `<span><span class="nm">${esc(r.q)}</span>`+
      `<span class="bb"><span style="position:absolute;inset:0 auto 0 0;width:${(r.a*100).toFixed(0)}%;background:${colOf(r.q)}"></span></span></span>`+
      `<span class="vv">${r.a.toFixed(3)}</span>`;
    b.onclick=()=>{sel=sel===r.q?null:r.q;render();};
    box.appendChild(b);
  });
}

function drawQDetail(){
  const c=C(),box=$('qdetail');
  if(!sel){
    box.innerHTML=`<p class="hint" style="margin:0">Pick a query on the right to isolate its trajectory.</p>`;
    return;
  }
  const text=c.qtext[sel]||'(proposition not recorded for this chain)';
  box.innerHTML=`<div class="qhead"><span class="sw" style="background:${colOf(sel)}"></span>`+
    `<span style="color:${colOf(sel)}">${esc(sel)}</span>`+
    `<span class="dim">${esc(CNAME[CLUSTER[sel]]||'')}</span></div>`+
    `<div class="qtext">${esc(text)}</div>`;
}

function drawCot(){
  const c=C(),box=$('cot');box.textContent='';
  const before=document.createElement('span');
  before.textContent=c.toks.slice(0,pos).join('');
  const hlt=document.createElement('span');hlt.className='hl';
  hlt.textContent=c.toks[pos]||'';
  const after=document.createElement('span');
  after.textContent=c.toks.slice(pos+1).join('');
  box.appendChild(before);box.appendChild(hlt);box.appendChild(after);
  const bt=box.getBoundingClientRect(),ht=hlt.getBoundingClientRect();
  box.scrollTop+=(ht.top-bt.top)-box.clientHeight/2;
  $('cotwarn').innerHTML=c.truncated
    ?`<div class="warn">This chain hit the generation limit at ${num(c.n)} tokens, so it stops before the model finished.</div>`
    :'';
}

function drawZbar(){
  const c=C();
  $('zbar').innerHTML=`<button class="zbtn" id="rz">reset zoom</button>`+
    `<span>showing ${num(hi-lo+1)} of ${num(c.n)} tokens</span>`+
    `<span class="dim">scroll to zoom · drag to scrub · double-click to reset</span>`;
  $('rz').onclick=()=>{lo=0;hi=c.n-1;render();};
}

function drawMeta(){
  const c=C();
  $('question').textContent=c.question;
  const seen={};
  $('legend').innerHTML=QS().length<=QSLOTS
    ?QS().map(q=>`<span><i style="background:${colOf(q)}"></i>${esc(q)}</span>`).join('')
    :QS().map(q=>CLUSTER[q]).filter(k=>k&&!seen[k]&&(seen[k]=1))
      .map(k=>`<span><i style="background:var(--c-${k})"></i>${CNAME[k]}</span>`).join('');
}

function drawScrub(){
  $('scrub').innerHTML=`<span>token <span class="tokchip">${num(pos)}</span></span>`+
    `<span class="dim">${sel?'showing '+sel:'click a query to isolate it'}</span>`;
}

function render(){
  const c=C();
  if(!c)return;
  if(hi<=lo){lo=0;hi=c.n-1;}
  pos=Math.max(lo,Math.min(hi,pos));
  drawMeta();drawChart();drawS();drawQList();drawQDetail();drawCand();
  drawCot();drawZbar();drawScrub();
}

// The click lands in whichever token's span contains it, so the cursor always sits on a whole token.
function xToTok(ev,cv){
  const c=C(),r=cv.getBoundingClientRect(),PL=42,PR=8;
  const w=r.width-PL-PR,f=Math.max(0,Math.min(1,((ev.clientX-r.left)-PL)/w));
  const e=charEdges(c),ch=e[lo]+f*(e[hi+1]-e[lo]);
  let a=lo,b=hi;
  while(a<b){const m=(a+b+1)>>1;if(e[m]<=ch)a=m;else b=m-1;}
  return a;
}
['chart','schart','schart2','schart3'].forEach(id=>{
  const cv=$(id);let down=false;
  const scrubTo=e=>{if(!C())return;pos=xToTok(e,cv);
    drawChart();drawS();drawQList();drawQDetail();drawCand();
    drawCot();drawScrub();};
  cv.addEventListener('pointerdown',e=>{down=true;cv.setPointerCapture(e.pointerId);scrubTo(e);});
  cv.addEventListener('pointermove',e=>{if(down)scrubTo(e);});
  cv.addEventListener('pointerup',()=>down=false);
  cv.addEventListener('pointercancel',()=>down=false);
  cv.addEventListener('dblclick',()=>{if(!C())return;lo=0;hi=C().n-1;render();});
  cv.addEventListener('wheel',e=>{
    if(!C())return;
    e.preventDefault();
    const c=C(),at=xToTok(e,cv),k=e.deltaY<0?0.72:1/0.72;
    let span=Math.max(24,Math.min(c.n,Math.round((hi-lo+1)*k)));
    lo=Math.max(0,Math.min(c.n-span,Math.round(at-(at-lo)*k)));
    hi=Math.min(c.n-1,lo+span-1);
    render();
  },{passive:false});
});

async function openExperiment(i){
  const mine=++ticket;
  const experiment=experiments[i];
  status(`Loading ${experiment.name}…`);
  let sets;
  try{sets=await loadIndex(client,experiment.id);}
  catch(err){if(mine===ticket)status(`Could not list ${experiment.name}. ${err.message}`,true);return;}
  if(mine!==ticket)return;
  SETS=sets;run=0;cur=0;sel=null;
  RUNS_INIT();CASES_INIT();drawRunWarn();
  if(!SETS.length)return status('This experiment holds no belief sets.');
  openChain();
}

async function enter(){
  $('login').hidden=true;$('app').hidden=false;
  $('who').textContent=client.user;
  experiments=await client.experiments();
  const box=$('experiment');box.textContent='';
  experiments.forEach((e,i)=>{
    const o=document.createElement('option');o.value=i;o.textContent=e.name;box.appendChild(o);
  });
  if(!experiments.length)return status('This account sees no lm-mindreader experiments.',true);
  await openExperiment(0);
}

function showLogin(message){
  $('app').hidden=true;$('login').hidden=false;
  $('loginerror').textContent=message||'';
  $('user').focus();
}

$('loginform').addEventListener('submit',async e=>{
  e.preventDefault();
  $('loginerror').textContent='Signing in…';
  try{
    client=await Mlflow.login($('user').value.trim(),$('password').value);
    $('password').value='';
    await enter();
  }catch(err){showLogin(err.message);}
});
$('logout').addEventListener('click',()=>{forgetLogin();location.reload();});
$('experiment').addEventListener('change',e=>openExperiment(Number(e.target.value)));
$('find').addEventListener('input',e=>{filter=e.target.value;CASES_INIT();});
window.addEventListener('resize',()=>{if(C()){drawChart();drawS();}});

async function start(){
  GRAN_INIT();
  client=Mlflow.restore();
  if(!client)return showLogin();
  try{await enter();}
  catch(err){
    if(err.status===401)forgetLogin();
    showLogin(err.message);
  }
}

start();
