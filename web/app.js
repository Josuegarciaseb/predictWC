/* ===================================================================
   PredictWC — lógica del dashboard
   Banderas reales vía flagcdn.com · Chart.js · PapaParse
   =================================================================== */

/* nombre (en inglés, como en el CSV) -> código ISO 3166-1 alpha-2
   (England/Scotland/Wales usan los subcódigos de flagcdn) */
const ISO = {
  Argentina:"ar", Spain:"es", France:"fr", England:"gb-eng", Colombia:"co",
  Brazil:"br", Portugal:"pt", Germany:"de", Netherlands:"nl", Morocco:"ma",
  Mexico:"mx", Japan:"jp", Norway:"no", Ecuador:"ec", Uruguay:"uy",
  Switzerland:"ch", Italy:"it", Croatia:"hr", Denmark:"dk", Belgium:"be",
  "United States":"us", Austria:"at", Australia:"au", Senegal:"sn", Paraguay:"py",
  "Ivory Coast":"ci", "Curaçao":"cw", Sweden:"se", Canada:"ca", Qatar:"qa",
  "South Korea":"kr", Iran:"ir", "Saudi Arabia":"sa", "South Africa":"za",
  Tunisia:"tn", Ghana:"gh", Egypt:"eg", Nigeria:"ng", Algeria:"dz",
  Cameroon:"cm", Panama:"pa", "Costa Rica":"cr", Jamaica:"jm", Honduras:"hn",
  Peru:"pe", Chile:"cl", Bolivia:"bo", Venezuela:"ve", Scotland:"gb-sct",
  Wales:"gb-wls", Poland:"pl", Serbia:"rs", Turkey:"tr", Ukraine:"ua",
  Greece:"gr", "Czech Republic":"cz", Hungary:"hu", "New Zealand":"nz",
  "Cape Verde":"cv", Jordan:"jo", Uzbekistan:"uz", "DR Congo":"cd",
  Iraq:"iq", Slovakia:"sk", Romania:"ro", Finland:"fi", Ireland:"ie"
};
const ES = {
  Spain:"España", France:"Francia", England:"Inglaterra", Brazil:"Brasil",
  Germany:"Alemania", Netherlands:"Países Bajos", "Ivory Coast":"Costa de Marfil",
  Sweden:"Suecia", Japan:"Japón", Norway:"Noruega", Italy:"Italia", Croatia:"Croacia",
  Denmark:"Dinamarca", Belgium:"Bélgica", "United States":"Estados Unidos",
  Switzerland:"Suiza", Morocco:"Marruecos", Mexico:"México"
};
const name = n => ES[n] || n;
const flagUrl = n => ISO[n] ? `https://flagcdn.com/${ISO[n]}.svg` : "";
const flag = (n, cls="flag") => {
  const u = flagUrl(n);
  return u ? `<img class="${cls}" src="${u}" alt="" loading="lazy" decoding="async">` : "";
};
const pct = x => (x * 100).toFixed(1).replace(".", ",") + "%";

/* tokens (espejo de styles.css) */
const C = {
  text:"#e8ecf2", muted:"#9aa6b6", grid:"rgba(37,46,59,.85)",
  accent:"#f5a524", blue:"#4b8bf4", gold:"#f5c542",
  home:"#4b8bf4", draw:"#6b7686", away:"#f5a524",
  line:["#4b8bf4","#f5a524","#3fb950","#a78bfa","#ec4899","#22b8cf","#f0506e","#e0b341"]
};
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
Chart.defaults.color = C.muted;
Chart.defaults.font.family = "'Archivo', sans-serif";
Chart.defaults.animation = REDUCED ? false : { duration: 600 };

const store = {};
const loadCSV = file => new Promise((res, rej) =>
  Papa.parse(`../outputs/${file}`, {
    download:true, header:true, dynamicTyping:true, skipEmptyLines:true,
    complete:r => res(r.data), error:rej
  }));
const loadJSON = file => fetch(`../outputs/${file}`).then(r => r.json());

function skeleton(id, tall){
  const cv = document.getElementById(id);
  const sk = document.createElement("div");
  sk.className = "skeleton sk-bar" + (tall ? " tall" : "");
  sk.dataset.for = id; cv.parentElement.appendChild(sk); cv.style.display = "none";
}
const unskeleton = id => {
  document.querySelectorAll(`.skeleton[data-for="${id}"]`).forEach(e=>e.remove());
  document.getElementById(id).style.display = "";
};

async function init(){
  skeleton("champChart"); skeleton("eloChart", true); skeleton("roundsChart", true);
  try{
    const [elo, mc, f4, matrices] = await Promise.all([
      loadCSV("ranking_elo_actual.csv"),
      loadCSV("predicciones_fase5_montecarlo.csv"),
      loadCSV("predicciones_fase4_stacking_ensemble.csv"),
      loadJSON("matrices_marcador.json").catch(() => ({})),
    ]);
    store.elo = elo; store.mc = mc; store.f4 = f4; store.matrices = matrices;
    renderChamp(); renderPodium(); renderElo(); renderRounds(); renderMatches();
    document.getElementById("loadStatus").textContent =
      `${mc.length} selecciones · ${f4.length} partidos · ${elo.length} equipos en el ranking Elo`;
  }catch(e){
    ["champChart","eloChart","roundsChart"].forEach(unskeleton);
    document.getElementById("loadStatus").innerHTML =
      `<div class="error-box">No se pudieron cargar los CSV (¿abriste el archivo con <code>file://</code>?). Sírvelo con un servidor local:<br>
       <code>py -m http.server 8000</code> &nbsp;·&nbsp; abre <code>http://localhost:8000/web/</code> &nbsp;·&nbsp; o ejecuta <code>web\\start.bat</code></div>`;
    console.error(e);
  }
}

/* ---- TABS ---- */
const tabs = [...document.querySelectorAll(".navlink")];
function selectTab(tab){
  tabs.forEach(t=>{
    const on = t === tab;
    t.setAttribute("aria-selected", on);
    const p = document.getElementById(t.dataset.tab);
    p.classList.toggle("active", on); p.hidden = !on;
  });
}
tabs.forEach((t,i)=>{
  t.addEventListener("click", ()=>selectTab(t));
  t.addEventListener("keydown", e=>{
    if(e.key!=="ArrowRight" && e.key!=="ArrowLeft") return;
    e.preventDefault();
    const n = (i + (e.key==="ArrowRight"?1:-1) + tabs.length) % tabs.length;
    tabs[n].focus(); selectTab(tabs[n]);
  });
});

/* ---- CHAMPION ---- */
function renderChamp(){
  unskeleton("champChart");
  const top = [...store.mc].sort((a,b)=>b.prob_campeon-a.prob_campeon).slice(0,12);
  document.getElementById("champSummary").textContent =
    "Favorito: " + name(top[0].seleccion) + " con " + pct(top[0].prob_campeon) +
    ". Le siguen " + top.slice(1,4).map(d=>`${name(d.seleccion)} ${pct(d.prob_campeon)}`).join(", ") + ".";
  new Chart(document.getElementById("champChart"), {
    type:"bar",
    data:{ labels: top.map(d=>name(d.seleccion)),
      datasets:[{ data: top.map(d=>d.prob_campeon*100),
        backgroundColor: top.map((_,i)=> i===0?C.gold : C.blue), borderRadius:3, barThickness:17 }] },
    options:{ indexAxis:"y", responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false}, tooltip:{callbacks:{label:c=>c.parsed.x.toFixed(1)+"% campeón"}} },
      scales:{ x:{grid:{color:C.grid}, ticks:{callback:v=>v+"%"}}, y:{grid:{display:false}, ticks:{font:{weight:600}}} } }
  });
}

/* ---- PODIUM ---- */
function renderPodium(){
  const top = [...store.mc].sort((a,b)=>b.prob_campeon-a.prob_campeon).slice(0,5);
  const cls = ["g","s","b","",""], max = top[0].prob_campeon;
  document.getElementById("podium").innerHTML = top.map((d,i)=>`
    <div class="pod-row ${cls[i]}">
      <span class="pos">${i+1}</span>
      ${flag(d.seleccion)}
      <div style="flex:1">
        <div class="pod-name">${name(d.seleccion)}</div>
        <div class="pod-bartrack"><i style="width:${d.prob_campeon/max*100}%"></i></div>
      </div>
      <span class="pod-prob">${pct(d.prob_campeon)}</span>
    </div>`).join("");
}

/* ---- ELO ---- */
function renderElo(){
  unskeleton("eloChart");
  const top = store.elo.slice(0,15);
  new Chart(document.getElementById("eloChart"), {
    type:"bar",
    data:{ labels: top.map(d=>name(d.seleccion)),
      datasets:[{ data: top.map(d=>d.elo), backgroundColor:C.blue, borderRadius:3, barThickness:17 }] },
    options:{ indexAxis:"y", responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false}, tooltip:{callbacks:{label:c=>"Elo "+Math.round(c.parsed.x)}} },
      scales:{ x:{grid:{color:C.grid}, min:1800, suggestedMax:2200}, y:{grid:{display:false}, ticks:{font:{weight:600}}} } }
  });
  document.getElementById("eloTable").innerHTML = `
    <thead><tr><th class="r">#</th><th>Selección</th><th class="r">Elo</th></tr></thead>
    <tbody>${store.elo.map(d=>`
      <tr><td class="r rk">${d.rank}</td>
      <td>${flag(d.seleccion)}${name(d.seleccion)}</td>
      <td class="r val">${Math.round(d.elo)}</td></tr>`).join("")}
    </tbody>`;
}

/* ---- ROUNDS ---- */
let roundsChart, selected = [];
function renderRounds(){
  const top = [...store.mc].sort((a,b)=>b.prob_campeon-a.prob_campeon);
  selected = top.slice(0,5).map(d=>d.seleccion);
  document.getElementById("teamChips").innerHTML = top.slice(0,16).map(d=>{
    const on = selected.includes(d.seleccion);
    const col = on ? C.line[selected.indexOf(d.seleccion) % C.line.length] : "transparent";
    return `<button class="chip" aria-pressed="${on}" data-team="${d.seleccion}">
      <span class="swatch" style="background:${col}"></span>${flag(d.seleccion)}${name(d.seleccion)}</button>`;
  }).join("");
  document.querySelectorAll(".chip").forEach(c => c.onclick = () => {
    const t = c.dataset.team;
    selected = selected.includes(t) ? selected.filter(x=>x!==t) : [...selected, t];
    refreshChips(); drawRounds();
  });
  drawRounds();
}
function refreshChips(){
  document.querySelectorAll(".chip").forEach(c=>{
    const t = c.dataset.team, on = selected.includes(t);
    c.setAttribute("aria-pressed", on);
    c.querySelector(".swatch").style.background = on ? C.line[selected.indexOf(t) % C.line.length] : "transparent";
  });
}
function drawRounds(){
  const labels = ["Octavos 32","Octavos 16","Cuartos","Semis","Final","Campeón"];
  const keys = ["prob_octavos_32","prob_octavos_16","prob_cuartos","prob_semis","prob_final","prob_campeon"];
  const ds = selected.map((t,i)=>{
    const row = store.mc.find(d=>d.seleccion===t), col = C.line[i % C.line.length];
    return { label:name(t), data:keys.map(k=>row[k]*100),
      borderColor:col, backgroundColor:col+"22", tension:.3, borderWidth:2,
      pointRadius:3, pointHoverRadius:6, pointBackgroundColor:col, fill:false };
  });
  if(roundsChart) roundsChart.destroy();
  unskeleton("roundsChart");
  roundsChart = new Chart(document.getElementById("roundsChart"), {
    type:"line", data:{labels, datasets:ds},
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{position:"bottom", labels:{boxWidth:12, padding:14, usePointStyle:true}},
        tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.parsed.y.toFixed(1)}%`}} },
      scales:{ y:{grid:{color:C.grid}, min:0, max:100, ticks:{callback:v=>v+"%"}}, x:{grid:{color:C.grid}} } }
  });
}

/* ---- MATCHES ---- */
function matchCard(m){
  const l=m.prob_local_ensemble*100, d=m.prob_empate_ensemble*100, v=m.prob_visita_ensemble*100;
  const top3 = (m.top3_marcadores||"").split("|").map(s=>s.trim()).join("   ·   ");
  const key = `${m.local}|${m.visitante}`;
  const hasMx = store.matrices && store.matrices[key];
  return `<article class="match${hasMx?" clickable":""}" data-q="${(m.local+' '+m.visitante).toLowerCase()}"
      ${hasMx?`data-key="${key}" role="button" tabindex="0" aria-label="Ver distribución de probabilidades de ${name(m.local)} vs ${name(m.visitante)}"`:""}>
    <div class="match-date"><svg class="ic" viewBox="0 0 24 24" aria-hidden="true"><use href="#i-calendar"/></svg>${m.fecha}</div>
    <div class="match-teams">
      <div class="mt">${flag(m.local)}<span class="nm">${name(m.local)}</span></div>
      <span class="score-pred" aria-label="marcador más probable ${m.marcador_mas_probable}">${m.marcador_mas_probable}</span>
      <div class="mt away">${flag(m.visitante)}<span class="nm">${name(m.visitante)}</span></div>
    </div>
    <div class="prob-bar" role="img" aria-label="Local ${l.toFixed(0)}%, empate ${d.toFixed(0)}%, visita ${v.toFixed(0)}%">
      <div class="seg l" style="width:${l}%"></div><div class="seg d" style="width:${d}%"></div><div class="seg v" style="width:${v}%"></div>
    </div>
    <div class="prob-legend">
      <span><i style="background:var(--home)"></i>Local <b>${l.toFixed(0)}%</b></span>
      <span><i style="background:var(--draw)"></i>Empate <b>${d.toFixed(0)}%</b></span>
      <span><i style="background:var(--away)"></i>Visita <b>${v.toFixed(0)}%</b></span>
    </div>
    <div class="match-top3"><b>Marcadores más probables</b>${top3}</div>
    ${hasMx?`<div class="match-cta">Ver distribución completa →</div>`:""}
  </article>`;
}
function renderMatches(){
  const list = document.getElementById("matchList");
  list.innerHTML = store.f4.map(matchCard).join("");
  document.getElementById("matchSearch").addEventListener("input", e=>{
    const q = e.target.value.toLowerCase().trim();
    list.querySelectorAll(".match").forEach(c => c.style.display = c.dataset.q.includes(q) ? "" : "none");
  });
  const open = el => { const m = store.f4.find(x=>`${x.local}|${x.visitante}`===el.dataset.key); if(m) openHeatmap(m); };
  list.querySelectorAll(".match.clickable").forEach(c=>{
    c.addEventListener("click", ()=>open(c));
    c.addEventListener("keydown", e=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); open(c); } });
  });
}

/* ---- HEATMAP MODAL ---- */
function heatColor(t){ // t en [0,1] -> blanco a rojo (como el ejemplo)
  const r=255, g=Math.round(255-216*t), b=Math.round(255-216*t);
  return `rgb(${r},${g},${b})`;
}
function openHeatmap(m){
  const data = store.matrices[`${m.local}|${m.visitante}`];
  if(!data) return;
  const M = data.matriz, n = M.length;
  let vmax = 0; M.forEach(r=>r.forEach(p=>{ if(p>vmax) vmax=p; }));

  let cells = "";
  // esquina + cabecera de columnas (goles de local)
  cells += `<div class="hm-corner">V \\ L</div>`;
  for(let j=0;j<n;j++) cells += `<div class="hm-head">${j}</div>`;
  // filas: goles de visita
  for(let i=0;i<n;i++){
    cells += `<div class="hm-head">${i}</div>`;
    for(let j=0;j<n;j++){
      const p = M[j][i];               // M[local][visita]; fila=visita(i), col=local(j)
      const t = vmax ? p/vmax : 0;
      const txt = t>0.6 ? "#fff" : "#1a1a1a";
      cells += `<div class="hm-cell" style="background:${heatColor(t)};color:${txt}">${(p*100).toFixed(1)}<span>%</span></div>`;
    }
  }

  const ov = document.getElementById("hmOverlay");
  ov.querySelector(".hm-title").innerHTML =
    `${flag(m.local)} ${name(m.local)} <span class="hm-vs">vs</span> ${name(m.visitante)} ${flag(m.visitante)}`;
  ov.querySelector(".hm-sub").textContent = `${m.fecha} · Mundial 2026 · modelo Dixon-Coles`;
  ov.querySelector(".hm-grid").style.gridTemplateColumns = `auto repeat(${n}, 1fr)`;
  ov.querySelector(".hm-grid").innerHTML = cells;
  ov.querySelector(".hm-foot").innerHTML =
    `<span>Marcador más probable <b>${data.marcador}</b> (${(data.prob_marcador*100).toFixed(1)}%)</span>
     <span><i style="background:var(--home)"></i>Gana ${name(m.local)} <b>${(data.prob_local*100).toFixed(0)}%</b></span>
     <span><i style="background:var(--draw)"></i>Empate <b>${(data.prob_empate*100).toFixed(0)}%</b></span>
     <span><i style="background:var(--away)"></i>Gana ${name(m.visitante)} <b>${(data.prob_visita*100).toFixed(0)}%</b></span>`;
  ov.querySelector(".hm-axis-x").textContent = `Goles de Local — ${name(m.local)}`;
  ov.querySelector(".hm-axis-y").textContent = `Goles de Visita — ${name(m.visitante)}`;
  ov.hidden = false;
  document.body.style.overflow = "hidden";
  ov.querySelector(".hm-close").focus();
}
function closeHeatmap(){
  document.getElementById("hmOverlay").hidden = true;
  document.body.style.overflow = "";
}
document.getElementById("hmOverlay").addEventListener("click", e=>{
  if(e.target.id==="hmOverlay" || e.target.closest(".hm-close")) closeHeatmap();
});
document.addEventListener("keydown", e=>{
  if(e.key==="Escape" && !document.getElementById("hmOverlay").hidden) closeHeatmap();
});

init();
