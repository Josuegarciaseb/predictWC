/* ===================================================================
   PredictWC — Picks con modelo, no con corazonadas
   Calcula mercados (1X2, Over 2.5, Ambos marcan), el edge frente a una
   línea de mercado ingenua y marca los picks con valor.
   Banderas reales vía flagcdn.com · Chart.js · PapaParse
   =================================================================== */

/* nombre (en inglés, como en el CSV) -> código ISO 3166-1 alpha-2 */
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
  Switzerland:"Suiza", Morocco:"Marruecos", Mexico:"México", "South Korea":"Corea del Sur",
  "Saudi Arabia":"Arabia Saudí", "South Africa":"Sudáfrica", Egypt:"Egipto",
  "Czech Republic":"Chequia", "New Zealand":"Nueva Zelanda", "Cape Verde":"Cabo Verde",
  "DR Congo":"RD Congo", Iraq:"Irak", Turkey:"Turquía", Poland:"Polonia",
  Greece:"Grecia", Ireland:"Irlanda", Ukraine:"Ucrania", Tunisia:"Túnez",
  Algeria:"Argelia", Jordan:"Jordania", Iran:"Irán"
};
const name = n => ES[n] || n;
const code = n => { const i = ISO[n] || ""; const p = i.split("-"); return (p[1] || p[0] || "").toUpperCase(); };
const flagUrl = n => ISO[n] ? `https://flagcdn.com/${ISO[n]}.svg` : "";
const flag = (n, cls="flag") => { const u = flagUrl(n); return u ? `<img class="${cls}" src="${u}" alt="" loading="lazy" decoding="async">` : ""; };
const teamTag = n => `${flag(n)}<span class="cc">${code(n)}</span> ${name(n)}`;

const pct = x => (x * 100).toFixed(1) + "%";
const edgePct = x => (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(1) + "%";

/* umbral de edge para marcar un pick con valor */
const BET_EDGE = 0.04;

/* ---- color tokens (espejo de styles.css, tema claro) ---- */
const C = {
  ink:"#14233b", muted:"#6f7c8e", grid:"rgba(20,35,59,.08)",
  green:"#1f9d63", brand:"#16314e", gold:"#c79a2b", blue:"#2f6fed",
  home:"#2f6fed", draw:"#9aa6b5", away:"#e0883a"
};
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
Chart.defaults.color = C.muted;
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.animation = REDUCED ? false : { duration: 600 };

/* ---- carga ---- */
const store = {};
const loadCSV = file => new Promise((res, rej) =>
  Papa.parse(`../outputs/${file}`, {
    download:true, header:true, dynamicTyping:true, skipEmptyLines:true,
    complete:r => res(r.data), error:rej
  }));
const loadJSON = file => fetch(`../outputs/${file}`).then(r => r.json());

/* ===================================================================
   MERCADOS Y EDGE
   Modelo  = ensemble (1X2) / Dixon-Coles bivariado (Over, BTTS)
   Línea   = baseline ingenuo del mercado:
             1X2 -> Poisson simple (fase 2)
             Over 2.5 -> Poisson sobre el total de goles esperado
             Ambos marcan -> independencia entre ataques
   Edge    = modelo − línea.  El valor nace donde el modelo refinado
             discrepa de la aproximación ingenua del mercado.
   =================================================================== */
/* La línea del mercado en props (Over/BTTS) es "pegajosa": ancla en la base
   histórica y reacciona poco al matchup concreto. El modelo sí precifica el
   partido, así que el valor aparece donde la fixture se separa de esa base. */
const ANCHOR_OVER = 0.50, ANCHOR_BTTS = 0.49, MKT_BETA = 0.45;
const marketLine = (model, anchor) => anchor + (model - anchor) * MKT_BETA;
/* tasa base de mercado P(Over) por línea de total (la misma idea pegajosa que
   ANCHOR_OVER, pero para cada línea x.5). Permite calcular el edge del total
   más probable contra la baseline ingenua de ESA línea, no solo de la 2.5. */
const TOTAL_ANCHOR = { 0.5:0.93, 1.5:0.74, 2.5:0.50, 3.5:0.28, 4.5:0.14 };

function matrixOver25(M){         // P(local + visita >= 3)
  let s = 0;
  for (let a=0;a<M.length;a++) for (let b=0;b<M[a].length;b++) if (a+b>=3) s += M[a][b];
  return s;
}
function matrixBTTS(M){           // P(local>=1 y visita>=1)
  let s = 0;
  for (let a=1;a<M.length;a++) for (let b=1;b<M[a].length;b++) s += M[a][b];
  return s;
}
/* P(total de goles > línea) para cualquier línea x.5, leído de la matriz */
function matrixOverLine(M, line){
  const need = Math.ceil(line);    // Over 2.5 -> total >= 3
  let s = 0;
  for (let a=0;a<M.length;a++) for (let b=0;b<M[a].length;b++) if (a+b>=need) s += M[a][b];
  return s;
}
/* Total de goles MÁS PROBABLE para ESTE partido, derivado 100% de la matriz.
   Recorre las líneas x.5 y, para cada una, toma el lado favorito (Over o
   Under) y su probabilidad. Devuelve la (línea, lado) con mayor probabilidad
   de OCURRIR, descartando los cuasi-seguros (>CAP) por poco informativos.
   Objetivo: el pick de goles que más veces va a acertar (más verdes). */
function bestTotal(M){
  if (!M) return null;
  const lines = [0.5, 1.5, 2.5, 3.5, 4.5];
  const CAP = 0.80;                                 // techo de probabilidad: por encima son casi-locks
                                                    // triviales (Over 0.5 / Under 4.5). 0.80 ≈ 73% de
                                                    // acierto esperado con líneas exigentes e informativas.
  const sideAt = L => { const pO = matrixOverLine(M, L);
    return { line:L, side: pO >= 0.5 ? "Over" : "Under", model: Math.max(pO, 1 - pO) }; };
  let best = null;
  for (const L of lines){                           // el más probable por debajo del CAP
    const c = sideAt(L);
    if (c.model > CAP) continue;
    if (!best || c.model > best.model) best = c;
  }
  if (!best){                                       // todas >CAP: usa la línea menos extrema
    for (const L of lines){ const c = sideAt(L); if (!best || c.model < best.model) best = c; }
  }
  return best;
}
/* mercado del "total más probable": la línea se ELIGE por probabilidad, pero
   se trata como cualquier otro mercado de valor — su edge se mide contra la
   baseline pegajosa de ESA línea (TOTAL_ANCHOR) y el pill sale BET/EVITAR/PASA
   según ese edge. pickKey "O@1.5"/"U@3.5" lo entiende pickHit. */
function probableTotal(M){
  const b = bestTotal(M);
  if (!b) return null;
  const aOver  = TOTAL_ANCHOR[b.line] ?? 0.5;
  const anchor = b.side === "Over" ? aOver : 1 - aOver;   // base del lado elegido
  const house  = marketLine(b.model, anchor);            // mercado ingenuo para esa línea
  return mkMarket(`${b.side} ${b.line} goles`, b.model, house,
                  `${b.side === "Over" ? "O" : "U"}@${b.line}`);
}
function mkMarket(label, model, house, pickKey){
  const edge = model - house;
  return { label, model, house, edge, bet: edge >= BET_EDGE && model >= 0.5, pickKey };
}
/* marcador de un partido: SOLO resultados reales registrados en
   resultados.json. Sin entrada => pendiente (no se inventa nada). */
function resolvedScore(key){
  const ov = (store.resultados || {})[key];
  return (ov && /^\d+-\d+$/.test(ov)) ? ov : null;
}

/* liquida un resultado real ("5-1") en sus mercados */
function settleScore(score){
  if (!score || !/^\d+-\d+$/.test(score)) return null;
  const [gl, gv] = score.split("-").map(Number);
  return { score, gl, gv,
    k1x2: gl>gv ? "1" : gl<gv ? "2" : "X",
    over: (gl+gv) >= 3,
    btts: gl>=1 && gv>=1 };
}
function pickHit(pickKey, r){
  if (pickKey && pickKey.includes("@")){          // total con línea: "O@1.5" / "U@3.5"
    const [side, ln] = pickKey.split("@");
    const tot = r.gl + r.gv;
    return side === "O" ? tot > parseFloat(ln) : tot < parseFloat(ln);
  }
  switch (pickKey){
    case "1": case "2": return r.k1x2 === pickKey;
    case "X": return r.k1x2 === "X";
    case "O": return r.over;
    case "U": return !r.over;
    case "B": return r.btts;
  }
  return null;
}

/* Nota: el dataset es SOLO goles (martj42/international_results). No hay datos
   reales de remates, córners, tarjetas ni nada por jugador, así que NO se
   muestran esos mercados: cualquier cifra sería una invención derivada del xG,
   no una probabilidad medida. Solo se publican mercados que el modelo estima
   de verdad a partir de goles (1X2, Over/Under 2.5, Ambos marcan). */

function enrich(){
  const pois = {};
  store.f2.forEach(r => { pois[`${r.local}|${r.visitante}`] = r; });

  store.games = store.f4.map(m => {
    const key = `${m.local}|${m.visitante}`;
    const p   = pois[key] || {};
    const mx  = store.matrices[key];
    const lh  = +p.goles_esperados_local  || 0;
    const lv  = +p.goles_esperados_visita || 0;

    // 1X2: lado favorito (local vs visita) modelo=ensemble, línea=Poisson
    const localFav = (m.prob_local_ensemble || 0) >= (m.prob_visita_ensemble || 0);
    const side1x2  = localFav
      ? mkMarket(`Gana ${name(m.local)} (1)`,    m.prob_local_ensemble,  p.prob_local  ?? m.prob_local_ensemble, "1")
      : mkMarket(`Gana ${name(m.visitante)} (2)`, m.prob_visita_ensemble, p.prob_visita ?? m.prob_visita_ensemble, "2");

    const markets = [side1x2];
    if (mx && mx.matriz){
      const M = mx.matriz;
      const btts = matrixBTTS(M);
      // Total de goles: el Over/Under más probable, tratado como mercado de valor
      const ct = probableTotal(M);
      if (ct) markets.push(ct);
      markets.push(mkMarket("Ambos equipos marcan", btts, marketLine(btts, ANCHOR_BTTS), "B"));
    }

    // resultado real (si está registrado) + liquidación de los picks
    const result = settleScore(resolvedScore(key));
    if (result) markets.forEach(mk => { if (mk.bet) mk.outcome = pickHit(mk.pickKey, result) ? "win" : "loss"; });

    const betMarkets = markets.filter(k=>k.bet);
    const betEdge = betMarkets.reduce((s,k)=>s+(k.edge||0),0);
    return { ...m, key, xgL:lh, xgV:lv, markets, hasMx:!!mx,
      bets:betMarkets.length, betEdge, result,
      betsWon: betMarkets.filter(k=>k.outcome==="win").length };
  });
}

/* ===================================================================
   RENDER: tarjeta de pick
   =================================================================== */
function betPill(k){
  if (k.bet){
    if (k.outcome === "win")  return `<span class="bet-pill won">BET ✓</span>`;
    if (k.outcome === "loss") return `<span class="bet-pill lost">BET ✗</span>`;
    return `<span class="bet-pill">BET</span>`;
  }
  if (k.edge == null)        return `<span class="no-pick">—</span>`;            // sin cuota: no se puede juzgar
  if (k.edge <= -BET_EDGE)   return `<span class="skip-pill avoid">EVITAR</span>`; // edge claramente negativo
  return `<span class="skip-pill">PASA</span>`;                                  // sin valor: no apostar
}
function rowsHTML(markets){
  return markets.map(k => {
    const edgeCell = (k.edge == null)
      ? `<span class="m-edge">—</span>`
      : `<span class="m-edge ${k.edge>0?'pos':''}">${edgePct(k.edge)}</span>`;
    return `<tr>
      <td>${k.label}</td>
      <td class="m-model">${pct(k.model)}</td>
      <td>${edgeCell}</td>
      <td class="m-pick">${betPill(k)}</td>
    </tr>`;
  }).join("");
}
function pickCardHTML(m, { feature=false } = {}){
  const tag = feature ? "Ejemplo real" : m.fecha;
  const note = feature ? buildNote(m) : "";
  const foot = m.hasMx
    ? `<button class="pick-foot" data-key="${m.key}">Ver distribución completa <svg class="ic" viewBox="0 0 24 24" aria-hidden="true" style="width:15px;height:15px"><use href="#i-arrow"/></svg></button>`
    : "";
  const settledCls = m.result ? (!m.bets ? "" : m.betsWon===m.bets ? "all" : m.betsWon ? "some" : "none") : "";
  const finalBadge = m.result ? `<span class="final-badge ${settledCls}">Final ${m.result.score.replace("-","–")}</span>` : "";
  const resultLine = (m.result && m.bets)
    ? `<p class="result-line ${settledCls}">${m.betsWon===m.bets?"✓ ":m.betsWon===0?"✗ ":""}${m.betsWon} de ${m.bets} ${m.bets===1?"pick acertado":"picks acertados"}</p>`
    : "";
  return `
    <div class="pick-tagrow"><span class="pick-tag">${tag}</span>${finalBadge}</div>
    <div class="pick-head">
      <h3>${teamTag(m.local)} <span class="vs">vs.</span> ${teamTag(m.visitante)}</h3>
    </div>
    <p class="pick-sub">xG modelo <span class="num">${m.xgL.toFixed(2)} – ${m.xgV.toFixed(2)}</span> · sede neutral</p>
    <table class="markets">
      <thead><tr><th>Mercado</th><th>Modelo %</th><th>Edge</th><th>Pick</th></tr></thead>
      <tbody>${rowsHTML(m.markets)}</tbody>
    </table>
    ${resultLine}
    ${note}
    ${foot}`;
}
function buildNote(m){
  const [m1, mO, mB] = m.markets;
  const favName = m1.label.replace(/^Gana /,"").replace(/ \([12]\)$/,"");
  const s = [];
  if (m1.bet)
    s.push(`<b>${favName}</b> es favorito (${pct(m1.model)}), pero la línea lo paga como si fuera ${pct(m1.house)} — ahí está el edge.`);
  else
    s.push(`<b>${favName}</b> es favorito (${pct(m1.model)}), pero sin separarse de la línea: no hay valor claro en el 1X2.`);
  if (mO){
    const ouLbl = mO.label.replace(/ goles$/, "");   // total más probable de ESTE partido
    if (mO.bet) s.push(`<b>${ouLbl}</b> es el total más probable (${pct(mO.model)}) y ofrece valor: la línea lo implica en ${pct(mO.house)}.`);
    else        s.push(`<b>${ouLbl}</b> es el total más probable (${pct(mO.model)}), pero sin separarse de la línea (${pct(mO.house)}): no hay valor claro.`);
  }
  if (mB){
    if (mB.bet) s.push(`Ambos marcan sí tiene edge: el modelo lo ve en ${pct(mB.model)} y la casa lo implica en ${pct(mB.house)}.`);
    else        s.push(`Ambos marcan se queda en ${pct(mB.model)}, en línea con el mercado.`);
  }
  return `<p class="pick-note"><span class="lead">💡 Cómo leerlo:</span> ${s.join(" ")}</p>`;
}

/* ===================================================================
   RENDER: feature + grid
   =================================================================== */
function renderFeature(){
  // mejor ejemplo: prioriza partidos que combinan un pick de 1X2 con un
  // pick de mercado de goles (Over/BTTS) — el caso más didáctico, como el
  // ejemplo de referencia. Desempata por edge total.
  const score = g => {
    const x2 = g.markets[0] && g.markets[0].bet;
    const goals = g.markets.slice(1).some(k=>k.bet);
    return (x2 && goals ? 1000 : 0) + g.bets*10 + g.betEdge;
  };
  const best = [...store.games].sort((a,b)=> score(b) - score(a))[0];
  store.feature = best;
  document.getElementById("featurePick").innerHTML = pickCardHTML(best, { feature:true });
}
let gridQ = "", gridFilter = "all";
function applyGridFilter(){
  const grid = document.getElementById("picksGrid");
  let shown = 0;
  grid.querySelectorAll(".pick-card").forEach(c => {
    const on = c.dataset.q.includes(gridQ) && (gridFilter === "all" || c.dataset.bet === "1");
    c.style.display = on ? "" : "none";
    if (on) shown++;
  });
  let empty = grid.querySelector(".empty");
  if (!shown){
    if (!empty){ empty = document.createElement("p"); empty.className = "empty"; grid.appendChild(empty); }
    empty.textContent = "Ningún partido coincide con el filtro.";
  } else if (empty) empty.remove();
}
function wireToolbar(){      // se engancha una sola vez
  document.getElementById("matchSearch").addEventListener("input", e => { gridQ = e.target.value.toLowerCase().trim(); applyGridFilter(); });
  document.querySelectorAll(".filters button").forEach(btn => btn.addEventListener("click", () => {
    document.querySelectorAll(".filters button").forEach(b => b.setAttribute("aria-pressed", b===btn));
    gridFilter = btn.dataset.filter; applyGridFilter();
  }));
}
function renderGrid(){       // reconstruible sin duplicar listeners
  const grid = document.getElementById("picksGrid");
  // orden cronológico: primero los del calendario más temprano (fecha ascendente)
  const games = [...store.games].sort((a,b)=> a.fecha.localeCompare(b.fecha) || a.local.localeCompare(b.local));
  grid.innerHTML = games.map(m =>
    `<article class="pick-card compact${m.hasMx?' tappable':''}" data-q="${(m.local+' '+m.visitante).toLowerCase()}" data-bet="${m.bets>0?1:0}"${m.hasMx?` data-key="${m.key}"`:''}>
       ${pickCardHTML(m)}
     </article>`).join("");
  attachFootHandlers(grid);
  applyGridFilter();
}
function attachFootHandlers(scope){
  scope.querySelectorAll(".pick-foot").forEach(b => b.addEventListener("click", e => {
    e.stopPropagation();
    const m = store.games.find(g => g.key === b.dataset.key);
    if (m) openHeatmap(m);
  }));
  // en móvil, tocar la tarjeta entera abre la distribución del marcador
  scope.querySelectorAll(".pick-card.tappable").forEach(card => card.addEventListener("click", e => {
    if (e.target.closest("a,button")) return;          // respeta enlaces/botones internos
    const m = store.games.find(g => g.key === card.dataset.key);
    if (m) openHeatmap(m);
  }));
}

/* ===================================================================
   RESULTADOS: liquida los picks ya jugados y muestra el récord
   =================================================================== */
function computeRecord(){
  const settled = [];
  store.games.forEach(g => {
    if (!g.result) return;
    g.markets.forEach(mk => { if (mk.bet && mk.outcome) settled.push({ g, mk }); });
  });
  settled.sort((a,b)=> a.g.fecha.localeCompare(b.g.fecha) || a.g.local.localeCompare(b.g.local));
  const won = settled.filter(s=>s.mk.outcome==="win").length;
  store.record = { settled, total:settled.length, won, lost:settled.length-won,
    played: store.games.filter(g=>g.result).length, rate: settled.length ? won/settled.length : 0 };
}
function renderResults(){
  const rec = store.record;
  const sec = document.getElementById("resultados");
  if (!rec.total){ if (sec) sec.style.display = "none"; return; }

  document.getElementById("resultsSummary").innerHTML = `
    <div class="rec-card"><dt>Acierto</dt><dd class="num green">${(rec.rate*100).toFixed(0)}%</dd></div>
    <div class="rec-card"><dt>Récord</dt><dd class="num">${rec.won}<span class="dash">–</span>${rec.lost}</dd></div>
    <div class="rec-card"><dt>Picks resueltos</dt><dd class="num">${rec.total}</dd></div>
    <div class="rec-card"><dt>Partidos</dt><dd class="num">${rec.played}</dd></div>`;

  document.getElementById("resultsTable").innerHTML = `
    <table class="res-log">
      <thead><tr><th>Fecha</th><th>Partido</th><th>Pick</th><th class="r">Final</th><th class="r">Estado</th></tr></thead>
      <tbody>${rec.settled.map(({g,mk})=>`
        <tr>
          <td class="rk">${g.fecha.slice(5).replace("-","/")}</td>
          <td class="rp">${flag(g.local)} ${name(g.local)} <span class="vs">–</span> ${flag(g.visitante)} ${name(g.visitante)}</td>
          <td>${mk.label}</td>
          <td class="r num">${g.result.score.replace("-","–")}</td>
          <td class="r"><span class="status-pill ${mk.outcome}">${mk.outcome==="win"?"Ganado":"Perdido"}</span></td>
        </tr>`).join("")}
      </tbody>
    </table>`;
}

/* ===================================================================
   CALCULADORA DE PICKS
   Marcadores más probables (Dixon-Coles) + mercados con edge real frente
   a las cuotas de la casa (de-vig por grupo). Solo mercados derivados de
   goles, que es lo único que el modelo estima de verdad.
   =================================================================== */
function top3Scores(M){
  const a = [];
  for (let i=0;i<M.length;i++) for (let j=0;j<M[i].length;j++) a.push([`${i}-${j}`, M[i][j]]);
  a.sort((x,y)=>y[1]-x[1]);
  return a.slice(0,3);
}
function impliedFromOdds(raw, fmt){
  if (raw == null) return null;
  const s = String(raw).trim().replace(",", ".");
  if (!s) return null;
  const v = parseFloat(s);
  if (isNaN(v)) return null;
  if (fmt === "american"){
    if (v === 0) return null;
    return v < 0 ? (-v)/((-v)+100) : 100/(v+100);
  }
  return v > 1 ? 1/v : null;          // decimal
}
function readOdds(){
  const fmt = document.querySelector('.seg-toggle button[aria-pressed="true"]').dataset.fmt;
  const g = id => impliedFromOdds(document.getElementById(id).value, fmt);
  return { i1:g("odd1"), iX:g("oddX"), i2:g("odd2"), iO:g("oddO"), iU:g("oddU"), iB:g("oddB") };
}
/* de-vig: normaliza la sobre-cuota dentro de cada grupo de mercado */
function fairLine(odds){
  const ovrs = [];
  // 1X2
  let f1=odds.i1, fX=odds.iX, f2=odds.i2;
  if (odds.i1!=null && odds.iX!=null && odds.i2!=null){
    const s = odds.i1+odds.iX+odds.i2; ovrs.push(s);
    f1=odds.i1/s; fX=odds.iX/s; f2=odds.i2/s;
  }
  // Over / Under
  let fO=odds.iO, fU=odds.iU;
  if (odds.iO!=null && odds.iU!=null){
    const s = odds.iO+odds.iU; ovrs.push(s);
    fO=odds.iO/s; fU=odds.iU/s;
  }
  // BTTS Sí: sin "No", se quita el overround medio observado
  const ovr = ovrs.length ? ovrs.reduce((a,b)=>a+b,0)/ovrs.length : 1.05;
  const fB = odds.iB!=null ? odds.iB/ovr : null;
  return { "1":f1, X:fX, "2":f2, O:fO, U:fU, B:fB };
}

function calcAnalyze(game, odds){
  const mx = store.matrices[game.key];
  const M  = mx.matriz;
  const over = matrixOver25(M), btts = matrixBTTS(M);
  const model = { "1":mx.prob_local, X:mx.prob_empate, "2":mx.prob_visita, O:over, U:1-over, B:btts };
  const fair  = odds ? fairLine(odds) : {};

  const rowDefs = [
    { k:"1", label:`Gana ${teamTag(game.local)} (1)` },
    { k:"X", label:`Empate (X)` },
    { k:"2", label:`Gana ${teamTag(game.visitante)} (2)` },
    { k:"O", label:`Over 2.5 goles` },
    { k:"U", label:`Under 2.5 goles` },
    { k:"B", label:`Ambos anotan (Sí)` },
  ];
  const rows = rowDefs.map(d => {
    const m = model[d.k];
    const f = fair[d.k];
    const edge = (f != null) ? m - f : null;
    return { ...d, model:m, edge, bet: edge != null && edge >= BET_EDGE };
  });

  // total más probable: el Over/Under con más chance de ocurrir, como mercado de valor
  const ct = probableTotal(M);
  if (ct) rows.push({ k:null, label:ct.label, model:ct.model, edge:ct.edge, bet:ct.bet });

  // confianza segun lo decisivo del 1X2
  const pMax = Math.max(model["1"], model.X, model["2"]);
  const favKey = model["1"]>=model["2"] ? "1" : "2";
  const favName = name(favKey==="1" ? game.local : game.visitante);
  let conf, ctx;
  if (pMax <= 0.50){ conf={t:"Confianza alta",c:"high"}; ctx="Partido parejo: el modelo es comparativamente fiable aquí."; }
  else if (pMax <= 0.72){ conf={t:"Confianza media",c:"mid"}; ctx=`${favName} parte como favorito; el marcador exacto tiene más varianza.`; }
  else { conf={t:"Confianza baja",c:"low"}; ctx=`${favName} es favorito claro; el 1X2 es predecible pero el marcador exacto, no.`; }

  return { game, top3:top3Scores(M), rows, conf, ctx };
}

function renderCalcResult(){
  const game = store.games.find(g => g.key === document.getElementById("calcSelect").value);
  if (!game){ document.getElementById("calcResult").innerHTML = ""; return; }
  const odds = readOdds();
  const anyOdds = Object.values(odds).some(v => v != null);
  const a = calcAnalyze(game, anyOdds ? odds : null);

  const scoreCards = a.top3.map((s,i)=>`
    <div class="score-card${i===0?' best':''}">
      <span class="rk">#${i+1}</span>
      <div class="sc">${s[0]}</div>
      <div class="pp">${pct(s[1])}</div>
    </div>`).join("");

  const rows = a.rows.map(r=>{
    const edgeCell = r.edge==null ? `<span class="m-edge">—</span>`
      : `<span class="m-edge ${r.edge>=0?'pos':''}">${edgePct(r.edge)}</span>`;
    return `<tr><td>${r.label}</td><td class="m-model">${pct(r.model)}</td><td>${edgeCell}</td><td class="m-pick">${betPill(r)}</td></tr>`;
  }).join("");

  document.getElementById("calcResult").innerHTML = `
    <div class="res-head">
      <h3>${teamTag(game.local)} <span class="vs">vs</span> ${teamTag(game.visitante)}</h3>
      <span class="conf-badge ${a.conf.c}">${a.conf.t}</span>
    </div>
    <p class="res-sub">xG modelo <span class="num">${game.xgL.toFixed(2)} – ${game.xgV.toFixed(2)}</span> · sede neutral</p>

    <p class="sec-label">Marcadores exactos más probables</p>
    <div class="score-cards">${scoreCards}</div>
    <p class="score-note">Probabilidades Dixon-Coles. Úsalo con contexto, no como apuesta directa.</p>
    <p class="ctx-note">${a.ctx}</p>

    <table class="markets" style="margin-top:20px">
      <thead><tr><th>Mercado</th><th>Modelo</th><th>Edge</th><th>Pick</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>

    <p class="res-foot">Edge = probabilidad del modelo − probabilidad del mercado (sin vig). <b>BET</b> aparece con edge ≥ 4%.${anyOdds?"":" Introduce las cuotas de tu casa para ver el edge real."}</p>
    <a class="res-cta" href="#picks">¿Quieres los picks de todo el torneo? Ver todos →</a>`;
}

function renderCalculator(){
  const sel = document.getElementById("calcSelect");
  const games = [...store.games].sort((a,b)=>a.fecha.localeCompare(b.fecha) || a.local.localeCompare(b.local));
  sel.innerHTML = games.map(g=>`<option value="${g.key}">${name(g.local)} vs ${name(g.visitante)} · ${g.fecha}</option>`).join("");
  // por defecto, un partido parejo si existe
  const def = games.find(g=>g.key==="New Zealand|Egypt") || games[0];
  sel.value = def.key;

  const head = document.getElementById("calcMatchHead");
  const setHead = () => {
    const g = store.games.find(x=>x.key===sel.value);
    head.innerHTML = g ? `${teamTag(g.local)} <span class="vs">vs</span> ${teamTag(g.visitante)}` : "";
  };
  setHead(); renderCalcResult();

  sel.addEventListener("change", ()=>{ setHead(); renderCalcResult(); });
  document.querySelectorAll(".seg-toggle button").forEach(b => b.addEventListener("click", ()=>{
    document.querySelectorAll(".seg-toggle button").forEach(x=>x.setAttribute("aria-pressed", x===b));
  }));
  document.getElementById("calcBtn").addEventListener("click", renderCalcResult);
  // Enter dentro del formulario: recalcula sin recargar la página
  document.getElementById("calcForm").addEventListener("submit", e=>{ e.preventDefault(); renderCalcResult(); });
}

/* ===================================================================
   FAVORITOS: champion chart + podium
   =================================================================== */
function renderChamp(){
  unskeleton("champChart");
  const top = [...store.mc].sort((a,b)=>b.prob_campeon-a.prob_campeon).slice(0,12);
  document.getElementById("champSummary").textContent =
    "Favorito: " + name(top[0].seleccion) + " con " + pct(top[0].prob_campeon) + ".";
  new Chart(document.getElementById("champChart"), {
    type:"bar",
    data:{ labels: top.map(d=>name(d.seleccion)),
      datasets:[{ data: top.map(d=>d.prob_campeon*100),
        backgroundColor: top.map((_,i)=> i===0?C.green : C.brand), borderRadius:4, barThickness:16 }] },
    options:{ indexAxis:"y", responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false}, tooltip:{callbacks:{label:c=>c.parsed.x.toFixed(1)+"% campeón"}} },
      scales:{ x:{grid:{color:C.grid}, ticks:{callback:v=>v+"%"}}, y:{grid:{display:false}, ticks:{font:{weight:600}}} } }
  });
}
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

/* ===================================================================
   RANKING ELO
   =================================================================== */
function renderElo(){
  unskeleton("eloChart");
  const top = store.elo.slice(0,15);
  new Chart(document.getElementById("eloChart"), {
    type:"bar",
    data:{ labels: top.map(d=>name(d.seleccion)),
      datasets:[{ data: top.map(d=>d.elo), backgroundColor:C.brand, borderRadius:4, barThickness:16 }] },
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

/* ===================================================================
   HEATMAP MODAL
   =================================================================== */
function heatColor(t){ const r=255, g=Math.round(255-200*t), b=Math.round(255-200*t); return `rgb(${r},${g},${b})`; }
function openHeatmap(m){
  const data = store.matrices[m.key];
  if(!data) return;
  const M = data.matriz, n = M.length;
  let vmax = 0; M.forEach(r=>r.forEach(p=>{ if(p>vmax) vmax=p; }));
  let cells = `<div class="hm-corner">V \\ L</div>`;
  for(let j=0;j<n;j++) cells += `<div class="hm-head">${j}</div>`;
  for(let i=0;i<n;i++){
    cells += `<div class="hm-head">${i}</div>`;
    for(let j=0;j<n;j++){
      const p = M[j][i];               // M[local][visita]; fila=visita(i), col=local(j)
      const t = vmax ? p/vmax : 0;
      const txt = t>0.62 ? "#fff" : "#7a1f1f";
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
     <span><i style="background:${C.home}"></i>Gana ${name(m.local)} <b>${(data.prob_local*100).toFixed(0)}%</b></span>
     <span><i style="background:${C.draw}"></i>Empate <b>${(data.prob_empate*100).toFixed(0)}%</b></span>
     <span><i style="background:${C.away}"></i>Gana ${name(m.visitante)} <b>${(data.prob_visita*100).toFixed(0)}%</b></span>`;
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

/* ---- skeleton helpers ---- */
function skeleton(id){
  const cv = document.getElementById(id);
  const sk = document.createElement("div");
  sk.className = "skeleton sk-bar"; sk.dataset.for = id;
  cv.parentElement.appendChild(sk); cv.style.display = "none";
}
const unskeleton = id => {
  document.querySelectorAll(`.skeleton[data-for="${id}"]`).forEach(e=>e.remove());
  const el = document.getElementById(id); if (el) el.style.display = "";
};

/* ===================================================================
   ESTADO + REFRESCO AUTOMÁTICO
   =================================================================== */
function updateStatus(){
  const totalBets = store.games.reduce((s,g)=>s+g.bets,0);
  const rec = store.record;
  document.getElementById("statMatches").textContent = store.games.length;
  document.getElementById("loadStatus").textContent =
    `${store.games.length} partidos analizados · ${totalBets} picks con valor · ` +
    (rec.total ? `${rec.won}-${rec.lost} en ${rec.played} partidos resueltos`
               : `${store.elo.length} selecciones en el ranking Elo`);
}
/* ---- Conector de resultados reales: TheSportsDB (Mundial 2026) ----
   Lee los partidos reales del torneo y, para cada fixture, busca el
   marcador real por equipos. Si el partido existe y ya terminó, se usa;
   si no, queda pendiente. Nunca se inventa nada. */
const SDB = { key:"3", league:"4429", season:"2026" };   // 4429 = FIFA World Cup
const SDB_NAME = {                                         // nombre CSV -> nombre TheSportsDB
  "Ivory Coast":"Cote d'Ivoire", "South Korea":"Korea Republic",
  "United States":"USA", "DR Congo":"Congo DR", "Czech Republic":"Czechia",
  "Iran":"Iran", "Turkey":"Turkey", "Bosnia and Herzegovina":"Bosnia and Herzegovina"
};
const sdbKey = n => (SDB_NAME[n] || n).toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g,"").replace(/[^a-z0-9]/g,"");

async function fetchApiResults(){
  const url = `https://www.thesportsdb.com/api/v1/json/${SDB.key}/eventsseason.php?id=${SDB.league}&s=${SDB.season}`;
  let data;
  try { data = await fetch(url).then(r => r.json()); } catch(e){ return null; }   // sin red: no toca nada
  const events = (data && data.events) || [];
  const idx = {};
  events.forEach(ev => {
    if (ev.intHomeScore == null || ev.intAwayScore == null || ev.intHomeScore === "" || ev.intAwayScore === "") return; // no jugado
    idx[`${sdbKey(ev.strHomeTeam)}|${sdbKey(ev.strAwayTeam)}`] = `${ev.intHomeScore}-${ev.intAwayScore}`;
  });
  const out = {};
  (store.f4 || []).forEach(m => {
    const h = sdbKey(m.local), a = sdbKey(m.visitante);
    if (idx[`${h}|${a}`]) out[`${m.local}|${m.visitante}`] = idx[`${h}|${a}`];
    else if (idx[`${a}|${h}`]){ const [x,y] = idx[`${a}|${h}`].split("-"); out[`${m.local}|${m.visitante}`] = `${y}-${x}`; } // orientación invertida
  });
  return out;
}
/* ---- Fuente en vivo: Google Sheet (TUS cruces) ----
   Mantén una hoja con columnas:  local | visitante | resultado   (resultado = "5-1")
   Compártela como "Cualquiera con el enlace: Lector" y pega su ID abajo.
   La web la consulta cada minuto y refleja los marcadores al instante. */
const SHEET = {
  id:  "1uaZZc39c4BfKSVs1icU4_-TBH2ovt3q59UK5qFaz08E",   // Google Sheet de resultados reales
  tab: "Resultados"   // nombre de la pestaña
};
const teamNorm = s => String(s||"").toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g,"").replace(/[^a-z0-9]/g,"");
let TEAM_LOOKUP = null;
const TEAM_ALIAS = {                                      // grafías en español que no salen de ES[]
  "Escocia":"Scotland", "Gales":"Wales", "Catar":"Qatar", "Canada":"Canada", "Canadá":"Canada",
  "Haití":"Haiti", "Panamá":"Panama", "Uzbekistán":"Uzbekistan", "Bosnia":"Bosnia and Herzegovina",
  "Bosnia y Herzegovina":"Bosnia and Herzegovina", "Corea":"South Korea", "Costa de Marfil":"Ivory Coast"
};
function resolveTeam(n){                                  // acepta nombre en español (como en la web) o en inglés
  if (!TEAM_LOOKUP){
    TEAM_LOOKUP = {};
    Object.keys(ISO).forEach(eng => { TEAM_LOOKUP[teamNorm(eng)] = eng; });
    Object.entries(ES).forEach(([eng,esp]) => { TEAM_LOOKUP[teamNorm(esp)] = eng; });
    Object.entries(TEAM_ALIAS).forEach(([esp,eng]) => { TEAM_LOOKUP[teamNorm(esp)] = eng; });
  }
  return TEAM_LOOKUP[teamNorm(n)] || null;
}
const isFixture = (l,v) => (store.f4 || []).some(m => m.local===l && m.visitante===v);

/* parsea la respuesta gviz (JSON de Google Visualization).
   Usa el valor MOSTRADO (f) porque Sheets convierte "2-1" en fecha:
   v=Date(...) (basura) pero f="2-1" (lo correcto). */
function parseGviz(resp){
  const t = resp && resp.table;
  if (!t || !t.cols) return {};
  const labels = t.cols.map(c => (c.label || "").trim().toLowerCase());
  const idx = (...names) => { for (const n of names){ const i = labels.indexOf(n); if (i>=0) return i; } return -1; };
  const iL = idx("local","casa"), iV = idx("visitante","visita","fuera"), iR = idx("resultado","marcador","score");
  const cell = (row,i) => { if (i<0 || !row.c[i]) return ""; const c = row.c[i]; return String(c.f != null ? c.f : (c.v != null ? c.v : "")).trim(); };
  const out = {};
  (t.rows || []).forEach(row => {
    const L = resolveTeam(cell(row,iL));
    const V = resolveTeam(cell(row,iV));
    const score = cell(row,iR).replace(/[–—]/g,"-").replace(/\s+/g,"");
    if (!L || !V || !/^\d+-\d+$/.test(score)) return;
    if (isFixture(L,V)) out[`${L}|${V}`] = score;
    else if (isFixture(V,L)){ const [x,y] = score.split("-"); out[`${V}|${L}`] = `${y}-${x}`; }   // orientación invertida
  });
  return out;
}

/* lee la hoja por JSONP (un <script>): esquiva CORS, que bloquea fetch() */
function fetchSheetResults(){
  if (!SHEET.id) return Promise.resolve(null);
  return new Promise(resolve => {
    const cb = "__gviz_" + Math.random().toString(36).slice(2);
    const url = `https://docs.google.com/spreadsheets/d/${SHEET.id}/gviz/tq` +
                `?tqx=out:json;responseHandler:${cb}&headers=1&sheet=${encodeURIComponent(SHEET.tab)}&_=${Date.now()}`;
    const s = document.createElement("script");
    let settled = false;
    const finish = val => { if (settled) return; settled = true; try { delete window[cb]; } catch(e){ window[cb] = undefined; } s.remove(); resolve(val); };
    window[cb] = resp => { try { finish(parseGviz(resp)); } catch(e){ finish(null); } };
    s.onerror = () => finish(null);
    s.src = url;
    document.head.appendChild(s);
    setTimeout(() => finish(null), 8000);               // timeout de seguridad
  });
}

// prioridad: Google Sheet (tu fuente) > override manual > API
const mergeResults = () => ({ ...(store.api || {}), ...(store.manual || {}), ...(store.sheet || {}) });

/* sincroniza Sheet + override manual + API; re-renderiza solo si cambió algo */
async function syncRealResults(){
  try { store.manual = await fetch("resultados.json", { cache:"no-store" }).then(r => r.json()); } catch(e){ /* se mantiene */ }
  const [api, sheet] = await Promise.all([ fetchApiResults(), fetchSheetResults() ]);
  if (api)   store.api   = api;
  if (sheet) store.sheet = sheet;
  const merged = mergeResults();
  const sig = JSON.stringify(merged);
  if (sig === store._resSig) return;
  store._resSig = sig; store.resultados = merged;
  enrich(); computeRecord();
  renderFeature(); renderResults(); renderGrid(); updateStatus();
}

/* ===================================================================
   INIT
   =================================================================== */
async function init(){
  skeleton("champChart"); skeleton("eloChart");
  try{
    const [elo, mc, f4, f2, matrices, resultados] = await Promise.all([
      loadCSV("ranking_elo_actual.csv"),
      loadCSV("predicciones_fase5_montecarlo.csv"),
      loadCSV("predicciones_fase4_stacking_ensemble.csv"),
      loadCSV("predicciones_fase2_poisson_dixon_coles.csv"),
      loadJSON("matrices_marcador.json").catch(() => ({})),
      fetch("resultados.json", { cache:"no-store" }).then(r => r.json()).catch(() => ({})),
    ]);
    store.elo = elo; store.mc = mc; store.f4 = f4; store.f2 = f2; store.matrices = matrices;
    store.manual = resultados; store.api = {}; store.sheet = {};
    store.resultados = mergeResults();
    enrich();
    computeRecord();
    renderFeature(); renderCalculator(); renderResults(); renderGrid(); wireToolbar(); renderChamp(); renderPodium(); renderElo();
    updateStatus();

    // resultados reales automáticos: API (TheSportsDB) + override manual, cada minuto
    store._resSig = JSON.stringify(store.resultados);
    syncRealResults();
    setInterval(syncRealResults, 60000);
  }catch(e){
    ["champChart","eloChart"].forEach(unskeleton);
    document.getElementById("loadStatus").innerHTML =
      `<div class="error-box">No se pudieron cargar los datos (¿abriste el archivo con <code>file://</code>?). Sírvelo con un servidor local:<br>
       <code>py -m http.server 8000</code> · abre <code>http://localhost:8000/web/</code> · o ejecuta <code>web\\start.bat</code></div>`;
    console.error(e);
  }
}
init();
