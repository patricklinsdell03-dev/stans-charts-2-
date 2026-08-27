"""Static, self-contained HTML dashboard. No external assets, no build step."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

COLS = ["ticker", "name", "market", "index", "sector", "close", "stage", "trend_score",
        "stage2_readiness", "stage4_readiness", "px_vs_ma_pct", "ma_slope_pct",
        "mansfield_rs", "rs_slope", "resistance", "support", "dist_to_resistance_pct",
        "dist_to_support_pct", "base_age_weeks", "base_age_recent", "base_width_pct",
        "tightness", "vol_ratio", "vol_ratio_3w", "vol_dryup", "down_vol_share",
        "atr_pct", "stop_suggestion", "stop_risk_pct", "ret_13w_pct", "ret_52w_pct",
        "prior_trend_pct", "sector_rs", "sector_rs_slope", "sector_rank_pct",
        "rs_vs_sector", "group_factor", "group_factor_dn", "grade",
        "signal", "signal_age_weeks", "up_parts", "dn_parts", "plan",
        "explain", "prompt_vals"]


def _sanitise(o):
    """
    Recursively replace NaN and infinity with null.

    json.dumps writes bare NaN by default, which Python reads back happily and a
    browser will not: the payload is inlined as a JavaScript object literal, and
    `nan` there is an undefined identifier that kills the whole script. The page
    renders blank with one console error and nothing else to say why.
    """
    if isinstance(o, dict):
        return {k: _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, np.integer):
        return int(o)
    return o


def _clean(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    d = df.copy()
    for c in COLS:
        if c not in d:
            d[c] = None
    d = d[COLS].replace({np.nan: None})
    recs = d.to_dict("records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, pd.Timestamp):
                r[k] = v.strftime("%Y-%m-%d")
    return recs


def build_html(lists, scan, calibration, asof, generated,
               market=None, sectors=None, regime_history=None,
               positions=None) -> str:
    from . import precheck as PC
    hist = []
    if regime_history is not None and len(regime_history):
        h = regime_history.dropna()
        hist = [{"d": str(pd.Timestamp(i).date()),
                 "s": round(float(r["regime_score"]), 1),
                 "b": round(float(r["breadth_above_ma"]), 1)}
                for i, r in h.iterrows()]
    payload = {
        "asof": str(pd.Timestamp(asof).date()) if pd.notna(asof) else "unknown",
        "generated": generated.strftime("%Y-%m-%d %H:%M UTC"),
        "universe": int(len(scan)),
        "stage_counts": scan["stage"].value_counts().to_dict() if not scan.empty else {},
        "lists": {k: _clean(v) for k, v in (lists or {}).items()},
        "calibration": calibration or {},
        "market": market or {},
        "sectors": sectors or [],
        "regime_history": hist,
        "positions": positions or [],
        "prompt_template": PC.TEMPLATE,
    }
    return TEMPLATE.replace("__DATA__", json.dumps(_sanitise(payload),
                                                   default=str, allow_nan=False))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage Tracker</title>
<style>
:root{
  --bg:#fbfaf9; --panel:#fff; --line:#e6e2dd; --ink:#1c1a18; --mute:#6c6560;
  --up:#1f7a4d; --up-bg:#e7f4ec; --dn:#a8302a; --dn-bg:#fbeceb;
  --warn:#8a6410; --warn-bg:#faf1dd; --accent:#2f5fa8;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#141312; --panel:#1c1b19; --line:#2e2b28; --ink:#eeebe7; --mute:#9a938c;
  --up:#4ec98a; --up-bg:#12291f; --dn:#f2857e; --dn-bg:#2b1614;
  --warn:#d9ab4a; --warn-bg:#2a2313; --accent:#7aa5e8;
}}
:root[data-theme="dark"]{
  --bg:#141312; --panel:#1c1b19; --line:#2e2b28; --ink:#eeebe7; --mute:#9a938c;
  --up:#4ec98a; --up-bg:#12291f; --dn:#f2857e; --dn-bg:#2b1614;
  --warn:#d9ab4a; --warn-bg:#2a2313; --accent:#7aa5e8;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:26px 16px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--mute);font-size:13px;margin-bottom:20px}
.regime{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:16px 18px;margin-bottom:18px;display:grid;
 grid-template-columns:minmax(200px,1fr) minmax(220px,auto);gap:16px;align-items:center}
@media (max-width:700px){.regime{grid-template-columns:1fr}}
.regime .lab{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}
.regime .val{font-size:26px;font-weight:650;letter-spacing:-.02em;margin:2px 0 6px}
.regime .guide{font-size:13.5px;color:var(--mute);line-height:1.5;max-width:62ch}
.regime .facts{font-size:12.5px;color:var(--mute);margin-top:8px}
.regime .facts b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.spark{width:100%;height:74px;display:block}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:10px;margin-bottom:22px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.kpi .v{font-size:21px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.kpi .l{font-size:11px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.tabs{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:12px}
.tab{border:1px solid var(--line);background:var(--panel);color:var(--mute);
 padding:7px 13px;border-radius:999px;font-size:13px;cursor:pointer}
.tab.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
input,select{background:var(--panel);border:1px solid var(--line);color:var(--ink);
 border-radius:8px;padding:7px 10px;font-size:13px}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:960px}
th{position:sticky;top:0;background:var(--panel);text-align:right;padding:9px 10px;
 border-bottom:1px solid var(--line);color:var(--mute);font-weight:600;cursor:pointer;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;
 font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}
tr.row:hover{background:color-mix(in srgb,var(--ink) 4%,transparent)}
.tk{font-weight:600}
.nm{color:var(--mute);font-size:12px;max-width:200px;overflow:hidden;text-overflow:ellipsis;
 display:inline-block;vertical-align:bottom}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;border:1px solid var(--line)}
.s2{background:var(--up-bg);color:var(--up);border-color:transparent}
.s4{background:var(--dn-bg);color:var(--dn);border-color:transparent}
.gr{display:inline-grid;place-items:center;width:20px;height:20px;border-radius:5px;
 font-size:11px;font-weight:700}
.gA{background:var(--up-bg);color:var(--up)} .gB{background:var(--warn-bg);color:var(--warn)}
.gC{background:var(--dn-bg);color:var(--dn)} .gD{background:var(--line);color:var(--mute)}
.pos{color:var(--up)} .neg{color:var(--dn)}
.meter{position:relative;height:6px;width:56px;background:var(--line);border-radius:3px;
 display:inline-block;vertical-align:middle;margin-right:6px}
.meter i{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:var(--accent)}
.note{font-size:12px;color:var(--mute);margin:10px 2px 0;line-height:1.5}
details{margin-top:22px;background:var(--panel);border:1px solid var(--line);
 border-radius:10px;padding:12px 14px}
summary{cursor:pointer;font-weight:600;font-size:14px}
.cal{font-size:12.5px;margin-top:10px;overflow-x:auto}
.cal table{min-width:520px}
.empty{padding:26px;text-align:center;color:var(--mute);font-size:13px}
tr.row{cursor:pointer}
tr.det td{background:color-mix(in srgb,var(--ink) 3%,transparent);white-space:normal;text-align:left;
 padding:14px 16px 18px}
/* The detail row lives inside a table that scrolls sideways, so without this the
   plan panel is laid out across the full table width and half of it sits off
   screen until you scroll. Sticking it to the left edge keeps it readable. */
/* Width comes from --panelw, set in JS to the visible width of the scrolling
   table container. 100vw is the wrong measure here: the table sits inside a
   centred wrapper with its own padding, so a viewport-derived width overhangs
   the right edge by however much margin the page has, which is exactly what it
   did before this was measured rather than assumed. */
.planwrap{position:sticky;left:0;width:min(100%,var(--panelw,88vw));
 max-width:var(--panelw,88vw)}
/* The detail row lives in a table that is wider than the viewport, so anything
   inside it inherits that width unless it is told not to. Without the explicit
   normal/anywhere pair the sentences run straight out past the panel border and
   off the right edge of the screen. */
.say{background:var(--panel);border:1px solid var(--line);border-radius:10px;
 padding:16px 18px;margin-bottom:14px;max-width:100%;white-space:normal;
 overflow-wrap:anywhere}
.say *{white-space:normal}
.say h4{margin:0 0 6px;font-size:16px;letter-spacing:-.01em}
.say p{margin:8px 0;font-size:14.5px;line-height:1.6}
.say .lab{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);
 margin-top:14px;display:block}
.say .do{font-weight:600}
.say ul{margin:6px 0 0;padding-left:18px;font-size:13.5px;color:var(--mute)}
.say li{margin:5px 0;line-height:1.5}
.copybar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px}
button.cp{font:inherit;font-size:13px;background:var(--ink);color:var(--bg);border:0;
 border-radius:999px;padding:7px 15px;cursor:pointer;font-weight:600}
button.cp.alt{background:var(--panel);color:var(--ink);border:1px solid var(--line);font-weight:500}
button.cp:hover{opacity:.88}
.cpnote{font-size:12px;color:var(--mute)}
pre.prompt{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);
 border-radius:8px;padding:12px 14px;font-size:11.5px;line-height:1.5;max-height:300px;
 overflow:auto;margin-top:10px;display:none}
.plan{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:12px}
.lvl{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 11px}
.lvl .l{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--mute)}
.lvl .v{font-size:17px;font-weight:650;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.lvl .s{font-size:11.5px;color:var(--mute);margin-top:1px}
.lvl.buy .v{color:var(--accent)} .lvl.stop .v{color:var(--dn)} .lvl.tgt .v{color:var(--up)}
.seq{margin:0;padding-left:20px;font-size:13px;color:var(--ink)}
.seq li{margin:5px 0;line-height:1.5}
.warn{background:var(--warn-bg);border:1px solid var(--line);border-radius:8px;
 padding:9px 12px;font-size:12.5px;color:var(--warn);margin:10px 0}
.posrow td{white-space:nowrap}
.act{font-weight:700;font-size:12px;letter-spacing:.02em}
.act.exit{color:var(--dn)} .act.trim{color:var(--warn)} .act.hold{color:var(--mute)}
.act.move{color:var(--accent)}
</style></head><body><div class="wrap">
<h1>Weinstein Stage Tracker</h1>
<div class="sub" id="sub"></div>
<div class="regime" id="regime"></div>
<div class="kpis" id="kpis"></div>
<div class="tabs" id="tabs"></div>
<div class="bar">
  <input id="q" placeholder="filter ticker, name or sector" style="min-width:210px">
  <select id="mkt"><option value="">all markets</option><option>US</option><option>UK</option></select>
  <select id="idx"><option value="">all indices</option></select>
  <select id="grd"><option value="">any grade</option><option>A</option><option>B</option><option>C</option></select>
  <label style="font-size:12.5px;color:var(--mute)"><input type="checkbox" id="lead" style="vertical-align:middle"> leading groups only</label>
</div>
<div class="tablewrap"><table id="tbl"><thead></thead><tbody></tbody></table></div>
<div class="note" id="note"></div>

<details open><summary>Open positions and what the plan says this week</summary>
<div class="cal" id="pos"></div></details>

<details open><summary>Sector relative strength</summary>
<div class="cal" id="sec"></div></details>

<details id="caldet"><summary>Calibration: what these signals did historically</summary>
<div class="cal" id="cal"></div></details>

<details><summary>How each column is built</summary>
<div class="note" style="line-height:1.7">
<b>Grade</b> combines the three things Weinstein requires. A is a textbook break by a leader in a leading group while the market supports it. B is a textbook break with a neutral group. C is a textbook break with the group or the market against it, which is the case he tells you to pass on. D failed a price, volume or relative strength rule outright.<br>
<b>Stage</b> comes from the slope of the 30 week moving average first and the continuum score second. A flat average after a decline is Stage 1, the same flat average after an advance is Stage 3.<br>
<b>Trend</b> is the continuum score from minus one hundred to plus one hundred: 35 per cent the volatility adjusted slope of the 30 week average, 30 per cent Mansfield relative strength, 20 per cent price relative to the average, 15 per cent the four week change in relative strength.<br>
<b>Ready 2</b> and <b>Ready 4</b> are pre-break scores out of one hundred, scaled by a stage context factor and a group factor. They describe setups that have not broken yet.<br>
<b>RS</b> is Mansfield relative strength against the market index. <b>Sec RS</b> is the same measure applied to the stock's equal weighted sector composite. <b>vs Sec</b> is the stock measured against its own sector, so a positive number means it leads its group rather than merely riding it.<br>
<b>Vol</b> is the best weekly volume over the breakout week and the two before it, as a multiple of the ten week average. Textbook wants at least 2.0 on a Stage 2 breakout. Breakdowns carry no volume requirement.<br>
<b>Trigger</b> is the pivot plus a quarter of a weekly ATR, floored at 0.4 per cent, so a one tick poke through the base high does not count. <b>Stop</b> sits below base support and the 30 week average, and is pulled in if that would put it more than twelve per cent away. <b>T1</b> is the height of the base projected from the pivot, with its distance from the trigger shown in R. Click any row for the full plan.
</div></details>
</div>
<script>
const D = __DATA__;
const LISTS = [
 ["breakouts","Stage 2 breaks","Price closed above the base high this week or in the last four. Sorted by grade, so the leaders in leading groups sit at the top."],
 ["watch_stage2","Approaching Stage 2","Highest readiness scores that have not broken out yet, after the group factor."],
 ["breakdowns","Stage 4 breaks","Price closed below base support. No volume requirement applies."],
 ["watch_stage4","Approaching Stage 4","Stage 3 tops with the highest breakdown readiness."],
];
let cur="breakouts", sortKey=null, sortDir=-1;
const f=(v,d=1)=> v===null||v===undefined||Number.isNaN(v) ? "—" : Number(v).toFixed(d);
const sgn=(v,d=1)=> v===null||v===undefined ? "—"
 : `<span class="${v>0?'pos':v<0?'neg':''}">${v>0?'+':''}${Number(v).toFixed(d)}</span>`;
const meter=v=> v===null||v===undefined ? "—"
 : `<span class="meter"><i style="width:${Math.max(0,Math.min(100,v))}%"></i></span>${Number(v).toFixed(0)}`;
const stagePill=s=>`<span class="pill ${/Stage 2/.test(s)?'s2':/Stage 4|Stage 3 to 4/.test(s)?'s4':''}">${s}</span>`;
const gradeBox=g=> g?`<span class="gr g${g}">${g}</span>`:"—";

const COLDEF=[
 ["ticker","Ticker",r=>`<span class="tk">${r.ticker}</span><br><span class="nm">${r.name||""}</span>`],
 ["grade","Grade",r=>gradeBox(r.grade)],
 ["stage","Stage",r=>stagePill(r.stage)],
 ["close","Close",r=>f(r.close,2)],
 ["trend_score","Trend",r=>sgn(r.trend_score,0)],
 ["stage2_readiness","Ready 2",r=>meter(r.stage2_readiness)],
 ["stage4_readiness","Ready 4",r=>meter(r.stage4_readiness)],
 ["mansfield_rs","RS",r=>sgn(r.mansfield_rs,1)],
 ["sector_rs","Sec RS",r=>sgn(r.sector_rs,1)],
 ["sector_rank_pct","Sec rank",r=>r.sector_rank_pct===null?"—":f(r.sector_rank_pct,0)+"%"],
 ["rs_vs_sector","vs Sec",r=>sgn(r.rs_vs_sector,1)],
 ["ma_slope_pct","MA slope",r=>sgn(r.ma_slope_pct,1)],
 ["dist_to_resistance_pct","To pivot",r=>f(r.dist_to_resistance_pct,1)+"%"],
 ["base_age_recent","Base wks",r=>f(r.base_age_recent ?? r.base_age_weeks,0)],
 ["vol_ratio_3w","Vol",r=>f(r.vol_ratio_3w,1)+"x"],
 ["plan_trigger","Trigger",r=>r.plan?f(r.plan.trigger,2):"—"],
 ["plan_stop","Stop",r=>r.plan?f(r.plan.stop,2):"—"],
 ["plan_risk","Risk",r=>r.plan?f(r.plan.risk_pct,1)+"%":"—"],
 ["plan_t1r","T1",r=>r.plan?f(r.plan.t1,2)+` <span style="color:var(--mute)">${f(r.plan.t1_r,1)}R</span>`:"—"],
 ["signal_age_weeks","Signal",r=>{
    if(!r.signal) return "—";
    const c=r.signal.confirmed?"confirmed":"unconfirmed";
    const w=r.signal_age_weeks===0?"this week":r.signal_age_weeks+"w ago";
    return `<span class="pill ${r.signal.kind==='stage2_breakout'?'s2':'s4'}" title="${(r.signal.notes||'').replace(/"/g,'')}">${c} · ${w}</span>`;}],
 ["sector","Sector",r=>`<span class="nm">${r.sector||""}</span>`],
];

function rows(){
  let r=(D.lists[cur]||[]).slice();
  const q=document.getElementById("q").value.toLowerCase().trim();
  const mk=document.getElementById("mkt").value, ix=document.getElementById("idx").value;
  const gd=document.getElementById("grd").value, ld=document.getElementById("lead").checked;
  if(q) r=r.filter(x=>(x.ticker+" "+(x.name||"")+" "+(x.sector||"")).toLowerCase().includes(q));
  if(mk) r=r.filter(x=>x.market===mk);
  if(ix) r=r.filter(x=>x.index===ix);
  if(gd) r=r.filter(x=>x.grade===gd);
  if(ld) r=r.filter(x=>(x.sector_rs??-99)>0 && (x.rs_vs_sector??-99)>0);
  const K=k=>({plan_trigger:x=>x.plan&&x.plan.trigger,plan_stop:x=>x.plan&&x.plan.stop,
    plan_risk:x=>x.plan&&x.plan.risk_pct,plan_t1r:x=>x.plan&&x.plan.t1_r}[k]||(x=>x[k]));
  if(sortKey) r.sort((a,b)=>{const g=K(sortKey),A=g(a),B=g(b);
    if(A===null||A===undefined) return 1; if(B===null||B===undefined) return -1;
    return (A>B?1:A<B?-1:0)*sortDir;});
  return r;
}
function sizePanels(){
  const w=document.querySelector(".tablewrap");
  if(w) document.documentElement.style.setProperty("--panelw",(w.clientWidth-34)+"px");
}
function draw(){
  sizePanels();
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("on",t.dataset.k===cur));
  document.getElementById("note").textContent=LISTS.find(l=>l[0]===cur)[2];
  const thead=document.querySelector("#tbl thead");
  thead.innerHTML="<tr>"+COLDEF.map(c=>`<th data-k="${c[0]}">${c[1]}${sortKey===c[0]?(sortDir>0?" ▲":" ▼"):""}</th>`).join("")+"</tr>";
  thead.querySelectorAll("th").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(sortKey===k) sortDir*=-1; else {sortKey=k;sortDir=-1;} draw();});
  const r=rows(), tb=document.querySelector("#tbl tbody");
  tb.innerHTML=r.length?r.map((x,i)=>`<tr class='row' data-i='${i}'>`+COLDEF.map(c=>`<td>${c[2](x)}</td>`).join("")+"</tr>").join("")
    :`<tr><td colspan="${COLDEF.length}"><div class="empty">Nothing in this list for the week ending ${D.asof} under the current filters. Confirmed Weinstein breaks are rare in most weeks, and rarer still when the market regime is against them.</div></td></tr>`;
  wireRows(r);
}
function sparkline(hist){
  if(!hist||hist.length<8) return "";
  const W=340,H=74,P=6;
  const xs=(i)=>P+(W-2*P)*i/(hist.length-1);
  const ys=(v)=>P+(H-2*P)*(1-(v+100)/200);
  const pts=hist.map((h,i)=>`${xs(i).toFixed(1)},${ys(h.s).toFixed(1)}`).join(" ");
  const zero=ys(0).toFixed(1);
  const last=hist[hist.length-1];
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="Market regime score over the last three years, currently ${last.s}.">
    <line x1="${P}" x2="${W-P}" y1="${zero}" y2="${zero}" stroke="var(--line)" stroke-width="1"/>
    <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-linejoin="round"/>
    <circle cx="${xs(hist.length-1).toFixed(1)}" cy="${ys(last.s).toFixed(1)}" r="3.5"
      fill="${last.s>=0?'var(--up)':'var(--dn)'}"/>
    <text x="${W-P}" y="${H-1}" text-anchor="end" font-size="10" fill="var(--mute)">regime score, 3 years</text>
  </svg>`;
}
function drawRegime(){
  const m=D.market||{}; const el=document.getElementById("regime");
  if(!m.regime){ el.style.display="none"; return; }
  const col=m.regime==="Bull"?"var(--up)":m.regime==="Bear"?"var(--dn)":
            (m.regime==="Improving"?"var(--up)":m.regime==="Deteriorating"?"var(--dn)":"var(--ink)");
  el.innerHTML=`<div>
      <div class="lab">Market regime</div>
      <div class="val" style="color:${col}">${m.regime} <span style="font-size:15px;color:var(--mute)">${m.regime_score>0?'+':''}${m.regime_score}</span></div>
      <div class="guide">${m.guidance||""}</div>
      <div class="facts"><b>${f(m.breadth_above_ma,0)}%</b> of the universe above its own 30 week line ·
        index <b>${sgn(m.index_px_vs_ma_pct,1)}%</b> vs its average, slope <b>${sgn(m.index_ma_slope_pct,2)}%</b> ·
        <b>${f(m.stage2_share,0)}%</b> in Stage 2, <b>${f(m.stage4_share,0)}%</b> in Stage 4</div>
    </div><div>${sparkline(D.regime_history)}</div>`;
}
function drawSectors(){
  const s=D.sectors||[]; const el=document.getElementById("sec");
  if(!s.length){ el.innerHTML="<p class='note'>No sector composites were built for this scan.</p>"; return; }
  el.innerHTML=`<p class="note">Each sector is an equal weighted composite of its own constituents, measured against the same index the stocks are measured against. Rank is the percentile of relative strength within that market. Weinstein's rule is to buy leaders inside the top of this table and to leave the bottom of it alone.</p>
  <table><thead><tr><th style="text-align:left">Sector</th><th>Market</th><th>Sec RS</th><th>4w change</th><th>MA slope</th><th>Rank</th><th>Names</th><th>In Stage 2</th><th>In Stage 4</th></tr></thead><tbody>`
   + s.map(r=>`<tr><td style="text-align:left">${r.sector}</td><td>${r.market}</td>
      <td>${sgn(r.sector_rs,1)}</td><td>${sgn(r.sector_rs_slope,1)}</td><td>${sgn(r.ma_slope_pct,1)}</td>
      <td>${r.rank_pct===null?"—":f(r.rank_pct,0)+"%"}</td><td>${r.n}</td>
      <td class="pos">${r.in_stage2}</td><td class="neg">${r.in_stage4}</td></tr>`).join("")+"</tbody></table>";
}
function drawCal(){
  const c=D.calibration||{}, el=document.getElementById("cal");
  if(!c.signals||!c.signals.length){ el.innerHTML="<p class='note'>Calibration was not run for this scan.</p>"; return; }
  const t1=`<table><thead><tr><th style="text-align:left">Signal</th><th>Confirmed</th><th>n</th>
    <th>Median excess 4w</th><th>13w</th><th>26w</th><th>Win rate 13w</th><th>p90 26w</th><th>p10 26w</th><th>Stopped out 13w</th></tr></thead><tbody>`+
    c.signals.map(s=>`<tr><td style="text-align:left">${s.kind}</td><td>${s.confirmed?"yes":"no"}</td><td>${s.n}</td>
    <td>${f(s.median_excess_4w)}%</td><td>${f(s.median_excess_13w)}%</td><td>${f(s.median_excess_26w)}%</td>
    <td>${f(s.win_rate_13w)}%</td><td>${f(s.p90_excess_26w)}%</td><td>${f(s.p10_excess_26w)}%</td>
    <td>${s.stopped_out_13w_pct===null||s.stopped_out_13w_pct===undefined?"—":f(s.stopped_out_13w_pct)+"%"}</td></tr>`).join("")+"</tbody></table>";
  const disc=(d,label,what)=> !d||!d.n ? "" :
    `<p class="note" style="margin-top:18px"><b>${label}</b> The score's own ranking power against ${what}, next to what distance to the level achieves on its own. The increment is the part the other eighty points of the score contribute. An increment near zero means this is a distance screen and nothing more.</p>
     <table><thead><tr><th style="text-align:left">Measure</th><th>AUC</th></tr></thead><tbody>
     <tr><td style="text-align:left">Full readiness score</td><td>${f(d.auc_score,3)}</td></tr>
     <tr><td style="text-align:left">Distance to the level alone</td><td>${f(d.auc_proximity_only,3)}</td></tr>
     <tr><td style="text-align:left">Score with distance removed</td><td>${f(d.auc_residual,3)}</td></tr>
     <tr><td style="text-align:left"><b>Increment over distance alone</b></td><td><b>${f(d.increment,3)}</b></td></tr>
     <tr><td style="text-align:left">Base rate / rows</td><td>${f(d.base_rate,1)}% / ${d.n}</td></tr>
     </tbody></table>`;
  const unc=(rows)=> !rows||!rows.length ? "" :
    `<p class="note" style="margin-top:18px"><b>How much of this is signal?</b> Median 13 week excess with a 95% interval from resampling whole calendar weeks, which keeps the fact that hundreds of signals fire together intact. Independent episodes, not rows, is the honest sample size: over eight years at a 13 week horizon there are about 32 of them however many rows the screen produced.</p>
     <table><thead><tr><th style="text-align:left">Signal</th><th>Confirmed</th><th>Rows</th><th>Distinct weeks</th><th>Independent episodes</th><th>Median 13w</th><th>95% interval</th></tr></thead><tbody>`+
     rows.map(r=>`<tr><td style="text-align:left">${r.kind}</td><td>${r.confirmed?"yes":"no"}</td><td>${r.rows}</td>
     <td>${r.distinct_weeks}</td><td><b>${r.independent_episodes}</b></td><td>${f(r.median_excess_13w)}%</td>
     <td>${f(r.ci_low)}% to ${f(r.ci_high)}%</td></tr>`).join("")+"</tbody></table>";
  const budget=(b)=> !b||!b.statistics_reported ? "" :
    `<div class="note" style="margin-top:18px;padding:10px 12px;border:1px solid var(--line);border-radius:8px">
     <b>Comparison budget.</b> This panel reports ${b.statistics_reported} statistics. If every filter here were inert, roughly ${b.expected_false_positives_at_5pct} of them would still look convincing at the usual five per cent threshold. To mean anything on its own a single number would need to clear ${b.bonferroni_z} standard errors, not two.</div>`;
  const split=(rows,key,title,blurb)=> !rows||!rows.length ? "" :
    `<p class="note" style="margin-top:18px"><b>${title}</b> ${blurb}</p>
     <table><thead><tr><th style="text-align:left">Signal</th><th style="text-align:left">${key}</th><th>n</th>
     <th>Median excess 13w</th><th>Win rate 13w</th><th>Median excess 26w</th></tr></thead><tbody>`+
     rows.map(r=>`<tr><td style="text-align:left">${r.kind}</td><td style="text-align:left">${r[key]}</td><td>${r.n}</td>
     <td>${f(r.median_excess_13w)}%</td><td>${f(r.win_rate_13w)}%</td><td>${f(r.median_excess_26w)}%</td></tr>`).join("")+"</tbody></table>";
  const dec=d=> !d||!d.length ? "" :
    `<table><thead><tr><th style="text-align:left">Decile</th><th>Score range</th><th>n</th><th>Break within 8 weeks</th></tr></thead><tbody>`+
    d.map(r=>`<tr><td style="text-align:left">${r.decile}</td><td>${f(r.score_lo)}–${f(r.score_hi)}</td><td>${r.n}</td><td>${f(r.hit_rate)}%</td></tr>`).join("")+"</tbody></table>";
  el.innerHTML=`<p class="note">Excess return is measured against the stock's own market index, so a breakout that rises with everything else does not count as a win. Sample: ${c.sample_signals} signals across ${c.sample_weeks} ticker-weeks.</p>`
    + t1
    + split(c.by_regime,"regime","Does the market regime matter?",
        "Confirmed signals split by the regime in force the week they fired. If the columns are flat, the regime filter is costing you signals without buying anything.")
    + split(c.by_group,"group_bucket","Does the group matter?",
        "The same signals split by sector strength and whether the stock led its own group. This is the direct test of Weinstein's leaders-in-leading-groups rule.")
    + unc(c.uncertainty)
    + disc(c.discrimination_up, "Does Stage 2 readiness lead the break?", "a confirmed breakout in the next eight weeks")
    + disc(c.discrimination_dn, "Does Stage 4 readiness lead the break?", "a confirmed breakdown in the next eight weeks")
    + `<p class="note" style="margin-top:18px"><b>Decile table, for description only.</b> Kept because it shows the shape of the sample, not because it is evidence. Its cut points are pooled across the whole history, and its target is close to a restatement of the score's own distance term, so a pure random walk passes it convincingly. Read the increment above instead.</p>`
    + dec(c.stage2_deciles) + dec(c.stage4_deciles)
    + budget(c.comparisons);
}
function fillPrompt(x){
  let t = D.prompt_template || "";
  const v = x.prompt_vals || {};
  Object.keys(v).forEach(k=>{ t = t.split("{"+k+"}").join(v[k]); });
  return t;
}
function sayPanel(x){
  const e = x.explain;
  if(!e) return "";
  const cav = (e.caveats||[]).map(c=>`<li>${c}</li>`).join("");
  return `<div class="say">
    <h4>${e.headline}</h4>
    <p>${e.what}</p>
    <p>${e.group}</p>
    ${e.grade_note?`<p>${e.grade_note}</p>`:""}
    <span class="lab">What to do</span>
    <p class="do">${e.action}</p>
    <span class="lab">What would make this wrong</span>
    <p>${e.invalidate}</p>
    ${cav?`<span class="lab">Worth knowing</span><ul>${cav}</ul>`:""}
    <div class="copybar">
      <button class="cp" type="button" data-cp="${x.ticker}">Copy pre-trade check</button>
      <button class="cp alt" type="button" data-pv="${x.ticker}">Show it</button>
      <span class="cpnote">Paste into Claude before you place the order. It checks live prices, re-derives the four conditions and looks for what a weekly scan cannot see.</span>
    </div>
    <pre class="prompt" id="pr-${x.ticker}">${fillPrompt(x).replace(/</g,"&lt;")}</pre>
  </div>`;
}
function planPanel(x){
  const p=x.plan;
  if(!p) return `<div class="planwrap">${sayPanel(x)}<div class="note">No plan: this row has no usable base geometry, so there is no pivot to trade against.</div></div>`;
  const up=p.side==="long", cur=up?"buy":"sell";
  const sz=p.size||{};
  const money = sz.shares!==undefined
    ? `<div class="lvl"><div class="l">Size</div><div class="v">${sz.shares}</div>
       <div class="s">shares · ${f(sz.exposure,0)} exposure · ${f(sz.risk_pct_of_account,1)}% of account at risk</div></div>`
    : `<div class="lvl"><div class="l">Size</div><div class="v">${f(sz.shares_per_1000_risked,1)}</div>
       <div class="s">shares per 1000 of risk · ${f(sz.exposure_per_1000_risked,0)} exposure</div></div>`;
  return `<div class="planwrap">${sayPanel(x)}<div class="plan">
    <div class="lvl buy"><div class="l">${up?"Buy stop":"Sell stop"}</div><div class="v">${f(p.trigger,2)}</div>
      <div class="s">${f(p.trigger_pct_from_close,2)}% from the close · pivot ${f(p.pivot,2)}</div></div>
    <div class="lvl buy"><div class="l">${up?"Pullback buy":"Pullback sell"}</div>
      <div class="v">${f(p.pullback_low,2)}–${f(p.pullback_high,2)}</div>
      <div class="s">old ${up?"resistance as support":"support as resistance"}</div></div>
    <div class="lvl stop"><div class="l">Initial stop</div><div class="v">${f(p.stop,2)}</div>
      <div class="s">${f(p.risk_pct,1)}% risk · ${f(p.risk_per_share,2)} per share</div></div>
    <div class="lvl tgt"><div class="l">Target 1</div><div class="v">${f(p.t1,2)}</div>
      <div class="s">${f(p.t1_r,1)}R · sell part here</div></div>
    <div class="lvl tgt"><div class="l">Target 2</div><div class="v">${f(p.t2,2)}</div>
      <div class="s">${f(p.t2_r,1)}R · measured move ×2</div></div>
    ${money}
  </div>
  ${(p.notes||[]).map(n=>`<div class="warn">${n}</div>`).join("")}
  <div class="note" style="margin:0 0 8px"><b>Entry style matters.</b> A resting ${cur} stop at ${f(p.trigger,2)} fills the moment price touches it on any day, including spikes that are back inside the base by Friday. Weinstein's rule is a weekly close beyond the pivot on at least double volume, which means checking on Friday and dealing on Monday. The level is the same, the fills are not: the resting order takes more trades and more of them fail.</div>
  <ol class="seq">${(p.sequence||[]).map(t=>`<li>${t}</li>`).join("")}</ol></div>`;
}
function wireRows(r){
  document.querySelectorAll("#tbl tbody tr.row").forEach(tr=>tr.onclick=()=>{
    const nxt=tr.nextElementSibling;
    if(nxt&&nxt.classList.contains("det")){ nxt.remove(); return; }
    document.querySelectorAll("#tbl tbody tr.det").forEach(d=>d.remove());
    const det=document.createElement("tr"); det.className="det";
    const rec=r[+tr.dataset.i];
    det.innerHTML=`<td colspan="${COLDEF.length}">${planPanel(rec)}</td>`;
    tr.after(det);
    const cp=det.querySelector("[data-cp]"), pv=det.querySelector("[data-pv]");
    if(cp) cp.onclick=(ev)=>{ev.stopPropagation();
      const txt=fillPrompt(rec);
      const done=()=>{cp.textContent="Copied";setTimeout(()=>cp.textContent="Copy pre-trade check",1800);};
      if(navigator.clipboard&&navigator.clipboard.writeText){
        navigator.clipboard.writeText(txt).then(done).catch(()=>{
          document.getElementById("pr-"+rec.ticker).style.display="block";
          cp.textContent="Select it below";});
      } else {
        document.getElementById("pr-"+rec.ticker).style.display="block";
        cp.textContent="Select it below";
      }};
    if(pv) pv.onclick=(ev)=>{ev.stopPropagation();
      const el=document.getElementById("pr-"+rec.ticker);
      const open=el.style.display==="block";
      el.style.display=open?"none":"block";
      pv.textContent=open?"Show it":"Hide it";};
    det.querySelectorAll("pre,button").forEach(el=>el.onclick=el.onclick||((ev)=>ev.stopPropagation()));
  });
}
function drawPositions(){
  const ps=D.positions||[], el=document.getElementById("pos");
  if(!ps.length){ el.innerHTML=`<p class="note">No open positions. Create a <code>positions.csv</code> beside the project with the columns <code>ticker, entry_date, entry_price, shares, initial_stop</code> and the weekly run will tell you what the plan says to do with each one.</p>`; return; }
  const cls=a=>a.startsWith("EXIT")?"exit":a.startsWith("TRIM")?"trim":a.startsWith("MOVE")?"move":"hold";
  el.innerHTML=`<p class="note">The plan is rebuilt from the bar at your entry date rather than from this week's, so the targets and the initial stop are the ones the trade was actually taken on.</p>
   <table><thead><tr><th style="text-align:left">Ticker</th><th>Entry</th><th>Now</th><th>Open</th><th>R</th>
   <th>Stop now</th><th>T1</th><th style="text-align:left">Action</th><th style="text-align:left">Why</th></tr></thead><tbody>`
   + ps.map(p=>`<tr class="posrow"><td style="text-align:left"><b>${p.ticker}</b><br><span class="nm">${p.entry_date||""}</span></td>
     <td>${f(p.entry_price,2)}</td><td>${f(p.close,2)}</td>
     <td class="${(p.open_pct||0)>=0?'pos':'neg'}">${f(p.open_pct,1)}%</td>
     <td>${p.open_r===null||p.open_r===undefined?"—":f(p.open_r,2)+"R"}</td>
     <td>${f(p.stop_now,2)}</td><td>${f(p.t1,2)}${p.t1_hit?' <span class="pill s2">hit</span>':''}</td>
     <td style="text-align:left" class="act ${cls(p.action||"HOLD")}">${p.action||"HOLD"}</td>
     <td style="text-align:left"><span class="nm" style="max-width:280px">${p.reason||""}</span></td></tr>`).join("")
   + "</tbody></table>";
}
function init(){
  document.getElementById("sub").textContent =
    `Week ending ${D.asof} · ${D.universe} tickers scored · ${(D.sectors||[]).length} sector composites · generated ${D.generated}`;
  const sc=D.stage_counts||{};
  const kpi=[["Stage 2 breaks",(D.lists.breakouts||[]).length],
    ["Grade A",(D.lists.breakouts||[]).filter(r=>r.grade==="A").length],
    ["Stage 4 breaks",(D.lists.breakdowns||[]).length],
    ["In Stage 2",sc["Stage 2"]||0],["In Stage 4",sc["Stage 4"]||0],
    ["Basing",(sc["Stage 1"]||0)+(sc["Stage 1 to 2"]||0)+(sc["Stage 4 to 1"]||0)],
    ["Topping",(sc["Stage 3"]||0)+(sc["Stage 3 to 4"]||0)+(sc["Stage 2 to 3"]||0)]];
  document.getElementById("kpis").innerHTML=kpi.map(k=>
    `<div class="kpi"><div class="v">${k[1]}</div><div class="l">${k[0]}</div></div>`).join("");
  document.getElementById("tabs").innerHTML=LISTS.map(l=>
    `<button class="tab" data-k="${l[0]}">${l[1]} <span style="opacity:.6">${(D.lists[l[0]]||[]).length}</span></button>`).join("");
  document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{cur=t.dataset.k;sortKey=null;draw();});
  const idxs=new Set(); Object.values(D.lists).flat().forEach(r=>r.index&&idxs.add(r.index));
  document.getElementById("idx").innerHTML='<option value="">all indices</option>'+
    [...idxs].sort().map(i=>`<option>${i}</option>`).join("");
  ["q","mkt","idx","grd"].forEach(id=>document.getElementById(id).oninput=draw);
  document.getElementById("lead").onchange=draw;
  drawRegime(); drawSectors(); drawPositions(); drawCal(); draw();
  window.addEventListener("resize", sizePanels);
}
init();
</script></body></html>"""
