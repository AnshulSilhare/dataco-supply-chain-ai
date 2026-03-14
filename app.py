import streamlit as st
import joblib
import os
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
import base64
import numpy as np
import streamlit.components.v1 as components

# ──────────────────────────────────────────────
# 1. PAGE CONFIG (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Nexus AI Logistics",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# 2. PATHS
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
model_path  = os.path.join(BASE_DIR, "dataco_rf_model.joblib")
scaler_path = os.path.join(BASE_DIR, "dataco_scaler.joblib")
cols_path   = os.path.join(BASE_DIR, "dataco_columns.joblib")

# ──────────────────────────────────────────────
# 3. DEMO CSV — embedded, no external file needed
#    Uses only valid regions/modes so it predicts
#    cleanly right out of the box.
# ──────────────────────────────────────────────
DEMO_CSV_CONTENT = """\
Shipping Mode,Order Region,Days_Scheduled,Order_Item_Quantity,Sales,Order_Profit_Per_Order
Standard Class,Southeast Asia,5,2,250.50,40.20
Second Class,South Asia,4,1,120.00,18.50
First Class,Oceania,2,3,450.75,80.40
Standard Class,Eastern Asia,6,4,300.60,45.00
Same Day,West Asia,1,1,75.00,12.00
Second Class,South Asia,4,2,210.30,35.10
Standard Class,Southeast Asia,5,5,520.00,95.30
First Class,West Asia,2,2,180.00,30.20
Standard Class,South Asia,6,3,340.00,50.00
Second Class,Oceania,4,1,95.00,15.00
Standard Class,Eastern Asia,5,2,260.00,38.20
First Class,Southeast Asia,2,4,480.00,85.60
Standard Class,Oceania,6,3,330.00,52.30
Second Class,West Asia,4,2,210.00,33.50
Same Day,South Asia,1,1,90.00,14.20
Standard Class,Eastern Asia,5,2,240.00,36.00
First Class,Southeast Asia,2,3,410.00,75.50
Second Class,West Asia,4,2,190.00,28.40
Standard Class,South Asia,6,4,350.00,55.00
First Class,Oceania,2,1,160.00,25.10
"""
DEMO_CSV_BYTES = DEMO_CSV_CONTENT.encode("utf-8")

# ──────────────────────────────────────────────
# 4. VALIDATION CONSTANTS
# ──────────────────────────────────────────────
REQUIRED_COLS    = ["Shipping Mode", "Order Region"]
OPTIONAL_COLS    = ["Days_Scheduled", "Order_Item_Quantity", "Sales", "Order_Profit_Per_Order"]
VALID_SHIP_MODES = ["Standard Class", "First Class", "Second Class", "Same Day"]
VALID_REGIONS    = ["Southeast Asia", "South Asia", "Oceania", "Eastern Asia", "West Asia"]
DEFAULT_VALS     = {
    "Days_Scheduled": 3,
    "Order_Item_Quantity": 1,
    "Sales": 150.0,
    "Order_Profit_Per_Order": 20.0,
}

# ──────────────────────────────────────────────
# 5. VALIDATION FUNCTION
#    Never raises. Returns (df, warnings, errors).
# ──────────────────────────────────────────────
def validate_and_clean(df: pd.DataFrame):
    errors, warnings = [], []

    if df.empty:
        errors.append("FILE IS EMPTY — upload a file with at least one data row.")
        return df, warnings, errors

    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        errors.append(
            f"Missing required column(s): **{', '.join(missing_required)}**  \n"
            f"Your file must contain: `Shipping Mode` and `Order Region`."
        )
        return df, warnings, errors

    missing_optional = [c for c in OPTIONAL_COLS if c not in df.columns]
    if missing_optional:
        warnings.append(
            f"Optional column(s) not found: `{'`, `'.join(missing_optional)}`  \n"
            f"Using defaults: {DEFAULT_VALS}"
        )

    bad_modes = df[~df["Shipping Mode"].isin(VALID_SHIP_MODES)]["Shipping Mode"].unique().tolist()
    if bad_modes:
        warnings.append(
            f"Unknown Shipping Mode value(s): `{bad_modes}`  \n"
            f"Affected rows defaulted to **Standard Class**."
        )
        df["Shipping Mode"] = df["Shipping Mode"].where(
            df["Shipping Mode"].isin(VALID_SHIP_MODES), "Standard Class"
        )

    bad_regions = df[~df["Order Region"].isin(VALID_REGIONS)]["Order Region"].unique().tolist()
    if bad_regions:
        warnings.append(
            f"Unknown Order Region value(s): `{bad_regions}`  \n"
            f"Affected rows defaulted to **Southeast Asia**."
        )
        df["Order Region"] = df["Order Region"].where(
            df["Order Region"].isin(VALID_REGIONS), "Southeast Asia"
        )

    for col in [c for c in OPTIONAL_COLS if c in df.columns]:
        coerced   = pd.to_numeric(df[col], errors="coerce")
        bad_count = int(coerced.isna().sum())
        if bad_count > 0:
            warnings.append(
                f"Column `{col}` has {bad_count} non-numeric value(s) — "
                f"replaced with default ({DEFAULT_VALS[col]})."
            )
            df[col] = coerced.fillna(DEFAULT_VALS[col])

    key_cols  = [c for c in REQUIRED_COLS if c in df.columns]
    null_rows = df[df[key_cols].isnull().all(axis=1)]
    if len(null_rows) > 0:
        warnings.append(f"{len(null_rows)} completely empty row(s) found and removed.")
        df = df.drop(null_rows.index).reset_index(drop=True)

    return df, warnings, errors


# ══════════════════════════════════════════════
# 6. GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg:        #03050a;
    --surface:   rgba(8, 14, 28, 0.82);
    --border:    rgba(0, 229, 255, 0.12);
    --border-hi: rgba(0, 229, 255, 0.45);
    --cyan:      #00e5ff;
    --violet:    #7c3aed;
    --rose:      #f43f5e;
    --green:     #00ffa3;
    --amber:     #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-body: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
}
html, body, [class*="css"] { font-family: var(--font-body) !important; background: var(--bg) !important; color: var(--text) !important; }
.stApp {
    background:
        repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,0.012) 2px,rgba(0,229,255,0.012) 4px),
        radial-gradient(ellipse 120% 80% at 20% 10%, rgba(124,58,237,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 80% 60% at 80% 80%, rgba(0,229,255,0.10) 0%, transparent 55%),
        #03050a !important;
    min-height: 100vh; overflow-x: hidden;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"]    { display: none; }
section[data-testid="stMain"] > div,
div[data-testid="stVerticalBlock"] { gap: 0 !important; }

.glass {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 2rem 2.25rem; backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    box-shadow: 0 0 0 1px rgba(0,229,255,0.04) inset, 0 24px 48px -12px rgba(0,0,0,0.7);
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
    transition: border-color .35s ease, box-shadow .35s ease;
}
.glass::before { content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,var(--cyan),transparent); opacity:0.4; }
.glass:hover { border-color:var(--border-hi);
    box-shadow: 0 0 0 1px rgba(0,229,255,0.08) inset, 0 32px 64px -12px rgba(0,0,0,0.8), 0 0 40px -10px rgba(0,229,255,0.12); }

/* Validation message boxes */
.val-error {
    background:rgba(244,63,94,0.07); border:1px solid rgba(244,63,94,0.45);
    border-left:4px solid #f43f5e; border-radius:10px; padding:1rem 1.25rem;
    font-family:var(--font-mono); font-size:.78rem; color:#fca5a5; margin:.5rem 0 1rem; line-height:1.8;
}
.val-warn {
    background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.35);
    border-left:4px solid #f59e0b; border-radius:10px; padding:.85rem 1.25rem;
    font-family:var(--font-mono); font-size:.76rem; color:#fcd34d; margin:.4rem 0; line-height:1.8;
}
.val-ok {
    background:rgba(0,255,163,0.05); border:1px solid rgba(0,255,163,0.3);
    border-left:4px solid #00ffa3; border-radius:10px; padding:.85rem 1.25rem;
    font-family:var(--font-mono); font-size:.76rem; color:#6ee7b7; margin:.4rem 0;
}
.demo-banner {
    display:flex; align-items:center; gap:12px;
    background:rgba(0,229,255,0.05); border:1px solid rgba(0,229,255,0.25);
    border-radius:12px; padding:.85rem 1.25rem; margin-bottom:1.25rem;
    font-family:var(--font-mono); font-size:.75rem; color:#94a3b8; letter-spacing:.5px;
}
.demo-dot { width:8px; height:8px; border-radius:50%; background:var(--cyan);
    box-shadow:0 0 8px var(--cyan); flex-shrink:0; animation:pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:.3} }

/* Hero */
.hero-wrap { text-align:center; padding:3.5rem 1rem 2.5rem; }
.hero-badge {
    display:inline-block; font-family:var(--font-mono); font-size:.72rem; letter-spacing:3px;
    color:var(--cyan); border:1px solid var(--border-hi); border-radius:40px; padding:6px 18px;
    margin-bottom:1.4rem; background:rgba(0,229,255,0.06); text-transform:uppercase;
    animation:pulse-badge 3s ease-in-out infinite;
}
@keyframes pulse-badge { 0%,100%{box-shadow:0 0 0 0 rgba(0,229,255,0.3)} 50%{box-shadow:0 0 0 8px rgba(0,229,255,0)} }
.hero-title {
    font-family:var(--font-body); font-weight:800; font-size:clamp(2.8rem,6vw,4.8rem);
    line-height:1.05; letter-spacing:-1px; margin:0 0 .8rem;
    background:linear-gradient(135deg,#ffffff 0%,var(--cyan) 50%,var(--violet) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub { font-family:var(--font-mono); font-size:.95rem; color:var(--muted);
    letter-spacing:.5px; margin:0 auto; max-width:560px; min-height:1.5rem; }

/* KPI */
.kpi-cell { background:rgba(0,229,255,0.04); border:1px solid var(--border); border-radius:14px;
    padding:1.25rem 1rem; text-align:center; position:relative; overflow:hidden;
    transition:border-color .3s,transform .3s; }
.kpi-cell::after { content:''; position:absolute; top:0; left:-100%; width:60%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(0,229,255,0.07),transparent);
    animation:shimmer 4s ease-in-out infinite; }
@keyframes shimmer { 0%{left:-100%} 50%,100%{left:160%} }
.kpi-cell:hover { border-color:var(--border-hi); transform:translateY(-3px); }
.kpi-label { font-family:var(--font-mono); font-size:.68rem; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-bottom:.6rem; }
.kpi-num   { font-family:var(--font-mono); font-size:2rem; font-weight:700; color:var(--cyan); text-shadow:0 0 20px rgba(0,229,255,0.45); }
.kpi-unit  { font-size:.9rem; color:var(--muted); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:rgba(0,229,255,0.04) !important; border:1px solid var(--border) !important; border-radius:12px !important; gap:8px !important; padding:6px !important; }
.stTabs [data-baseweb="tab"] { font-family:var(--font-mono) !important; font-size:.8rem !important; letter-spacing:1px !important; color:var(--muted) !important; border-radius:8px !important; border:none !important; padding:8px 20px !important; transition:all .25s ease !important; }
.stTabs [aria-selected="true"] { background:rgba(0,229,255,0.12) !important; color:var(--cyan) !important; box-shadow:0 0 12px rgba(0,229,255,0.2) !important; }
div[data-testid="stTabsContent"] { padding-top:1.5rem !important; }

/* Inputs */
.stSelectbox label,.stNumberInput label,.stSlider label { font-family:var(--font-mono) !important; font-size:.75rem !important; letter-spacing:1.5px !important; text-transform:uppercase !important; color:var(--muted) !important; margin-bottom:4px !important; }
.stSelectbox > div > div, .stNumberInput > div > div > input { background:rgba(0,5,15,0.7) !important; border:1px solid var(--border) !important; border-radius:10px !important; color:var(--text) !important; font-family:var(--font-mono) !important; font-size:.88rem !important; transition:border-color .25s,box-shadow .25s !important; }
.stSelectbox > div > div:focus-within, .stNumberInput > div > div:focus-within { border-color:var(--cyan) !important; box-shadow:0 0 0 3px rgba(0,229,255,0.12) !important; }
.stSlider > div[data-baseweb] > div > div { background:linear-gradient(90deg,var(--cyan),var(--violet)) !important; }

/* Button */
.stButton > button { font-family:var(--font-mono) !important; font-size:.8rem !important; letter-spacing:2px !important; text-transform:uppercase !important; background:transparent !important; border:1px solid var(--cyan) !important; color:var(--cyan) !important; border-radius:10px !important; padding:.75rem 1.5rem !important; width:100% !important; position:relative !important; overflow:hidden !important; transition:all .3s ease !important; box-shadow:0 0 16px rgba(0,229,255,0.15),inset 0 0 16px rgba(0,229,255,0.04) !important; }
.stButton > button:hover { color:#fff !important; border-color:var(--cyan) !important; box-shadow:0 0 32px rgba(0,229,255,0.35),inset 0 0 24px rgba(0,229,255,0.1) !important; transform:translateY(-2px) !important; }

/* Terminal */
.terminal { background:#000; border:1px solid rgba(0,229,255,0.25); border-radius:12px; padding:1.25rem 1.5rem; font-family:var(--font-mono); font-size:.8rem; color:var(--green); min-height:160px; overflow:hidden; box-shadow:0 0 40px rgba(0,255,163,0.08) inset,0 0 0 1px rgba(0,255,163,0.06); position:relative; }
.terminal::before { content:'● NEXUS NEURAL CORE  ○ READY'; position:absolute; top:10px; left:50%; transform:translateX(-50%); font-size:.65rem; letter-spacing:2px; color:rgba(0,255,163,0.4); }
.t-line { margin:6px 0; animation:fadeInLine .15s ease both; }
@keyframes fadeInLine { from{opacity:0;transform:translateX(-6px)} to{opacity:1;transform:none} }
.t-prompt { color:rgba(0,229,255,0.6); }
.t-cursor { display:inline-block; animation:blink 1s step-end infinite; }
@keyframes blink { 50%{opacity:0} }

/* Result cards */
.result-danger { background:rgba(244,63,94,0.08); border:1px solid rgba(244,63,94,0.5); border-radius:14px; padding:1.5rem; text-align:center; position:relative; overflow:hidden; animation:pulse-danger 2.5s ease-in-out infinite; }
@keyframes pulse-danger { 0%,100%{box-shadow:0 0 0 0 rgba(244,63,94,0.4)} 50%{box-shadow:0 0 0 14px rgba(244,63,94,0)} }
.result-success { background:rgba(0,255,163,0.06); border:1px solid rgba(0,255,163,0.4); border-radius:14px; padding:1.5rem; text-align:center; position:relative; overflow:hidden; }
.result-success::after { content:''; position:absolute; inset:0; background:linear-gradient(120deg,transparent 0%,rgba(0,255,163,0.1) 40%,transparent 80%); background-size:200% auto; animation:shine-success 3s linear infinite; }
@keyframes shine-success { 0%{background-position:-200% center} 100%{background-position:200% center} }
.result-icon  { font-size:2.4rem; display:block; margin-bottom:.5rem; }
.result-label { font-family:var(--font-mono); font-size:.75rem; letter-spacing:3px; text-transform:uppercase; opacity:.7; margin-top:.4rem; }

/* Feature bars */
.feat-bars { display:flex; flex-direction:column; gap:12px; margin-top:1rem; }
.feat-row  { display:flex; align-items:center; gap:12px; }
.feat-name { width:130px; flex-shrink:0; font-family:var(--font-mono); font-size:.72rem; color:var(--muted); text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.feat-track { flex:1; height:8px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden; }
.feat-fill  { height:100%; width:0%; border-radius:4px; background:linear-gradient(90deg,var(--cyan),var(--violet)); box-shadow:0 0 10px rgba(0,229,255,0.4); animation:grow-bar 1.1s cubic-bezier(.1,.6,.2,1) forwards; }
@keyframes grow-bar { to{width:var(--w)} }
.feat-pct  { width:36px; flex-shrink:0; font-family:var(--font-mono); font-size:.68rem; color:var(--cyan); text-align:right; }

/* Section headers */
.sec-header { font-family:var(--font-body); font-size:1.2rem; font-weight:700; color:#fff; margin-bottom:1.2rem; display:flex; align-items:center; gap:10px; }
.sec-header::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--border-hi),transparent); }
.sec-sub { font-family:var(--font-mono); font-size:.75rem; letter-spacing:1.5px; color:var(--muted); text-transform:uppercase; margin-bottom:.75rem; }

/* Misc */
hr { border-color:var(--border) !important; margin:2rem 0 !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#03050a; }
::-webkit-scrollbar-thumb { background:#1e293b; border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:#334155; }
.stAlert   { border-radius:10px !important; font-family:var(--font-mono) !important; font-size:.8rem !important; }
.stSuccess { border-color:rgba(0,255,163,.3) !important; background:rgba(0,255,163,.06) !important; }
.stError   { border-color:rgba(244,63,94,.3) !important; background:rgba(244,63,94,.06) !important; }
.stInfo    { border-color:rgba(0,229,255,.3) !important; background:rgba(0,229,255,.06) !important; }
.stWarning { border-color:rgba(245,158,11,.3) !important; background:rgba(245,158,11,.06) !important; }
.stDataFrame { border:1px solid var(--border) !important; border-radius:12px !important; overflow:hidden !important; }
.stDataFrame th { background:rgba(0,229,255,0.08) !important; font-family:var(--font-mono) !important; font-size:.75rem !important; }
.stDataFrame td { font-family:var(--font-mono) !important; font-size:.78rem !important; }
[data-testid="stFileUploader"] { border:1px dashed var(--border-hi) !important; border-radius:14px !important; background:rgba(0,229,255,0.03) !important; padding:1rem !important; transition:background .3s,border-color .3s !important; }
[data-testid="stFileUploader"]:hover { background:rgba(0,229,255,0.07) !important; border-color:var(--cyan) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color:var(--muted) !important; font-family:var(--font-mono) !important; font-size:.8rem !important; }
[data-testid="stDownloadButton"] > button { background:transparent !important; border:1px solid rgba(0,255,163,0.4) !important; color:var(--green) !important; font-family:var(--font-mono) !important; font-size:.78rem !important; letter-spacing:1.5px !important; border-radius:10px !important; transition:all .25s ease !important; }
[data-testid="stDownloadButton"] > button:hover { background:rgba(0,255,163,0.08) !important; box-shadow:0 0 20px rgba(0,255,163,0.25) !important; }
[data-testid="stExpander"] { border:1px solid var(--border) !important; border-radius:12px !important; background:rgba(0,5,15,0.5) !important; }
[data-testid="stExpander"] summary { font-family:var(--font-mono) !important; font-size:.8rem !important; color:var(--muted) !important; letter-spacing:1px !important; }
.stSpinner > div { border-top-color:var(--cyan) !important; }
[data-testid="stMetric"] { background:rgba(0,229,255,0.04); border:1px solid var(--border); border-radius:12px; padding:.75rem 1rem !important; }
[data-testid="stMetricLabel"] { font-family:var(--font-mono) !important; font-size:.7rem !important; letter-spacing:1.5px !important; text-transform:uppercase !important; color:var(--muted) !important; }
[data-testid="stMetricValue"] { font-family:var(--font-mono) !important; font-size:1.6rem !important; color:var(--cyan) !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 7. PARTICLE CANVAS
# ──────────────────────────────────────────────
components.html("""
<canvas id="c" style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;opacity:0.5;"></canvas>
<script>
(function(){
  const cv=document.getElementById('c'),cx=cv.getContext('2d');
  let W,H,P=[],N=[];
  function resize(){ W=cv.width=window.innerWidth; H=cv.height=window.innerHeight; }
  window.addEventListener('resize',resize); resize();
  for(let i=0;i<55;i++) P.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,r:Math.random()*1.5+.5,a:Math.random()});
  for(let i=0;i<12;i++) N.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.18,vy:(Math.random()-.5)*.18});
  function draw(){
    cx.clearRect(0,0,W,H);
    for(let i=0;i<N.length;i++){
      let n=N[i]; n.x+=n.vx; n.y+=n.vy;
      if(n.x<0||n.x>W)n.vx*=-1; if(n.y<0||n.y>H)n.vy*=-1;
      for(let j=i+1;j<N.length;j++){
        let m=N[j],d=Math.hypot(n.x-m.x,n.y-m.y);
        if(d<160){ cx.save(); cx.globalAlpha=(1-d/160)*0.3; cx.strokeStyle='#00e5ff'; cx.lineWidth=.6;
          cx.beginPath(); cx.moveTo(n.x,n.y); cx.lineTo(m.x,m.y); cx.stroke(); cx.restore(); }
      }
      cx.save(); cx.globalAlpha=0.7; cx.fillStyle='#00e5ff'; cx.shadowBlur=8; cx.shadowColor='#00e5ff';
      cx.beginPath(); cx.arc(n.x,n.y,2,0,Math.PI*2); cx.fill(); cx.restore();
    }
    for(let p of P){
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>W)p.vx*=-1; if(p.y<0||p.y>H)p.vy*=-1;
      cx.save(); cx.globalAlpha=p.a*0.4; cx.fillStyle='#7c3aed';
      cx.beginPath(); cx.arc(p.x,p.y,p.r,0,Math.PI*2); cx.fill(); cx.restore();
    }
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""", height=0)

# ──────────────────────────────────────────────
# 8. HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">⬡ PREDICTIVE LOGISTICS ENGINE v3.0</div>
    <h1 class="hero-title">NEXUS AI</h1>
    <div class="hero-sub" id="typewriter-el">Initializing intelligence layer...</div>
</div>
""", unsafe_allow_html=True)
components.html("""
<script>
  const el=window.parent.document.getElementById('typewriter-el');
  if(el){ const msg="Real-time delivery risk intelligence · Route deviation analysis · Fleet optimization";
    let i=0; el.textContent='';
    function tick(){ if(i<msg.length){ el.textContent+=msg[i++]; setTimeout(tick,32); } }
    setTimeout(tick,600); }
</script>
""", height=0)

# ──────────────────────────────────────────────
# 9. KPI RIBBON
# ──────────────────────────────────────────────
components.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&display=swap');
.kb{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:0 0 8px;font-family:'Space Mono',monospace;}
.kc{background:rgba(0,229,255,0.04);border:1px solid rgba(0,229,255,0.12);border-radius:14px;padding:1.25rem 1rem;text-align:center;position:relative;overflow:hidden;transition:border-color .3s,transform .3s;cursor:default;}
.kc:hover{border-color:rgba(0,229,255,0.45);transform:translateY(-3px);}
.kc::after{content:'';position:absolute;top:0;left:-100%;width:60%;height:100%;background:linear-gradient(90deg,transparent,rgba(0,229,255,0.07),transparent);animation:sh 4s ease-in-out infinite;}
@keyframes sh{0%{left:-100%}50%,100%{left:160%}}
.kl{font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:#64748b;margin-bottom:.6rem;}
.kv{font-size:1.9rem;font-weight:700;color:#00e5ff;text-shadow:0 0 18px rgba(0,229,255,0.5);}
.ku{font-size:.85rem;color:#64748b;}
</style>
<div class="kb">
  <div class="kc"><div class="kl">Model Accuracy</div><div class="kv"><span class="cnt" data-t="73.31" data-d="1">0.0</span><span class="ku">%</span></div></div>
  <div class="kc"><div class="kl">Routes Monitored</div><div class="kv"><span class="cnt" data-t="180519" data-d="0">0</span></div></div>
  <div class="kc"><div class="kl">Avg Inference Time</div><div class="kv"><span class="cnt" data-t="0.4" data-d="1">0.0</span><span class="ku">s</span></div></div>
</div>
<script>
document.querySelectorAll('.cnt').forEach(el=>{
  const target=+el.dataset.t,dec=+el.dataset.d,steps=108;
  let step=0;
  const id=setInterval(()=>{ step++; const cur=target*(step/steps);
    if(step>=steps){el.textContent=dec>0?target.toFixed(dec):target.toLocaleString();clearInterval(id);return;}
    el.textContent=dec>0?cur.toFixed(dec):Math.ceil(cur).toLocaleString();
  },1000/60);
});
</script>
""", height=145)

# ──────────────────────────────────────────────
# 10. LOAD MODEL
# ──────────────────────────────────────────────
try:
    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    model_columns = joblib.load(cols_path)
    st.success("⬡  AI BRAIN ONLINE — Random Forest ensemble loaded successfully")
except Exception as e:
    st.error(f"⬡  MODEL ERROR — {e}")
    st.stop()

st.markdown("<hr>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 11. TABS
# ──────────────────────────────────────────────
tab_single, tab_bulk, tab_about = st.tabs(["⬡  SINGLE ORDER", "⬡  BULK BATCH", "⬡  SYSTEM INFO"])

# ════════════════════════════════════════════════
# TAB 1 — SINGLE ORDER
# ════════════════════════════════════════════════
with tab_single:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">⬡ Configure Shipment Parameters</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        shipping_mode = st.selectbox("Shipping Mode Carrier", VALID_SHIP_MODES,
                                     help="Carrier class affects baseline transit SLA")
        order_region  = st.selectbox("Destination Hub Region", VALID_REGIONS,
                                     help="Regional hub affects historical delay probability")
    with col_r:
        days_scheduled      = st.number_input("Scheduled Transit Days", min_value=0, max_value=10,
                                              value=3, help="SLA-agreed delivery window")
        order_item_quantity = st.slider("Freight Quantity (Units)", 1, 5, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⬡  Advanced Parameters (optional)"):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            adv_sales  = st.number_input("Order Value ($)", min_value=0.0, value=150.0, step=10.0)
        with adv_col2:
            adv_profit = st.number_input("Profit Per Order ($)", min_value=0.0, value=20.0, step=5.0)

    run_btn = st.button("⬡  RUN PREDICTIVE ANALYSIS", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run_btn:
        term_ph = st.empty()
        logs = [
            ("Initializing Nexus Neural Core v3.0",        "ok"),
            ("Loading Random Forest ensemble (n=100)",     "ok"),
            ("Extracting logistics feature tensor",        "ok"),
            ("Encoding categorical embeddings → one-hot", "ok"),
            ("Applying StandardScaler normalization",      "ok"),
            ("Running probability inference pipeline",     "ok"),
            ("Computing feature attribution scores",       "ok"),
            ("Building route deviation projection",        "ok"),
            ("Fusing multi-axis risk vectors",             "ok"),
            ("ANALYSIS COMPLETE",                          "done"),
        ]
        rendered = ""
        for msg, kind in logs:
            color  = "#00ffa3" if kind == "ok" else "#00e5ff"
            marker = "✓" if kind == "ok" else "■"
            rendered += f'<div class="t-line"><span class="t-prompt">[NEXUS] </span><span style="color:{color}">{marker} {msg}</span></div>'
            term_ph.markdown(
                f'<div class="terminal" style="padding-top:2rem">{rendered}'
                f'<div class="t-line"><span class="t-prompt">[NEXUS] </span><span class="t-cursor">█</span></div></div>',
                unsafe_allow_html=True)
            time.sleep(0.22)
        term_ph.empty()

        try:
            expected_cols = model.feature_names_in_
            data_dict = {col: 0 for col in expected_cols}
            if "Days_Scheduled"         in data_dict: data_dict["Days_Scheduled"]         = days_scheduled
            if "Order_Item_Quantity"     in data_dict: data_dict["Order_Item_Quantity"]     = order_item_quantity
            if "Sales"                  in data_dict: data_dict["Sales"]                  = adv_sales
            if "Order_Profit_Per_Order"  in data_dict: data_dict["Order_Profit_Per_Order"]  = adv_profit
            ship_col = f"Shipping_Mode_{shipping_mode}"
            reg_col  = f"Order_Region_{order_region}"
            if ship_col in data_dict: data_dict[ship_col] = 1
            if reg_col  in data_dict: data_dict[reg_col]  = 1

            input_df    = pd.DataFrame([data_dict])[expected_cols]
            scaler_cols = scaler.feature_names_in_
            input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
            prediction  = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]
            risk_pct    = probability * 100

            st.toast("⬡  Analysis complete", icon="✅")
            st.markdown("<hr>", unsafe_allow_html=True)
            r1, r2 = st.columns([1, 1], gap="large")

            with r1:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.markdown('<div class="sec-header">⬡ Risk Assessment</div>', unsafe_allow_html=True)

                if prediction[0] == 1:
                    st.markdown("""<div class="result-danger">
                        <span class="result-icon">🚩</span>
                        <div style="font-family:var(--font-mono);font-size:1.05rem;font-weight:700;color:#fca5a5;letter-spacing:2px;">HIGH RISK DETECTED</div>
                        <div class="result-label">Shipment highly likely to be delayed</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="result-success">
                        <span class="result-icon">✅</span>
                        <div style="font-family:var(--font-mono);font-size:1.05rem;font-weight:700;color:#6ee7b7;letter-spacing:2px;">ON-TIME PREDICTION</div>
                        <div class="result-label">Shipment on track for standard delivery</div>
                    </div>""", unsafe_allow_html=True)

                dash_total      = 157
                dash_offset_end = dash_total - (dash_total * probability)
                gauge_color     = "#f43f5e" if probability > 0.65 else "#f59e0b" if probability > 0.35 else "#00ffa3"

                components.html(f"""
                <style>
                  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&display=swap');
                  .g-wrap{{position:relative;width:220px;margin:1.25rem auto 0;}}
                  .g-wrap svg{{display:block;overflow:visible;width:100%;}}
                  .g-track{{fill:none;stroke:rgba(255,255,255,0.07);stroke-width:10;stroke-linecap:round;}}
                  .g-fill{{fill:none;stroke:{gauge_color};stroke-width:10;stroke-linecap:round;
                           stroke-dasharray:{dash_total};stroke-dashoffset:{dash_total};
                           filter:drop-shadow(0 0 6px {gauge_color});
                           transition:stroke-dashoffset 1.4s cubic-bezier(.2,.8,.2,1);}}
                  .g-label{{position:absolute;left:50%;top:81.25%;transform:translate(-50%,-50%);text-align:center;pointer-events:none;}}
                  .g-pct{{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;color:{gauge_color};text-shadow:0 0 18px {gauge_color}80;line-height:1;display:block;}}
                  .g-sub{{font-family:'Space Mono',monospace;font-size:.58rem;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-top:4px;display:block;}}
                </style>
                <div class="g-wrap">
                  <svg viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">
                    <path d="M 10 65 A 50 50 0 0 1 110 65" class="g-track"/>
                    <path d="M 10 65 A 50 50 0 0 1 110 65" class="g-fill" id="gf"/>
                  </svg>
                  <div class="g-label">
                    <span class="g-pct">{risk_pct:.1f}%</span>
                    <span class="g-sub">Delay&nbsp;Probability</span>
                  </div>
                </div>
                <script>
                  setTimeout(function(){{
                    var el=document.getElementById('gf');
                    if(el) el.style.strokeDashoffset='{dash_offset_end:.4f}';
                  }},80);
                </script>
                """, height=200)

                conf_color = "#00ffa3" if probability < 0.35 else "#f59e0b" if probability < 0.65 else "#f43f5e"
                conf_label = "LOW RISK" if probability < 0.35 else "MODERATE" if probability < 0.65 else "HIGH RISK"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;margin-top:1.2rem;
                            font-family:var(--font-mono);font-size:.72rem;color:var(--muted);letter-spacing:1px;">
                    <span>CONFIDENCE BAND</span>
                    <span style="color:{conf_color}">{conf_label}</span>
                </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with r2:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.markdown('<div class="sec-header">⬡ Explainable AI</div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-sub">Top Driving Factors</div>', unsafe_allow_html=True)

                importances   = model.feature_importances_
                feature_names = model.feature_names_in_
                imp_df = (pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                          .sort_values('Importance', ascending=False).head(6))
                max_imp   = imp_df['Importance'].max()
                bars_html = '<div class="feat-bars">'
                for _, row in imp_df.iterrows():
                    label = (str(row['Feature'])
                             .replace('Category_Name_','').replace('Customer_Segment_','')
                             .replace('Shipping_Mode_','').replace('Order_Region_',''))
                    w         = (row['Importance'] / max_imp) * 100
                    pct_label = f"{row['Importance']*100:.1f}"
                    bars_html += f"""<div class="feat-row">
                      <div class="feat-name" title="{label}">{label}</div>
                      <div class="feat-track"><div class="feat-fill" style="--w:{w:.1f}%"></div></div>
                      <div class="feat-pct">{pct_label}</div></div>"""
                bars_html += '</div>'
                st.markdown(bars_html, unsafe_allow_html=True)

                st.markdown('<div class="sec-sub" style="margin-top:1.5rem">Risk Vector Profile</div>', unsafe_allow_html=True)
                radar_df  = imp_df.head(5).copy()
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=radar_df['Importance'] * 100,
                    theta=radar_df['Feature'].str.replace(r'.+_', '', regex=True),
                    fill='toself', fillcolor='rgba(124,58,237,0.25)',
                    line={'color':'#00e5ff','width':2}, marker={'color':'#00e5ff','size':5},
                ))
                fig_radar.update_layout(
                    polar={'bgcolor':'rgba(0,0,0,0)',
                           'radialaxis':{'visible':False,'range':[0,max_imp*110]},
                           'angularaxis':{'tickfont':{'color':'#64748b','size':9,'family':'Space Mono'}}},
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8',family='Space Mono'),
                    margin=dict(l=30,r=30,t=20,b=20), height=220,
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar':False})

                st.markdown('<div class="sec-sub">Route Deviation Projection</div>', unsafe_allow_html=True)
                extra       = round(probability * 5) if prediction[0] == 1 else 0
                actual_days = days_scheduled + extra
                tl_df = pd.DataFrame([
                    {"Task":"Scheduled SLA","Start":0,"Finish":days_scheduled,"Type":"Scheduled"},
                    {"Task":"AI Projected", "Start":0,"Finish":actual_days,   "Type":"Delayed" if extra>0 else "Scheduled"},
                ])
                fig_tl = px.bar(tl_df, x="Finish", y="Task", color="Type", orientation='h',
                                color_discrete_map={"Scheduled":"#00e5ff","Delayed":"#f43f5e"}, opacity=0.85)
                fig_tl.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color':'#94a3b8','family':'Space Mono','size':10},
                    margin={'l':0,'r':10,'t':5,'b':30},
                    xaxis={'title':'Days After Dispatch','showgrid':True,'gridcolor':'rgba(255,255,255,0.06)','color':'#64748b'},
                    yaxis={'title':'','color':'#64748b'}, showlegend=False, height=140,
                )
                st.plotly_chart(fig_tl, use_container_width=True, config={'displayModeBar':False})
                st.info(f"⬡  **Insight**: {'At-risk shipment — consider expediting or re-routing.' if prediction[0]==1 else 'All signals nominal — no intervention required.'}")
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⬡  INFERENCE ERROR — {e}")

# ════════════════════════════════════════════════
# TAB 2 — BULK BATCH
# ════════════════════════════════════════════════
with tab_bulk:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">⬡ Enterprise Batch Processing</div>', unsafe_allow_html=True)

    # Required format reference card
    st.markdown("""
    <div style="background:rgba(0,229,255,0.03);border:1px solid rgba(0,229,255,0.15);
                border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.25rem;">
        <div style="font-family:var(--font-mono);font-size:.7rem;letter-spacing:2px;
                    color:#64748b;text-transform:uppercase;margin-bottom:.6rem;">⬡ Required Column Format</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">
            <span style="font-family:var(--font-mono);font-size:.72rem;padding:3px 10px;background:rgba(244,63,94,0.12);border:1px solid rgba(244,63,94,0.35);border-radius:6px;color:#fca5a5;">Shipping Mode ✱</span>
            <span style="font-family:var(--font-mono);font-size:.72rem;padding:3px 10px;background:rgba(244,63,94,0.12);border:1px solid rgba(244,63,94,0.35);border-radius:6px;color:#fca5a5;">Order Region ✱</span>
            <span style="font-family:var(--font-mono);font-size:.72rem;padding:3px 10px;background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.2);border-radius:6px;color:#94a3b8;">Days_Scheduled</span>
            <span style="font-family:var(--font-mono);font-size:.72rem;padding:3px 10px;background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.2);border-radius:6px;color:#94a3b8;">Order_Item_Quantity</span>
            <span style="font-family:var(--font-mono);font-size:.72rem;padding:3px 10px;background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.2);border-radius:6px;color:#94a3b8;">Sales</span>
            <span style="font-family:var(--font-mono);font-size:.72rem;padding:3px 10px;background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.2);border-radius:6px;color:#94a3b8;">Order_Profit_Per_Order</span>
        </div>
        <div style="font-family:var(--font-mono);font-size:.65rem;color:#475569;margin-top:.6rem;">
            ✱ Required &nbsp;·&nbsp; Others optional (defaults used if missing)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Valid values expander
    with st.expander("⬡  Valid values for Shipping Mode & Order Region"):
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("""<div class="sec-sub">Shipping Mode</div>
            <ul style="font-family:var(--font-mono);font-size:.78rem;color:#94a3b8;line-height:2.2;margin:0;padding-left:1.2rem;">
              <li>Standard Class</li><li>First Class</li><li>Second Class</li><li>Same Day</li>
            </ul>""", unsafe_allow_html=True)
        with vc2:
            st.markdown("""<div class="sec-sub">Order Region</div>
            <ul style="font-family:var(--font-mono);font-size:.78rem;color:#94a3b8;line-height:2.2;margin:0;padding-left:1.2rem;">
              <li>Southeast Asia</li><li>South Asia</li><li>Oceania</li><li>Eastern Asia</li><li>West Asia</li>
            </ul>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Demo download banner
    st.markdown("""
    <div class="demo-banner">
        <div class="demo-dot"></div>
        <span>No file yet? Download the demo CSV below — it shows correct formatting and runs predictions immediately.</span>
    </div>
    """, unsafe_allow_html=True)
    dl1, _ = st.columns([1, 3])
    with dl1:
        st.download_button("⬡  DOWNLOAD DEMO CSV", data=DEMO_CSV_BYTES,
                           file_name="nexus_demo_batch.csv", mime="text/csv",
                           use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Upload Your CSV File</div>', unsafe_allow_html=True)
    uploaded   = st.file_uploader("", type=["csv"], key="bulk_uploader")
    using_demo = uploaded is None

    if using_demo:
        st.markdown("""<div class="val-ok">
            ⬡ No file uploaded — demo data loaded automatically.
            Upload your own CSV above to replace it.
        </div>""", unsafe_allow_html=True)
        raw_source = io.BytesIO(DEMO_CSV_BYTES)
    else:
        raw_source = uploaded

    # ── Read CSV — never crash ───────────────────
    read_ok = False
    try:
        bulk_df_raw = pd.read_csv(raw_source)
        read_ok     = True
    except Exception as read_err:
        st.markdown(f"""<div class="val-error">
            ⬡ COULD NOT READ FILE — {read_err}<br><br>
            Make sure your file is a valid, UTF-8 encoded CSV and try again.
        </div>""", unsafe_allow_html=True)

    if read_ok:
        # ── Validate ─────────────────────────────
        bulk_df, val_warnings, val_errors = validate_and_clean(bulk_df_raw.copy())

        # Show all errors — DO NOT run prediction
        if val_errors:
            for err in val_errors:
                st.markdown(f'<div class="val-error">⬡ ERROR — {err}</div>',
                            unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:var(--font-mono);font-size:.75rem;color:#64748b;
                        text-align:center;padding:1.2rem 0 .5rem;">
                Fix the error(s) above and re-upload your file. No prediction was run.
            </div>""", unsafe_allow_html=True)

        else:
            # Show all warnings — continue to predict
            for w in val_warnings:
                st.markdown(f'<div class="val-warn">⬡ WARNING — {w}</div>',
                            unsafe_allow_html=True)

            demo_label = " (demo)" if using_demo else ""
            st.markdown(f'<div class="sec-sub" style="margin-top:1rem">Preview{demo_label} — {len(bulk_df):,} records ready</div>',
                        unsafe_allow_html=True)
            st.dataframe(bulk_df.head(8), use_container_width=True)

            btn_label = "⬡  RUN DEMO PREDICTION" if using_demo else "⬡  EXECUTE ENTERPRISE BATCH PREDICTION"
            if st.button(btn_label, use_container_width=True):
                pred_ok = False
                try:
                    with st.spinner("⬡  Nexus Engine classifying batch..."):
                        expected_cols = model.feature_names_in_
                        processed     = []
                        for _, row in bulk_df.iterrows():
                            d = {col: 0 for col in expected_cols}
                            if "Days_Scheduled"         in expected_cols: d["Days_Scheduled"]         = row.get("Days_Scheduled",         DEFAULT_VALS["Days_Scheduled"])
                            if "Order_Item_Quantity"     in expected_cols: d["Order_Item_Quantity"]     = row.get("Order_Item_Quantity",    DEFAULT_VALS["Order_Item_Quantity"])
                            if "Sales"                  in expected_cols: d["Sales"]                  = row.get("Sales",                  DEFAULT_VALS["Sales"])
                            if "Order_Profit_Per_Order"  in expected_cols: d["Order_Profit_Per_Order"]  = row.get("Order_Profit_Per_Order", DEFAULT_VALS["Order_Profit_Per_Order"])
                            ship_col = f"Shipping_Mode_{row.get('Shipping Mode','Standard Class')}"
                            reg_col  = f"Order_Region_{row.get('Order Region','Southeast Asia')}"
                            if ship_col in d: d[ship_col] = 1
                            if reg_col  in d: d[reg_col]  = 1
                            processed.append(d)

                        in_df = pd.DataFrame(processed)[expected_cols].copy()
                        in_df[scaler.feature_names_in_] = scaler.transform(in_df[scaler.feature_names_in_])
                        preds = model.predict(in_df)
                        probs = model.predict_proba(in_df)[:, 1]
                    pred_ok = True

                except Exception as pred_err:
                    st.markdown(f"""<div class="val-error">
                        ⬡ PREDICTION ENGINE ERROR — {pred_err}<br><br>
                        This is likely a data type mismatch. Ensure all numeric columns contain only numbers and try again.
                    </div>""", unsafe_allow_html=True)

                if pred_ok:
                    bulk_df["⬡ Risk Verdict"]  = ["🚩 Late Delivery Risk" if p == 1 else "✅ On Time" for p in preds]
                    bulk_df["⬡ Risk Score %"] = (probs * 100).round(2)
                    n_at_risk = int(np.sum(preds == 1))
                    n_on_time = int(np.sum(preds == 0))
                    avg_risk  = float(probs.mean() * 100)

                    s1, s2, s3 = st.columns(3)
                    s1.metric("At-Risk Shipments",  f"{n_at_risk:,}",
                              delta=f"{n_at_risk/len(preds)*100:.1f}% of batch", delta_color="inverse")
                    s2.metric("On-Time Predicted",  f"{n_on_time:,}",
                              delta=f"{n_on_time/len(preds)*100:.1f}% of batch")
                    s3.metric("Average Risk Score", f"{avg_risk:.1f}%")

                    if n_at_risk == 0:
                        st.balloons()

                    st.success(f"⬡  Batch complete — {len(bulk_df):,} records classified")
                    st.dataframe(bulk_df, use_container_width=True)

                    export_name = "nexus_demo_predictions.csv" if using_demo else "nexus_batch_predictions.csv"
                    st.download_button(
                        "⬡  EXPORT CLASSIFIED RECORDS (.CSV)",
                        data=bulk_df.to_csv(index=False).encode("utf-8"),
                        file_name=export_name, mime="text/csv", use_container_width=True,
                    )

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 3 — SYSTEM INFO
# ════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">⬡ System Architecture</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown("""<div class="sec-sub">Core Engine</div>
        <table style="width:100%;font-family:var(--font-mono);font-size:.78rem;border-collapse:collapse;color:#e2e8f0">
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><td style="padding:.5rem 0;color:#64748b">Algorithm</td><td>Random Forest Classifier</td></tr>
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><td style="padding:.5rem 0;color:#64748b">Estimators</td><td>100 trees</td></tr>
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><td style="padding:.5rem 0;color:#64748b">Feature Set</td><td>One-hot + Scaled numerics</td></tr>
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><td style="padding:.5rem 0;color:#64748b">Accuracy</td><td style="color:#00e5ff">73.31%</td></tr>
          <tr><td style="padding:.5rem 0;color:#64748b">Dataset</td><td>DataCo Global Supply Chain</td></tr>
        </table>""", unsafe_allow_html=True)
    with a2:
        st.markdown("""<div class="sec-sub">Performance Metrics</div>
        <table style="width:100%;font-family:var(--font-mono);font-size:.78rem;border-collapse:collapse;color:#e2e8f0">
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><td style="padding:.5rem 0;color:#64748b">Routes Indexed</td><td style="color:#00ffa3">180,519</td></tr>
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><td style="padding:.5rem 0;color:#64748b">Avg Inference</td><td>0.4s</td></tr>
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><td style="padding:.5rem 0;color:#64748b">Batch Throughput</td><td>&gt;10k rows/min</td></tr>
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><td style="padding:.5rem 0;color:#64748b">Normalizer</td><td>StandardScaler</td></tr>
          <tr><td style="padding:.5rem 0;color:#64748b">Framework</td><td>scikit-learn · Streamlit</td></tr>
        </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Accepted Input Values</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown("""
        <table style="width:100%;font-family:var(--font-mono);font-size:.75rem;border-collapse:collapse;color:#e2e8f0">
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><th style="color:#00e5ff;text-align:left;padding:.4rem 0">Shipping Mode</th></tr>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:.35rem 0;color:#94a3b8">Standard Class</td></tr>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:.35rem 0;color:#94a3b8">First Class</td></tr>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:.35rem 0;color:#94a3b8">Second Class</td></tr>
          <tr><td style="padding:.35rem 0;color:#94a3b8">Same Day</td></tr>
        </table>""", unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <table style="width:100%;font-family:var(--font-mono);font-size:.75rem;border-collapse:collapse;color:#e2e8f0">
          <tr style="border-bottom:1px solid rgba(0,229,255,0.1)"><th style="color:#00e5ff;text-align:left;padding:.4rem 0">Order Region</th></tr>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:.35rem 0;color:#94a3b8">Southeast Asia</td></tr>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:.35rem 0;color:#94a3b8">South Asia</td></tr>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:.35rem 0;color:#94a3b8">Oceania</td></tr>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:.35rem 0;color:#94a3b8">Eastern Asia</td></tr>
          <tr><td style="padding:.35rem 0;color:#94a3b8">West Asia</td></tr>
        </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:var(--font-mono);font-size:.72rem;color:#334155;text-align:center;letter-spacing:2px;">
    NEXUS AI LOGISTICS ENGINE · BUILT ON DATACO GLOBAL SUPPLY CHAIN DATASET · © 2025
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
