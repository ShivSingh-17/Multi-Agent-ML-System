import streamlit as st
import sys, os, json, builtins, tempfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from graph.workflow import build_workflow
from state.ml_state import MLState

st.set_page_config(
    page_title="AutoML Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { 
    background: #f0f4fb !important;
    font-family: 'Inter', sans-serif;
}

/* Hide sidebar completely */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { 
    padding: 1.8rem 2rem 4rem !important; 
    max-width: 100% !important;
}

/* ── Left Panel Card ── */
.input-panel {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    height: fit-content;
    position: sticky;
    top: 20px;
}
.panel-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid #f0f4f8;
}
.panel-divider { height: 1px; background: #f0f4f8; margin: 16px 0; }
.panel-info {
    font-size: 12.5px;
    color: #94a3b8;
    line-height: 2;
}
.panel-info span { color: #475569; font-weight: 600; }

/* ── Header ── */
.mc-header {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(79,70,229,0.22);
}
.mc-title    { font-size: 1.5rem; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
.mc-subtitle { color: rgba(255,255,255,0.72); font-size: 13px; margin-top: 3px; }
.mc-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #fff;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px; padding: 5px 13px;
    background: rgba(255,255,255,0.15);
    font-weight: 500; white-space: nowrap;
}

/* ── Section label ── */
.section-label {
    font-size: 10.5px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #a0aec0; margin: 20px 0 10px;
    display: flex; align-items: center; gap: 10px;
}
.section-label::after { content:''; flex:1; height:1px; background:#e2e8f0; }

/* ── Agent Cards ── */
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-bottom: 6px;
}
.agent-card {
    background: #fff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 16px;
    position: relative; overflow: hidden;
    transition: all 0.3s ease;
    min-height: 150px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.agent-card::before {
    content:''; position:absolute;
    top:0; left:0; right:0; height:3px;
    border-radius:14px 14px 0 0;
    transition: background 0.3s;
}
.agent-card.pending::before  { background: #e2e8f0; }
.agent-card.running  { border-color:#fed7aa; box-shadow:0 4px 18px rgba(251,146,60,0.14); }
.agent-card.running::before  { background:linear-gradient(90deg,#fb923c,#fbbf24); animation:sh 1.4s ease infinite; }
.agent-card.complete { border-color:#bbf7d0; box-shadow:0 4px 18px rgba(34,197,94,0.09); }
.agent-card.complete::before { background:linear-gradient(90deg,#22c55e,#16a34a); }
@keyframes sh { 0%,100%{opacity:1} 50%{opacity:0.5} }

.agent-num  { font-family:'JetBrains Mono',monospace; font-size:9.5px; color:#c5cfe0; letter-spacing:2px; margin-bottom:10px; }
.agent-icon { font-size:19px; width:36px; height:36px; background:#f8fafc; border:1.5px solid #e2e8f0; border-radius:9px; display:flex; align-items:center; justify-content:center; }
.agent-name { font-weight:700; font-size:13px; color:#1a202c; margin:9px 0 4px; }
.agent-desc { font-size:11px; color:#a0aec0; line-height:1.55; }
.agent-status { margin-top:11px; font-size:11px; font-weight:600; display:flex; align-items:center; gap:6px; }
.status-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.status-dot.pending  { background:#d1d9e6; }
.status-dot.running  { background:#fb923c; box-shadow:0 0 0 3px #ffedd5; animation:pl 1.1s ease infinite; }
.status-dot.complete { background:#22c55e; box-shadow:0 0 0 3px #dcfce7; }
@keyframes pl { 0%,100%{opacity:1} 50%{opacity:0.4} }
.status-text-pending  { color:#c5cfe0; }
.status-text-running  { color:#ea6f0c; }
.status-text-complete { color:#16a34a; }

/* ── Terminal ── */
.terminal-wrap {
    background:#1a1b2e; border-radius:14px;
    overflow:hidden; margin-bottom:6px;
    box-shadow:0 6px 24px rgba(0,0,0,0.13);
}
.terminal-topbar {
    background:#141525; padding:10px 16px;
    display:flex; align-items:center; gap:10px;
    border-bottom:1px solid #252640;
}
.terminal-dots { display:flex; gap:7px; }
.td  { width:11px; height:11px; border-radius:50%; }
.td-r{background:#ff5f57;}.td-y{background:#febc2e;}.td-g{background:#28c840;}
.terminal-label { font-family:'JetBrains Mono',monospace; font-size:10px; color:#3d3d5c; letter-spacing:2px; text-transform:uppercase; margin-left:8px; }
.terminal-body {
    font-family:'JetBrains Mono',monospace; font-size:12px; line-height:1.8;
    padding:18px 20px; min-height:260px; max-height:420px;
    overflow-y:auto; white-space:pre-wrap; word-break:break-word;
}
.terminal-body::-webkit-scrollbar { width:4px; }
.terminal-body::-webkit-scrollbar-thumb { background:#2e2e50; border-radius:2px; }
.log-divider{color:#252640;} .log-agent{color:#a78bfa;font-weight:600;}
.log-info{color:#60cdff;}    .log-success{color:#5af7a0;}
.log-warning{color:#ffd16a;} .log-best{color:#ff9a5c;font-weight:600;}
.log-metric{color:#e2e8f0;}  .log-dim{color:#2e3060;}
.log-default{color:#7878a0;}

/* ── Metrics ── */
.metrics-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:6px; }
.metric-tile  { background:#fff; border:1.5px solid #e2e8f0; border-radius:13px; padding:20px 16px; text-align:center; box-shadow:0 1px 6px rgba(0,0,0,0.04); }
.metric-tile-value { font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:700; background:linear-gradient(135deg,#4f46e5,#0ea5e9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.metric-tile-label { font-size:9.5px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#a0aec0; margin-top:6px; }

/* ── Model Table ── */
.model-table-wrap { background:#fff; border:1.5px solid #e2e8f0; border-radius:13px; overflow:hidden; margin-bottom:6px; box-shadow:0 1px 6px rgba(0,0,0,0.04); }
.model-table { width:100%; border-collapse:collapse; font-family:'Inter',sans-serif; font-size:13px; }
.model-table th { background:#f8fafc; color:#a0aec0; font-size:9.5px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; padding:12px 16px; text-align:left; border-bottom:1.5px solid #e2e8f0; }
.model-table td { padding:13px 16px; border-bottom:1px solid #f4f6fb; color:#4a5568; vertical-align:middle; }
.model-table tr:last-child td { border-bottom:none; }
.model-table tr:hover td { background:#f9fafc; }
.model-table tr.best-row td { color:#1a202c; background:#fffbeb; font-weight:500; }
.best-badge { display:inline-block; background:linear-gradient(135deg,#4f46e5,#0ea5e9); color:#fff; border-radius:20px; padding:2px 9px; font-size:8.5px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-left:7px; }
.score-bar-bg   { background:#f0f4f8; border-radius:3px; height:4px; margin-top:5px; }
.score-bar-fill { height:4px; border-radius:3px; background:linear-gradient(90deg,#4f46e5,#0ea5e9); }
.score-bar-fill.best { background:linear-gradient(90deg,#f59e0b,#fb923c); }

/* ── Report ── */
.report-panel { background:#fff; border:1.5px solid #e2e8f0; border-radius:13px; padding:28px 30px; font-size:13.5px; line-height:1.9; margin-bottom:6px; box-shadow:0 1px 6px rgba(0,0,0,0.04); }
.report-panel h1 { font-size:1.1rem; color:#1a202c; font-weight:700; margin:18px 0 7px; }
.report-panel h2 { font-size:0.95rem; color:#4f46e5; font-weight:700; margin:16px 0 6px; }
.report-panel h3 { font-size:0.88rem; color:#0ea5e9; font-weight:600; margin:12px 0 5px; }
.report-panel        { color: #374151 !important; }
.report-panel p      { color: #374151 !important; margin-bottom: 10px; }
.report-panel li     { color: #4a5568 !important; margin-bottom: 4px; }
.report-panel ul, .report-panel ol { color: #4a5568 !important; padding-left: 20px; }
.report-panel strong { color: #1a202c !important; font-weight: 700; }
.report-panel span   { color: #374151 !important; }
.report-panel code { background:#f0f4ff; color:#4f46e5; padding:1px 6px; border-radius:4px; font-family:'JetBrains Mono',monospace; font-size:11.5px; border:1px solid #c7d2fe; }

/* Report inline table */
.report-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin: 14px 0;
    font-family: 'Inter', sans-serif;
}
.report-table th {
    background: #f0f4ff;
    color: #4f46e5;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid #c7d2fe;
}
.report-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #f0f4f8;
    color: #374151 !important;
    font-size: 13px;
}
.report-table tr:last-child td { border-bottom: none; }
.report-table tr:hover td      { background: #fafbff; }
.report-table tr.best-report-row td {
    background: #fffbeb;
    font-weight: 600;
    color: #1a202c !important;
}

/* ── Streamlit widget overrides ── */
.stTextInput input, .stTextArea textarea {
    background:#f7f9fc !important; border:1.5px solid #dde3ee !important;
    color:#1a202c !important; border-radius:9px !important;
    font-family:'Inter',sans-serif !important; font-size:13.5px !important;
    caret-color: #1a202c !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color:#a5b4fc !important;
    box-shadow:0 0 0 3px #eef2ff !important;
    background:#fff !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background:#f7f9fc !important; border:1.5px solid #dde3ee !important;
    border-radius:9px !important; color:#1a202c !important;
}
.stSelectbox [data-baseweb="select"] * { color:#1a202c !important; font-size:13.5px !important; }
label { color:#4a5568 !important; font-size:11.5px !important; font-weight:600 !important; letter-spacing:0.3px !important; font-family:'Inter',sans-serif !important; }
.stFileUploader section { background:#f7f9fc !important; border:2px dashed #c5cfe0 !important; border-radius:10px !important; }
.stFileUploader p { color:#4a5568 !important; font-size:13px !important; }
.stButton > button {
    background:linear-gradient(135deg,#4f46e5,#0ea5e9) !important;
    color:#fff !important; border:none !important; border-radius:9px !important;
    padding:12px 20px !important; font-family:'Inter',sans-serif !important;
    font-weight:700 !important; font-size:13px !important; width:100% !important;
    box-shadow:0 4px 14px rgba(79,70,229,0.3) !important;
    transition:opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity:0.88 !important; transform:translateY(-1px) !important; }
.stDownloadButton > button {
    background:#f0f4ff !important; color:#4f46e5 !important;
    border:1.5px solid #c7d2fe !important; border-radius:9px !important;
    font-family:'Inter',sans-serif !important; font-size:12.5px !important;
    font-weight:600 !important; width:100% !important;
}

</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
AGENTS = [
    {"id":"data_analyst",     "num":"01","icon":"🔍","name":"Data Analyst",     "desc":"Loads dataset · detects problem type · identifies patterns & correlations"},
    {"id":"feature_engineer", "num":"02","icon":"⚙️","name":"Feature Engineer", "desc":"Handles missing values · encodes categoricals · drops bad columns"},
    {"id":"model_trainer",    "num":"03","icon":"🤖","name":"Model Trainer",     "desc":"Trains multiple models · 5-fold CV · selects best performer"},
    {"id":"report_writer",    "num":"04","icon":"📝","name":"Report Writer",     "desc":"Synthesises all findings into a structured technical report"},
]
NODE_ORDER = ["data_analyst","feature_engineer","model_trainer","report_writer"]

# ── Log Capture ───────────────────────────────────────────────────────────────
_log_buffer     = []
_original_print = builtins.print

def _patched_print(*args, sep=" ", end="\n", **kwargs):
    text = sep.join(str(a) for a in args)
    _log_buffer.append(text)
    _original_print(*args, sep=sep, end=end, **kwargs)

def colorize(line):
    esc = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    if not esc.strip(): return "<br>"
    if "="*8 in esc: return f'<span class="log-divider">{esc}</span>\n'
    if any(h in esc for h in ["AGENT 1:","AGENT 2:","AGENT 3:","AGENT 4:",
                               "DATA ANALYST","FEATURE ENGINEER","MODEL TRAINER","REPORT WRITER"]):
        return f'<span class="log-agent">{esc}</span>\n'
    if esc.strip().startswith("✅"): return f'<span class="log-success">{esc}</span>\n'
    if "⭐" in esc or "BEST" in esc: return f'<span class="log-best">{esc}</span>\n'
    if "⚠️" in esc or "Error" in esc: return f'<span class="log-warning">{esc}</span>\n'
    if any(c in esc for c in ["📂","📊","🎯","🧠","🏆","📈","📤","📥","📋","🔢","🔧","🔠"]):
        return f'<span class="log-info">{esc}</span>\n'
    if "•" in esc or "→" in esc: return f'<span class="log-metric">{esc}</span>\n'
    return f'<span class="log-default">{esc}</span>\n'

def render_terminal(logs):
    body = "".join(colorize(l) for l in logs) if logs else '<span class="log-dim">// waiting for pipeline execution...</span>'
    return f"""<div class="terminal-wrap">
  <div class="terminal-topbar">
    <div class="terminal-dots"><div class="td td-r"></div><div class="td td-y"></div><div class="td td-g"></div></div>
    <span class="terminal-label">agent execution log</span>
  </div>
  <div class="terminal-body">{body}</div>
</div>"""

def render_pipeline(completed, running=None):
    cards = ""
    for a in AGENTS:
        aid = a["id"]
        if   aid in completed:  sc,dc,st,tc = "complete","complete","✓  Complete","status-text-complete"
        elif aid == running:    sc,dc,st,tc = "running","running","◌  Running...","status-text-running"
        else:                   sc,dc,st,tc = "pending","pending","—  Pending","status-text-pending"
        cards += f"""<div class="agent-card {sc}">
          <div class="agent-num">AGENT {a['num']}</div>
          <div class="agent-icon">{a['icon']}</div>
          <div class="agent-name">{a['name']}</div>
          <div class="agent-desc">{a['desc']}</div>
          <div class="agent-status"><div class="status-dot {dc}"></div><span class="{tc}">{st}</span></div>
        </div>"""
    return f'<div class="pipeline-grid">{cards}</div>'

def render_metrics(model_data, preprocessing_data):
    best_val   = model_data.get("best_metric_value", 0)
    metric_lbl = model_data.get("best_metric_label","score").upper()
    best_name  = model_data.get("best_model","—").split()[0]
    n_features = len(preprocessing_data.get("final_columns",[])) - 1
    n_rows     = preprocessing_data.get("processed_shape",{}).get("rows",0)
    tiles = [(f"{best_val:.4f}",metric_lbl),(best_name,"Best Model"),(str(n_features),"Features"),(str(n_rows),"Rows Trained")]
    html  = '<div class="metrics-grid">'
    for val,lbl in tiles:
        html += f'<div class="metric-tile"><div class="metric-tile-value">{val}</div><div class="metric-tile-label">{lbl}</div></div>'
    return html + "</div>"

def render_model_table(model_data):
    results   = model_data.get("all_results",{})
    best      = model_data.get("best_model","")
    prob_type = model_data.get("problem_type","classification")
    if prob_type == "classification":
        cols = [("Model",None),("CV Score","cv_mean_score"),("CV Std","cv_std"),("Test Acc","test_accuracy"),("F1","f1_score")]
    else:
        cols = [("Model",None),("CV R²","cv_mean_score"),("CV Std","cv_std"),("Test R²","r2_score"),("MAE","mae"),("RMSE","rmse")]
    header = "".join(f"<th>{c[0]}</th>" for c in cols)
    rows   = ""
    for name,res in sorted(results.items(),key=lambda x:x[1].get("primary_metric",0),reverse=True):
        if "error" in res: continue
        is_best = (name == best)
        badge   = '<span class="best-badge">Best</span>' if is_best else ""
        cells   = f'<td><strong>{name}</strong>{badge}</td>'
        for _,key in cols[1:]:
            val   = res.get(key,0)
            bar_w = int(min(abs(val),1.0)*100)
            bc    = "best" if is_best else ""
            cells += f'<td>{val:.4f}<div class="score-bar-bg"><div class="score-bar-fill {bc}" style="width:{bar_w}%"></div></div></td>'
        rows += f'<tr class="{"best-row" if is_best else ""}">{cells}</tr>'
    return f'<div class="model-table-wrap"><table class="model-table"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'

# ── Layout: Two-column (no sidebar) ──────────────────────────────────────────
left_col, right_col = st.columns([1, 2.8], gap="large")

# ── LEFT PANEL — Inputs ───────────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⚙ Configuration</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    df_preview, col_options = None, []
    if uploaded:
        try:
            df_preview  = pd.read_csv(uploaded)
            col_options = list(df_preview.columns)
            uploaded.seek(0)
        except Exception as e:
            st.error(f"CSV error: {e}")

    target_col   = st.selectbox("Target Column", options=col_options if col_options else ["—"])
    problem_stmt = st.text_area("Problem Statement",
        placeholder="e.g. Predict whether a passenger survived the Titanic disaster",
        height=90)

    st.markdown('<div class="panel-divider"></div>', unsafe_allow_html=True)
    run_clicked = st.button("▶  Run Pipeline")
    st.markdown('<div class="panel-divider"></div>', unsafe_allow_html=True)

    st.markdown("""<div class="panel-info">
    <span>🔍 Agent 1</span> — Data Analyst<br>
    <span>⚙️ Agent 2</span> — Feature Engineer<br>
    <span>🤖 Agent 3</span> — Model Trainer<br>
    <span>📝 Agent 4</span> — Report Writer<br><br>
    LangGraph · Groq<br>llama-3.3-70b-versatile
    </div>""", unsafe_allow_html=True)

    if df_preview is not None:
        st.markdown('<div class="panel-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df_preview.head(5), use_container_width=True, height=190)

    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT PANEL — Pipeline + Results ─────────────────────────────────────────
with right_col:
    # Header
    st.markdown("""<div class="mc-header">
      <div>
        <div class="mc-title">AutoML Agent System</div>
        <div class="mc-subtitle">Multi-Agent Machine Learning Pipeline — LangGraph + Groq</div>
      </div>
      <div class="mc-tag">v1.0 · 4 Agents</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Pipeline Status</div>', unsafe_allow_html=True)
    pipeline_ph = st.empty()
    pipeline_ph.markdown(render_pipeline([],None), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Execution Log</div>', unsafe_allow_html=True)
    terminal_ph = st.empty()
    terminal_ph.markdown(render_terminal([]), unsafe_allow_html=True)

    results_ph  = st.empty()
    report_ph   = st.empty()
    download_ph = st.empty()

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run_clicked:
    if not uploaded:
        with left_col: st.error("Please upload a CSV file.")
    elif not problem_stmt.strip():
        with left_col: st.error("Please enter a problem statement.")
    elif target_col == "—":
        with left_col: st.error("Please select a target column.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue()); tmp_path = tmp.name

        _log_buffer.clear()
        builtins.print = _patched_print

        initial_state: MLState = {
            "file_path":tmp_path,"problem_statement":problem_stmt,"target_column":target_col,
            "problem_type":None,"analysis_result":None,"preprocessing_result":None,
            "model_result":None,"report":None,"current_agent":None,"error":None,
        }

        completed, final_state = [], None

        with right_col:
            pipeline_ph.markdown(render_pipeline([],NODE_ORDER[0]), unsafe_allow_html=True)
            terminal_ph.markdown(render_terminal(["// pipeline initialising..."]), unsafe_allow_html=True)

        try:
            app = build_workflow()
            for chunk in app.stream(initial_state):
                node        = list(chunk.keys())[0]
                final_state = chunk[node]
                if node not in completed: completed.append(node)
                idx = NODE_ORDER.index(node)
                nxt = NODE_ORDER[idx+1] if idx+1 < len(NODE_ORDER) else None
                with right_col:
                    pipeline_ph.markdown(render_pipeline(completed, nxt), unsafe_allow_html=True)
                    terminal_ph.markdown(render_terminal(_log_buffer),    unsafe_allow_html=True)
        except Exception as e:
            with right_col: st.error(f"Pipeline error: {e}")
        finally:
            builtins.print = _original_print

        if final_state and final_state.get("model_result"):
            md = json.loads(final_state["model_result"])
            pp = json.loads(final_state.get("preprocessing_result","{}"))
            with right_col:
                results_ph.markdown(
                    '<div class="section-label">Results</div>' + render_metrics(md,pp) +
                    '<div class="section-label">Model Comparison</div>' + render_model_table(md),
                    unsafe_allow_html=True
                )

        if final_state and final_state.get("report"):
            def render_report_html(text):
                import re

                def inline(s):
                    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
                    s = re.sub(r'`(.+?)`', r'<code>\1</code>', s)
                    return s

                def parse_table(table_lines):
                    # Filter out separator rows (--- lines)
                    rows = [l for l in table_lines if not re.match(r'^\|[\s\-|]+\|$', l)]
                    if not rows:
                        return ""
                    header_cells = [c.strip() for c in rows[0].strip('|').split('|')]
                    header_html = "".join(f"<th>{inline(c)}</th>" for c in header_cells)
                    body_html = ""
                    for row in rows[1:]:
                        cells = [c.strip() for c in row.strip('|').split('|')]
                        row_html = "".join(f"<td>{inline(c)}</td>" for c in cells)
                        body_html += f"<tr>{row_html}</tr>"
                    return f'<table class="report-table"><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>'

                lines = text.split('\n')
                output = []
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if not line:
                        i += 1
                        continue

                    # ── Markdown table detection ──
                    if line.startswith('|') and line.endswith('|'):
                        table_lines = []
                        while i < len(lines) and lines[i].strip().startswith('|'):
                            table_lines.append(lines[i].strip())
                            i += 1
                        output.append(parse_table(table_lines))
                        continue

                    line = inline(line)

                    if line.startswith('### '):
                        output.append(f'<h3>{line[4:]}</h3>')
                    elif line.startswith('## '):
                        output.append(f'<h2>{line[3:]}</h2>')
                    elif line.startswith('# '):
                        output.append(f'<h1>{line[2:]}</h1>')
                    elif line.startswith('* ') or line.startswith('- '):
                        items = []
                        while i < len(lines) and (
                                lines[i].strip().startswith('* ') or lines[i].strip().startswith('- ')):
                            item = inline(lines[i].strip()[2:])
                            items.append(f'<li>{item}</li>')
                            i += 1
                        output.append('<ul>' + ''.join(items) + '</ul>')
                        continue
                    else:
                        output.append(f'<p>{line}</p>')
                    i += 1

                return ''.join(output)

            with right_col:
                report_ph.markdown(
                    '<div class="section-label">Final Report</div>' +
                    f'<div class="report-panel">{render_report_html(final_state["report"])}</div>',
                    unsafe_allow_html=True
                )
                download_ph.download_button(
                    label="⬇  Download Report",
                    data=final_state["report"],
                    file_name="automl_report.txt",
                    mime="text/plain"
                )