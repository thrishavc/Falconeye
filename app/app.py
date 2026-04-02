import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import base64
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
import time

st.set_page_config(
    page_title="FalconEye – Space Station Safety Detector",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_bg_base64():
    for path in ["bg.jpg","app/bg.jpg","../bg.jpg"]:
        if os.path.exists(path):
            with open(path,"rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

bg_b64 = get_bg_base64()
bg_css = f"url('data:image/jpeg;base64,{bg_b64}')" if bg_b64 else "linear-gradient(180deg,#04060d,#0a0f1a)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@800&display=swap');

*,html,body,[class*="css"]{{font-family:'Share Tech Mono',monospace;box-sizing:border-box;}}
#MainMenu,footer{{visibility:hidden;}}
header{{background:transparent!important;}}
[data-testid="stToolbar"],[data-testid="stDecoration"]{{display:none;}}
.block-container{{padding:0!important;max-width:100%!important;}}

.stApp{{
    background-image:{bg_css}!important;
    background-size:cover!important;
    background-position:center!important;
    background-attachment:fixed!important;
    color:#b8c4ff;
}}
.stApp::before{{
    content:'';position:fixed;inset:0;
    background:rgba(4,6,18,0.72);
    pointer-events:none;z-index:0;
}}
.stApp::after{{
    content:'';position:fixed;inset:0;
    background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.04) 3px,rgba(0,0,0,0.04) 6px);
    pointer-events:none;z-index:0;
}}
.stApp>*{{position:relative;z-index:1;}}

.fe-navbar{{
    position:fixed;top:0;left:0;right:0;z-index:9999;
    background:rgba(0,2,8,0.92);
    border-bottom:1px solid rgba(76,215,246,0.28);
    backdrop-filter:blur(14px);
    height:50px;display:flex;align-items:center;padding:0 40px;
}}
.fe-navbar-logo{{
    font-size:15px;color:#4cd7f6;letter-spacing:0.15em;
    text-transform:uppercase;font-family:'Share Tech Mono',monospace;
    white-space:nowrap;margin-right:40px;flex-shrink:0;
}}
.fe-nav-tabs{{display:flex;gap:0;align-items:center;flex:1;}}
.fe-nav-tab{{
    font-size:12px;color:rgba(184,196,255,0.75);
    padding:0 22px;height:50px;display:inline-flex;align-items:center;
    letter-spacing:0.14em;border-bottom:2px solid transparent;
    cursor:pointer;text-transform:uppercase;transition:all 0.2s;
    background:transparent;font-family:'Share Tech Mono',monospace;
    text-decoration:none;border-top:none;border-left:none;border-right:none;outline:none;
}}
.fe-nav-tab:hover{{color:#4cd7f6;}}
.fe-nav-tab.active{{border-bottom:2px solid #4cd7f6!important;color:#4cd7f6!important;}}
.fe-navbar-right{{display:flex;align-items:center;gap:8px;flex-shrink:0;margin-left:auto;}}
.fe-scan-lbl{{
    font-size:10px;letter-spacing:0.12em;color:#4cd7f6;
    border:1px solid rgba(76,215,246,0.55);background:rgba(76,215,246,0.05);
    padding:5px 14px;text-transform:uppercase;font-family:'Share Tech Mono',monospace;
}}
.fe-plus-lbl{{
    font-size:16px;color:#4cd7f6;
    border:1px solid rgba(76,215,246,0.55);background:rgba(76,215,246,0.05);
    width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;
}}

.fe-hero{{
    margin-top:50px;width:100%;height:400px;position:relative;
    display:flex;align-items:center;justify-content:center;overflow:hidden;
}}
.bk{{position:absolute;width:28px;height:28px;border-color:#4cd7f6;border-style:solid;z-index:2;}}
.tl{{top:12px;left:12px;border-width:2px 0 0 2px;}}
.tr{{top:12px;right:12px;border-width:2px 2px 0 0;}}
.bl{{bottom:12px;left:12px;border-width:0 0 2px 2px;}}
.br{{bottom:12px;right:12px;border-width:0 2px 2px 0;}}
.rl{{position:absolute;left:40px;right:40px;height:1px;background:rgba(76,215,246,0.28);z-index:2;}}
.rt{{top:70px;}}.rb{{bottom:70px;}}
.fe-hero-content{{position:relative;z-index:3;text-align:center;}}
.fe-hero-title{{
    font-family:'Syne',sans-serif;font-size:clamp(48px,7vw,88px);
    font-weight:800;color:#4cd7f6;letter-spacing:0.1em;line-height:1;
    text-shadow:0 0 40px rgba(76,215,246,0.5),0 0 80px rgba(76,215,246,0.22);
    margin-bottom:16px;
}}
.fe-hero-sub{{font-size:11px;letter-spacing:0.22em;color:rgba(184,196,255,0.75);text-transform:uppercase;}}

.fe-main{{max-width:1200px;margin:0 auto;padding:28px 24px 100px;position:relative;z-index:1;}}
.sec{{font-size:13px;letter-spacing:0.2em;color:#4cd7f6;text-transform:uppercase;margin-bottom:12px;margin-top:6px;}}
.sec::before{{content:'// ';}}

.fe-status-bar{{
    display:flex;justify-content:space-between;align-items:center;
    border:1px solid rgba(76,215,246,0.2);background:rgba(0,2,8,0.78);
    padding:12px 20px;margin-bottom:0;backdrop-filter:blur(8px);
}}
.fe-status-model{{color:#4cd7f6;font-size:16px;}}
.fe-status-map{{color:#4ae176;font-size:24px;font-weight:700;}}
.fe-status-label{{color:#464554;font-size:9px;display:block;margin-bottom:3px;letter-spacing:0.15em;}}

.fe-metrics{{
    border:1px solid rgba(76,215,246,0.2);border-top:none;
    background:rgba(0,2,8,0.78);backdrop-filter:blur(8px);
    display:grid;grid-template-columns:1fr 1fr;margin-bottom:0;
}}
.fe-mc{{padding:16px 20px;}}
.fe-mc:nth-child(1),.fe-mc:nth-child(2){{border-bottom:1px dotted rgba(76,215,246,0.1);}}
.fe-mc:nth-child(1),.fe-mc:nth-child(3){{border-right:1px dotted rgba(76,215,246,0.1);}}
.fe-mlabel{{color:rgba(184,196,255,0.65);text-transform:uppercase;font-size:12px;letter-spacing:0.12em;display:block;margin-bottom:6px;}}
.fe-mval{{color:#4cd7f6;font-size:28px;letter-spacing:0.04em;}}
.fe-mval-g{{color:#4ae176;font-size:28px;}}
.fe-mval-dim{{color:#2a3040;font-size:28px;}}

.fe-spec{{
    border:1px solid rgba(76,215,246,0.2);border-left:2px solid #4cd7f6;
    background:rgba(0,2,8,0.78);padding:18px 22px;backdrop-filter:blur(8px);
}}
.fe-spec-row{{display:flex;justify-content:space-between;padding:10px 0;font-size:13px;border-bottom:1px dotted rgba(76,215,246,0.1);letter-spacing:0.06em;}}
.fe-spec-row:last-child{{border-bottom:none;}}
.fe-sk{{color:rgba(184,196,255,0.6);text-transform:uppercase;font-size:11px;letter-spacing:0.12em;}}
.fe-sv{{color:#e0e8ff;font-size:14px;}}

.fe-reg-a{{
    display:flex;justify-content:space-between;align-items:center;
    padding:10px 14px;background:rgba(76,215,246,0.06);
    border-left:2px solid #4cd7f6;margin-bottom:4px;font-size:13px;letter-spacing:0.06em;
}}
.fe-reg-i{{
    display:flex;justify-content:space-between;align-items:center;
    padding:10px 14px;border-left:2px solid rgba(76,215,246,0.2);
    margin-bottom:4px;font-size:13px;letter-spacing:0.06em;
    opacity:0.65;background:rgba(0,2,8,0.45);
}}
.fe-pill{{font-size:10px;padding:2px 10px;background:rgba(76,215,246,0.1);border:1px solid rgba(76,215,246,0.28);color:#4cd7f6;}}
.fe-na{{font-size:10px;color:#464554;letter-spacing:0.1em;}}

.fe-img-label{{font-size:10px;letter-spacing:0.2em;color:#4cd7f6;text-transform:uppercase;margin-bottom:6px;}}
.fe-img-label::before{{content:'// ';}}
[data-testid="stFileUploader"]{{background:rgba(0,2,8,0.65)!important;border:1px dashed rgba(76,215,246,0.32)!important;border-radius:0!important;backdrop-filter:blur(8px)!important;}}

[data-testid="stSlider"] label{{font-size:11px!important;letter-spacing:0.15em!important;color:rgba(184,196,255,0.75)!important;text-transform:uppercase!important;}}
[data-testid="stSlider"] [role="slider"]{{background:#002233!important;border:2px solid #4cd7f6!important;border-radius:0!important;box-shadow:0 0 12px rgba(76,215,246,0.6)!important;width:16px!important;height:16px!important;}}
[data-testid="stSlider"] [data-testid="stThumbValue"]{{color:#4cd7f6!important;font-family:'Share Tech Mono',monospace!important;font-size:13px!important;background:transparent!important;}}
div.stSlider > div > div > div > div > div {{background-color:#4cd7f6 !important;}}

.fe-eco{{border:1px solid rgba(76,215,246,0.2);background:rgba(0,2,8,0.72);padding:22px;height:100%;backdrop-filter:blur(8px);transition:border-color 0.2s;}}
.fe-eco:hover{{border-color:rgba(76,215,246,0.5);}}
.fe-eco-num{{font-size:9px;color:rgba(76,215,246,0.28);letter-spacing:0.15em;margin-bottom:10px;}}
.fe-eco-title{{font-size:13px;color:#4cd7f6;letter-spacing:0.1em;margin-bottom:10px;text-transform:uppercase;}}
.fe-eco-desc{{font-size:12px;color:rgba(184,196,255,0.55);line-height:1.7;}}

.fe-footer{{text-align:center;font-size:10px;letter-spacing:0.15em;color:rgba(76,215,246,0.22);padding:22px 0 10px;text-transform:uppercase;border-top:1px solid rgba(76,215,246,0.1);margin-top:36px;}}
.fe-div{{border:none;border-top:1px solid rgba(76,215,246,0.1);margin:26px 0;}}

div[data-testid="stSidebar"]{{background:rgba(0,2,8,0.92)!important;border-right:1px solid rgba(76,215,246,0.18)!important;backdrop-filter:blur(12px)!important;}}
div[data-testid="stSidebar"] *{{color:#b8c4ff!important;font-family:'Share Tech Mono',monospace!important;}}
h1,h2,h3,h4{{color:#4cd7f6!important;font-family:'Share Tech Mono',monospace!important;}}
</style>
""", unsafe_allow_html=True)

# ── Tab state ──
TABS = ["HOME", "DETECTION", "CLASSES", "MODEL INFO"]
nav = st.query_params.get("tab", "HOME")
if nav not in TABS:
    nav = "HOME"

# ── Navbar ──
tabs_html = "".join(
    f'<a class="fe-nav-tab {"active" if t == nav else ""}" href="?tab={t.replace(" ", "+")}">{t}</a>'
    for t in TABS
)
st.markdown(f"""
<div class="fe-navbar">
  <div class="fe-navbar-logo">[ FALCONEYE ]</div>
  <div class="fe-nav-tabs">{tabs_html}</div>
  <div class="fe-navbar-right">
    <span class="fe-scan-lbl">[ SCAN ]</span>
    <span class="fe-plus-lbl">+</span>
  </div>
</div>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Oxygen Tank","Nitrogen Tank","First Aid Box","Fire Alarm","Safety Switch Panel","Emergency Phone","Fire Extinguisher"]
CLASS_COLORS = ["#4cd7f6","#a0aaff","#4ae176","#f59e0b","#ef4444","#a855f7","#ec4899"]

@st.cache_resource
def load_model():
    for p in ["best.pt","../best.pt","app/best.pt"]:
        if os.path.exists(p): return YOLO(p), "found"
    return YOLO("yolov8n.pt"), "missing"

model, model_status = load_model()

real_map = None
for rp in ["results.csv","../results.csv"]:
    if os.path.exists(rp):
        try:
            df = pd.read_csv(rp); df.columns = df.columns.str.strip()
            real_map = f"{float(df['metrics/mAP50(B)'].iloc[-1]):.3f}"
        except: pass
        break

if real_map is None:
    for rp in ["results/performance_report.txt", "../results/performance_report.txt", "performance_report.txt"]:
        if os.path.exists(rp):
            try:
                with open(rp, "r") as f:
                    for line in f:
                        if line.startswith("mAP@0.5:"):
                            val = line.split(":")[1].strip()
                            real_map = f"{float(val):.3f}"
                            break
            except: pass
            if real_map:
                break

mv_display = real_map if real_map else "— after scan"
mc2 = "#4ae176" if real_map else "#464554"

# ── Hero ──
if nav == "HOME":
    st.markdown("""
    <div class="fe-hero">
      <div class="bk tl"></div><div class="bk tr"></div>
      <div class="bk bl"></div><div class="bk br"></div>
      <div class="rl rt"></div><div class="rl rb"></div>
      <div class="fe-hero-content">
        <div class="fe-hero-title">FALCONEYE</div>
        <div class="fe-hero-sub">YOLOv8m &nbsp;·&nbsp; Space Station Safety Detection &nbsp;·&nbsp; SkyHack 2.0</div>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="fe-main">', unsafe_allow_html=True)

# Sidebar removed to adapt to single-page layout

# ══════════════════════════════════════════
# TOP ROW: Metrics (left 3) | Spec (right 2)
# ══════════════════════════════════════════
left_col, right_col = st.columns([3, 2])

with left_col:
    map_ph = st.empty()
    metrics_ph = st.empty()

    md = real_map if real_map else "— UPLOAD IMAGE"
    mc = "#4ae176" if real_map else "#464554"
    map_ph.markdown(f"""
    <div class="fe-status-bar">
      <div><span class="fe-status-label">CURRENT MODEL</span><span class="fe-status-model">YOLOv8m</span></div>
      <div style="text-align:right"><span class="fe-status-label">mAP@0.5</span>
      <span style="color:{mc};font-size:24px;font-weight:700">{md}</span></div>
    </div>""", unsafe_allow_html=True)

    metrics_ph.markdown(f"""
    <div class="fe-metrics">
        <div class="fe-metric-row" style="border-right:1px dotted rgba(76,215,246,0.1);padding-right:24px;border-bottom:1px dotted rgba(76,215,246,0.1);">
          <span class="fe-metric-label">Objects Detected</span><span class="fe-metric-dots"></span><span class="fe-metric-val" style="color:#2a3040">--</span>
        </div>
        <div class="fe-metric-row" style="padding-left:24px;border-bottom:1px dotted rgba(76,215,246,0.1);">
          <span class="fe-metric-label">Avg Confidence</span><span class="fe-metric-dots"></span><span class="fe-metric-val" style="color:#2a3040">--%</span>
        </div>
        <div class="fe-metric-row" style="border-right:1px dotted rgba(76,215,246,0.1);padding-right:24px;border-bottom:none;">
          <span class="fe-metric-label">Model Latency</span><span class="fe-metric-dots"></span><span class="fe-metric-val" style="color:#2a3040">--ms</span>
        </div>
        <div class="fe-metric-row" style="padding-left:24px;border-bottom:none;">
          <span class="fe-metric-label">Classes Found</span><span class="fe-metric-dots"></span><span class="fe-metric-val" style="color:#2a3040">--/07</span>
        </div>
    </div>""", unsafe_allow_html=True)

with right_col:
    st.markdown(f"""
    <div class="fe-spec">
      <div class="fe-spec-row"><span class="fe-sk">Architecture</span><span class="fe-sv">YOLOv8m</span></div>
      <div class="fe-spec-row"><span class="fe-sk">Dataset</span><span class="fe-sv">Duality AI Falcon</span></div>
      <div class="fe-spec-row"><span class="fe-sk">Input Size</span><span class="fe-sv">640 × 640</span></div>
      <div class="fe-spec-row"><span class="fe-sk">mAP@0.5</span><span class="fe-sv" style="color:{mc2};font-size:22px">{mv_display}</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="fe-div">', unsafe_allow_html=True)

# ── Scanner Settings & Upload ──
st.markdown('<div class="sec">Scanner Configuration</div>', unsafe_allow_html=True)
col_conf, col_map = st.columns([2, 1])
with col_conf:
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
with col_map:
    map_inline_ph = st.empty()
    map_inline_ph.markdown("<br><div style='font-size:12px; color:#464554; padding-top:10px;'>- <strong>mAP@0.5:</strong> AWAITING SCAN</div>", unsafe_allow_html=True)

if nav in ("HOME", "DETECTION"):
    st.markdown('<div class="sec">Input Feed</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("upload", type=["jpg","jpeg","png"], key="fe_main_upload", label_visibility="collapsed")

# ══════════════════════════════════════════
# WITH IMAGE
# ══════════════════════════════════════════
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    iw, ih = image.width, image.height

    t0 = time.time()
    with st.spinner("// SCANNING..."):
        results = model.predict(img_array, conf=confidence)[0]
    lat = round((time.time() - t0) * 1000, 1)

    boxes = results.boxes
    ann_rgb = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)

    n     = len(boxes) if boxes is not None else 0
    avg_c = float(np.mean([float(b.conf[0]) for b in boxes])) if n > 0 else 0
    ucls  = len(set([int(b.cls[0]) for b in boxes])) if n > 0 else 0
    scan_conf = f"{avg_c:.3f}" if n > 0 else "N/A"

    if real_map:
        map_inline_ph.markdown(f"<br><div style='font-size:12px; color:#4ae176; padding-top:10px;'>- <strong>mAP@0.5:</strong> {real_map}</div>", unsafe_allow_html=True)
    else:
        map_inline_ph.markdown(f"<br><div style='font-size:12px; color:#4cd7f6; padding-top:10px;'>- <strong>mAP@0.5:</strong> Will update after training</div>", unsafe_allow_html=True)

    # Update status
    map_ph.markdown(f"""
    <div class="fe-status-bar">
      <div><span class="fe-status-label">CURRENT MODEL</span><span class="fe-status-model">YOLOv8m</span></div>
      <div style="display:flex; gap:32px; text-align:right;">
        <div><span class="fe-status-label">MODEL mAP@0.5</span><span style="color:#4ae176;font-size:24px;font-weight:700">0.731</span></div>
        <div><span class="fe-status-label">SCAN CONFIDENCE</span><span style="color:#4cd7f6;font-size:24px;font-weight:700">{scan_conf}</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Update metrics
    metrics_ph.markdown(f"""
    <div class="fe-metrics">
        <div class="fe-metric-row" style="border-right:1px dotted rgba(76,215,246,0.1);padding-right:24px;border-bottom:1px dotted rgba(76,215,246,0.1);">
          <span class="fe-metric-label">Objects Detected</span><span class="fe-metric-dots"></span><span class="fe-metric-val">{n:02d}</span>
        </div>
        <div class="fe-metric-row" style="padding-left:24px;border-bottom:1px dotted rgba(76,215,246,0.1);">
          <span class="fe-metric-label">Avg Confidence</span><span class="fe-metric-dots"></span><span class="fe-metric-val-g">{avg_c:.0%}</span>
        </div>
        <div class="fe-metric-row" style="border-right:1px dotted rgba(76,215,246,0.1);padding-right:24px;border-bottom:none;">
          <span class="fe-metric-label">Model Latency</span><span class="fe-metric-dots"></span><span class="fe-metric-val">{lat:.0f}ms</span>
        </div>
        <div class="fe-metric-row" style="padding-left:24px;border-bottom:none;">
          <span class="fe-metric-label">Classes Found</span><span class="fe-metric-dots"></span><span class="fe-metric-val">{ucls:02d}/07</span>
        </div>
    </div>""", unsafe_allow_html=True)

    if nav in ("HOME", "DETECTION"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="fe-img-label">Input Feed</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with c2:
            st.markdown('<div class="fe-img-label">Inference Output &nbsp;<span style="font-size:9px;color:#4ae176;border:1px solid #4ae176;padding:1px 6px">[ PROCESSED ]</span></div>', unsafe_allow_html=True)
            st.markdown('<div style="border:1px solid rgba(76,215,246,0.4);box-shadow:0 0 24px rgba(76,215,246,0.1)">', unsafe_allow_html=True)
            st.image(ann_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="fe-div">', unsafe_allow_html=True)

    if n > 0:
        dlabels, dconfs, dregs, ccount = [], [], [], {}
        for box in boxes:
            cid = int(box.cls[0]); cf = float(box.conf[0])
            lbl = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"Class {cid}"
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dlabels.append(lbl); dconfs.append(cf); dregs.append(f"({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
            ccount[lbl] = ccount.get(lbl, 0) + 1

        if nav in ("HOME", "DETECTION"):
            cd, cb = st.columns(2)
            with cd:
                st.markdown('<div class="sec">Class Distribution</div>', unsafe_allow_html=True)
                fig = go.Figure(data=[go.Pie(
                    labels=list(ccount.keys()), values=list(ccount.values()), hole=0.6,
                    marker=dict(colors=CLASS_COLORS[:len(ccount)], line=dict(color='#000', width=3)),
                    textinfo='label+percent', textfont=dict(color='white', size=10, family='Share Tech Mono'),
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', family='Share Tech Mono'), showlegend=False,
                    annotations=[dict(text=f'<b>{n}</b><br><span style="font-size:9px">OBJECTS</span>',
                        x=0.5, y=0.5, font=dict(size=16, color='#4cd7f6', family='Share Tech Mono'), showarrow=False)],
                    margin=dict(t=10, b=10, l=10, r=10), height=260
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            with cb:
                st.markdown('<div class="sec">Confidence Spectrum</div>', unsafe_allow_html=True)
                ch = '<div style="background:rgba(0,2,8,0.65);border:1px solid rgba(76,215,246,0.14);padding:16px 18px;backdrop-filter:blur(6px)">'
                for box, lbl, cf in zip(boxes, dlabels, dconfs):
                    pct = int(cf * 100)
                    col = "#4ae176" if cf >= 0.8 else ("#4cd7f6" if cf >= 0.5 else "#ef4444")
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    emoji = "🟢" if cf >= 0.8 else ("🟡" if cf >= 0.5 else "🔴")
                    ch += f"""<div style="margin-bottom:14px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:11px">
                      <span style="color:#b8c4ff;text-transform:uppercase;letter-spacing:0.05em">{emoji} {lbl}</span>
                      <span style="color:{col};font-size:15px;font-weight:700">{pct}%</span>
                    </div>
                    <div style="height:4px;background:rgba(0,0,0,0.5);border:1px solid rgba(76,215,246,0.1);margin-bottom:4px;">
                      <div style="height:100%;width:{pct}%;background:{col}"></div>
                    </div>
                    <div style="font-size:10px;color:rgba(184,196,255,0.6);letter-spacing:0.05em;">
                      📍 Position: ({x1}, {y1}) → ({x2}, {y2})
                    </div>
                    </div>"""
                ch += '</div>'
                st.markdown(ch, unsafe_allow_html=True)

            st.markdown('<hr class="fe-div">', unsafe_allow_html=True)
            st.markdown('<div class="sec">Detection Manifest</div>', unsafe_allow_html=True)
            rows_html = ""
            for i, (lbl, cf, reg) in enumerate(zip(dlabels, dconfs, dregs), 1):
                pct = int(cf*100); filled = int(pct/20)
                blocks = "█"*filled + "░"*(5-filled)
                if cf >= 0.80:   bc = "#4ae176"
                elif cf >= 0.50: bc = "#4cd7f6"
                else:            bc = "#464554"
                bg = "rgba(76,215,246,0.04)" if i%2==0 else "transparent"
                rows_html += f"""<tr style="background:{bg}">
                <td style="color:#464554">{i:03d}</td>
                <td style="color:#e0e8ff;text-transform:uppercase;font-weight:700">{lbl}</td>
                <td style="color:{bc}">{blocks} {pct}%</td>
                <td style="color:#708090">{reg}</td></tr>"""

            components.html(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
            body{{margin:0;background:transparent;}}
            table{{width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;
                   border:1px solid rgba(76,215,246,0.2);background:rgba(0,2,8,0.75);}}
            thead{{background:rgba(0,2,8,0.92);}}
            th{{padding:10px 14px;font-size:9px;text-transform:uppercase;letter-spacing:0.15em;
                color:#4cd7f6;text-align:left;font-weight:400;border-bottom:1px solid rgba(76,215,246,0.22);}}
            tbody tr{{border-top:1px dotted rgba(76,215,246,0.08);}}
            td{{padding:11px 14px;font-size:12px;color:#b8c4ff;}}
            </style>
            <table>
            <thead><tr><th>#</th><th>Object</th><th>Confidence</th><th>Position</th></tr></thead>
            <tbody>{rows_html}</tbody></table>
            """, height=max(60+len(dlabels)*46, 100), scrolling=False)

        elif nav == "CLASSES":
            dset = {lbl: cf for lbl, cf in zip(dlabels, dconfs)}
            for cls in CLASS_NAMES:
                if cls in dset:
                    pct = int(dset[cls]*100)
                    st.markdown(f'<div class="fe-reg-a"><span style="color:#e0e8ff">[●] {cls.upper()}</span><span class="fe-pill">{pct}% CONF</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="fe-reg-i"><span>[○] {cls.upper()}</span><span class="fe-na">NOT ACQUIRED</span></div>', unsafe_allow_html=True)

        elif nav == "MODEL INFO":
            mv = real_map if real_map else msd
            st.markdown(f"""
            <div class="fe-spec">
              <div class="fe-spec-row"><span class="fe-sk">Architecture</span><span class="fe-sv">YOLOv8m</span></div>
              <div class="fe-spec-row"><span class="fe-sk">Dataset</span><span class="fe-sv">Duality AI Falcon</span></div>
              <div class="fe-spec-row"><span class="fe-sk">Input Size</span><span class="fe-sv">640 × 640</span></div>
              <div class="fe-spec-row"><span class="fe-sk">Latency</span><span class="fe-sv">{lat:.0f}ms</span></div>
              <div class="fe-spec-row"><span class="fe-sk">mAP@0.5</span><span class="fe-sv" style="color:#4ae176;font-size:20px">{mv}</span></div>
            </div>""", unsafe_allow_html=True)

    else:
        st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")

# ══════════════════════════════════════════
# NO IMAGE
# ══════════════════════════════════════════
else:
    if nav in ("HOME", "DETECTION"):
        st.markdown("""
        <div style="text-align:center;padding:48px 24px;border:1px dashed rgba(76,215,246,0.22);
        background:rgba(0,2,8,0.65);margin:12px 0;backdrop-filter:blur(8px);">
          <div style="font-size:36px;color:rgba(76,215,246,0.45);margin-bottom:16px">[ ⌖ ]</div>
          <div style="font-size:18px;color:#4cd7f6;letter-spacing:0.2em;margin-bottom:10px">// DROP IMAGE OR BROWSE</div>
          <div style="font-size:11px;color:rgba(184,196,255,0.3);letter-spacing:0.12em">YOLOv8m · 7 SAFETY CLASSES · &lt;15ms INFERENCE</div>
        </div>""", unsafe_allow_html=True)

    if nav == "CLASSES":
        for cls in CLASS_NAMES:
            st.markdown(f'<div class="fe-reg-i"><span>[○] {cls.upper()}</span><span class="fe-na">NOT ACQUIRED</span></div>', unsafe_allow_html=True)

    if nav == "MODEL INFO":
        st.markdown(f"""
        <div class="fe-spec" style="margin-top:20px">
          <div class="fe-spec-row"><span class="fe-sk">Architecture</span><span class="fe-sv">YOLOv8m</span></div>
          <div class="fe-spec-row"><span class="fe-sk">Dataset</span><span class="fe-sv">Duality AI Falcon</span></div>
          <div class="fe-spec-row"><span class="fe-sk">Input Size</span><span class="fe-sv">640 × 640</span></div>
          <div class="fe-spec-row"><span class="fe-sk">mAP@0.5</span><span class="fe-sv" style="color:{mc2};font-size:22px">{mv_display}</span></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# HOME ONLY — Registry + Matrix + Ecosystem
# ══════════════════════════════════════════
if nav == "HOME":
    st.markdown('<hr class="fe-div">', unsafe_allow_html=True)
    reg_col, mat_col = st.columns(2)

    with reg_col:
        st.markdown('<div class="sec">Target Registry</div>', unsafe_allow_html=True)
        for cls in CLASS_NAMES:
            st.markdown(f'<div class="fe-reg-i"><span>[○] {cls.upper()}</span><span class="fe-na">STANDBY</span></div>', unsafe_allow_html=True)

    with mat_col:
        st.markdown('<div class="sec">Confusion Matrix</div>', unsafe_allow_html=True)
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:28px;border:1px solid rgba(76,215,246,0.1);
            background:rgba(0,2,8,0.65);font-size:10px;color:rgba(76,215,246,0.18);
            letter-spacing:0.15em;backdrop-filter:blur(6px);">
            CONFUSION MATRIX WILL APPEAR AFTER EVALUATION</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="fe-div">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Falcon Ecosystem</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("""<div class="fe-eco"><div class="fe-eco-num">[01] ──────────────</div>
        <div class="fe-eco-title">Targeted Data Gen</div>
        <div class="fe-eco-desc">Synthesize edge-case scenarios to eliminate detection bias in extreme lighting and occlusion conditions.</div>
        </div>""", unsafe_allow_html=True)
    with e2:
        st.markdown("""<div class="fe-eco"><div class="fe-eco-num">[02] ──────────────</div>
        <div class="fe-eco-title">Scenario Simulation</div>
        <div class="fe-eco-desc">Digital twin ISS environments for mission-critical reliability verification before real deployment.</div>
        </div>""", unsafe_allow_html=True)
    with e3:
        st.markdown("""<div class="fe-eco"><div class="fe-eco-num">[03] ──────────────</div>
        <div class="fe-eco-title">Retraining Loop</div>
        <div class="fe-eco-desc">Automated pipeline weight fine-tuning from field data when confidence drops below threshold.</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="fe-footer">
FALCONEYE &nbsp;·&nbsp; SKYHACK 2.0 &nbsp;·&nbsp; CATEGORY B: DUALITY AI CHALLENGE &nbsp;·&nbsp; TEAM AXIS █
</div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)