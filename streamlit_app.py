
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import tempfile
import os
import sys
from datetime import datetime
from collections import defaultdict
from PIL import Image
import io
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.detector import VisionAIDetector

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI — Object Detection & Tracking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

:root {
    --bg: #0a0c10;
    --surface: #12151c;
    --border: #1e2330;
    --accent: #00e5ff;
    --accent2: #ff3d7f;
    --accent3: #7eff8a;
    --text: #e0e8f0;
    --muted: #5a6a80;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-val { font-size: 2.2rem; font-weight: 800; color: var(--accent); font-family: 'Space Mono', monospace; }
.metric-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }

.alert-box {
    background: rgba(255,61,127,0.1);
    border: 1px solid var(--accent2);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.85rem;
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
}

.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.status-live { background: rgba(126,255,138,0.15); color: var(--accent3); border: 1px solid var(--accent3); }
.status-idle { background: rgba(90,106,128,0.15); color: var(--muted); border: 1px solid var(--muted); }

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00e5ff22, #7eff8a11) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(0,229,255,0.2) !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.3) !important;
}

.stSlider > div { color: var(--text) !important; }
.stSelectbox > div { background: var(--surface) !important; border-color: var(--border) !important; }

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "detector": None,
        "running": False,
        "alerts": [],
        "frame_history": [],
        "analytics_history": [],
        "total_detected": 0,
        "fps_history": [],
        "class_history": defaultdict(int),
        "show_heatmap": False,
        "zone_mode": False,
        "zones_defined": {},
        "source_type": "Webcam",
        "uploaded_video": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Alert callback ─────────────────────────────────────────────────────────────
def on_alert(msg, cls_name, zone):
    st.session_state.alerts.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg": msg, "cls": cls_name, "zone": zone
    })
    if len(st.session_state.alerts) > 50:
        st.session_state.alerts = st.session_state.alerts[:50]


# ── Sidebar Controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 VisionAI")
    st.markdown('<span class="status-badge status-live">● SYSTEM READY</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Model selection
    st.markdown("### 🧠 Model")
    model_key = st.selectbox("Detection Model", list(VisionAIDetector.MODELS.keys()), index=0)

    # Source selection
    st.markdown("### 📡 Input Source")
    source_type = st.selectbox("Source", ["Webcam", "Video File", "Image"])
    st.session_state.source_type = source_type

    uploaded_file = None
    if source_type in ["Video File", "Image"]:
        ft = ["mp4","avi","mov","mkv"] if source_type=="Video File" else ["jpg","jpeg","png","bmp","webp"]
        uploaded_file = st.file_uploader(f"Upload {source_type}", type=ft)
        st.session_state.uploaded_video = uploaded_file

    # Detection settings
    st.markdown("### ⚙️ Detection")
    conf = st.slider("Confidence Threshold", 0.1, 0.95, 0.40, 0.05)
    iou = st.slider("IoU Threshold", 0.1, 0.95, 0.45, 0.05)

    # Tracking
    st.markdown("### 🎯 Tracking")
    use_tracking = st.toggle("Deep SORT Tracking", value=True)
    use_trails = st.toggle("Motion Trails", value=True)
    show_heatmap = st.toggle("Heatmap Overlay", value=False)
    st.session_state.show_heatmap = show_heatmap

    # Alert classes
    st.markdown("### ⚠️ Alert Triggers")
    COCO_CLASSES = ["person","car","truck","bus","motorcycle","bicycle","dog","cat","weapon","knife","scissors"]
    alert_classes = st.multiselect("Alert on detection of:", COCO_CLASSES, default=["person"])

    # Performance
    st.markdown("### ⚡ Performance")
    frame_skip = st.slider("Frame Skip (speed up)", 0, 5, 0)
    resize_pct = st.slider("Resize % (speed up)", 25, 100, 100, 25)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ START", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹ STOP", use_container_width=True)

    if st.button("💾 Export Logs", use_container_width=True):
        if st.session_state.detector:
            path = st.session_state.detector.save_logs()
            st.success(f"Saved: {path}")

    if st.button("🌡 Reset Heatmap", use_container_width=True):
        if st.session_state.detector:
            st.session_state.detector.reset_heatmap()
            st.success("Heatmap cleared")

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.detector = None
        st.session_state.running = False
        st.session_state.analytics_history = []
        st.session_state.fps_history = []
        st.session_state.class_history = defaultdict(int)
        st.session_state.alerts = []
        st.rerun()


# ── Handle start/stop ──────────────────────────────────────────────────────────
if start_btn:
    if st.session_state.detector is None or st.session_state.detector.model_key != model_key:
        with st.spinner(f"Loading {model_key}..."):
            det = VisionAIDetector(
                model_key=model_key, conf=conf, iou=iou,
                use_tracking=use_tracking, use_trails=use_trails,
                log_dir="logs"
            )
            for cls in alert_classes:
                det.add_alert_class(cls)
            det.register_alert_callback(on_alert)
            st.session_state.detector = det
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False


# ── Header ─────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown("# 🎯 VisionAI")
    st.markdown("*Advanced Object Detection · Tracking · Analytics*")
with col_h2:
    status_html = '<span class="status-badge status-live">● LIVE</span>' if st.session_state.running else '<span class="status-badge status-idle">◉ IDLE</span>'
    st.markdown(f"<div style='text-align:right;padding-top:20px'>{status_html}</div>", unsafe_allow_html=True)

st.markdown("---")

# ── Main layout ────────────────────────────────────────────────────────────────
col_video, col_right = st.columns([3,2])

with col_video:
    video_placeholder = st.empty()
    
    # Zone editor UI
    if st.session_state.zone_mode:
        st.info("Zone editor: Define zones in the code or future click UI (coming soon)")

with col_right:
    # Real-time metrics
    st.markdown("### 📊 Live Metrics")
    m1,m2,m3,m4 = st.columns(4)
    fps_ph = m1.empty()
    obj_ph = m2.empty()
    frame_ph = m3.empty()
    alert_ph = m4.empty()

    # Class distribution
    st.markdown("### 📈 Class Distribution")
    chart_ph = st.empty()

    # Timeline
    st.markdown("### 📉 Detection Timeline")
    timeline_ph = st.empty()


# ── Alerts panel ───────────────────────────────────────────────────────────────
with st.expander("⚠️ Alert Log", expanded=False):
    alert_panel_ph = st.empty()

# ── Analytics tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋 Session Stats", "🌡 Heatmap", "📦 Raw Data", "ℹ️ Model Info"])


# ── Helper: render metrics ─────────────────────────────────────────────────────
def render_metrics(fps, obj_count, frame_id, alert_count):
    fps_ph.markdown(f'<div class="metric-card"><div class="metric-val">{fps:.0f}</div><div class="metric-label">FPS</div></div>', unsafe_allow_html=True)
    obj_ph.markdown(f'<div class="metric-card"><div class="metric-val">{obj_count}</div><div class="metric-label">Objects</div></div>', unsafe_allow_html=True)
    frame_ph.markdown(f'<div class="metric-card"><div class="metric-val">{frame_id}</div><div class="metric-label">Frames</div></div>', unsafe_allow_html=True)
    alert_ph.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ff3d7f">{alert_count}</div><div class="metric-label">Alerts</div></div>', unsafe_allow_html=True)


def render_class_chart(class_counts):
    if not class_counts:
        chart_ph.info("No detections yet")
        return
    df = pd.DataFrame({"Class":list(class_counts.keys()),"Count":list(class_counts.values())})
    df = df.sort_values("Count",ascending=True)
    fig = px.bar(df, x="Count", y="Class", orientation="h",
                 color="Count", color_continuous_scale=["#1a1f2e","#00e5ff"],
                 template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),
        height=200,
        showlegend=False,
        coloraxis_showscale=False,
        font=dict(family="Space Mono",color="#e0e8f0",size=10),
        xaxis=dict(gridcolor="#1e2330"),
        yaxis=dict(gridcolor="#1e2330"),
    )
    chart_ph.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key=f"class_chart_{int(time.time()*10)}")
def render_timeline(fps_history):
    if len(fps_history) < 3:
        timeline_ph.info("Collecting data...")
        return
    fig = go.Figure(go.Scatter(
        y=fps_history[-60:], mode="lines", fill="tozeroy",
        line=dict(color="#00e5ff",width=2),
        fillcolor="rgba(0,229,255,0.1)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0), height=130,
        xaxis=dict(showticklabels=False,gridcolor="#1e2330"),
        yaxis=dict(title="FPS",gridcolor="#1e2330",color="#5a6a80",title_font_size=10),
        font=dict(family="Space Mono",color="#e0e8f0"),
    )
    timeline_ph.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key=f"timeline_chart_{int(time.time()*10)}")


def render_alerts():
    if not st.session_state.alerts:
        alert_panel_ph.info("No alerts yet.")
        return
    html = ""
    for a in st.session_state.alerts[:10]:
        html += f'<div class="alert-box">[{a["time"]}] {a["msg"]}</div>'
    alert_panel_ph.markdown(html,unsafe_allow_html=True)


# ── Image processing ───────────────────────────────────────────────────────────
def process_image(uploaded):
    det = st.session_state.detector
    if det is None: return
    img = Image.open(uploaded).convert("RGB")
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if resize_pct < 100:
        w = int(frame.shape[1]*resize_pct/100)
        h = int(frame.shape[0]*resize_pct/100)
        frame = cv2.resize(frame,(w,h))
    vis, analytics = det.process_frame(frame, show_heatmap=show_heatmap)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    video_placeholder.image(vis_rgb, use_column_width=True)
    render_metrics(analytics.fps, analytics.total_detections, analytics.frame_id, len(st.session_state.alerts))
    render_class_chart(det.class_counts_all)
    render_alerts()


# ── Video / Webcam processing loop ─────────────────────────────────────────────
def run_video_loop(cap):
    det = st.session_state.detector
    frame_idx = 0
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret: break

            # Frame skip
            if frame_skip > 0 and frame_idx % (frame_skip+1) != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            if resize_pct < 100:
                w = int(frame.shape[1]*resize_pct/100)
                h = int(frame.shape[0]*resize_pct/100)
                frame = cv2.resize(frame,(w,h))

            vis, analytics = det.process_frame(frame, show_heatmap=show_heatmap)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            # Update state
            st.session_state.fps_history.append(analytics.fps)
            for cls,cnt in analytics.class_counts.items():
                st.session_state.class_history[cls] += cnt

            # Render
            video_placeholder.image(vis_rgb, channels="RGB", use_container_width=True)
            render_metrics(analytics.fps, analytics.total_detections, analytics.frame_id, len(st.session_state.alerts))
            render_class_chart(dict(st.session_state.class_history))
            render_timeline(st.session_state.fps_history)
            render_alerts()

    finally:
        cap.release()


# ── Main execution ─────────────────────────────────────────────────────────────
if st.session_state.running and st.session_state.detector:
    if source_type == "Image" and uploaded_file:
        process_image(uploaded_file)

    elif source_type == "Video File" and uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)
        run_video_loop(cap)
        os.unlink(tfile.name)

    elif source_type == "Webcam":
        cap = cv2.VideoCapture("http://192.168.0.100:8080/video")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            st.error("❌ Webcam not accessible. Try Video File mode instead.")
            st.session_state.running = False
        else:
            run_video_loop(cap)

elif not st.session_state.running:
    video_placeholder.markdown("""
    <div style="background:#12151c;border:1px dashed #1e2330;border-radius:12px;
    padding:60px;text-align:center;color:#5a6a80;">
        <div style="font-size:3rem;margin-bottom:16px">🎯</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#e0e8f0">
            VisionAI System Idle
        </div>
        <div style="margin-top:8px;font-size:0.85rem">
            Configure settings in the sidebar and click ▶ START
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Analytics Tabs ─────────────────────────────────────────────────────────────
with tab1:
    if st.session_state.detector:
        stats = st.session_state.detector.get_session_stats()
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Frames Processed", stats.get("total_frames",0))
        c2.metric("Total Detections", sum(stats.get("class_counts",{}).values()))
        c3.metric("Classes Detected", len(stats.get("class_counts",{})))
        if stats.get("class_counts"):
            df = pd.DataFrame({"Class":list(stats["class_counts"].keys()),
                               "Count":list(stats["class_counts"].values())})
            fig = px.pie(df,values="Count",names="Class",hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                        template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("Start a session to see stats")

with tab2:
    if st.session_state.detector:
        st.info("Heatmap is overlaid on the live video when the Heatmap toggle is enabled.")
        hmap = st.session_state.detector.heatmap.map
        if hmap.max() > 0:
            norm = cv2.normalize(hmap,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            st.image(cv2.cvtColor(colored,cv2.COLOR_BGR2RGB), caption="Movement Heatmap", use_column_width=True)
        else:
            st.info("Run detection to build the heatmap")
    else:
        st.info("Start a session to see heatmap")

with tab3:
    if st.session_state.detector and st.session_state.detector.logger.records:
        records = st.session_state.detector.logger.records
        df = pd.DataFrame(records)
        st.dataframe(df.tail(200), use_container_width=True)
        csv_data = df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv_data, "detections.csv", "text/csv")
    else:
        st.info("No detection data yet")

with tab4:
    st.markdown("### Model Information")
    for name, file in VisionAIDetector.MODELS.items():
        curr = "← **ACTIVE**" if (st.session_state.detector and st.session_state.detector.model_key == name) else ""
        st.markdown(f"- `{file}` — **{name}** {curr}")
    st.markdown("""
    ### Architecture
    - **Detection**: YOLOv8 family (Ultralytics)
    - **Tracking**: Deep SORT with MobileNet embedder  
    - **Analytics**: Real-time class counting, FPS monitoring
    - **Zones**: Polygon-based restricted area detection
    - **Heatmap**: Cumulative movement visualization
    - **Logging**: CSV + JSON export with timestamps

    ### COCO Classes Detected
    80 object classes including: person, vehicle, animals, furniture, electronics, food, sports equipment, and more.

    ### GitHub
    ```
    Repository: VisionAI_ObjectDetection
    streamlit run streamlit_app.py
    ```
    """)
-------------------------