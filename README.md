# 🎯 VisionAI — Advanced Object Detection & Tracking System

> **CodeAlpha AI Internship | Task 4: Object Detection & Tracking**  
> Production-grade AI surveillance system with real-time analytics dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com)
[![Deep SORT](https://img.shields.io/badge/Tracking-DeepSORT-purple)](https://github.com/levan92/deep_sort_realtime)

---

## 🚀 Features

### 🧠 Detection Models
| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLOv8n | ⚡⚡⚡⚡⚡ | ★★★ | Real-time webcam |
| YOLOv8s | ⚡⚡⚡⚡ | ★★★★ | Balanced |
| YOLOv8m | ⚡⚡⚡ | ★★★★ | General purpose |
| YOLOv8l | ⚡⚡ | ★★★★★ | High accuracy |
| YOLOv8x | ⚡ | ★★★★★ | Best accuracy |

### 🎯 Tracking
- **Deep SORT** with MobileNet feature embedder
- **Persistent Track IDs** across frames
- **Re-identification** (Deep SORT cosine distance)
- **Motion trails** with fade effect

### 📊 Analytics Dashboard
- Real-time FPS counter
- Per-class object counts (live)
- FPS timeline chart
- Session-wide class distribution (pie chart)
- Cumulative detection log

### 🌡 Heatmap System
- Movement accumulation map
- JET colormap overlay
- Reset between sessions

### ⚠️ Alert System
- Per-class alert triggers (configurable)
- Zone violation alerts
- Alert log with timestamps
- Extensible callback architecture

### 🔐 Smart Zone Detection
- Polygon-based restricted zones
- Named zones with visual overlay
- Real-time violation detection
- Per-zone alert toggle

### 📡 REST API (FastAPI)
- `POST /detect/image` — detect objects in image
- `GET /session/stats` — session statistics
- `POST /session/export` — download logs
- `POST /zones/add` — add detection zones
- `POST /alerts/classes` — configure alerts

### 🧾 Logging & Export
- CSV log: every detection (timestamp, class, confidence, bbox, zone)
- JSON session export
- In-app data table with download button

### ⚙️ Performance Controls
- Frame skip (reduce processing load)
- Resize percentage (speed/quality tradeoff)
- Confidence & IoU threshold sliders

---

## 📁 Project Structure

```
VisionAI/
├── streamlit_app.py        # 🖥️  Main dashboard UI
├── requirements.txt        # 📦 Dependencies
├── README.md               # 📖 This file
├── app/
│   ├── detector.py         # 🧠 Core detection engine
│   └── api.py              # 🌐 FastAPI REST endpoints
├── utils/
│   └── metrics.py          # 📏 mAP, Precision/Recall, FPS benchmark
├── logs/                   # 📋 Auto-generated detection logs
└── exports/                # 📤 Exported JSON reports
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

### 3. Run the REST API (optional)
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# API docs at: http://localhost:8000/docs
```

---

## 🎮 How to Use

1. **Select Model** from the sidebar (YOLOv8n is fastest for webcam)
2. **Choose Source**: Webcam, Video File, or Image
3. **Configure Detection**: Adjust confidence and IoU thresholds
4. **Enable Tracking**: Toggle Deep SORT + motion trails
5. **Set Alerts**: Select object classes to trigger alerts
6. **Click ▶ START** to begin detection
7. **View Analytics** in the right panel and bottom tabs
8. **Export Logs** with the sidebar button

---

## 🔧 Extending the System

### Add Custom Zone
```python
from app.detector import VisionAIDetector

det = VisionAIDetector()
# Define a polygon zone (pixel coordinates)
det.zone_mgr.add_zone(
    name="Restricted Area",
    polygon=[[100,100],[400,100],[400,300],[100,300]],
    color=(0,0,255),
    alert=True
)
```

### Register Alert Callback
```python
def my_alert_handler(message, class_name, zone):
    print(f"ALERT: {message}")
    # Send email, SMS, webhook, etc.

det.register_alert_callback(my_alert_handler)
det.add_alert_class("person")
```

### Use via API
```python
import requests

with open("image.jpg","rb") as f:
    r = requests.post(
        "http://localhost:8000/detect/image",
        files={"file": f},
        params={"model": "YOLOv8n (Fastest)", "confidence": 0.4}
    )
print(r.json())
```

---

## 📊 Evaluation Metrics

```python
from utils.metrics import evaluate_detections, fps_benchmark

# Compute mAP
results = evaluate_detections(ground_truth_boxes, predicted_boxes, iou_threshold=0.5)
print(f"mAP@0.5: {results['mAP']}")

# FPS Benchmark
benchmark = fps_benchmark(detector, test_frame, n_runs=50)
print(f"Mean FPS: {benchmark['mean_fps']}")
```

---

## 🚢 Deployment

### Streamlit Cloud
```bash
# Push to GitHub, connect repo at share.streamlit.io
# Set requirements.txt — auto-deploys
```

### HuggingFace Spaces
```bash
# Create Space with Streamlit SDK
# Upload files — auto-deploys
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
```

---

## 📚 Tech Stack
- **YOLOv8** (Ultralytics) — State-of-the-art object detection
- **Deep SORT** (deep-sort-realtime) — Multi-object tracking
- **OpenCV** — Video capture & image processing
- **Streamlit** — Interactive dashboard UI
- **FastAPI** — REST API backend
- **Plotly** — Real-time interactive charts
- **Pandas** — Data logging & export
- **NumPy** — Array & heatmap operations

---

## 👤 Author
**[Your Name]**  
CodeAlpha AI Internship — Task 4  
LinkedIn: [Your LinkedIn]  
GitHub: [Your GitHub]

---

*Built with ❤️ for CodeAlpha AI Internship*
