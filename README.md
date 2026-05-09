<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00e5ff,100:7eff8a&height=200&section=header&text=VisionAI&fontSize=80&fontColor=fff&animation=fadeIn&fontAlignY=35&desc=Advanced%20Object%20Detection%20%26%20Tracking%20System&descAlignY=55&descSize=18" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![DeepSORT](https://img.shields.io/badge/Deep_SORT-Tracking-7B2FBE?style=for-the-badge&logo=opencv&logoColor=white)](https://github.com/levan92/deep_sort_realtime)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

<br/>

> 🎓 **CodeAlpha AI Internship — Task 4: Object Detection & Tracking**
> 
> A production-grade AI surveillance system featuring real-time detection, multi-object tracking, analytics dashboard, REST API, and smart zone alerts.

<br/>

---

</div>

## 📸 Demo

<div align="center">

| Live Detection | Analytics Dashboard |
|:-:|:-:|
| Real-time YOLOv8 bounding boxes with Track IDs | Streamlit dashboard with live charts |
| Deep SORT multi-object tracking | Class distribution & FPS timeline |
| Motion trails & heatmap overlay | Alert log & CSV export |

</div>

---

## ✨ Features at a Glance

<div align="center">

| 🧠 Detection | 🎯 Tracking | 📊 Analytics | ⚠️ Alerts |
|:---:|:---:|:---:|:---:|
| YOLOv8 n/s/m/l/x | Deep SORT | Live FPS counter | Per-class triggers |
| 80 COCO classes | Persistent Track IDs | Class distribution | Zone violations |
| Confidence slider | Motion trails | FPS timeline | Timestamped log |
| IoU threshold | Re-identification | Session stats | Callback system |

| 🔐 Zones | 🌡️ Heatmap | 🧾 Logging | 🚀 Deployment |
|:---:|:---:|:---:|:---:|
| Polygon zones | Movement map | CSV export | Streamlit Cloud |
| Named zones | JET colormap | JSON reports | HuggingFace |
| Visual overlay | Per-session reset | In-app table | Docker ready |
| Alert on entry | Cumulative | Download button | FastAPI REST |

</div>

---

## 🧠 Model Comparison

<div align="center">

| Model | Speed | Accuracy | Best For |
|:---:|:---:|:---:|:---:|
| `YOLOv8n` | ⚡⚡⚡⚡⚡ | ★★★☆☆ | Real-time webcam |
| `YOLOv8s` | ⚡⚡⚡⚡☆ | ★★★★☆ | Balanced use |
| `YOLOv8m` | ⚡⚡⚡☆☆ | ★★★★☆ | General purpose |
| `YOLOv8l` | ⚡⚡☆☆☆ | ★★★★★ | High accuracy |
| `YOLOv8x` | ⚡☆☆☆☆ | ★★★★★ | Best accuracy |

</div>

---

## 📁 Project Structure

```
CodeAlpha_ObjectDetection/
│
├── 📄 streamlit_app.py          ← Main dashboard UI (Streamlit)
├── 📄 simple.py                 ← Lightweight OpenCV-only version
├── 📄 requirements.txt          ← All dependencies
├── 📄 Dockerfile                ← Container deployment
├── 📄 README.md                 ← You are here
│
├── 📂 app/
│   ├── 🧠 detector.py           ← Core engine (YOLOv8 + DeepSORT + Zones)
│   └── 🌐 api.py                ← FastAPI REST endpoints
│
├── 📂 utils/
│   └── 📏 metrics.py            ← mAP, Precision/Recall, FPS benchmark
│
├── 📂 logs/                     ← Auto-generated CSV detection logs
└── 📂 exports/                  ← JSON session reports
```

---

## ⚡ Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SarfrazTechie/CodeAlpha_ObjectDetection.git
cd CodeAlpha_ObjectDetection
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run — choose your mode

**🖥️ Full Dashboard (Streamlit):**
```bash
python -m streamlit run streamlit_app.py
```

**⚡ Simple & Fast (OpenCV only):**
```bash
python simple.py
```

**🌐 REST API (FastAPI):**
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# Swagger docs → http://localhost:8000/docs
```

---

## 🎮 How to Use

```
1. Select Model     →  YOLOv8n (fastest) to YOLOv8x (most accurate)
2. Choose Source    →  Webcam / Video File / Image
3. Set Thresholds   →  Confidence & IoU sliders
4. Enable Tracking  →  Deep SORT + Motion Trails toggle
5. Set Alerts       →  Pick classes to trigger alerts (person, car, etc.)
6. Click ▶ START    →  Detection begins!
7. View Analytics   →  Live charts update in real-time
8. Export Logs      →  Download CSV with all detections
```

---

## 🔌 REST API Endpoints

```http
POST   /detect/image      →  Detect objects in uploaded image
GET    /session/stats     →  Current session statistics
POST   /session/export    →  Download JSON log
POST   /zones/add         →  Add polygon detection zone
DELETE /zones/{name}      →  Remove a zone
POST   /alerts/classes    →  Set alert trigger classes
GET    /models            →  List available models
```

**Example API call:**
```python
import requests

with open("photo.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/detect/image",
        files={"file": f},
        params={"model": "YOLOv8n (Fastest)", "confidence": 0.4}
    )
print(r.json())
```

---

## 🔧 Advanced Usage

### Add a Smart Zone
```python
from app.detector import VisionAIDetector

det = VisionAIDetector()
det.zone_mgr.add_zone(
    name="Restricted Area",
    polygon=[[100,100],[400,100],[400,300],[100,300]],
    color=(0, 0, 255),
    alert=True
)
```

### Custom Alert Handler
```python
def my_alert(message, class_name, zone):
    print(f"🚨 {message}")
    # → Send email, SMS, webhook, etc.

det.register_alert_callback(my_alert)
det.add_alert_class("person")
```

### Evaluate Model Performance
```python
from utils.metrics import evaluate_detections, fps_benchmark

results = evaluate_detections(ground_truth, predictions, iou_threshold=0.5)
print(f"mAP@0.5: {results['mAP']}")

bench = fps_benchmark(detector, test_frame, n_runs=50)
print(f"Mean FPS: {bench['mean_fps']}")
```

---

## 🚢 Deployment Options

<div align="center">

| Platform | Command | Free Tier |
|:---:|:---:|:---:|
| **Streamlit Cloud** | Push to GitHub → connect at share.streamlit.io | ✅ Yes |
| **HuggingFace Spaces** | Create Space → Streamlit SDK → upload files | ✅ Yes |
| **Docker** | `docker build -t visionai . && docker run -p 8501:8501 visionai` | ✅ Yes |
| **AWS / Azure** | Deploy container to cloud VM | 💳 Paid |

</div>

---

## 🛠️ Tech Stack

<div align="center">

| Library | Version | Purpose |
|:---:|:---:|:---:|
| `ultralytics` | 8.4+ | YOLOv8 detection engine |
| `deep-sort-realtime` | 1.3+ | Multi-object tracking |
| `opencv-python` | 4.11 | Video capture & processing |
| `streamlit` | 1.46 | Interactive dashboard |
| `fastapi` | 0.103+ | REST API backend |
| `plotly` | 6.7 | Real-time charts |
| `pandas` | 2.3 | Data logging & export |
| `torch` | 2.7 | Deep learning backend |
| `numpy` | 1.26 | Array operations |

</div>

---

## 📊 Evaluation Metrics Supported

- ✅ **mAP** (mean Average Precision) @ IoU 0.5
- ✅ **Precision & Recall** per class
- ✅ **FPS Benchmark** (mean, min, max, std)
- ✅ **Confusion Matrix** ready
- ✅ **Detection Log** with timestamps & confidence scores

---

<div align="center">

## 👤 Author

**Sarfraz**  
CodeAlpha AI Internship — Task 4  

[![GitHub](https://img.shields.io/badge/GitHub-SarfrazTechie-181717?style=for-the-badge&logo=github)](https://github.com/SarfrazTechie)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com)

<br/>

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7eff8a,100:00e5ff&height=100&section=footer" width="100%"/>

*Built with ❤️ for CodeAlpha AI Internship*

</div>
