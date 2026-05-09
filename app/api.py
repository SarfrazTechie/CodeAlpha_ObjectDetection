
"""
VisionAI REST API
FastAPI endpoint for detection results, session stats, alerts
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import io
import os
import sys
import base64
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.detector import VisionAIDetector

app = FastAPI(
    title="VisionAI API",
    description="REST API for YOLOv8 Object Detection & Tracking",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
_detector: Optional[VisionAIDetector] = None

def get_detector(model_key="YOLOv8n (Fastest)", conf=0.4):
    global _detector
    if _detector is None:
        _detector = VisionAIDetector(model_key=model_key, conf=conf)
    return _detector


@app.get("/")
async def root():
    return {"message": "VisionAI API running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/models")
async def list_models():
    return {"models": list(VisionAIDetector.MODELS.keys())}


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model: str = "YOLOv8n (Fastest)",
    confidence: float = 0.4,
    return_image: bool = False
):
    """Detect objects in an uploaded image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    det = get_detector(model, confidence)
    vis, analytics = det.process_frame(frame)

    response = {
        "frame_id": analytics.frame_id,
        "total_detections": analytics.total_detections,
        "fps": round(analytics.fps, 2),
        "class_counts": analytics.class_counts,
        "detections": analytics.detections,
        "zone_violations": analytics.zone_violations,
    }

    if return_image:
        _, buf = cv2.imencode(".jpg", vis)
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        response["annotated_image_b64"] = img_b64

    return JSONResponse(response)


@app.get("/session/stats")
async def session_stats():
    """Get current session statistics."""
    det = get_detector()
    stats = det.get_session_stats()
    return JSONResponse(stats)


@app.post("/session/export")
async def export_session():
    """Export session logs to JSON file."""
    det = get_detector()
    path = det.save_logs()
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json",
                            filename=os.path.basename(path))
    raise HTTPException(404, "Log file not found")


@app.post("/alerts/classes")
async def set_alert_classes(classes: list[str]):
    """Set classes to trigger alerts for."""
    det = get_detector()
    for cls in classes:
        det.add_alert_class(cls)
    return {"message": f"Alert classes set: {classes}"}


@app.post("/zones/add")
async def add_zone(name: str, polygon: list[list[int]], alert: bool = True):
    """Add a detection zone by polygon coordinates."""
    det = get_detector()
    det.zone_mgr.add_zone(name, polygon, alert=alert)
    return {"message": f"Zone '{name}' added"}


@app.delete("/zones/{name}")
async def remove_zone(name: str):
    """Remove a detection zone."""
    det = get_detector()
    det.zone_mgr.remove_zone(name)
    return {"message": f"Zone '{name}' removed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
