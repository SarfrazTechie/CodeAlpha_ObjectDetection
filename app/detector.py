"""
VisionAI Core Detection Engine
YOLOv8 + Deep SORT + Zones + Heatmaps + Analytics + Alerts
"""

import cv2
import numpy as np
import time
import json
import csv
import os
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
import threading

@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list
    center: tuple
    timestamp: float
    frame_id: int
    in_zone: bool = False
    zone_name: str = ""

@dataclass
class FrameAnalytics:
    frame_id: int
    timestamp: float
    fps: float
    total_detections: int
    class_counts: dict
    active_tracks: int
    zone_violations: list
    detections: list = field(default_factory=list)

COLORS = [
    (255,56,56),(255,157,151),(255,112,31),(255,178,29),(207,210,49),
    (72,249,10),(146,204,23),(61,219,134),(26,147,52),(0,212,187),
    (44,153,168),(0,194,255),(52,69,147),(100,115,255),(0,24,236),
    (132,56,255),(82,0,133),(203,56,255),(255,149,200),(255,55,199),
]

def get_color(idx):
    return COLORS[int(idx) % len(COLORS)]


class HeatmapAccumulator:
    def __init__(self, shape=(480,640)):
        self.map = np.zeros(shape, dtype=np.float32)
        self.shape = shape

    def update(self, cx, cy, radius=20):
        h,w = self.shape
        cx,cy = max(0,min(w-1,int(cx))), max(0,min(h-1,int(cy)))
        cv2.circle(self.map,(cx,cy),radius,1.0,-1)

    def get_overlay(self, frame):
        resized = cv2.resize(self.map,(frame.shape[1],frame.shape[0]))
        norm = cv2.normalize(resized,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        alpha = (norm/255.0*0.6)[...,np.newaxis]
        return (frame.astype(np.float32)*(1-alpha)+colored.astype(np.float32)*alpha).astype(np.uint8)

    def reset(self):
        self.map = np.zeros(self.shape, dtype=np.float32)


class ZoneManager:
    def __init__(self):
        self.zones = {}

    def add_zone(self, name, polygon, color=(0,0,255), alert=True):
        self.zones[name] = {"polygon":np.array(polygon,dtype=np.int32),"color":color,"alert":alert}

    def remove_zone(self, name):
        self.zones.pop(name,None)

    def point_in_zone(self, cx, cy):
        violations = []
        for name,z in self.zones.items():
            if cv2.pointPolygonTest(z["polygon"],(float(cx),float(cy)),False) >= 0:
                violations.append(name)
        return violations

    def draw_zones(self, frame):
        overlay = frame.copy()
        for name,z in self.zones.items():
            cv2.fillPoly(overlay,[z["polygon"]],z["color"])
            cv2.polylines(frame,[z["polygon"]],True,z["color"],2)
            M = cv2.moments(z["polygon"])
            if M["m00"]>0:
                cx,cy = int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])
                cv2.putText(frame,name,(cx-20,cy),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        return cv2.addWeighted(overlay,0.25,frame,0.75,0)


class TrailTracker:
    def __init__(self, maxlen=40):
        self.trails = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, tid, center):
        self.trails[tid].append(center)

    def draw(self, frame, color_fn=None):
        for tid, pts in self.trails.items():
            pts_list = list(pts)
            col = color_fn(tid) if color_fn else (0,255,200)
            for i in range(1,len(pts_list)):
                alpha = i/len(pts_list)
                cv2.line(frame,pts_list[i-1],pts_list[i],col,max(1,int(2*alpha)))
        return frame

    def cleanup(self, active_ids):
        dead = [k for k in list(self.trails.keys()) if k not in active_ids]
        for k in dead:
            del self.trails[k]


class DetectionLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir,exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir,f"detections_{ts}.csv")
        self.json_path = os.path.join(log_dir,f"session_{ts}.json")
        self.records = []
        self.fieldnames = ["timestamp","frame_id","track_id","class_name","confidence","x1","y1","x2","y2","zone"]
        with open(self.csv_path,"w",newline="") as f:
            csv.DictWriter(f,fieldnames=self.fieldnames).writeheader()

    def log(self, det):
        row={"timestamp":datetime.fromtimestamp(det.timestamp).isoformat(),"frame_id":det.frame_id,
             "track_id":det.track_id,"class_name":det.class_name,"confidence":round(det.confidence,3),
             "x1":det.bbox[0],"y1":det.bbox[1],"x2":det.bbox[2],"y2":det.bbox[3],"zone":det.zone_name}
        self.records.append(row)
        with open(self.csv_path,"a",newline="") as f:
            csv.DictWriter(f,fieldnames=self.fieldnames).writerow(row)

    def save_json(self):
        with open(self.json_path,"w") as f:
            json.dump(self.records,f,indent=2)
        return self.json_path

    def get_summary(self):
        if not self.records: return {}
        cc = defaultdict(int)
        for r in self.records: cc[r["class_name"]]+=1
        return {"total_detections":len(self.records),"unique_classes":dict(cc),"csv_path":self.csv_path}


class FPSCounter:
    def __init__(self,window=30):
        self.ts = deque(maxlen=window)

    def tick(self):
        self.ts.append(time.time())
        if len(self.ts)<2: return 0.0
        return (len(self.ts)-1)/(self.ts[-1]-self.ts[0])


class VisionAIDetector:
    MODELS = {
        "YOLOv8n (Fastest)":"yolov8n.pt",
        "YOLOv8s (Balanced)":"yolov8s.pt",
        "YOLOv8m (Accurate)":"yolov8m.pt",
        "YOLOv8l (High Acc)":"yolov8l.pt",
        "YOLOv8x (Best)":"yolov8x.pt",
    }

    def __init__(self, model_key="YOLOv8n (Fastest)", conf=0.4, iou=0.45,
                 use_tracking=True, use_trails=True, selected_classes=None, log_dir="logs"):
        self.conf = conf
        self.iou = iou
        self.use_tracking = use_tracking
        self.use_trails = use_trails
        self.selected_classes = selected_classes
        self.model = None
        self.tracker = None
        self.model_key = None
        self.load_model(model_key)
        self.heatmap = HeatmapAccumulator()
        self.zone_mgr = ZoneManager()
        self.trails = TrailTracker()
        self.fps_counter = FPSCounter()
        self.logger = DetectionLogger(log_dir)
        self.frame_id = 0
        self.class_counts_all = defaultdict(int)
        self.class_timeline = []
        self.alert_callbacks = []
        self.alert_classes = set()
        self._lock = threading.Lock()

    def load_model(self, model_key):
        from ultralytics import YOLO
        self.model = YOLO(self.MODELS.get(model_key,"yolov8n.pt"))
        self.model_key = model_key
        self._init_tracker()

    def _init_tracker(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(max_age=30,n_init=3,nms_max_overlap=1.0,
                max_cosine_distance=0.3,nn_budget=None,embedder="mobilenet",half=True,bgr=True)
        except Exception as e:
            print(f"[Tracker] Deep SORT unavailable ({e}), using YOLO only.")
            self.tracker = None

    def add_alert_class(self, cls):
        self.alert_classes.add(cls.lower())

    def register_alert_callback(self, fn):
        self.alert_callbacks.append(fn)

    def _trigger_alert(self, cls, zone=""):
        msg = f"ALERT: '{cls}' detected" + (f" in zone '{zone}'" if zone else "")
        for cb in self.alert_callbacks:
            try: cb(msg,cls,zone)
            except: pass

    def process_frame(self, frame, show_heatmap=False):
        with self._lock:
            self.frame_id += 1
            t = time.time()
            fps = self.fps_counter.tick()

            results = self.model(frame,conf=self.conf,iou=self.iou,
                                 classes=self.selected_classes,verbose=False)[0]

            raw_dets = []
            yolo_boxes = []
            for box in results.boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                raw_dets.append(([x1,y1,x2-x1,y2-y1],conf,cls_id))
                yolo_boxes.append((x1,y1,x2,y2,conf,cls_id,cls_name))

            detections_out = []

            if self.use_tracking and self.tracker and raw_dets:
                tracks = self.tracker.update_tracks(raw_dets,frame=frame)
                active_ids = set()
                for track in tracks:
                    if not track.is_confirmed(): continue
                    tid = track.track_id
                    active_ids.add(tid)
                    ltrb = track.to_ltrb()
                    x1,y1,x2,y2 = map(int,ltrb)
                    cx,cy = (x1+x2)//2,(y1+y2)//2
                    cls_id = getattr(track,'det_class',0) or 0
                    cls_name = self.model.names.get(int(cls_id),"object")
                    det_conf = getattr(track,'det_conf',0.5) or 0.5
                    zones = self.zone_mgr.point_in_zone(cx,cy)
                    zone_name = zones[0] if zones else ""
                    det = Detection(tid,int(cls_id),cls_name,float(det_conf),[x1,y1,x2,y2],(cx,cy),t,self.frame_id,bool(zones),zone_name)
                    detections_out.append(det)
                    self.trails.update(tid,(cx,cy))
                    self.heatmap.update(cx,cy)
                    self.class_counts_all[cls_name] += 1
                    self.class_timeline.append((t,cls_name))
                    self.logger.log(det)
                    if cls_name.lower() in self.alert_classes: self._trigger_alert(cls_name,zone_name)
                    for z in zones:
                        if self.zone_mgr.zones.get(z,{}).get("alert"): self._trigger_alert(cls_name,z)
                self.trails.cleanup(active_ids)
            else:
                for i,(x1,y1,x2,y2,conf,cls_id,cls_name) in enumerate(yolo_boxes):
                    cx,cy = (x1+x2)//2,(y1+y2)//2
                    zones = self.zone_mgr.point_in_zone(cx,cy)
                    zone_name = zones[0] if zones else ""
                    det = Detection(i,cls_id,cls_name,conf,[x1,y1,x2,y2],(cx,cy),t,self.frame_id,bool(zones),zone_name)
                    detections_out.append(det)
                    self.heatmap.update(cx,cy)
                    self.class_counts_all[cls_name] += 1
                    self.class_timeline.append((t,cls_name))
                    self.logger.log(det)

            vis = frame.copy()
            vis = self.zone_mgr.draw_zones(vis)
            if self.use_trails and self.use_tracking:
                vis = self.trails.draw(vis,color_fn=lambda tid:get_color(tid))

            for det in detections_out:
                x1,y1,x2,y2 = det.bbox
                color = get_color(det.track_id if self.use_tracking else det.class_id)
                cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
                label = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
                if det.in_zone: label += f" ZONE:{det.zone_name}"
                (lw,lh),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
                cv2.rectangle(vis,(x1,y1-lh-8),(x1+lw+4,y1),color,-1)
                cv2.putText(vis,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

            cf = defaultdict(int)
            for d in detections_out: cf[d.class_name] += 1
            vis = self._draw_hud(vis,fps,cf,len(detections_out))
            if show_heatmap: vis = self.heatmap.get_overlay(vis)

            return vis, FrameAnalytics(self.frame_id,t,fps,len(detections_out),
                dict(cf),len(detections_out),
                [d.zone_name for d in detections_out if d.in_zone],
                [asdict(d) for d in detections_out])

    def _draw_hud(self,frame,fps,class_counts,total):
        h,w = frame.shape[:2]
        panel = frame.copy()
        cv2.rectangle(panel,(8,8),(240,36+20*max(1,len(class_counts))),(0,0,0),-1)
        frame = cv2.addWeighted(panel,0.55,frame,0.45,0)
        y = 28
        cv2.putText(frame,f"FPS:{fps:.1f}  Objects:{total}",(14,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,180),1)
        y += 20
        for cls,cnt in class_counts.items():
            cv2.putText(frame,f"  {cls}:{cnt}",(14,y),cv2.FONT_HERSHEY_SIMPLEX,0.48,(200,230,255),1)
            y += 18
        ml = f"Model:{self.model_key}"
        (tw,_),_ = cv2.getTextSize(ml,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
        cv2.putText(frame,ml,(w-tw-10,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,255),1)
        cv2.putText(frame,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),(14,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(160,160,160),1)
        return frame

    def get_session_stats(self):
        return {"total_frames":self.frame_id,"class_counts":dict(self.class_counts_all),"log_summary":self.logger.get_summary()}

    def save_logs(self): return self.logger.save_json()
    def reset_heatmap(self): self.heatmap.reset()
