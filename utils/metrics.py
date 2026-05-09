
"""
VisionAI Evaluation Metrics
mAP, Precision, Recall, Confusion Matrix, FPS benchmark
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict


def compute_iou(boxA: list, boxB: list) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += (p.max() if p.size > 0 else 0.0)
    return ap / 11.0


def evaluate_detections(
    gt_boxes: List[Tuple[str, list]],  # [(class, [x1,y1,x2,y2]), ...]
    pred_boxes: List[Tuple[str, float, list]],  # [(class, conf, [x1,y1,x2,y2]), ...]
    iou_threshold: float = 0.5
) -> Dict:
    """
    Compute per-class AP and mAP.
    Returns dict with per-class results and mAP.
    """
    classes = list(set([g[0] for g in gt_boxes] + [p[0] for p in pred_boxes]))
    results = {}

    for cls in classes:
        gt_cls = [b for c,b in gt_boxes if c == cls]
        pred_cls = sorted([(c,b) for c,cf,b in pred_boxes if c == cls],
                          key=lambda x: -pred_boxes[[p[0] for p in pred_boxes].index(cls)][1] if cls in [p[0] for p in pred_boxes] else 0)

        if not gt_cls:
            results[cls] = {"ap": 0.0, "precision": 0.0, "recall": 0.0}
            continue

        tp = np.zeros(len(pred_cls))
        fp = np.zeros(len(pred_cls))
        matched = set()

        for i,(conf,pbox) in enumerate([(cf,b) for c,cf,b in pred_boxes if c==cls]):
            best_iou = 0; best_j = -1
            for j,gbox in enumerate(gt_cls):
                if j in matched: continue
                iou = compute_iou(pbox,gbox)
                if iou > best_iou:
                    best_iou = iou; best_j = j
            if best_iou >= iou_threshold and best_j not in matched:
                tp[i] = 1; matched.add(best_j)
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / len(gt_cls)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        ap = compute_ap(recalls, precisions)
        results[cls] = {
            "ap": round(ap,4),
            "precision": round(precisions[-1] if len(precisions) else 0.0, 4),
            "recall": round(recalls[-1] if len(recalls) else 0.0, 4),
        }

    mAP = np.mean([r["ap"] for r in results.values()])
    return {"per_class": results, "mAP": round(float(mAP),4)}


def fps_benchmark(detector, test_frame: np.ndarray, n_runs: int = 50) -> Dict:
    """Benchmark FPS across models."""
    import time
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        detector.process_frame(test_frame.copy())
        times.append(time.perf_counter() - t0)
    times = times[5:]  # Warm-up discard
    return {
        "mean_fps": round(1.0/np.mean(times),2),
        "min_fps": round(1.0/np.max(times),2),
        "max_fps": round(1.0/np.min(times),2),
        "std_ms": round(np.std(times)*1000,2),
        "mean_ms": round(np.mean(times)*1000,2),
        "n_runs": len(times),
    }
