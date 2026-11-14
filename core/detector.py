"""
YOLOv8 Human Detection & Tracking Engine for Drone Search and Rescue.

Filters specifically for human detections with confidence >= 80% (0.80)
and integrates multi-object tracking for persistent disaster survivor monitoring.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class DroneHumanDetector:
    """
    Wraps YOLOv8 model with strict confidence filtering (>= 80%)
    and persistent ID tracking.
    """
    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.80,
        iou_threshold: float = 0.45,
        human_class_id: int = 0,
        inference_size: int = 640,
        half_precision: bool = False,
        enable_tracking: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.human_class_id = human_class_id
        self.inference_size = inference_size
        self.enable_tracking = enable_tracking

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        logger.info(f"Loading YOLOv8 weights from: {model_path}...")
        self.model = YOLO(model_path)

        # Select fastest available execution device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.half = half_precision and (self.device == "cuda")

        # Fuse model layers for maximum inference speed on Raspberry Pi / CPU
        try:
            self.model.fuse()
        except Exception:
            pass

        logger.info(f"Detector ready on {self.device}. Conf threshold: {self.confidence_threshold * 100:.1f}%")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on frame and return list of detected humans meeting
        the confidence threshold (>= 80%).

        Args:
            frame: OpenCV BGR image (numpy ndarray).

        Returns:
            List of dictionaries:
            [{
                "track_id": int or None,
                "confidence": float (>= 0.80),
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "bottom_center": (cx, y2),
                "width": int,
                "height": int
            }, ...]
        """
        detections = []
        h, w = frame.shape[:2]

        with torch.no_grad():
            if self.enable_tracking:
                results = self.model.track(
                    source=frame,
                    classes=[self.human_class_id],
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.inference_size,
                    persist=True,
                    verbose=False
                )
            else:
                results = self.model.predict(
                    source=frame,
                    classes=[self.human_class_id],
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.inference_size,
                    verbose=False
                )

        if not results or len(results) == 0:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        track_ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)

        for box, conf, cls, tid in zip(xyxy, confidences, classes, track_ids):
            if int(cls) != self.human_class_id:
                continue

            # Strict 80% filter check
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(float, box)
            # Clamp to frame boundary
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(0.0, min(float(w - 1), x2))
            y2 = max(0.0, min(float(h - 1), y2))

            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections.append({
                "track_id": int(tid) if tid is not None else None,
                "confidence": float(conf),
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "bottom_center": (cx, y2),
                "width": bw,
                "height": bh
            })

        return detections
