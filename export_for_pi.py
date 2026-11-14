#!/usr/bin/env python3
"""
Model Export Tool for Raspberry Pi Deployment.

Converts standard YOLOv8 PyTorch weights to optimized runtime formats:
- ONNX: High performance cross-platform runtime
- NCNN: Ultra-fast ARM NEON accelerated inference tailored for Raspberry Pi 4/5
- TFLite: Lightweight runtime for Edge TPU and embedded devices
"""

import argparse
import os
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PiExporter")

def export_model(model_path: str = "models/yolov8n.pt", target_format: str = "onnx", imgsz: int = 640):
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found.")
        return

    logger.info(f"Loading {model_path}...")
    model = YOLO(model_path)

    logger.info(f"Exporting to format: {target_format} (image size: {imgsz})...")
    try:
        exported_path = model.export(format=target_format, imgsz=imgsz)
        logger.info(f"Export successful! Optimized model saved at: {exported_path}")
        logger.info(f"You can now point 'model_path' in config/default_config.yaml to this file.")
    except Exception as e:
        logger.error(f"Export failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 for Raspberry Pi acceleration")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", help="Path to .pt weights")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "ncnn", "tflite", "engine", "openvino"], help="Target format")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    args = parser.parse_args()

    export_model(args.model, args.format, args.imgsz)
