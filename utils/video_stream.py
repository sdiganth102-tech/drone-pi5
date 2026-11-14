"""
Unified Video Stream Reader for Raspberry Pi CSI Camera, USB Webcam, and Video Files.
"""

import time
import logging
import threading
import queue
from typing import Optional, Union, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class VideoStream:
    """
    Threaded, low-latency video capture stream compatible with:
    - Raspberry Pi Camera (Picamera2 / libcamera)
    - Standard USB UVC Webcams
    - Pre-recorded drone video files and RTSP drone telemetry streams
    """
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        threaded: bool = True
    ):
        self.source = source
        self.width = width
        self.height = height
        self.threaded = threaded

        self.cap = None
        self.picam2 = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread = None

        self._init_source()

    def _init_source(self):
        # Attempt Picamera2 initialization if on Raspberry Pi with CSI camera
        if self.source == "picam2" or (isinstance(self.source, int) and self.source == -1):
            try:
                from picamera2 import Picamera2
                logger.info("Initializing Raspberry Pi Camera via Picamera2...")
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                logger.info("Picamera2 started successfully.")
                return
            except Exception as e:
                logger.warning(f"Picamera2 init failed ({e}), falling back to standard OpenCV VideoCapture.")

        # Standard OpenCV VideoCapture (USB webcam, v4l2, or video file)
        logger.info(f"Opening OpenCV VideoCapture source: {self.source}...")
        self.cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")

    def start(self):
        if self.threaded and self.cap is not None and isinstance(self.source, int):
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()
        return self

    def _capture_worker(self):
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                else:
                    time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.picam2 is not None:
            try:
                rgb_frame = self.picam2.capture_array()
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                return True, bgr_frame
            except Exception as e:
                logger.error(f"Picamera2 capture error: {e}")
                return False, None

        if self.threaded and self.thread is not None and self.thread.is_alive():
            try:
                frame = self.frame_queue.get(timeout=1.0)
                return True, frame
            except queue.Empty:
                return False, None

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return ret, frame

        return False, None

    def release(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
            self.picam2 = None
