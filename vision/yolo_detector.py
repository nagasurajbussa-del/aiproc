"""
YOLOv8 object detection module.

Runs YOLO inference on a background thread so that the main camera
loop is never blocked by heavy model execution.  Frame skipping
is applied to further reduce GPU / CPU load.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import numpy as np

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import get_logger

log = get_logger("yolo")


class YOLODetector:
    """
    Async-threaded YOLOv8 object detector with frame skipping.

    Detections covered:
        6-9. Suspicious objects (phone, book, laptop, second person, remote)
    """

    def __init__(self, config: ProctorConfig, state: ProctorState) -> None:
        self.cfg = config
        self.state = state
        self._model = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_flags: List[str] = []
        self._running = False
        self._frame_count = 0

    def load_model(self) -> None:
        """Load the YOLOv8 model (called once at startup)."""
        from ultralytics import YOLO

        log.info("Loading YOLOv8 from %s …", self.cfg.yolo_model_path)
        self._model = YOLO(self.cfg.yolo_model_path)

        # Force GPU if available
        if self.cfg.yolo_use_gpu:
            log.info("YOLO using GPU (CUDA)")
        else:
            log.info("YOLO using CPU")

        log.info("YOLOv8 ready ✓")

    def _inference_loop(self) -> None:
        """Background thread: continuously infer on the latest frame."""
        while self._running:
            frame = None
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    self._latest_frame = None

            if frame is None:
                time.sleep(0.01)
                continue

            flags = self._run_yolo(frame)
            with self._lock:
                self._latest_flags = flags
                self.state.latest_yolo_flags = flags

    def _run_yolo(self, frame: np.ndarray) -> List[str]:
        """Run YOLO inference on a single frame."""
        if self._model is None:
            return []
        flags: List[str] = []
        try:
            device = "0" if self.cfg.yolo_use_gpu else "cpu"
            results = self._model(
                frame,
                conf=self.cfg.yolo_confidence,
                verbose=False,
                device=device,
            )
            seen: set = set()
            for result in results:
                for box in result.boxes:
                    label = result.names[int(box.cls)].lower()
                    conf = float(box.conf)
                    for key, msg in self.cfg.suspicious_classes.items():
                        if key in label and key not in seen:
                            seen.add(key)
                            flags.append(f"{msg} ({conf:.0%})")
        except Exception as exc:
            log.error("YOLO inference error: %s", exc)
        return flags

    def start(self) -> None:
        """Start the background inference thread."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="yolo-worker",
        )
        self._thread.start()
        log.info("YOLO background thread started.")

    def stop(self) -> None:
        """Stop the background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("YOLO background thread stopped.")

    def submit_frame(self, frame: np.ndarray) -> None:
        """
        Submit a frame for async inference.

        Frame skipping: only every N-th frame is actually processed.
        """
        self._frame_count += 1
        if self._frame_count % self.cfg.yolo_frame_skip != 0:
            return
        with self._lock:
            self._latest_frame = frame

    def get_latest_flags(self) -> List[str]:
        """Return the most recent YOLO detection flags (non-blocking)."""
        with self._lock:
            return list(self._latest_flags)

    def run_sync(self, frame: np.ndarray) -> List[str]:
        """Synchronous fallback (used when threading is undesirable)."""
        return self._run_yolo(frame)
