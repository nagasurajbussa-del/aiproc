"""
Shared utilities for the AI Proctor system.

- Structured logging (replaces all ``print()`` calls)
- Folder setup
- Haar cascade loading
- Frame pre-processing (compute once, share everywhere)
- FPS limiter
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ai_proctor.core.config import ProctorConfig

# ─── Structured Logger ────────────────────────────────────────────

_LOG_FORMAT = (
    "%(asctime)s │ %(levelname)-7s │ %(name)-18s │ %(message)s"
)
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger with consistent formatting."""
    logger = logging.getLogger(f"proctor.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger("utils")


# ─── Folder Setup ─────────────────────────────────────────────────

def setup_folders(config: ProctorConfig) -> None:
    """Create all required output directories."""
    for folder in (
        config.base_folder,
        config.flagged_folder,
        config.report_folder,
        config.snap_folder,
    ):
        os.makedirs(folder, exist_ok=True)
    log.info("Output folders ready: %s/", config.base_folder)


# ─── Haar Cascade Loading ─────────────────────────────────────────

def load_cascade(filename: str) -> Optional[cv2.CascadeClassifier]:
    """
    Load a Haar cascade from OpenCV's built-in data directory.

    Falls back to downloading from GitHub if the local file is missing.
    """
    builtin = os.path.join(cv2.data.haarcascades, filename)
    if os.path.exists(builtin):
        cascade = cv2.CascadeClassifier(builtin)
        if not cascade.empty():
            return cascade

    if not os.path.exists(filename):
        import urllib.request

        url = (
            "https://raw.githubusercontent.com/opencv/opencv/master"
            f"/data/haarcascades/{filename}"
        )
        log.info("Downloading cascade: %s", filename)
        urllib.request.urlretrieve(url, filename)

    cascade = cv2.CascadeClassifier(filename)
    return cascade if not cascade.empty() else None


# ─── Frame Pre-Processing ─────────────────────────────────────────

@dataclass
class PreprocessedFrame:
    """
    Holds all pre-computed versions of a single video frame.

    Compute once at the start of each snapshot cycle so that
    individual detectors never repeat grayscale / blur work.
    """

    raw: np.ndarray
    gray: np.ndarray
    gray_eq: np.ndarray
    blurred: np.ndarray
    height: int
    width: int
    frame_id: int


def preprocess_frame(frame: np.ndarray, frame_id: int) -> PreprocessedFrame:
    """Convert a raw BGR frame into all required representations."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    h, w = frame.shape[:2]
    return PreprocessedFrame(
        raw=frame,
        gray=gray,
        gray_eq=gray_eq,
        blurred=blurred,
        height=h,
        width=w,
        frame_id=frame_id,
    )


# ─── FPS Limiter ──────────────────────────────────────────────────

class FPSLimiter:
    """Throttle the main loop to a target FPS."""

    def __init__(self, target_fps: int = 30) -> None:
        self._interval = 1.0 / max(target_fps, 1)
        self._last_time = 0.0
        self._actual_fps = 0.0
        self._frame_count = 0
        self._fps_update_time = time.time()

    def wait(self) -> None:
        """Sleep until the next frame is due."""
        now = time.time()
        elapsed = now - self._last_time
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_time = time.time()

        # Track actual FPS
        self._frame_count += 1
        fps_elapsed = time.time() - self._fps_update_time
        if fps_elapsed >= 1.0:
            self._actual_fps = self._frame_count / fps_elapsed
            self._frame_count = 0
            self._fps_update_time = time.time()

    @property
    def actual_fps(self) -> float:
        """Return the measured FPS over the last second."""
        return self._actual_fps


# ─── Snapshot Saver ───────────────────────────────────────────────

def save_snapshot(
    frame: np.ndarray,
    folder: str,
    prefix: str = "snap",
) -> str:
    """Save a JPEG snapshot and return the file path."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(folder, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path
