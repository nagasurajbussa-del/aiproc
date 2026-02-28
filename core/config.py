"""
Centralized configuration for the AI Proctor system.

All magic numbers are extracted here. Every value can be overridden
via an environment variable with the ``PROCTOR_`` prefix.

Example:
    ``PROCTOR_EXAM_DURATION_MINUTES=90 python -m ai_proctor.main``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


def _env(key: str, default: str) -> str:
    """Read ``PROCTOR_<key>`` from environment, falling back to *default*."""
    return os.environ.get(f"PROCTOR_{key}", default)


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes")


# ─── GPU detection ────────────────────────────────────────────────
def detect_gpu() -> bool:
    """Return ``True`` if a CUDA-capable GPU is available for YOLO."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@dataclass(frozen=True)
class ProctorConfig:
    """Immutable, centralized configuration for the entire system."""

    # ── Exam ──────────────────────────────────────────────────────
    exam_duration_minutes: int = _env_int("EXAM_DURATION_MINUTES", 60)
    screenshot_interval: int = _env_int("SCREENSHOT_INTERVAL", 5)
    camera_index: int = _env_int("CAMERA_INDEX", 0)
    frame_width: int = _env_int("FRAME_WIDTH", 640)
    frame_height: int = _env_int("FRAME_HEIGHT", 480)
    fps_limit: int = _env_int("FPS_LIMIT", 30)
    demo_mode: bool = _env_bool("DEMO_MODE", False)

    # ── Folders ───────────────────────────────────────────────────
    base_folder: str = _env("BASE_FOLDER", "proctor_data")
    flagged_folder: str = ""
    report_folder: str = ""
    snap_folder: str = ""

    # ── Face Detection ────────────────────────────────────────────
    face_scale_factor: float = _env_float("FACE_SCALE_FACTOR", 1.05)
    face_min_neighbors: int = _env_int("FACE_MIN_NEIGHBORS", 3)
    face_min_size: tuple[int, int] = (40, 40)

    # ── Gaze ──────────────────────────────────────────────────────
    gaze_offset_threshold: float = _env_float("GAZE_OFFSET_THRESHOLD", 52.0)
    consecutive_away_limit: int = _env_int("CONSECUTIVE_AWAY_LIMIT", 4)
    gaze_direction_limit: int = _env_int("GAZE_DIRECTION_LIMIT", 6)

    # ── Movement ──────────────────────────────────────────────────
    movement_threshold: float = _env_float("MOVEMENT_THRESHOLD", 22.0)
    movement_window: int = _env_int("MOVEMENT_WINDOW", 10)
    movement_flag_count: int = _env_int("MOVEMENT_FLAG_COUNT", 7)

    # ── Mouth ─────────────────────────────────────────────────────
    mouth_diff_threshold: float = _env_float("MOUTH_DIFF_THRESHOLD", 7.0)
    mouth_window: int = _env_int("MOUTH_WINDOW", 5)
    mouth_flag_count: int = _env_int("MOUTH_FLAG_COUNT", 3)

    # ── Eyes ──────────────────────────────────────────────────────
    eye_closed_limit: int = _env_int("EYE_CLOSED_LIMIT", 4)

    # ── Brightness ────────────────────────────────────────────────
    brightness_threshold: float = _env_float("BRIGHTNESS_THRESHOLD", 20.0)
    brightness_change_limit: float = _env_float("BRIGHTNESS_CHANGE_LIMIT", 45.0)

    # ── Face Size / Lean ──────────────────────────────────────────
    face_size_change_ratio: float = _env_float("FACE_SIZE_CHANGE_RATIO", 1.7)
    face_shrink_ratio: float = _env_float("FACE_SHRINK_RATIO", 0.35)

    # ── Head Down ─────────────────────────────────────────────────
    head_down_threshold: float = _env_float("HEAD_DOWN_THRESHOLD", 78.0)

    # ── Desk Object ───────────────────────────────────────────────
    desk_diff_threshold: float = _env_float("DESK_DIFF_THRESHOLD", 55.0)
    desk_baseline_update: int = _env_int("DESK_BASELINE_UPDATE", 20)

    # ── Text Density ──────────────────────────────────────────────
    edge_density_threshold: float = _env_float("EDGE_DENSITY_THRESHOLD", 0.20)

    # ── Identity ──────────────────────────────────────────────────
    identity_diff_threshold: float = _env_float("IDENTITY_DIFF_THRESHOLD", 60.0)

    # ── Rough Sheet ───────────────────────────────────────────────
    rough_sheet_min_area: int = _env_int("ROUGH_SHEET_MIN_AREA", 5000)
    rough_sheet_white_ratio: float = _env_float("ROUGH_SHEET_WHITE_RATIO", 0.50)
    rough_sheet_line_threshold: int = _env_int("ROUGH_SHEET_LINE_THRESHOLD", 3)

    # ── Earphone ──────────────────────────────────────────────────
    earphone_line_threshold: int = _env_int("EARPHONE_LINE_THRESHOLD", 6)

    # ── YOLOv8 ────────────────────────────────────────────────────
    yolo_confidence: float = _env_float("YOLO_CONFIDENCE", 0.60)
    yolo_model_path: str = _env("YOLO_MODEL_PATH", "yolov8n.pt")
    yolo_use_gpu: bool = _env_bool("YOLO_USE_GPU", True)
    yolo_frame_skip: int = _env_int("YOLO_FRAME_SKIP", 2)

    # ── Audio ─────────────────────────────────────────────────────
    audio_sample_rate: int = _env_int("AUDIO_SAMPLE_RATE", 48000)
    audio_frame_duration: int = _env_int("AUDIO_FRAME_DURATION", 5)
    audio_calibration_duration: int = _env_int("AUDIO_CALIBRATION_DURATION", 5)
    audio_alert_duration: int = _env_int("AUDIO_ALERT_DURATION", 10)
    speech_threshold: float = _env_float("SPEECH_THRESHOLD", 0.004)
    spike_threshold: float = _env_float("SPIKE_THRESHOLD", 0.008)
    whisper_rms_factor: float = _env_float("WHISPER_RMS_FACTOR", 1.35)
    whisper_centroid_max: float = _env_float("WHISPER_CENTROID_MAX", 1800.0)
    muffled_centroid_max: float = _env_float("MUFFLED_CENTROID_MAX", 1300.0)
    muffled_zcr_max: float = _env_float("MUFFLED_ZCR_MAX", 0.045)
    shout_rms_factor: float = _env_float("SHOUT_RMS_FACTOR", 1.8)
    shout_centroid_min: float = _env_float("SHOUT_CENTROID_MIN", 2200.0)

    # ── Suspicion Scoring (weights) ───────────────────────────────
    suspicion_threshold: int = _env_int("SUSPICION_THRESHOLD", 100)

    suspicion_weights: Dict[str, int] = field(default_factory=lambda: {
        "camera_covered":       30,
        "no_face":              20,
        "multiple_faces":       25,
        "phone_detected":       40,
        "book_detected":        25,
        "laptop_detected":      35,
        "second_person":        25,
        "remote_detected":      20,
        "mouth_movement":       10,
        "eyes_closed":          10,
        "hand_over_mouth":      15,
        "head_down":            15,
        "lighting_change":       5,
        "leaning_forward":      10,
        "new_desk_object":      20,
        "text_density":         15,
        "repeated_gaze":        15,
        "face_size_change":     10,
        "earphone":             30,
        "identity_change":      50,
        "rough_sheet":          35,
        "body_movement":        10,
        "voice_detected":       10,
        "whisper":              10,
        "normal_speech":        15,
        "muffled_speech":       15,
        "shout":                20,
        "loud_spike":           15,
    })

    # ── Suspicious YOLO classes ───────────────────────────────────
    suspicious_classes: Dict[str, str] = field(default_factory=lambda: {
        "cell phone": "Phone detected",
        "book":       "Suspicious book/paper detected",
        "laptop":     "Second laptop/device detected",
        "person":     "Second person in frame",
        "remote":     "Remote/device detected",
    })

    # ── API ───────────────────────────────────────────────────────
    api_host: str = _env("API_HOST", "0.0.0.0")
    api_port: int = _env_int("API_PORT", 8000)

    # ── Confidence decay ──────────────────────────────────────────
    confidence_decay_rate: float = _env_float("CONFIDENCE_DECAY_RATE", 0.95)
    confidence_decay_interval: float = _env_float("CONFIDENCE_DECAY_INTERVAL", 30.0)

    def __post_init__(self) -> None:
        # Derive folder paths from base_folder
        object.__setattr__(self, "flagged_folder", os.path.join(self.base_folder, "flagged"))
        object.__setattr__(self, "report_folder", os.path.join(self.base_folder, "reports"))
        object.__setattr__(self, "snap_folder", os.path.join(self.base_folder, "snapshots"))

        # Auto-detect GPU
        if self.yolo_use_gpu and not detect_gpu():
            object.__setattr__(self, "yolo_use_gpu", False)
