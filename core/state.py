"""
Thread-safe singleton state manager for the AI Proctor system.

Replaces all global variables from the original monolithic script.
Every mutable runtime value lives here so that modules can share
state without import-time side effects.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai_proctor.core.config import ProctorConfig


@dataclass
class ViolationRecord:
    """A single recorded violation event."""

    timestamp: float
    violation_type: str
    subtype: str
    description: str
    confidence: float
    severity: int
    frame_id: int
    score_delta: int
    snapshot_path: Optional[str] = None


class ProctorState:
    """
    Thread-safe singleton that holds all mutable runtime state.

    Usage::

        state = ProctorState(config)
        state.add_violation(record)
        score = state.suspicion_score
    """

    _instance: Optional["ProctorState"] = None
    _lock_cls = threading.Lock()

    def __new__(cls, config: Optional[ProctorConfig] = None) -> "ProctorState":
        with cls._lock_cls:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Optional[ProctorConfig] = None) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._lock = threading.Lock()
        self.config = config or ProctorConfig()

        # ── Exam state ────────────────────────────────────────────
        self.exam_active: bool = False
        self.exam_start_time: float = 0.0
        self.total_snapshots: int = 0
        self.total_warnings: int = 0
        self.frame_counter: int = 0

        # ── Suspicion scoring ─────────────────────────────────────
        self.suspicion_score: float = 0.0
        self.peak_suspicion_score: float = 0.0
        self._last_decay_time: float = 0.0

        # ── Violation history ─────────────────────────────────────
        self.violations: List[ViolationRecord] = []
        self.violation_counts: Dict[str, int] = {}

        # ── Vision tracking state ─────────────────────────────────
        self.snapshot_history: List[Any] = []
        self.mouth_history: List[Any] = []
        self.consecutive_away: int = 0
        self.consecutive_eyes_closed: int = 0
        self.last_brightness: Optional[float] = None
        self.baseline_face_size: Optional[float] = None
        self.baseline_face_img: Optional[Any] = None
        self.baseline_desk_region: Optional[Any] = None
        self.desk_snapshot_counter: int = 0
        self.gaze_direction_counts: Dict[str, int] = {
            "left": 0, "right": 0, "up": 0, "down": 0,
        }

        # ── Audio state ───────────────────────────────────────────
        self.audio_voice_detected: bool = False
        self.audio_classification: str = "SILENCE"
        self.audio_suspicion_score: int = 0
        self.audio_suspicious_start: Optional[float] = None
        self.audio_calibrated: bool = False

        # ── Display state ─────────────────────────────────────────
        self.current_warning_text: str = ""
        self.warning_display_until: float = 0.0

        # ── YOLO async results ────────────────────────────────────
        self.latest_yolo_flags: List[str] = []

    # ── Thread-safe mutators ──────────────────────────────────────

    def add_violation(self, record: ViolationRecord) -> None:
        """Register a violation and update the suspicion score."""
        with self._lock:
            self.violations.append(record)
            self.total_warnings += 1
            self.suspicion_score += record.score_delta
            self.peak_suspicion_score = max(
                self.peak_suspicion_score, self.suspicion_score,
            )
            key = record.subtype or record.violation_type
            self.violation_counts[key] = self.violation_counts.get(key, 0) + 1

    def apply_score_decay(self) -> None:
        """Apply exponential decay to the suspicion score over time."""
        with self._lock:
            now = time.time()
            if self._last_decay_time == 0.0:
                self._last_decay_time = now
                return
            elapsed = now - self._last_decay_time
            if elapsed >= self.config.confidence_decay_interval:
                intervals = int(elapsed / self.config.confidence_decay_interval)
                self.suspicion_score *= (
                    self.config.confidence_decay_rate ** intervals
                )
                self._last_decay_time = now

    def is_cheating_alert(self) -> bool:
        """Return ``True`` if the suspicion score exceeds the threshold."""
        return self.suspicion_score >= self.config.suspicion_threshold

    def increment_snapshots(self) -> int:
        """Bump snapshot counter and return the new total."""
        with self._lock:
            self.total_snapshots += 1
            return self.total_snapshots

    def increment_frame(self) -> int:
        """Bump frame counter and return the new total."""
        with self._lock:
            self.frame_counter += 1
            return self.frame_counter

    def reset(self) -> None:
        """Reset all mutable state for a new exam session."""
        with self._lock:
            self.exam_active = False
            self.exam_start_time = 0.0
            self.total_snapshots = 0
            self.total_warnings = 0
            self.frame_counter = 0
            self.suspicion_score = 0.0
            self.peak_suspicion_score = 0.0
            self._last_decay_time = 0.0
            self.violations.clear()
            self.violation_counts.clear()
            self.snapshot_history.clear()
            self.mouth_history.clear()
            self.consecutive_away = 0
            self.consecutive_eyes_closed = 0
            self.last_brightness = None
            self.baseline_face_size = None
            self.baseline_face_img = None
            self.baseline_desk_region = None
            self.desk_snapshot_counter = 0
            self.gaze_direction_counts = {
                "left": 0, "right": 0, "up": 0, "down": 0,
            }
            self.audio_voice_detected = False
            self.audio_classification = "SILENCE"
            self.audio_suspicion_score = 0
            self.audio_suspicious_start = None
            self.audio_calibrated = False
            self.current_warning_text = ""
            self.warning_display_until = 0.0
            self.latest_yolo_flags = []

    @classmethod
    def destroy(cls) -> None:
        """Destroy the singleton instance (useful for testing)."""
        with cls._lock_cls:
            cls._instance = None
