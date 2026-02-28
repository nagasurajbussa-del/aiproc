"""
Event-based violation engine.

Converts raw detection flags (strings) into structured
``ViolationEvent`` dicts and pushes them into ``ProctorState``.
Acts as the central event bus between detectors and the scoring /
reporting layers.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState, ViolationRecord
from ai_proctor.core.utils import get_logger, save_snapshot

import numpy as np

log = get_logger("engine.violation")

# Map raw flag substrings → canonical subtype keys (used for scoring)
_FLAG_TO_SUBTYPE: Dict[str, str] = {
    "Camera covered":           "camera_covered",
    "body movement":            "body_movement",
    "No face detected":         "no_face",
    "Multiple faces":           "multiple_faces",
    "Mouth movement":           "mouth_movement",
    "Eyes closed":              "eyes_closed",
    "Hand covering mouth":      "hand_over_mouth",
    "Head down":                "head_down",
    "lighting change":          "lighting_change",
    "leaning too far":          "leaning_forward",
    "moved suddenly far":       "face_size_change",
    "Repeatedly looking":       "repeated_gaze",
    "Head turned":              "head_pose",
    "Head tilted":              "head_pose",
    "New object detected":      "new_desk_object",
    "text/paper density":       "text_density",
    "Rough sheet":              "rough_sheet",
    "earphone":                 "earphone",
    "identity change":          "identity_change",
    "Phone detected":           "phone_detected",
    "book/paper detected":      "book_detected",
    "laptop/device detected":   "laptop_detected",
    "Second person":            "second_person",
    "Remote/device":            "remote_detected",
    "Voice/talking":            "voice_detected",
}

# Severity by subtype (1 = low, 5 = critical)
_SEVERITY: Dict[str, int] = {
    "camera_covered":   4,
    "body_movement":    2,
    "no_face":          3,
    "multiple_faces":   4,
    "mouth_movement":   2,
    "eyes_closed":      2,
    "hand_over_mouth":  3,
    "head_down":        2,
    "lighting_change":  1,
    "leaning_forward":  2,
    "face_size_change": 2,
    "repeated_gaze":    3,
    "head_pose":        2,
    "new_desk_object":  3,
    "text_density":     3,
    "rough_sheet":      4,
    "earphone":         4,
    "identity_change":  5,
    "phone_detected":   5,
    "book_detected":    3,
    "laptop_detected":  4,
    "second_person":    4,
    "remote_detected":  3,
    "voice_detected":   2,
}


class ViolationEngine:
    """
    Converts raw detection flags into structured violation events.

    Usage::

        engine = ViolationEngine(config, state)
        events = engine.process_flags(flags, frame, frame_id)
    """

    def __init__(
        self,
        config: ProctorConfig,
        state: ProctorState,
        on_event: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self.cfg = config
        self.state = state
        self._on_event = on_event

    def _resolve_subtype(self, flag: str) -> str:
        """Map a raw flag string to a canonical subtype key."""
        for keyword, subtype in _FLAG_TO_SUBTYPE.items():
            if keyword.lower() in flag.lower():
                return subtype
        return "unknown"

    def _parse_confidence(self, flag: str) -> float:
        """Extract a confidence % from the flag string, if present."""
        import re

        match = re.search(r"(\d+)%", flag)
        if match:
            return int(match.group(1)) / 100.0
        # YOLO format: (85%)
        match = re.search(r"\((\d+)%\)", flag)
        if match:
            return int(match.group(1)) / 100.0
        return 0.85  # default high confidence

    def process_flags(
        self,
        flags: List[str],
        frame: Optional[np.ndarray] = None,
        frame_id: int = 0,
    ) -> List[Dict]:
        """
        Process a list of raw flag strings into structured events.

        Each flag becomes a ``ViolationRecord`` added to state and
        a structured dict emitted to the event bus.

        Returns:
            List of structured event dicts.
        """
        if not flags:
            return []

        events: List[Dict] = []
        ts = time.time()

        # Save a flagged snapshot
        snapshot_path: Optional[str] = None
        if frame is not None:
            snapshot_path = save_snapshot(
                frame, self.cfg.flagged_folder, prefix="flag",
            )

        for flag in flags:
            subtype = self._resolve_subtype(flag)
            confidence = self._parse_confidence(flag)
            severity = _SEVERITY.get(subtype, 2)
            score_delta = self.cfg.suspicion_weights.get(subtype, 10)

            record = ViolationRecord(
                timestamp=ts,
                violation_type="vision_violation",
                subtype=subtype,
                description=flag,
                confidence=confidence,
                severity=severity,
                frame_id=frame_id,
                score_delta=score_delta,
                snapshot_path=snapshot_path,
            )
            self.state.add_violation(record)

            event = {
                "type": "vision_violation",
                "subtype": subtype,
                "description": flag,
                "confidence": round(confidence, 2),
                "severity": severity,
                "score_delta": score_delta,
                "timestamp": ts,
                "frame_id": frame_id,
                "snapshot": snapshot_path,
                "suspicion_score": round(self.state.suspicion_score, 1),
            }
            events.append(event)

            if self._on_event:
                self._on_event(event)

        # Update display state
        combined = " | ".join(flags)
        self.state.current_warning_text = combined
        self.state.warning_display_until = time.time() + 8

        log.warning(
            "WARNING #%d — %s (score: %.1f)",
            self.state.total_warnings,
            combined,
            self.state.suspicion_score,
        )

        # Check cheating alert
        if self.state.is_cheating_alert():
            alert_event = {
                "type": "cheating_alert",
                "suspicion_score": round(self.state.suspicion_score, 1),
                "threshold": self.cfg.suspicion_threshold,
                "violation_count": self.state.total_warnings,
                "timestamp": ts,
            }
            events.append(alert_event)
            if self._on_event:
                self._on_event(alert_event)
            log.error(
                "🚨 CHEATING ALERT — score %.1f exceeds threshold %d",
                self.state.suspicion_score,
                self.cfg.suspicion_threshold,
            )

        return events
