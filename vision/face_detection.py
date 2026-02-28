"""
Face detection module.

Handles: face counting, no-face detection, camera covered, eyes closed,
mouth movement, hand over mouth, head down, leaning forward,
face size change, lighting change, and body movement.

All detectors operate on a shared ``PreprocessedFrame`` and read/write
state through the ``ProctorState`` singleton — zero global variables.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import PreprocessedFrame, get_logger

log = get_logger("face")


class FaceDetector:
    """
    Detects faces via Haar cascades and runs all face-dependent checks.

    Detections covered:
        1.  Camera covered / blocked
        2.  Frequent body movement
        3.  No face detected
        4.  Multiple faces
        11. Mouth movement
        12. Eyes closed
        13. Hand over mouth
        14. Head down
        15. Sudden lighting change
        16. Leaning forward
        20. Face size sudden change
    """

    def __init__(
        self,
        config: ProctorConfig,
        state: ProctorState,
        face_cascade: Optional[cv2.CascadeClassifier],
        eye_cascade: Optional[cv2.CascadeClassifier],
    ) -> None:
        self.cfg = config
        self.state = state
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade

    # ── Helpers ───────────────────────────────────────────────────

    def _find_faces(self, gray_eq: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using the Haar cascade."""
        if self.face_cascade is None:
            return []
        return list(
            self.face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=self.cfg.face_scale_factor,
                minNeighbors=self.cfg.face_min_neighbors,
                minSize=self.cfg.face_min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        )

    # ── Detection 1: Camera Covered ──────────────────────────────

    def check_camera_covered(self, pf: PreprocessedFrame) -> Optional[str]:
        """Return a flag if the camera appears blocked (very dark)."""
        if pf.gray.mean() < self.cfg.brightness_threshold:
            return "Camera covered or blocked"
        return None

    # ── Detection 2: Body Movement ───────────────────────────────

    def check_movement(self, pf: PreprocessedFrame) -> Optional[str]:
        """Flag frequent body movement across a sliding window."""
        s = self.state
        s.snapshot_history.append(pf.blurred)
        if len(s.snapshot_history) > self.cfg.movement_window:
            s.snapshot_history.pop(0)
        if len(s.snapshot_history) < 2:
            return None

        moved = sum(
            1
            for i in range(1, len(s.snapshot_history))
            if cv2.absdiff(s.snapshot_history[i - 1], s.snapshot_history[i]).mean()
            > self.cfg.movement_threshold
        )
        if moved >= self.cfg.movement_flag_count:
            return (
                f"Frequent body movement "
                f"({moved}/{self.cfg.movement_window} snapshots)"
            )
        return None

    # ── Detection 3/4: Face Count ────────────────────────────────

    def check_faces(
        self, pf: PreprocessedFrame,
    ) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
        """
        Count faces and flag anomalies (no face / multiple faces).

        Returns:
            A tuple of (flag_list, detected_faces).
        """
        flags: List[str] = []
        faces = self._find_faces(pf.gray_eq)
        s = self.state

        if len(faces) == 0:
            s.consecutive_away += 1
            if s.consecutive_away >= self.cfg.consecutive_away_limit:
                flags.append(
                    "No face detected — student may have stepped away"
                )
            return flags, faces

        # Reset away counter when a face is found
        s.consecutive_away = 0

        if len(faces) > 1:
            flags.append(
                f"Multiple faces detected ({len(faces)} people in frame)"
            )
        return flags, faces

    # ── Detection 11: Mouth Movement ─────────────────────────────

    def check_mouth_movement(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """Detect mouth motion that may indicate talking."""
        if not faces:
            return None
        s = self.state
        x, y, fw, fh = faces[0]
        y1 = y + int(fh * 0.65)
        y2 = min(y + fh, pf.height)
        x1 = x + int(fw * 0.20)
        x2 = min(x + int(fw * 0.80), pf.width)
        if y1 >= y2 or x1 >= x2:
            return None

        mouth = cv2.resize(pf.gray_eq[y1:y2, x1:x2], (60, 30))
        s.mouth_history.append(mouth)
        if len(s.mouth_history) > self.cfg.mouth_window:
            s.mouth_history.pop(0)
        if len(s.mouth_history) < 3:
            return None

        moved = sum(
            1
            for i in range(1, len(s.mouth_history))
            if cv2.absdiff(s.mouth_history[i - 1], s.mouth_history[i]).mean()
            > self.cfg.mouth_diff_threshold
        )
        if moved >= self.cfg.mouth_flag_count:
            return "Mouth movement detected — possible talking"
        return None

    # ── Detection 12: Eyes Closed ────────────────────────────────

    def check_eyes_closed(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """
        Detect prolonged eye closure.

        Bug-fix carried from v5: only runs when a face IS detected
        so that the absence of a face does not incorrectly flag
        "eyes closed."
        """
        s = self.state

        if not faces:
            s.consecutive_eyes_closed = 0
            return None
        if self.eye_cascade is None:
            return None

        x, y, fw, fh = faces[0]
        face_roi = pf.gray_eq[y : y + int(fh * 0.55), x : x + fw]
        if face_roi.size == 0:
            return None

        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20),
        )

        if len(eyes) == 0:
            s.consecutive_eyes_closed += 1
            if s.consecutive_eyes_closed >= self.cfg.eye_closed_limit:
                return "Eyes closed repeatedly — student not looking at screen"
        else:
            s.consecutive_eyes_closed = 0
        return None

    # ── Detection 13: Hand Over Mouth ────────────────────────────

    def check_hand_over_mouth(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """Detect skin-tone occlusion over the mouth region."""
        if not faces:
            return None
        x, y, fw, fh = faces[0]
        y1 = y + int(fh * 0.65)
        y2 = min(y + fh, pf.height)
        x1 = x + int(fw * 0.15)
        x2 = min(x + int(fw * 0.85), pf.width)
        if y1 >= y2 or x1 >= x2:
            return None

        hsv = cv2.cvtColor(pf.raw[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([25, 255, 255]))
        ratio = mask.sum() / (mask.size * 255)
        if ratio > 0.70:
            return "Hand covering mouth detected"
        return None

    # ── Detection 14: Head Down ──────────────────────────────────

    def check_head_down(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """Flag when the face centroid is in the bottom portion of the frame."""
        if not faces:
            return None
        _, y, _, fh = faces[0]
        cy = y + fh // 2
        if (cy / pf.height) * 100 > self.cfg.head_down_threshold:
            return "Head down — student may be reading desk notes"
        return None

    # ── Detection 15: Lighting Change ────────────────────────────

    def check_lighting_change(self, pf: PreprocessedFrame) -> Optional[str]:
        """Detect sudden brightness jumps between consecutive frames."""
        s = self.state
        brightness = float(pf.gray.mean())
        flag = None
        if s.last_brightness is not None:
            delta = abs(brightness - s.last_brightness)
            if delta > self.cfg.brightness_change_limit:
                flag = f"Sudden lighting change (delta: {delta:.0f})"
        s.last_brightness = brightness
        return flag

    # ── Detection 16: Leaning Forward ────────────────────────────

    def check_leaning_forward(
        self, faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """Flag if the face area suddenly grows (student lunges at screen)."""
        s = self.state
        if not faces:
            return None
        _, _, fw, fh = faces[0]
        size = fw * fh
        if s.baseline_face_size is None:
            s.baseline_face_size = float(size)
            return None
        if size > s.baseline_face_size * self.cfg.face_size_change_ratio:
            return "Student leaning too far forward — possible hidden device"
        return None

    # ── Detection 20: Face Size Sudden Change ────────────────────

    def check_face_size_change(
        self, faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """Flag if the face shrinks dramatically (student moved far away)."""
        s = self.state
        if not faces or s.baseline_face_size is None:
            return None
        _, _, fw, fh = faces[0]
        ratio = (fw * fh) / s.baseline_face_size
        if ratio < self.cfg.face_shrink_ratio:
            return "Student moved suddenly far from camera"
        return None

    # ── Run All ──────────────────────────────────────────────────

    def run_all(
        self, pf: PreprocessedFrame,
    ) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
        """
        Execute every face-related detection on a single preprocessed frame.

        Returns:
            ``(all_flags, detected_faces)``
        """
        flags: List[str] = []

        r = self.check_camera_covered(pf)
        if r:
            flags.append(r)

        r = self.check_movement(pf)
        if r:
            flags.append(r)

        face_flags, faces = self.check_faces(pf)
        flags.extend(face_flags)

        r = self.check_mouth_movement(pf, faces)
        if r:
            flags.append(r)

        r = self.check_eyes_closed(pf, faces)
        if r:
            flags.append(r)

        r = self.check_hand_over_mouth(pf, faces)
        if r:
            flags.append(r)

        r = self.check_head_down(pf, faces)
        if r:
            flags.append(r)

        r = self.check_lighting_change(pf)
        if r:
            flags.append(r)

        r = self.check_leaning_forward(faces)
        if r:
            flags.append(r)

        r = self.check_face_size_change(faces)
        if r:
            flags.append(r)

        return flags, faces
