"""
Gaze tracking module.

Handles: gaze offset detection, repeated same-direction gaze,
and head pose estimation heuristics.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import PreprocessedFrame, get_logger

log = get_logger("gaze")


class GazeTracker:
    """
    Tracks gaze direction and flags repeated off-center looking.

    Detections covered:
        5.  Looking away (gaze offset)
        19. Repeated same gaze direction
        *.  Head pose estimation (centroid-based heuristic)
    """

    def __init__(self, config: ProctorConfig, state: ProctorState) -> None:
        self.cfg = config
        self.state = state

    # ── Detection 19: Repeated Gaze Direction ────────────────────

    def check_repeated_gaze(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """Flag if the student repeatedly looks in the same direction."""
        if not faces:
            return None

        s = self.state
        x, y, fw, fh = faces[0]
        cx = x + fw // 2
        cy = y + fh // 2

        w_quarter = int(pf.width * 0.25)
        h_quarter = int(pf.height * 0.25)
        center_x = pf.width // 2
        center_y = pf.height // 2

        if cx < center_x - w_quarter:
            direction = "left"
        elif cx > center_x + w_quarter:
            direction = "right"
        elif cy < center_y - h_quarter:
            direction = "up"
        elif cy > center_y + h_quarter:
            direction = "down"
        else:
            # Looking at center — reset all counts
            s.gaze_direction_counts = {
                "left": 0, "right": 0, "up": 0, "down": 0,
            }
            return None

        s.gaze_direction_counts[direction] += 1
        if s.gaze_direction_counts[direction] >= self.cfg.gaze_direction_limit:
            s.gaze_direction_counts[direction] = 0
            return (
                f"Repeatedly looking {direction} "
                f"({self.cfg.gaze_direction_limit}+ times)"
            )
        return None

    # ── Head Pose Estimation (heuristic) ─────────────────────────

    def estimate_head_pose(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """
        Estimate approximate head yaw/pitch from face centroid position.

        This is a lightweight heuristic — not a full 3D pose solver,
        but effective for hackathon demonstration.
        """
        if not faces:
            return None

        x, y, fw, fh = faces[0]
        cx = x + fw // 2
        cy = y + fh // 2

        # Normalized offset from frame centre (0..100)
        off_x = abs(cx - pf.width // 2) / (pf.width // 2) * 100
        off_y = abs(cy - pf.height // 2) / (pf.height // 2) * 100

        if off_x > self.cfg.gaze_offset_threshold:
            side = "left" if cx < pf.width // 2 else "right"
            return f"Head turned {side} (offset {off_x:.0f}%)"
        if off_y > self.cfg.gaze_offset_threshold:
            vert = "up" if cy < pf.height // 2 else "down"
            return f"Head tilted {vert} (offset {off_y:.0f}%)"
        return None

    # ── Eye Tracking Ratio ───────────────────────────────────────

    def eye_tracking_ratio(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[float]:
        """
        Return a 0-1 ratio indicating how centred the student's gaze is.

        1.0 = looking straight at the camera, 0.0 = looking far away.
        Useful for the demo-mode confidence meter.
        """
        if not faces:
            return None
        x, y, fw, fh = faces[0]
        cx = x + fw // 2
        cy = y + fh // 2

        max_dist = np.sqrt((pf.width // 2) ** 2 + (pf.height // 2) ** 2)
        dist = np.sqrt(
            (cx - pf.width // 2) ** 2 + (cy - pf.height // 2) ** 2
        )
        return max(0.0, 1.0 - dist / max_dist)

    # ── Run All ──────────────────────────────────────────────────

    def run_all(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> List[str]:
        """Execute all gaze detections and return flag strings."""
        flags: List[str] = []

        r = self.check_repeated_gaze(pf, faces)
        if r:
            flags.append(r)

        r = self.estimate_head_pose(pf, faces)
        if r:
            flags.append(r)

        return flags
