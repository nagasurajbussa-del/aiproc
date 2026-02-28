"""
Desk monitoring module.

Handles: new object on desk, text/paper density detection.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import PreprocessedFrame, get_logger

log = get_logger("desk")


class DeskMonitor:
    """
    Monitor the desk region for suspicious changes.

    Detections covered:
        17. New object on desk
        18. High text / paper density
    """

    def __init__(self, config: ProctorConfig, state: ProctorState) -> None:
        self.cfg = config
        self.state = state

    # ── Detection 17: New Object on Desk ─────────────────────────

    def check_new_object(self, pf: PreprocessedFrame) -> Optional[str]:
        """
        Compare the lower 30 % of the frame against a slowly-updating
        baseline to detect genuinely new large objects.

        Bug-fix from v5: threshold raised to 55, baseline updates
        gradually to prevent slow lighting drift from accumulating.
        """
        s = self.state
        s.desk_snapshot_counter += 1

        desk = cv2.GaussianBlur(
            cv2.cvtColor(
                pf.raw[int(pf.height * 0.70) :, :], cv2.COLOR_BGR2GRAY,
            ),
            (21, 21),
            0,
        )

        if (
            s.baseline_desk_region is None
            or desk.shape != s.baseline_desk_region.shape
        ):
            s.baseline_desk_region = desk.copy()
            return None

        diff = cv2.absdiff(s.baseline_desk_region, desk)
        mean_diff = float(diff.mean())
        flag = None

        if mean_diff > self.cfg.desk_diff_threshold:
            flag = f"New object detected on desk (diff: {mean_diff:.1f})"

        # Gradual baseline update (only when scene is clean)
        if (
            s.desk_snapshot_counter % self.cfg.desk_baseline_update == 0
            and flag is None
        ):
            s.baseline_desk_region = cv2.addWeighted(
                s.baseline_desk_region, 0.8, desk, 0.2, 0,
            )

        return flag

    # ── Detection 18: Text / Paper Density ───────────────────────

    def check_text_density(self, pf: PreprocessedFrame) -> Optional[str]:
        """Flag high edge density that may indicate a cheat sheet."""
        edges = cv2.Canny(pf.gray, 50, 150)
        density = edges.sum() / (edges.size * 255)
        if density > self.cfg.edge_density_threshold:
            return (
                f"High text/paper density ({density:.2f}) "
                "— possible cheat sheet"
            )
        return None

    # ── Run All ──────────────────────────────────────────────────

    def run_all(self, pf: PreprocessedFrame) -> List[str]:
        """Execute all desk-related detections."""
        flags: List[str] = []

        r = self.check_new_object(pf)
        if r:
            flags.append(r)

        r = self.check_text_density(pf)
        if r:
            flags.append(r)

        return flags
