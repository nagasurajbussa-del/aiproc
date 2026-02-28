"""
Rough sheet & earphone detection module.

Uses colour segmentation, contour analysis, and Hough line detection
to identify paper / cheat sheets and earphone wires.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import PreprocessedFrame, get_logger

log = get_logger("rough_sheet")


class RoughSheetDetector:
    """
    Detects physical cheat aids (paper, notebooks, earphones).

    Detections covered:
        21. Earphone / wire near ear
        23. Rough sheet / cheat sheet
    """

    def __init__(self, config: ProctorConfig, state: ProctorState) -> None:
        self.cfg = config
        self.state = state

    # ── Detection 23: Rough Sheet / Cheat Sheet ──────────────────

    def check_rough_sheet(self, pf: PreprocessedFrame) -> Optional[str]:
        """
        Multi-colour paper detection with Hough line confirmation.

        Bug-fix from v5: lowered MIN_AREA (8000→5000),
        LINE_THRESHOLD (4→3), and wider angle tolerance.
        """
        gray = pf.gray
        hsv = cv2.cvtColor(pf.raw, cv2.COLOR_BGR2HSV)

        # Colour masks for paper types
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 160], dtype=np.uint8),
            np.array([180, 55, 255], dtype=np.uint8),
        )
        cream_mask = cv2.inRange(
            hsv,
            np.array([15, 0, 130], dtype=np.uint8),
            np.array([40, 90, 255], dtype=np.uint8),
        )
        yellow_mask = cv2.inRange(
            hsv,
            np.array([20, 80, 180], dtype=np.uint8),
            np.array([35, 255, 255], dtype=np.uint8),
        )

        paper_mask = cv2.bitwise_or(white_mask, cream_mask)
        paper_mask = cv2.bitwise_or(paper_mask, yellow_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
        cleaned = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        min_area = self.cfg.rough_sheet_min_area
        white_ratio_thresh = self.cfg.rough_sheet_white_ratio
        line_thresh = self.cfg.rough_sheet_line_threshold

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if min(rw, rh) == 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect < 1.1 or aspect > 6.0:
                continue

            roi_mask = paper_mask[ry : ry + rh, rx : rx + rw]
            if roi_mask.size == 0:
                continue
            white_ratio = roi_mask.sum() / (roi_mask.size * 255)
            if white_ratio < white_ratio_thresh:
                continue

            roi_gray = gray[ry : ry + rh, rx : rx + rw]
            roi_edges = cv2.Canny(roi_gray, 25, 90)

            lines = cv2.HoughLinesP(
                roi_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=20,
                minLineLength=max(20, int(rw * 0.20)),
                maxLineGap=15,
            )

            h_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 - x1 == 0:
                        continue
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if angle < 20 or angle > 160:
                        h_lines += 1

            if h_lines >= line_thresh:
                conf = min(100, h_lines * 12)
                return (
                    f"Rough sheet/cheat sheet detected "
                    f"({h_lines} text lines, {conf}% confidence)"
                )

        return None

    # ── Detection 21: Earphone Near Ear ──────────────────────────

    def check_earphone(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """
        Detect earphone wires near the ear region.

        Bug-fix from v5: line count threshold raised from 2 → 6 to
        eliminate false positives from hair and shadows.
        """
        if not faces:
            return None

        x, y, fw, fh = faces[0]

        # Left ear region
        ex1 = max(0, x - int(fw * 0.45))
        ex2 = x + int(fw * 0.05)
        ey1 = y + int(fh * 0.25)
        ey2 = min(y + int(fh * 0.65), pf.height)

        if ex1 >= ex2 or ey1 >= ey2:
            return None

        ear = cv2.cvtColor(pf.raw[ey1:ey2, ex1:ex2], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(ear, 40, 120)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=20,
            minLineLength=20,
            maxLineGap=4,
        )

        if lines is not None and len(lines) > self.cfg.earphone_line_threshold:
            return "Possible earphone/wire near ear"
        return None

    # ── Run All ──────────────────────────────────────────────────

    def run_all(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> List[str]:
        """Execute all rough-sheet and earphone detections."""
        flags: List[str] = []

        r = self.check_rough_sheet(pf)
        if r:
            flags.append(r)

        r = self.check_earphone(pf, faces)
        if r:
            flags.append(r)

        return flags
