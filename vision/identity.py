"""
Identity verification module.

Compares the current face with the baseline captured at exam start
and flags potential identity swaps.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import PreprocessedFrame, get_logger

log = get_logger("identity")


class IdentityVerifier:
    """
    Detection 22 — Student identity change.

    Captures a 64×64 grayscale baseline on the first face encounter
    and continuously compares subsequent crops against it.
    """

    def __init__(self, config: ProctorConfig, state: ProctorState) -> None:
        self.cfg = config
        self.state = state

    def check_identity(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[str]:
        """Return a flag if the detected face differs from the baseline."""
        if not faces:
            return None

        s = self.state
        x, y, fw, fh = faces[0]

        # Bounds check
        y_end = min(y + fh, pf.height)
        x_end = min(x + fw, pf.width)
        if y_end <= y or x_end <= x:
            return None

        crop = cv2.resize(pf.gray_eq[y:y_end, x:x_end], (64, 64))

        if s.baseline_face_img is None:
            s.baseline_face_img = crop.copy()
            log.info("Baseline face captured.")
            return None

        diff = cv2.absdiff(s.baseline_face_img, crop)
        mean_diff = float(diff.mean())

        if mean_diff > self.cfg.identity_diff_threshold:
            return (
                f"Possible identity change — different face detected "
                f"(diff: {mean_diff:.1f})"
            )
        return None

    def run_all(
        self,
        pf: PreprocessedFrame,
        faces: List[Tuple[int, int, int, int]],
    ) -> List[str]:
        """Execute identity check and return flag strings."""
        flags: List[str] = []
        r = self.check_identity(pf, faces)
        if r:
            flags.append(r)
        return flags
