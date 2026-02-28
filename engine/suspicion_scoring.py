"""
Suspicion scoring engine.

Implements weighted scoring per violation type, confidence decay
over time, temporal violation memory, multi-condition fusion,
and Bayesian-inspired confidence aggregation.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState, ViolationRecord
from ai_proctor.core.utils import get_logger

log = get_logger("engine.scoring")


class SuspicionScorer:
    """
    Advanced suspicion scoring with temporal awareness.

    Features:
        - Weighted scores per violation type (from config)
        - Exponential decay when no violations occur
        - Temporal memory: tracks violation frequency per window
        - Multi-condition fusion: bonus when multiple types co-occur
        - Bayesian-inspired cheat probability (0–100 %)
    """

    def __init__(self, config: ProctorConfig, state: ProctorState) -> None:
        self.cfg = config
        self.state = state

        # Temporal memory — sliding window of violation subtypes
        self._window_seconds = 60.0
        self._recent_violations: deque[Tuple[float, str]] = deque()

        # For multi-condition fusion bonus
        self._fusion_bonus = 15  # extra points when ≥3 types in window

    # ── Score Management ─────────────────────────────────────────

    def apply_decay(self) -> None:
        """
        Apply exponential decay to the suspicion score.

        Called periodically from the main loop.
        """
        self.state.apply_score_decay()

    def add_violation_score(self, record: ViolationRecord) -> float:
        """
        Add a weighted score for a violation, apply fusion bonuses,
        and return the new total score.
        """
        now = time.time()
        subtype = record.subtype

        # Track in temporal memory
        self._recent_violations.append((now, subtype))
        self._prune_old(now)

        # Multi-condition fusion: if ≥3 distinct violation types
        # in the current window, apply an extra bonus.
        distinct_types = set(v[1] for v in self._recent_violations)
        if len(distinct_types) >= 3:
            self.state.suspicion_score += self._fusion_bonus
            log.info(
                "Multi-condition fusion bonus +%d (types: %s)",
                self._fusion_bonus,
                ", ".join(sorted(distinct_types)),
            )

        return self.state.suspicion_score

    def _prune_old(self, now: float) -> None:
        """Remove violations older than the temporal window."""
        cutoff = now - self._window_seconds
        while (
            self._recent_violations
            and self._recent_violations[0][0] < cutoff
        ):
            self._recent_violations.popleft()

    # ── Bayesian Cheat Probability ───────────────────────────────

    def cheat_probability(self) -> float:
        """
        Return a 0-100 % cheat probability using a sigmoid function
        over the current suspicion score.

        The midpoint is at the configured threshold so that exactly
        at threshold the probability is ~50 %.
        """
        score = self.state.suspicion_score
        k = 0.05  # steepness — tuned for demo readability
        midpoint = self.cfg.suspicion_threshold
        prob = 1.0 / (1.0 + math.exp(-k * (score - midpoint)))
        return round(prob * 100, 1)

    # ── Frequency Analysis ───────────────────────────────────────

    def violation_frequency(self) -> Dict[str, int]:
        """
        Return per-subtype violation counts within the temporal window.
        """
        self._prune_old(time.time())
        freq: Dict[str, int] = defaultdict(int)
        for _, subtype in self._recent_violations:
            freq[subtype] += 1
        return dict(freq)

    # ── Summary Stats ────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return a JSON-serialisable summary of the scoring state."""
        return {
            "suspicion_score": round(self.state.suspicion_score, 1),
            "peak_score": round(self.state.peak_suspicion_score, 1),
            "cheat_probability_pct": self.cheat_probability(),
            "threshold": self.cfg.suspicion_threshold,
            "is_alert": self.state.is_cheating_alert(),
            "total_warnings": self.state.total_warnings,
            "violation_counts": dict(self.state.violation_counts),
            "recent_frequency": self.violation_frequency(),
        }
