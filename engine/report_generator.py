"""
Report generation module.

Produces text reports, JSON exports, and summary statistics
for completed exam sessions.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import get_logger

log = get_logger("engine.report")


class ReportGenerator:
    """
    Generates exam integrity reports in multiple formats.

    Supported outputs:
        - Plain-text report (human-readable)
        - JSON export (machine-readable)
        - Summary statistics dict (for API / dashboard)
    """

    def __init__(self, config: ProctorConfig, state: ProctorState) -> None:
        self.cfg = config
        self.state = state

    # ── Text Report ──────────────────────────────────────────────

    def generate_text_report(self) -> str:
        """
        Write a human-readable exam integrity report and return
        the file path.
        """
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self.cfg.report_folder, f"report_{ts}.txt")

        s = self.state
        lines = [
            "=" * 64,
            "         AI PROCTOR — EXAM INTEGRITY REPORT",
            "=" * 64,
            f"  Generated         : {ts}",
            f"  Total Snapshots   : {s.total_snapshots}",
            f"  Total Warnings    : {s.total_warnings}",
            f"  Flagged Events    : {len(s.violations)}",
            f"  Suspicion Score   : {s.suspicion_score:.1f}",
            f"  Peak Score        : {s.peak_suspicion_score:.1f}",
            f"  Cheating Alert    : {'YES' if s.is_cheating_alert() else 'NO'}",
            "=" * 64,
            "",
        ]

        if not s.violations:
            lines.append("  No suspicious activity detected.")
        else:
            lines.append("  FLAGGED EVENTS:")
            lines.append("")
            for i, v in enumerate(s.violations, 1):
                dt = datetime.fromtimestamp(v.timestamp).strftime(
                    "%Y-%m-%d %H:%M:%S",
                )
                lines.extend([
                    f"  [{i:03d}]  Time       : {dt}",
                    f"         Type       : {v.violation_type}",
                    f"         Subtype    : {v.subtype}",
                    f"         Reason     : {v.description}",
                    f"         Confidence : {v.confidence:.0%}",
                    f"         Severity   : {v.severity}/5",
                    f"         Score Δ    : +{v.score_delta}",
                    f"         Snapshot   : {v.snapshot_path or 'N/A'}",
                    "",
                ])

        lines.extend([
            "=" * 64,
            "  VIOLATION BREAKDOWN:",
            "",
        ])
        for subtype, count in sorted(s.violation_counts.items()):
            lines.append(f"    {subtype:25s} : {count}")

        lines.extend([
            "",
            "=" * 64,
            "  END OF REPORT",
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        log.info("Text report saved: %s", path)
        return path

    # ── JSON Export ───────────────────────────────────────────────

    def generate_json_report(self) -> str:
        """
        Export the full session data as a JSON file and return
        the file path.
        """
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self.cfg.report_folder, f"report_{ts}.json")

        s = self.state
        data: Dict[str, Any] = {
            "generated_at": ts,
            "total_snapshots": s.total_snapshots,
            "total_warnings": s.total_warnings,
            "suspicion_score": round(s.suspicion_score, 1),
            "peak_suspicion_score": round(s.peak_suspicion_score, 1),
            "cheating_alert": s.is_cheating_alert(),
            "violation_counts": s.violation_counts,
            "violations": [
                {
                    "timestamp": v.timestamp,
                    "type": v.violation_type,
                    "subtype": v.subtype,
                    "description": v.description,
                    "confidence": round(v.confidence, 2),
                    "severity": v.severity,
                    "frame_id": v.frame_id,
                    "score_delta": v.score_delta,
                    "snapshot": v.snapshot_path,
                }
                for v in s.violations
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        log.info("JSON report saved: %s", path)
        return path

    # ── Summary Statistics ───────────────────────────────────────

    def summary_stats(self) -> Dict[str, Any]:
        """
        Return a dict of summary statistics suitable for API
        responses or dashboard rendering.
        """
        s = self.state
        elapsed = time.time() - s.exam_start_time if s.exam_start_time else 0

        return {
            "exam_active": s.exam_active,
            "elapsed_seconds": round(elapsed, 1),
            "total_snapshots": s.total_snapshots,
            "total_warnings": s.total_warnings,
            "suspicion_score": round(s.suspicion_score, 1),
            "peak_suspicion_score": round(s.peak_suspicion_score, 1),
            "cheating_alert": s.is_cheating_alert(),
            "violation_counts": dict(s.violation_counts),
            "audio_classification": s.audio_classification,
            "audio_calibrated": s.audio_calibrated,
        }
