"""
Pydantic models for the FastAPI backend.

These models define the request / response schemas for all API
endpoints and WebSocket messages.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────

class ViolationType(str, Enum):
    VISION = "vision_violation"
    AUDIO = "audio_violation"
    AUDIO_ANALYSIS = "audio_analysis"
    CHEATING_ALERT = "cheating_alert"


class AudioClassification(str, Enum):
    SILENCE = "SILENCE"
    WHISPER = "WHISPER"
    NORMAL_SPEECH = "NORMAL_SPEECH"
    MUFFLED_SPEECH = "MUFFLED_SPEECH"
    SHOUT = "SHOUT"
    LOUD_SPIKE = "LOUD_SPIKE"


# ─── Event Models ─────────────────────────────────────────────────

class VisionViolationEvent(BaseModel):
    """Structured event emitted by the vision pipeline."""

    type: str = "vision_violation"
    subtype: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: int = Field(ge=1, le=5)
    score_delta: int
    timestamp: float
    frame_id: int
    snapshot: Optional[str] = None
    suspicion_score: float


class AudioViolationEvent(BaseModel):
    """Structured event emitted by the audio pipeline."""

    type: str = "audio_violation"
    classification: AudioClassification
    rms: float
    severity: int = Field(ge=0, le=5)
    stage: Optional[str] = None
    sustained_seconds: Optional[float] = None
    timestamp: float


class CheatingAlertEvent(BaseModel):
    """Emitted when the suspicion score exceeds the threshold."""

    type: str = "cheating_alert"
    suspicion_score: float
    threshold: int
    violation_count: int
    timestamp: float


# ─── API Request / Response ───────────────────────────────────────

class ExamStartRequest(BaseModel):
    """Request body for ``POST /start_exam``."""

    duration_minutes: int = Field(default=60, ge=1, le=300)
    demo_mode: bool = False


class ExamStartResponse(BaseModel):
    """Response for ``POST /start_exam``."""

    status: str = "started"
    exam_duration_minutes: int
    demo_mode: bool
    message: str


class ExamStopResponse(BaseModel):
    """Response for ``POST /stop_exam``."""

    status: str = "stopped"
    total_warnings: int
    suspicion_score: float
    report_path: Optional[str] = None
    json_report_path: Optional[str] = None


class ExamReport(BaseModel):
    """Full exam report."""

    generated_at: str
    total_snapshots: int
    total_warnings: int
    suspicion_score: float
    peak_suspicion_score: float
    cheating_alert: bool
    violation_counts: Dict[str, int]
    violations: List[Dict[str, Any]]


class HealthStatus(BaseModel):
    """Response for ``GET /health``."""

    status: str = "healthy"
    version: str
    exam_active: bool
    uptime_seconds: float


class SummaryStats(BaseModel):
    """Live summary statistics."""

    exam_active: bool
    elapsed_seconds: float
    total_snapshots: int
    total_warnings: int
    suspicion_score: float
    peak_suspicion_score: float
    cheating_alert: bool
    violation_counts: Dict[str, int]
    audio_classification: str
    audio_calibrated: bool


class EventListResponse(BaseModel):
    """Response for ``GET /events``."""

    total: int
    events: List[Dict[str, Any]]
