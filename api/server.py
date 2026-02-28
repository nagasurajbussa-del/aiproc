"""
FastAPI server for the AI Proctor system.

Provides REST endpoints and a WebSocket for real-time event
streaming.  Designed to run alongside the main proctoring loop
on a background thread.

Endpoints:
    POST  /start_exam   — begin a new exam session
    POST  /stop_exam    — end the current session
    GET   /events       — list all recorded events
    GET   /report       — generate and return the exam report
    GET   /health       — server health check
    GET   /stats        — live summary statistics
    WS    /ws/events    — real-time event stream
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ai_proctor import __version__
from ai_proctor.api.models import (
    EventListResponse,
    ExamReport,
    ExamStartRequest,
    ExamStartResponse,
    ExamStopResponse,
    HealthStatus,
    SummaryStats,
)
from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import get_logger
from ai_proctor.engine.report_generator import ReportGenerator

log = get_logger("api")

# ─── WebSocket Manager ───────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections for event broadcasting."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        log.info("WebSocket client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        log.info("WebSocket client disconnected (%d total)", len(self._connections))

    async def broadcast(self, data: Dict[str, Any]) -> None:
        """Send a JSON message to all connected clients."""
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ─── Shared instances (set by create_app) ─────────────────────────

_config: Optional[ProctorConfig] = None
_state: Optional[ProctorState] = None
_report_gen: Optional[ReportGenerator] = None
_ws_manager = ConnectionManager()
_event_log: List[Dict[str, Any]] = []
_start_time = time.time()
_loop: Optional[asyncio.AbstractEventLoop] = None

# Callback to receive events from the proctoring engine
def on_event(event: Dict[str, Any]) -> None:
    """Thread-safe event receiver — queues events for WebSocket broadcast."""
    _event_log.append(event)
    # Schedule broadcast on the async event loop
    if _loop is not None and _loop.is_running():
        asyncio.run_coroutine_threadsafe(_ws_manager.broadcast(event), _loop)


# ─── App Factory ──────────────────────────────────────────────────

def create_app(
    config: ProctorConfig,
    state: ProctorState,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _config, _state, _report_gen

    _config = config
    _state = state
    _report_gen = ReportGenerator(config, state)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _loop
        _loop = asyncio.get_running_loop()
        log.info("API server started on %s:%d", config.api_host, config.api_port)
        yield
        log.info("API server shutting down.")

    app = FastAPI(
        title="AI Proctor API",
        version=__version__,
        description="Real-time AI-powered exam proctoring system",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Endpoints ────────────────────────────────────────────────

    @app.post("/start_exam", response_model=ExamStartResponse)
    async def start_exam(req: ExamStartRequest) -> ExamStartResponse:
        """Start a new exam proctoring session."""
        assert _state is not None
        _state.reset()
        _state.exam_active = True
        _state.exam_start_time = time.time()
        _event_log.clear()
        log.info("Exam started via API (duration=%dm)", req.duration_minutes)
        return ExamStartResponse(
            exam_duration_minutes=req.duration_minutes,
            demo_mode=req.demo_mode,
            message=f"Exam session started for {req.duration_minutes} minutes.",
        )

    @app.post("/stop_exam", response_model=ExamStopResponse)
    async def stop_exam() -> ExamStopResponse:
        """Stop the current exam session and generate a report."""
        assert _state is not None and _report_gen is not None
        _state.exam_active = False
        txt_path = _report_gen.generate_text_report()
        json_path = _report_gen.generate_json_report()
        log.info("Exam stopped via API.")
        return ExamStopResponse(
            total_warnings=_state.total_warnings,
            suspicion_score=round(_state.suspicion_score, 1),
            report_path=txt_path,
            json_report_path=json_path,
        )

    @app.get("/events", response_model=EventListResponse)
    async def get_events(limit: int = 100, offset: int = 0) -> EventListResponse:
        """Return recorded events with pagination."""
        total = len(_event_log)
        events = _event_log[offset : offset + limit]
        return EventListResponse(total=total, events=events)

    @app.get("/report")
    async def get_report() -> Dict[str, Any]:
        """Generate and return the current exam report as JSON."""
        assert _report_gen is not None
        return _report_gen.summary_stats()

    @app.get("/health", response_model=HealthStatus)
    async def health() -> HealthStatus:
        """Health check endpoint."""
        assert _state is not None
        return HealthStatus(
            version=__version__,
            exam_active=_state.exam_active,
            uptime_seconds=round(time.time() - _start_time, 1),
        )

    @app.get("/stats", response_model=SummaryStats)
    async def stats() -> SummaryStats:
        """Return live summary statistics."""
        assert _report_gen is not None
        data = _report_gen.summary_stats()
        return SummaryStats(**data)

    @app.websocket("/ws/events")
    async def websocket_events(ws: WebSocket) -> None:
        """Real-time WebSocket stream of all proctoring events."""
        await _ws_manager.connect(ws)
        try:
            while True:
                # Keep connection alive; client can also send control msgs
                await ws.receive_text()
        except WebSocketDisconnect:
            _ws_manager.disconnect(ws)

    return app


# ─── Background Server Runner ────────────────────────────────────

def run_server_background(
    config: ProctorConfig,
    state: ProctorState,
) -> threading.Thread:
    """
    Start the FastAPI server on a background daemon thread.

    Returns the thread handle (for join on shutdown).
    """
    import uvicorn

    app = create_app(config, state)

    def _run() -> None:
        uvicorn.run(
            app,
            host=config.api_host,
            port=config.api_port,
            log_level="warning",
        )

    thread = threading.Thread(target=_run, daemon=True, name="api-server")
    thread.start()
    log.info("API server thread started.")
    return thread
