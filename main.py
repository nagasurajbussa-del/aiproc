"""
AI Proctor — Main Entry Point.

Unified launcher supporting three execution modes:

    1. **Standalone** (default) — runs vision + audio proctoring
       with an OpenCV window.  Add ``--demo`` for the demo overlay.

    2. **API mode** (``--api``) — starts the FastAPI server alongside
       the proctoring engine.

    3. **API-only** (``--api-only``) — starts only the FastAPI server
       without the vision/audio loop (useful for testing).

Usage::

    python -m ai_proctor.main                 # standalone
    python -m ai_proctor.main --demo          # demo mode with overlays
    python -m ai_proctor.main --api           # vision + audio + API
    python -m ai_proctor.main --api-only      # API server only
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import (
    FPSLimiter,
    get_logger,
    load_cascade,
    preprocess_frame,
    save_snapshot,
    setup_folders,
)
from ai_proctor.engine.report_generator import ReportGenerator
from ai_proctor.engine.suspicion_scoring import SuspicionScorer
from ai_proctor.engine.violation_engine import ViolationEngine
from ai_proctor.vision.desk_monitor import DeskMonitor
from ai_proctor.vision.face_detection import FaceDetector
from ai_proctor.vision.gaze_tracking import GazeTracker
from ai_proctor.vision.identity import IdentityVerifier
from ai_proctor.vision.rough_sheet_detector import RoughSheetDetector
from ai_proctor.vision.yolo_detector import YOLODetector

log = get_logger("main")


# ─── Demo-Mode Overlay ───────────────────────────────────────────

def draw_demo_overlay(
    frame: np.ndarray,
    state: ProctorState,
    scorer: SuspicionScorer,
    gaze_tracker: GazeTracker,
    faces: list,
    pf: "PreprocessedFrame",
) -> np.ndarray:
    """
    Render the hackathon demo overlay on top of the video feed.

    Includes:
        - Severity bar (colour-coded)
        - Suspicion score meter
        - Cheat probability %
        - Gaze confidence %
        - Live FPS (added by caller)
    """
    from ai_proctor.core.utils import PreprocessedFrame

    out = frame.copy()
    h, w = out.shape[:2]

    # ── Suspicion Score Bar ──────────────────────────────────────
    score = state.suspicion_score
    max_score = state.config.suspicion_threshold * 2
    ratio = min(score / max_score, 1.0)

    # Background
    bar_h = 28
    cv2.rectangle(out, (10, h - bar_h - 10), (w - 10, h - 10), (30, 30, 30), -1)

    # Colour: green → yellow → red
    if ratio < 0.5:
        color = (0, int(255 * (1 - ratio * 2)), int(255 * ratio * 2))
        color = (0, 200, 0)  # green
    elif ratio < 0.75:
        color = (0, 200, 255)  # yellow
    else:
        color = (0, 0, 255)  # red

    bar_w = int((w - 20) * ratio)
    cv2.rectangle(out, (10, h - bar_h - 10), (10 + bar_w, h - 10), color, -1)

    # Score text
    cv2.putText(
        out,
        f"SUSPICION: {score:.0f}/{state.config.suspicion_threshold}",
        (15, h - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # ── Cheat Probability ────────────────────────────────────────
    prob = scorer.cheat_probability()
    prob_color = (0, 200, 0) if prob < 30 else (0, 200, 255) if prob < 60 else (0, 0, 255)
    cv2.putText(
        out,
        f"CHEAT PROB: {prob:.0f}%",
        (w - 220, h - bar_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        prob_color,
        2,
    )

    # ── Gaze Confidence ──────────────────────────────────────────
    gaze_ratio = gaze_tracker.eye_tracking_ratio(pf, faces)
    if gaze_ratio is not None:
        gaze_pct = gaze_ratio * 100
        cv2.putText(
            out,
            f"GAZE: {gaze_pct:.0f}%",
            (w - 145, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 50),
            1,
        )

    # ── CHEATING ALERT flash ─────────────────────────────────────
    if state.is_cheating_alert():
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

        # Pulsing text
        pulse = int(abs(time.time() % 1.0 - 0.5) * 2 * 255)
        cv2.putText(
            out,
            "!! CHEATING ALERT !!",
            (w // 2 - 180, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, pulse),
            3,
        )

    return out


def draw_ui(
    frame: np.ndarray,
    state: ProctorState,
    start_time: float,
    duration_secs: float,
) -> np.ndarray:
    """Draw the standard proctoring UI overlay (non-demo mode)."""
    out = frame.copy()
    h, w = out.shape[:2]
    now = time.time()

    # Top bar
    cv2.rectangle(out, (0, 0), (w, 50), (20, 20, 35), -1)
    cv2.putText(
        out,
        "ONLINE EXAM — AI PROCTORED",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 210),
        2,
    )

    # Timer
    rem = max(0, duration_secs - (now - start_time))
    mins, secs = int(rem // 60), int(rem % 60)
    tcol = (0, 220, 100) if mins > 10 else (0, 120, 255)
    cv2.putText(
        out,
        f"{mins:02d}:{secs:02d}",
        (w - 90, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        tcol,
        2,
    )

    # Warning reason box
    if (
        state.total_warnings > 0
        and now < state.warning_display_until
    ):
        ov = out.copy()
        cv2.rectangle(ov, (0, 50), (w, 125), (0, 0, 140), -1)
        cv2.addWeighted(ov, 0.85, out, 0.15, 0, out)

        cv2.putText(
            out,
            f"! VIOLATION DETECTED  —  WARNING #{state.total_warnings}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 80, 80),
            2,
        )

        reason = state.current_warning_text
        if len(reason) > 65:
            mid = reason.rfind(" | ", 0, 65)
            mid = mid if mid > 0 else 65
            line1, line2 = reason[:mid], reason[mid:].lstrip(" | ")
        else:
            line1, line2 = reason, ""

        cv2.putText(
            out,
            f"Reason: {line1}",
            (10, 93),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 220, 100),
            1,
        )
        if line2:
            cv2.putText(
                out,
                f"        {line2}",
                (10, 113),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 220, 100),
                1,
            )

    # Bottom bar
    if state.total_warnings > 0:
        cv2.rectangle(out, (0, h - 42), (w, h), (0, 0, 170), -1)
        cv2.putText(
            out,
            "This exam is monitored. All violations are recorded and reported.",
            (8, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            out,
            f"Warnings: {state.total_warnings}  Score: {state.suspicion_score:.0f}",
            (w - 220, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 100),
            1,
        )

    return out


# ─── Main Proctoring Loop ────────────────────────────────────────

def run_proctor(config: ProctorConfig) -> None:
    """
    Main proctoring loop.

    Initialises all modules, opens the camera, and runs the
    detection + scoring pipeline until the exam ends or the user
    presses Q.
    """
    # ── State ────────────────────────────────────────────────────
    ProctorState.destroy()  # ensure clean singleton
    state = ProctorState(config)
    state.exam_active = True
    state.exam_start_time = time.time()

    # ── Folders ──────────────────────────────────────────────────
    setup_folders(config)

    # ── Load models ──────────────────────────────────────────────
    log.info("Loading models …")
    face_cascade = load_cascade("haarcascade_frontalface_default.xml")
    eye_cascade = load_cascade("haarcascade_eye.xml")
    log.info(
        "Face cascade: %s  |  Eye cascade: %s",
        "ready ✓" if face_cascade else "FAILED",
        "ready ✓" if eye_cascade else "FAILED",
    )

    # ── Vision detectors ─────────────────────────────────────────
    face_detector = FaceDetector(config, state, face_cascade, eye_cascade)
    gaze_tracker = GazeTracker(config, state)
    identity_verifier = IdentityVerifier(config, state)
    desk_monitor = DeskMonitor(config, state)
    rough_sheet = RoughSheetDetector(config, state)
    yolo_detector = YOLODetector(config, state)
    yolo_detector.load_model()
    yolo_detector.start()

    # ── Event callback (for API WebSocket) ───────────────────────
    event_callback = None
    api_thread = None

    if config.demo_mode or hasattr(config, '_api_mode'):
        try:
            from ai_proctor.api.server import on_event, run_server_background

            event_callback = on_event
            api_thread = run_server_background(config, state)
        except Exception as exc:
            log.warning("Could not start API server: %s", exc)

    # ── Engine ───────────────────────────────────────────────────
    violation_engine = ViolationEngine(config, state, on_event=event_callback)
    scorer = SuspicionScorer(config, state)
    report_gen = ReportGenerator(config, state)

    # ── Audio engine ─────────────────────────────────────────────
    audio_engine = None
    try:
        from ai_proctor.audio.audio_engine import AudioEngine

        audio_engine = AudioEngine(config, state, on_event=event_callback)
        audio_engine.start()
        log.info("Audio engine active ✓")
    except Exception as exc:
        log.warning("Audio not available: %s", exc)

    # ── Camera ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        log.error("Webcam not found (index=%d).", config.camera_index)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    fps_limiter = FPSLimiter(config.fps_limit)
    start_time = time.time()
    duration_secs = config.exam_duration_minutes * 60
    last_snap_t = time.time()

    log.info("=" * 60)
    log.info("  AI PROCTOR v6.0 — %d detection conditions", 23)
    log.info("  Exam: %d min | Snapshot interval: %ds", config.exam_duration_minutes, config.screenshot_interval)
    log.info("  Demo mode: %s", "ON" if config.demo_mode else "OFF")
    log.info("  Press Q to end.")
    log.info("=" * 60)

    # ── Graceful shutdown handler ────────────────────────────────
    def _shutdown(signum, frame):
        state.exam_active = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Main Loop ────────────────────────────────────────────────
    try:
        while state.exam_active:
            ret, frame = cap.read()
            if not ret:
                log.error("Lost webcam connection.")
                break

            now = time.time()
            if now - start_time >= duration_secs:
                log.info("Exam time up.")
                break

            frame_id = state.increment_frame()

            # Submit every frame to YOLO (it handles its own skipping)
            yolo_detector.submit_frame(frame)

            # Snapshot-based analysis
            if now - last_snap_t >= config.screenshot_interval:
                last_snap_t = now
                save_snapshot(frame, config.snap_folder)
                state.increment_snapshots()

                # Preprocess once
                pf = preprocess_frame(frame, frame_id)

                # Run all detectors
                flags: List[str] = []

                face_flags, faces = face_detector.run_all(pf)
                flags.extend(face_flags)

                flags.extend(gaze_tracker.run_all(pf, faces))
                flags.extend(identity_verifier.run_all(pf, faces))
                flags.extend(desk_monitor.run_all(pf))
                flags.extend(rough_sheet.run_all(pf, faces))

                # YOLO results (from async thread)
                flags.extend(yolo_detector.get_latest_flags())

                # Audio flag
                if state.audio_voice_detected:
                    flags.append("Voice/talking detected")

                # Process violations
                if flags:
                    violation_engine.process_flags(flags, frame, frame_id)
                else:
                    log.info(
                        "Snapshot %d — Clear", state.total_snapshots,
                    )

                # Apply score decay
                scorer.apply_decay()

            # ── Render UI ────────────────────────────────────────
            pf_display = preprocess_frame(frame, frame_id)
            display_frame = draw_ui(frame, state, start_time, duration_secs)
            if config.demo_mode:
                # Re-detect faces for overlay (cheap — reuse pf_display)
                _, faces_display = face_detector.check_faces(pf_display)
                display_frame = draw_demo_overlay(
                    display_frame, state, scorer, gaze_tracker,
                    faces_display, pf_display,
                )

            cv2.imshow("AI Proctor — Exam Monitor", display_frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                log.info("Exam ended manually (Q pressed).")
                break

            fps_limiter.wait()

    finally:
        # ── Cleanup ──────────────────────────────────────────────
        state.exam_active = False

        yolo_detector.stop()
        if audio_engine is not None:
            audio_engine.stop()

        cap.release()
        cv2.destroyAllWindows()

        # ── Reports ──────────────────────────────────────────────
        log.info("=" * 60)
        log.info("  EXAM ENDED — GENERATING REPORTS")
        log.info("=" * 60)

        txt_path = report_gen.generate_text_report()
        json_path = report_gen.generate_json_report()
        stats = report_gen.summary_stats()

        log.info("  Total Snapshots   : %d", state.total_snapshots)
        log.info("  Total Warnings    : %d", state.total_warnings)
        log.info("  Suspicion Score   : %.1f", state.suspicion_score)
        log.info("  Peak Score        : %.1f", state.peak_suspicion_score)
        log.info("  Cheating Alert    : %s", "YES" if state.is_cheating_alert() else "NO")
        log.info("  Cheat Probability : %.1f%%", scorer.cheat_probability())
        log.info("  Text Report       : %s", txt_path)
        log.info("  JSON Report       : %s", json_path)
        log.info("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Proctor — Smart Exam Proctoring System v6.0",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Enable demo mode (severity bar, cheat probability overlay)",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Start the FastAPI server alongside the proctoring engine",
    )
    parser.add_argument(
        "--api-only", action="store_true",
        help="Start only the FastAPI server (no vision/audio)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="API server port (default: 8000)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Exam duration in minutes (default: 60)",
    )

    args = parser.parse_args()

    # Build config from CLI args
    import os

    if args.demo:
        os.environ["PROCTOR_DEMO_MODE"] = "true"
    if args.port != 8000:
        os.environ["PROCTOR_API_PORT"] = str(args.port)
    if args.camera != 0:
        os.environ["PROCTOR_CAMERA_INDEX"] = str(args.camera)
    if args.duration != 60:
        os.environ["PROCTOR_EXAM_DURATION_MINUTES"] = str(args.duration)

    config = ProctorConfig()

    if args.api_only:
        # API-only mode
        import uvicorn

        from ai_proctor.api.server import create_app

        ProctorState.destroy()
        state = ProctorState(config)
        app = create_app(config, state)
        log.info("Starting API-only mode on :%d", config.api_port)
        uvicorn.run(app, host=config.api_host, port=config.api_port)
    else:
        if args.api:
            # Mark for API server startup inside run_proctor
            object.__setattr__(config, "_api_mode", True)
        run_proctor(config)


if __name__ == "__main__":
    main()
