"""
Unified audio engine.

Orchestrates: stream → features → classification → events.
Runs on its own daemon thread so it never blocks the vision pipeline.
Supports calibration, temporal smoothing, multi-stage alert escalation,
and sustained-suspicious-audio (>10 s) cheating alerts.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from ai_proctor.audio.audio_classifier import AudioClass, AudioClassifier
from ai_proctor.audio.audio_features import AudioFeatureExtractor
from ai_proctor.audio.audio_stream import AudioStream
from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.state import ProctorState
from ai_proctor.core.utils import get_logger

log = get_logger("audio.engine")

# Severity lookup per classification
_SEVERITY: Dict[str, int] = {
    AudioClass.SILENCE.value: 0,
    AudioClass.WHISPER.value: 2,
    AudioClass.NORMAL_SPEECH.value: 3,
    AudioClass.MUFFLED_SPEECH.value: 3,
    AudioClass.SHOUT.value: 5,
    AudioClass.LOUD_SPIKE.value: 4,
}


class AudioEngine:
    """
    Self-contained audio monitoring engine.

    Lifecycle::

        engine = AudioEngine(config, state)
        engine.start()          # opens mic, starts background thread
        …
        engine.stop()           # graceful shutdown

    Events are pushed into ``ProctorState`` and optionally forwarded
    to an external callback for the event bus.
    """

    def __init__(
        self,
        config: ProctorConfig,
        state: ProctorState,
        on_event: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self.cfg = config
        self.state = state
        self._on_event = on_event

        self._stream = AudioStream(sample_rate=config.audio_sample_rate)
        self._extractor = AudioFeatureExtractor(
            sample_rate=config.audio_sample_rate,
            window_size=5,
        )
        self._classifier = AudioClassifier(config)

        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Multi-stage alert escalation
        self._suspicious_start: Optional[float] = None
        self._alert_level = 0  # 0 = normal, 1 = warning, 2 = alert
        self._consecutive_suspicious = 0

    # ── Event emitter ────────────────────────────────────────────

    def _emit(self, event: Dict) -> None:
        """Push an event into state and optional callback."""
        if self._on_event:
            self._on_event(event)

    # ── Background loop ──────────────────────────────────────────

    def _monitor_loop(self) -> None:
        """Background thread entry point."""
        calibration_buffer: List[float] = []
        calibration_start = time.monotonic()
        calibrated = False

        frame_buffer: List[np.ndarray] = []
        frame_start = time.monotonic()

        while self._running:
            chunk = self._stream.get_chunk(timeout=2.0)
            if chunk is None:
                continue

            # ── Calibration phase ────────────────────────────────
            if not calibrated:
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                calibration_buffer.append(rms)

                elapsed = time.monotonic() - calibration_start
                if elapsed >= self.cfg.audio_calibration_duration:
                    baseline = float(np.mean(calibration_buffer))
                    self._classifier.calibrate(baseline)
                    calibrated = True
                    self.state.audio_calibrated = True
                    log.info("Audio calibration complete.")
                    frame_buffer.clear()
                    frame_start = time.monotonic()
                continue

            # ── Accumulate into frame ────────────────────────────
            frame_buffer.append(chunk)

            if time.monotonic() - frame_start >= self.cfg.audio_frame_duration:
                audio_data = np.concatenate(frame_buffer)
                frame_buffer.clear()
                frame_start = time.monotonic()

                self._process_frame(audio_data)

    def _process_frame(self, audio: np.ndarray) -> None:
        """Extract features, classify, and update state."""
        features = self._extractor.extract(audio)
        classification = self._classifier.classify(features)
        now = time.monotonic()

        # Update state
        self.state.audio_classification = classification.value
        suspicious = classification not in (AudioClass.SILENCE,)

        self.state.audio_voice_detected = suspicious

        # Build frame event
        event = {
            "type": "audio_analysis",
            "classification": classification.value,
            "rms": round(features.rms, 6),
            "spectral_centroid": round(features.spectral_centroid, 2),
            "zcr": round(features.zcr, 5),
            "rms_smooth": round(features.rms_smooth, 6),
            "severity": _SEVERITY.get(classification.value, 0),
            "timestamp": time.time(),
        }
        self._emit(event)

        # ── Multi-stage alert escalation ─────────────────────────
        if suspicious:
            self._consecutive_suspicious += 1
            self.state.audio_suspicion_score += _SEVERITY.get(
                classification.value, 0,
            )

            if self._suspicious_start is None:
                self._suspicious_start = now

            sustained = now - self._suspicious_start

            # Stage 1: warning after 5 seconds
            if sustained >= 5.0 and self._alert_level < 1:
                self._alert_level = 1
                self._emit({
                    "type": "audio_violation",
                    "classification": classification.value,
                    "rms": round(features.rms, 6),
                    "severity": 3,
                    "stage": "WARNING",
                    "sustained_seconds": round(sustained, 1),
                    "timestamp": time.time(),
                })
                log.warning(
                    "Audio WARNING — sustained %s for %.1fs",
                    classification.value,
                    sustained,
                )

            # Stage 2: cheating alert after 10 seconds
            if sustained >= self.cfg.audio_alert_duration and self._alert_level < 2:
                self._alert_level = 2
                self._emit({
                    "type": "audio_violation",
                    "classification": classification.value,
                    "rms": round(features.rms, 6),
                    "severity": 5,
                    "stage": "CHEATING_ALERT",
                    "sustained_seconds": round(sustained, 1),
                    "timestamp": time.time(),
                })
                log.error(
                    "🚨 CHEATING ALERT — sustained audio for %.1fs",
                    sustained,
                )
                # Reset timer for next alert cycle
                self._suspicious_start = now
                self._alert_level = 0
        else:
            # Silence → reset escalation
            self._suspicious_start = None
            self._alert_level = 0
            self._consecutive_suspicious = 0

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """Start the audio stream and monitoring thread."""
        try:
            self._stream.start()
        except Exception as exc:
            log.error("Cannot start audio: %s", exc)
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="audio-engine",
        )
        self._thread.start()
        log.info("Audio engine started.")

    def stop(self) -> None:
        """Gracefully shut down the audio engine."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._stream.stop()
        log.info("Audio engine stopped.")

    @property
    def is_running(self) -> bool:
        return self._running and self._stream.is_active
