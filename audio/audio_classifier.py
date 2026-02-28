"""
Audio classification module.

Classifies audio frames into one of six categories using
noise-adaptive thresholds computed during a calibration phase.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from ai_proctor.audio.audio_features import AudioFeatures
from ai_proctor.core.config import ProctorConfig
from ai_proctor.core.utils import get_logger

log = get_logger("audio.classifier")


class AudioClass(str, Enum):
    """Possible audio classification outcomes."""

    SILENCE = "SILENCE"
    WHISPER = "WHISPER"
    NORMAL_SPEECH = "NORMAL_SPEECH"
    MUFFLED_SPEECH = "MUFFLED_SPEECH"
    SHOUT = "SHOUT"
    LOUD_SPIKE = "LOUD_SPIKE"


class AudioClassifier:
    """
    Rule-based audio classifier with noise-adaptive thresholds.

    After a calibration period, the baseline noise level is stored
    and all thresholds are relative to it, making the system
    robust in different acoustic environments.
    """

    def __init__(self, config: ProctorConfig) -> None:
        self.cfg = config
        self._speech_thresh: float = config.speech_threshold
        self._spike_thresh: float = config.spike_threshold
        self._calibrated = False
        self._baseline_rms: float = 0.0

    def calibrate(self, baseline_rms: float) -> None:
        """
        Set noise-adaptive thresholds from the measured background.

        Called automatically by ``AudioEngine`` after the calibration
        window completes.
        """
        self._baseline_rms = baseline_rms
        # Adaptive thresholds: speech must be above background + margin
        self._speech_thresh = max(
            self.cfg.speech_threshold,
            baseline_rms * 1.8,
        )
        self._spike_thresh = max(
            self.cfg.spike_threshold,
            baseline_rms * 3.5,
        )
        self._calibrated = True
        log.info(
            "Calibrated — baseline RMS=%.5f  speech_thresh=%.5f  spike_thresh=%.5f",
            baseline_rms,
            self._speech_thresh,
            self._spike_thresh,
        )

    def classify(self, features: AudioFeatures) -> AudioClass:
        """
        Classify an audio frame using smoothed features.

        Classification hierarchy:
            1. SILENCE  — below speech threshold
            2. WHISPER  — just above threshold, low centroid
            3. MUFFLED  — moderate RMS, low centroid + ZCR
            4. NORMAL   — moderate RMS
            5. SHOUT    — very high RMS or centroid
            6. LOUD_SPIKE — high RMS but not quite shout
        """
        rms = features.rms_smooth
        centroid = features.centroid_smooth
        zcr = features.zcr_smooth

        if rms < self._speech_thresh:
            return AudioClass.SILENCE

        if rms < self._speech_thresh * self.cfg.whisper_rms_factor:
            if centroid <= self.cfg.whisper_centroid_max:
                return AudioClass.WHISPER

        if rms < self._spike_thresh:
            if (
                centroid <= self.cfg.muffled_centroid_max
                and zcr <= self.cfg.muffled_zcr_max
            ):
                return AudioClass.MUFFLED_SPEECH
            return AudioClass.NORMAL_SPEECH

        if (
            rms >= self._spike_thresh * self.cfg.shout_rms_factor
            or centroid >= self.cfg.shout_centroid_min
        ):
            return AudioClass.SHOUT

        return AudioClass.LOUD_SPIKE

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
