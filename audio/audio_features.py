"""
Audio feature extraction module.

Extracts RMS energy, spectral centroid, and zero-crossing rate
from raw audio chunks.  Supports rolling-window smoothing for
more stable classification.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from ai_proctor.core.utils import get_logger

log = get_logger("audio.features")


@dataclass
class AudioFeatures:
    """Extracted features for a single audio frame."""

    rms: float
    spectral_centroid: float
    zcr: float
    # Smoothed versions (rolling average)
    rms_smooth: float = 0.0
    centroid_smooth: float = 0.0
    zcr_smooth: float = 0.0


class AudioFeatureExtractor:
    """
    Extracts and smooths audio features.

    Uses a configurable rolling window to produce temporally
    stable values that reduce classification jitter.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        window_size: int = 5,
    ) -> None:
        self.sample_rate = sample_rate
        self._rms_history: deque[float] = deque(maxlen=window_size)
        self._centroid_history: deque[float] = deque(maxlen=window_size)
        self._zcr_history: deque[float] = deque(maxlen=window_size)

    def extract(self, audio: np.ndarray) -> AudioFeatures:
        """
        Compute RMS, spectral centroid, and ZCR for the given
        audio chunk, then return both raw and smoothed values.
        """
        rms = float(librosa.feature.rms(y=audio).mean())
        centroid = float(
            librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate,
            ).mean()
        )
        zcr = float(librosa.feature.zero_crossing_rate(y=audio).mean())

        # Update rolling windows
        self._rms_history.append(rms)
        self._centroid_history.append(centroid)
        self._zcr_history.append(zcr)

        return AudioFeatures(
            rms=rms,
            spectral_centroid=centroid,
            zcr=zcr,
            rms_smooth=float(np.mean(self._rms_history)),
            centroid_smooth=float(np.mean(self._centroid_history)),
            zcr_smooth=float(np.mean(self._zcr_history)),
        )

    def reset(self) -> None:
        """Clear all history buffers."""
        self._rms_history.clear()
        self._centroid_history.clear()
        self._zcr_history.clear()
