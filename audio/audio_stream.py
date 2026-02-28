"""
Audio stream capture module.

Handles microphone auto-detection and raw audio buffering via
``sounddevice``.  Provides a clean callback-based stream that
feeds audio chunks into a thread-safe queue.
"""

from __future__ import annotations

import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from ai_proctor.core.utils import get_logger

log = get_logger("audio.stream")


class AudioStream:
    """
    Manages the microphone input stream.

    Usage::

        stream = AudioStream(sample_rate=48000)
        stream.start()
        chunk = stream.get_chunk()  # blocks until data available
        stream.stop()
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        self.sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._device_index: Optional[int] = None

    # ── Auto-detect microphone ───────────────────────────────────

    @staticmethod
    def find_input_device() -> int:
        """Return the index of the first available input device."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                log.info("Using mic: %s (index=%d)", dev["name"], i)
                return i
        raise RuntimeError("No microphone detected")

    # ── Callback ─────────────────────────────────────────────────

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            log.warning("Audio stream status: %s", status)
        mono = np.mean(indata, axis=1)
        self._queue.put(mono.copy())

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone stream."""
        self._device_index = self.find_input_device()
        self._stream = sd.InputStream(
            device=self._device_index,
            channels=1,
            samplerate=self.sample_rate,
            callback=self._callback,
        )
        self._stream.start()
        log.info("Audio stream started (sr=%d)", self.sample_rate)

    def stop(self) -> None:
        """Close the microphone stream gracefully."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                log.warning("Error stopping audio stream: %s", exc)
            self._stream = None
        log.info("Audio stream stopped.")

    def get_chunk(self, timeout: float = 3.0) -> Optional[np.ndarray]:
        """
        Block until an audio chunk is available, or return ``None``
        on timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active
