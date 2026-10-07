import queue
import threading
import sounddevice as sd
import numpy as np
from typing import Optional, Callable

from config import AudioConfig

class AudioCapture:
    """
    Captures live microphone audio in 16 kHz mono using sounddevice.
    Buffers audio blocks into a thread-safe queue.
    """
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.audio_queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._is_running = False

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            pass  # Overflow or underflow warning if needed
        # Flatten to 1D mono float32 array
        audio_chunk = indata[:, 0].copy().astype(np.float32)
        self.audio_queue.put(audio_chunk)

    def start(self):
        if self._is_running:
            return

        self._is_running = True
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.chunk_size,
            device=self.config.device_index,
            channels=self.config.channels,
            dtype="float32",
            callback=self._audio_callback
        )
        self._stream.start()

    def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Retrieve the next audio chunk from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._is_running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @staticmethod
    def list_input_devices():
        """Returns a list of available audio input devices."""
        devices = sd.query_devices()
        input_devices = []
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devices.append((idx, dev["name"], dev["hostapi"]))
        return input_devices
