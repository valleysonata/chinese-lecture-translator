import collections
import time
import wave
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from silero_vad import load_silero_vad

from config import VADConfig, AudioConfig

class SileroVADSegmenter:
    """
    Processes real-time audio streams with Silero VAD to detect
    meaningful speech utterances without dropping natural pauses or fragmenting phrases.
    """
    def __init__(self, audio_config: Optional[AudioConfig] = None, vad_config: Optional[VADConfig] = None):
        self.audio_config = audio_config or AudioConfig()
        self.vad_config = vad_config or VADConfig()

        # Load Silero VAD (CPU inference is <1ms per 32ms chunk)
        self.model = load_silero_vad()
        self.model.eval()

        self.sample_rate = self.audio_config.sample_rate
        self.chunk_size = self.audio_config.chunk_size
        self.chunk_duration_s = self.chunk_size / self.sample_rate

        # Pre-speech buffer (ring buffer to retain audio leading up to speech)
        pre_chunks = max(1, int((self.vad_config.pre_speech_padding_ms / 1000.0) / self.chunk_duration_s))
        self.pre_buffer = collections.deque(maxlen=pre_chunks)

        # State tracking
        self.is_speaking = False
        self.current_utterance: list[np.ndarray] = []
        self.speech_start_time: float = 0.0
        self.silence_chunks_count: int = 0
        self.speech_chunks_count: int = 0

        # Calculations
        self.required_silence_chunks = int((self.vad_config.silence_duration_ms / 1000.0) / self.chunk_duration_s)
        self.min_speech_chunks = int((self.vad_config.min_speech_duration_ms / 1000.0) / self.chunk_duration_s)
        self.max_speech_chunks = int(self.vad_config.max_speech_duration_s / self.chunk_duration_s)

    def process_chunk(self, chunk: np.ndarray) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Process a single audio chunk (expected 512 samples float32).
        Returns:
            (utterance_data_dict_or_None, speech_probability)
        """
        if len(chunk) != self.chunk_size:
            # Pad or truncate if needed
            if len(chunk) < self.chunk_size:
                padded = np.zeros(self.chunk_size, dtype=np.float32)
                padded[:len(chunk)] = chunk
                chunk = padded
            else:
                chunk = chunk[:self.chunk_size]

        tensor_chunk = torch.from_numpy(chunk).float()

        with torch.no_grad():
            prob = self.model(tensor_chunk, self.sample_rate).item()

        is_speech = prob >= self.vad_config.threshold
        completed_utterance: Optional[Dict[str, Any]] = None

        if not self.is_speaking:
            self.pre_buffer.append(chunk)
            if is_speech:
                self.speech_chunks_count += 1
                if self.speech_chunks_count >= 2:  # Confirm speech onset
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                    self.current_utterance = list(self.pre_buffer)
                    self.silence_chunks_count = 0
            else:
                self.speech_chunks_count = 0
        else:
            self.current_utterance.append(chunk)

            if is_speech:
                self.silence_chunks_count = 0
            else:
                self.silence_chunks_count += 1

            total_chunks = len(self.current_utterance)

            # Check if utterance finalized (pause detected) or max duration reached
            reached_silence = self.silence_chunks_count >= self.required_silence_chunks
            reached_max = total_chunks >= self.max_speech_chunks

            if reached_silence or reached_max:
                duration_s = total_chunks * self.chunk_duration_s
                
                # Check if it meets minimum speech length
                if total_chunks >= self.min_speech_chunks:
                    audio_data = np.concatenate(self.current_utterance)
                    completed_utterance = {
                        "audio": audio_data,
                        "sample_rate": self.sample_rate,
                        "duration": duration_s,
                        "timestamp": self.speech_start_time,
                        "reason": "silence" if reached_silence else "max_duration"
                    }

                # Reset state
                self.is_speaking = False
                self.current_utterance = []
                self.speech_chunks_count = 0
                self.silence_chunks_count = 0
                self.pre_buffer.clear()

        return completed_utterance, prob

    @staticmethod
    def save_to_wav(audio_data: np.ndarray, file_path: Path | str, sample_rate: int = 16000) -> str:
        """Saves a float32 numpy audio array as a 16-bit PCM WAV file."""
        # Convert float32 [-1.0, 1.0] to int16
        clipped = np.clip(audio_data, -1.0, 1.0)
        pcm16 = (clipped * 32767).astype(np.int16)
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
            
        return str(path)
