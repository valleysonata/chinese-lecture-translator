import io
import time
import wave
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np

from groq import Groq
from providers.base import ASREngine
from config import ASRConfig, DEFAULT_CONFIG

class GroqASREngine(ASREngine):
    """
    Mandarin ASR using Groq's high-speed Whisper Cloud endpoint.
    Produces low-latency Mandarin transcripts with timestamps and metrics.
    """
    def __init__(self, api_key: Optional[str] = None, config: Optional[ASRConfig] = None):
        self.config = config or DEFAULT_CONFIG.asr
        self.api_key = api_key or DEFAULT_CONFIG.groq_api_key
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in your environment or in a .env file."
            )
            
        self.client = Groq(api_key=self.api_key, timeout=self.config.timeout_seconds)
        self.model = self.config.model

    def _to_wav_bytes(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Converts float32 numpy audio array to 16-bit PCM WAV in-memory bytes."""
        clipped = np.clip(audio_data, -1.0, 1.0)
        pcm16 = (clipped * 32767).astype(np.int16)
        
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
            
        return buf.getvalue()

    def transcribe(
        self,
        audio_data: Union[np.ndarray, bytes, str, Path],
        sample_rate: int = 16000,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribes speech audio into Mandarin text.
        audio_data can be a numpy array, raw wav bytes, or a file path.
        """
        start_time = time.time()
        
        # Prepare file payload for Groq
        if isinstance(audio_data, (str, Path)):
            file_path = Path(audio_data)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            with open(file_path, "rb") as f:
                wav_bytes = f.read()
            file_tuple = (file_path.name, wav_bytes, "audio/wav")
        elif isinstance(audio_data, np.ndarray):
            wav_bytes = self._to_wav_bytes(audio_data, sample_rate)
            file_tuple = ("speech.wav", wav_bytes, "audio/wav")
        elif isinstance(audio_data, bytes):
            file_tuple = ("speech.wav", audio_data, "audio/wav")
        else:
            raise TypeError(f"Unsupported audio_data type: {type(audio_data)}")

        kwargs: Dict[str, Any] = {
            "file": file_tuple,
            "model": self.model,
            "language": self.config.language,
            "temperature": self.config.temperature,
            "response_format": "verbose_json"
        }
        
        if prompt:
            kwargs["prompt"] = prompt

        try:
            response = self.client.audio.transcriptions.create(**kwargs)
            latency = time.time() - start_time
            
            transcript = getattr(response, "text", "") or ""
            duration = getattr(response, "duration", 0.0) or 0.0
            segments = getattr(response, "segments", []) or []

            return {
                "success": True,
                "transcript": transcript.strip(),
                "language": self.config.language,
                "latency": round(latency, 3),
                "duration": round(float(duration), 2),
                "segments": segments,
                "error": None
            }
        except Exception as e:
            latency = time.time() - start_time
            return {
                "success": False,
                "transcript": "",
                "language": self.config.language,
                "latency": round(latency, 3),
                "duration": 0.0,
                "segments": [],
                "error": str(e)
            }
