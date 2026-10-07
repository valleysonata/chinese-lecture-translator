from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


def classify_error(error: Exception) -> str:
    """Bucket provider exceptions so session logs distinguish rate limits and timeouts."""
    message = str(error).lower()
    if "429" in message or "rate limit" in message or "rate_limit" in message or "tokens per" in message or "tpm" in message:
        return "rate_limit"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "connection" in message or "unreachable" in message:
        return "connection"
    if "401" in message or "403" in message or "unauthorized" in message or "invalid api key" in message:
        return "auth"
    return "api_error"


class ASREngine(ABC):
    """Abstract interface for Speech-to-Text providers (Groq Cloud, Local Whisper, etc.)"""
    @abstractmethod
    def transcribe(self, audio_data: bytes | str, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribes speech audio into text.
        Returns:
            dict containing:
                'transcript': str,
                'latency': float,
                'timestamps': Optional[list]
        """
        pass

class TranslationEngine(ABC):
    """Abstract interface for Fast English Translation."""
    @abstractmethod
    def translate(self, mandarin_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Translates Mandarin text to natural English with optional context hints.
        Returns:
            dict containing:
                'translation': str,
                'latency': float
        """
        pass

class ContextEngine(ABC):
    """Abstract interface for slower lecture context explanation / interpretation."""
    @abstractmethod
    def explain(self, current_slide: Dict[str, Any], recent_transcripts: list[str]) -> Dict[str, Any]:
        """
        Generates high-level educational context/clarification based on slides and speech.
        """
        pass
