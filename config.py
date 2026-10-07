import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
STORAGE_DIR = PROJECT_ROOT / "storage"
AUDIO_CHUNKS_DIR = STORAGE_DIR / "audio_chunks"
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

AUDIO_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class AudioConfig:
    sample_rate: int = 16000          # 16 kHz standard for Whisper and Silero
    channels: int = 1                 # Mono
    chunk_size: int = 512             # 512 samples = 32ms frames for Silero VAD
    device_index: int | None = None   # None uses default input device

@dataclass
class VADConfig:
    threshold: float = 0.5            # Silero speech probability threshold
    min_speech_duration_ms: int = 250 # Ignore brief clicks / noises under 250ms
    silence_duration_ms: int = 800    # Pause duration to finalize utterance (800ms)
    max_speech_duration_s: float = 12.0 # Force split if speech exceeds this duration (avoids latency lag)
    pre_speech_padding_ms: int = 300  # Audio buffer before speech start (prevents word onset clipping)

@dataclass
class ASRConfig:
    provider: str = "groq"            # "groq" or "local"
    model: str = "whisper-large-v3"   # Groq model: whisper-large-v3 or whisper-large-v3-turbo
    language: str = "zh"              # Mandarin Chinese
    temperature: float = 0.0          # Low temperature for deterministic transcription
    timeout_seconds: float = 8.0      # Bounded timeout so stalled requests don't hang pipeline

@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

DEFAULT_CONFIG = AppConfig()
