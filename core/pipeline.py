"""
Shared live pipeline orchestration (Phase 5 extraction).

Owns the full chain: AudioCapture -> SileroVADSegmenter -> ASRWorker ->
TranslationWorker, plus session logging and drop/latency statistics.
Consumers (console script, PyQt overlay) provide an `on_event` callback and
drive the loop with process_once(); all cross-thread UI concerns stay outside.
"""
import time
import threading
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np

from config import DEFAULT_CONFIG, AUDIO_CHUNKS_DIR, SESSION_LOGS_DIR
from core.audio_capture import AudioCapture
from core.vad import SileroVADSegmenter
from core.asr import ASRWorker
from core.live_translator import TranslationWorker
from core.session_log import SessionLogger
from providers.groq_asr import GroqASREngine
from providers.groq_translation import GroqTranslationEngine


class LivePipeline:
    """
    Threaded live pipeline with a pull-based main loop.

    Event contract (delivered to `on_event` from worker threads):
      {"type": "utterance", "utt_index", "duration"}
      {"type": "asr",       "asr_index", ...ASR payload}
      {"type": "translation","trans_index", ...translation payload}
      {"type": "drop",      "stage", "reason", ...}
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        glossary_terms: Optional[list] = None,
        device_index: Optional[int] = None,
        wav_prefix: str = "seg",
        log_prefix: str = "session",
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        config=None,
    ):
        self.config = config or DEFAULT_CONFIG
        if device_index is not None:
            self.config.audio.device_index = device_index
        self.on_event = on_event
        self.glossary_terms = glossary_terms or []
        self.wav_prefix = wav_prefix

        self.print_lock = threading.Lock()

        self.stats: Dict[str, Any] = {
            "utt_count": 0,
            "asr_count": 0,
            "trans_count": 0,
            "dropped_asr": 0,
            "dropped_trans": 0,
            "empty_transcript": 0,
            "truncated_trans": 0,
            "rate_limited": 0,
            "total_asr_latency": 0.0,
            "total_trans_latency": 0.0,
            "total_delay": 0.0,
        }

        self.asr_engine = GroqASREngine()
        self.trans_engine = GroqTranslationEngine(system_prompt=system_prompt)
        self.trans_worker = TranslationWorker(engine=self.trans_engine, on_result=self._on_translation)
        self.asr_worker = ASRWorker(engine=self.asr_engine, on_result=self._on_asr)
        self.capture = AudioCapture(self.config.audio)
        self.segmenter = SileroVADSegmenter(self.config.audio, self.config.vad)

        # Session log opened last: a missing API key must not leave an empty file behind
        self.session_path = SESSION_LOGS_DIR / f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.session = SessionLogger(self.session_path)

        self._started = False
        self._stopped = False
        self._session_open = False
        self._start_time = 0.0

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start capture and both workers. Raises on audio device failure."""
        self.capture.start()
        self._started = True  # capture is running: stop() must unwind it even if a worker fails
        self.asr_worker.start()
        self.trans_worker.start()
        self._start_time = time.time()
        self._session_open = True
        self.session.log("session_start", {
            "asr_model": self.config.asr.model,
            "trans_model": self.config.translation.model,
            "silence_ms": self.config.vad.silence_duration_ms,
            "glossary_terms": len(self.glossary_terms),
        })

    def process_once(self, timeout: float = 0.05) -> Optional[Dict[str, Any]]:
        """
        Consume one audio chunk and run it through VAD.
        Returns a display-status dict, or None when no chunk was available.
        """
        chunk = self.capture.get_chunk(timeout=timeout)
        if chunk is None:
            return None

        rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
        utterance, prob = self.segmenter.process_chunk(chunk)

        if utterance is not None:
            self._handle_utterance(utterance)

        return {
            "status": "SPEAKING" if self.segmenter.is_speaking else "LISTENING",
            "rms": rms,
            "prob": prob,
            "asr_q": self.asr_worker.queue.qsize(),
            "trans_q": self.trans_worker.queue.qsize(),
        }

    def stop(self) -> Dict[str, Any]:
        """Stop all threads, flush the session log, and return the summary dict."""
        if self._stopped:
            return self._last_summary if hasattr(self, "_last_summary") else {}
        self._stopped = True

        if self._started:
            self.capture.stop()
            self.asr_worker.stop()
            self.trans_worker.stop()

        elapsed = time.time() - self._start_time if self._start_time else 0.0
        tc = self.stats["trans_count"]
        ac = self.stats["asr_count"]
        avg_asr = (self.stats["total_asr_latency"] / ac) if ac > 0 else 0.0
        avg_trans = (self.stats["total_trans_latency"] / tc) if tc > 0 else 0.0
        avg_delay = (self.stats["total_delay"] / tc) if tc > 0 else 0.0

        summary = {
            "runtime_s": round(elapsed, 1),
            "utterances": self.stats["utt_count"],
            "transcripts": ac,
            "translations": tc,
            "dropped_asr": self.stats["dropped_asr"],
            "dropped_translation": self.stats["dropped_trans"],
            "empty_transcripts": self.stats["empty_transcript"],
            "truncated_translations": self.stats["truncated_trans"],
            "rate_limit_errors": self.stats["rate_limited"],
            "dropped_audio_frames": self.capture.dropped_frames,
            "asr_worker_drops": self.asr_worker.dropped_count,
            "translation_worker_drops": self.trans_worker.dropped_count,
            "avg_asr_latency": round(avg_asr, 3),
            "avg_translation_latency": round(avg_trans, 3),
            "avg_total_delay": round(avg_delay, 3),
            "session_log": str(self.session_path),
        }
        if self._session_open:
            # A failed start leaves no session_start; don't emit an orphan session_end
            self.session.log("session_end", summary)
            self._session_open = False
        self.session.close()
        self._last_summary = summary
        return summary

    # ------------------------------------------------------------------ internals

    def _emit(self, event: Dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception as cb_err:
                print(f"[Pipeline Event Callback Error]: {cb_err}")

    def _handle_utterance(self, utterance: Dict[str, Any]) -> None:
        self.stats["utt_count"] += 1
        index = self.stats["utt_count"]
        dur = utterance["duration"]

        # Save local copy for debugging / offline ASR testing
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = AUDIO_CHUNKS_DIR / f"{self.wav_prefix}_{index:03d}_{ts_str}.wav"
        self.segmenter.save_to_wav(utterance["audio"], out_path, self.config.audio.sample_rate)

        self._emit({"type": "utterance", "utt_index": index, "duration": dur})

        if not self.asr_worker.submit_utterance(utterance):
            self.stats["dropped_asr"] += 1
            self.session.log("drop", {"stage": "asr", "reason": "queue_full", "audio_duration": dur})
            self._emit({"type": "drop", "stage": "asr", "reason": "queue_full", "utt_index": index})

    def _on_asr(self, payload: Dict[str, Any]) -> None:
        self.stats["asr_count"] += 1
        self.stats["total_asr_latency"] += payload["asr_latency"]
        payload["asr_index"] = self.stats["asr_count"]

        self.session.log("asr", {
            "success": payload["success"],
            "transcript": payload["transcript"],
            "asr_latency": payload["asr_latency"],
            "audio_duration": payload["audio_duration"],
            "total_delay": payload["total_delay"],
            "queue_backlog": payload["queue_backlog"],
            "error": payload["error"],
            "error_type": payload.get("error_type"),
        })

        if not payload["success"]:
            if payload.get("error_type") == "rate_limit":
                self.stats["rate_limited"] += 1
            self._emit({"type": "asr", **payload})
            return

        text = payload["transcript"]
        if not text.strip():
            # Hallucination filter emptied the transcript: skip translation, record it.
            self.stats["empty_transcript"] += 1
            self.session.log("drop", {"stage": "translation", "reason": "empty_transcript"})
            self._emit({"type": "asr", **payload})
            self._emit({"type": "drop", "stage": "translation", "reason": "empty_transcript"})
            return

        self._emit({"type": "asr", **payload})
        if not self.trans_worker.submit_transcript(payload):
            self.stats["dropped_trans"] += 1
            self.session.log("drop", {"stage": "translation", "reason": "queue_full", "transcript": text})
            self._emit({"type": "drop", "stage": "translation", "reason": "queue_full", "asr_index": payload["asr_index"]})

    def _on_translation(self, payload: Dict[str, Any]) -> None:
        self.stats["trans_count"] += 1
        self.stats["total_trans_latency"] += payload["translation_latency"]
        self.stats["total_delay"] += payload["total_delay"]
        payload["trans_index"] = self.stats["trans_count"]

        if payload["success"] and payload.get("finish_reason") == "length":
            self.stats["truncated_trans"] += 1
        if not payload["success"] and payload.get("error_type") == "rate_limit":
            self.stats["rate_limited"] += 1

        self.session.log("translation", {
            "success": payload["success"],
            "mandarin_transcript": payload["mandarin_transcript"],
            "english_translation": payload["english_translation"],
            "translation_latency": payload["translation_latency"],
            "asr_latency": payload["asr_latency"],
            "total_delay": payload["total_delay"],
            "queue_backlog": payload["queue_backlog"],
            "finish_reason": payload.get("finish_reason"),
            "error": payload["error"],
            "error_type": payload.get("error_type"),
        })
        self._emit({"type": "translation", **payload})
