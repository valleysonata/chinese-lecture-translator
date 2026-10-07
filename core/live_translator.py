import time
import queue
import threading
import collections
from typing import Optional, Callable, Dict, Any

from providers.base import TranslationEngine
from providers.groq_translation import GroqTranslationEngine
from config import TranslationConfig, DEFAULT_CONFIG

class TranslationWorker:
    """
    Asynchronous worker that translates Mandarin ASR transcripts into natural English.
    Operates in an independent background thread.
    CRITICAL: Never blocks the audio capture, VAD, or ASR loops.
    """
    def __init__(
        self,
        engine: Optional[TranslationEngine] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_queue_size: int = 5,
        context_history_len: int = 3
    ):
        self.engine = engine or GroqTranslationEngine()
        self.on_result = on_result
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue: queue.Queue = queue.Queue()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        # Rolling history of recent translations for coherence
        self.recent_translations = collections.deque(maxlen=context_history_len)

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def submit_transcript(self, asr_payload: Dict[str, Any]) -> bool:
        """
        Enqueues an ASR output dictionary for translation.
        Returns False if queue is saturated (prevents latency buildup).
        """
        if not asr_payload.get("transcript", "").strip():
            return False

        try:
            self.queue.put_nowait(asr_payload)
            return True
        except queue.Full:
            return False

    def _run_loop(self):
        while self._is_running:
            try:
                asr_payload = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            transcript = asr_payload["transcript"]
            audio_ts = asr_payload.get("audio_timestamp", time.time())
            audio_dur = asr_payload.get("audio_duration", 0.0)
            asr_lat = asr_payload.get("asr_latency", 0.0)
            current_backlog = self.queue.qsize()

            # Build recent translation context if available
            context_str = " -> ".join(self.recent_translations) if self.recent_translations else None

            trans_start = time.time()
            result = self.engine.translate(transcript, context=context_str)
            trans_end = time.time()

            trans_lat = round(trans_end - trans_start, 3)
            total_delay = round(trans_end - audio_ts, 3) if audio_ts > 0 else (asr_lat + trans_lat)

            english_text = result.get("translation", "")
            if result.get("success", False) and english_text:
                self.recent_translations.append(english_text)

            payload = {
                "success": result.get("success", False),
                "mandarin_transcript": transcript,
                "english_translation": english_text,
                "translation_latency": trans_lat,
                "asr_latency": asr_lat,
                "audio_duration": audio_dur,
                "total_delay": total_delay,
                "queue_backlog": current_backlog,
                "error": result.get("error")
            }

            self.result_queue.put(payload)
            if self.on_result:
                try:
                    self.on_result(payload)
                except Exception as cb_err:
                    print(f"[TranslationWorker Callback Error]: {cb_err}")

            self.queue.task_done()

    def get_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
