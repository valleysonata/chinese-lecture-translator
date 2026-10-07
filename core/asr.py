import time
import queue
import threading
from typing import Optional, Callable, Dict, Any
from providers.base import ASREngine
from providers.groq_asr import GroqASREngine
from config import ASRConfig, DEFAULT_CONFIG

class ASRWorker:
    """
    Background worker that receives segmented audio utterances,
    submits them to the configured ASREngine, and publishes
    transcription results without blocking the live audio/VAD loop.
    """
    def __init__(
        self,
        engine: Optional[ASREngine] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_queue_size: int = 5
    ):
        self.engine = engine or GroqASREngine()
        self.on_result = on_result
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue: queue.Queue = queue.Queue()
        self.dropped_count = 0  # Utterances rejected because the queue was saturated
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def submit_utterance(self, utterance: Dict[str, Any]) -> bool:
        """
        Enqueues an utterance dict from SileroVADSegmenter.
        Returns False if queue was full (backlog protection).
        """
        try:
            self.queue.put_nowait(utterance)
            return True
        except queue.Full:
            # Latency protection: utterance dropped, counted for the session summary
            self.dropped_count += 1
            return False

    def _run_loop(self):
        while self._is_running:
            try:
                utterance = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            audio_data = utterance["audio"]
            sr = utterance["sample_rate"]
            dur = utterance["duration"]
            audio_ts = utterance["timestamp"]
            current_queue_depth = self.queue.qsize()

            asr_start = time.time()
            result = self.engine.transcribe(audio_data, sample_rate=sr)
            asr_end = time.time()

            # Package enriched metric payload
            payload = {
                "transcript": result.get("transcript", ""),
                "success": result.get("success", False),
                "error": result.get("error"),
                "audio_duration": dur,
                "audio_timestamp": audio_ts,
                "asr_latency": round(asr_end - asr_start, 3),
                "total_delay": round(asr_end - audio_ts, 3),
                "queue_backlog": current_queue_depth,
                "segments": result.get("segments", [])
            }

            self.result_queue.put(payload)
            if self.on_result:
                try:
                    self.on_result(payload)
                except Exception as cb_err:
                    print(f"[ASRWorker Callback Error]: {cb_err}")

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
