import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class SessionLogger:
    """
    Thread-safe JSONL event logger for live pipeline sessions.

    Every event is one JSON object per line (UTF-8), flushed immediately so a
    Ctrl+C or crash never loses the session record. This is the artifact used to
    review transcript/translation quality and drop metrics after a lecture test.
    """

    def __init__(self, file_path: Path | str):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Append one event. `payload` values must be JSON-serializable primitives."""
        record: Dict[str, Any] = {"ts": round(time.time(), 3), "event": event}
        if payload:
            record.update(payload)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            if not self._fh.closed:
                self._fh.write(line + "\n")
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    @property
    def closed(self) -> bool:
        return self._fh.closed
