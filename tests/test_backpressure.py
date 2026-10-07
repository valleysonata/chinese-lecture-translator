import sys
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.asr import ASRWorker
from core.live_translator import TranslationWorker


def _dummy_engine():
    """Truthy placeholder engine; workers never start their threads in these tests."""
    return object()


def test_asr_queue_backpressure():
    print("=" * 60)
    print("  Backpressure: ASR Worker Queue")
    print("=" * 60)

    worker = ASRWorker(engine=_dummy_engine(), on_result=None, max_queue_size=1)

    first = worker.submit_utterance({"audio": None, "duration": 1.0})
    second = worker.submit_utterance({"audio": None, "duration": 1.0})

    assert first is True, "First submit must be accepted"
    assert second is False, "Second submit must be rejected on a full queue"
    assert worker.dropped_count == 1, f"Drop must be counted, got {worker.dropped_count}"
    assert worker.queue.qsize() == 1
    print("[OK] Queue full -> rejected and counted (dropped_count=1)\n")


def test_translation_queue_backpressure():
    print("=" * 60)
    print("  Backpressure: Translation Worker Queue")
    print("=" * 60)

    worker = TranslationWorker(engine=_dummy_engine(), on_result=None, max_queue_size=1)

    first = worker.submit_transcript({"transcript": "這個 node 不平衡"})
    second = worker.submit_transcript({"transcript": "需要做 right rotation"})

    assert first is True, "First submit must be accepted"
    assert second is False, "Second submit must be rejected on a full queue"
    assert worker.dropped_count == 1, f"Drop must be counted, got {worker.dropped_count}"

    # Empty transcripts are filtered before enqueueing: not a backlog drop
    empty = worker.submit_transcript({"transcript": "   "})
    assert empty is False, "Empty transcript must be rejected"
    assert worker.dropped_count == 1, "Empty transcript must not count as a backlog drop"
    print("[OK] Queue full -> rejected and counted (dropped_count=1)")
    print("[OK] Empty transcript rejected without counting as a drop\n")


def test_result_queue_not_consumed_in_live_mode():
    """Result queue exists but live mode uses callbacks; it must not block workers."""
    worker = TranslationWorker(engine=_dummy_engine(), on_result=None, max_queue_size=1)
    assert worker.get_result(timeout=0.01) is None
    print("[OK] get_result returns None when empty (non-blocking consumer)\n")


if __name__ == "__main__":
    test_asr_queue_backpressure()
    test_translation_queue_backpressure()
    test_result_queue_not_consumed_in_live_mode()
    print("All backpressure tests completed successfully!")
