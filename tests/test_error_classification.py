import sys
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from providers.base import classify_error


def test_rate_limit():
    print("=" * 60)
    print("  Error Classification")
    print("=" * 60)
    cases = [
        (Exception("Error code: 429 - rate limit reached for model"), "rate_limit"),
        (Exception("429 Too Many Requests"), "rate_limit"),
        (Exception("Rate limit reached for requests"), "rate_limit"),
        (Exception("Request timed out."), "timeout"),
        (Exception("Connection error."), "connection"),
        (Exception("401 Unauthorized: invalid api key"), "auth"),
        (Exception("something exploded"), "api_error"),
        (Exception(""), "api_error"),
    ]
    for exc, expected in cases:
        got = classify_error(exc)
        assert got == expected, f"expected {expected!r} for {str(exc)!r}, got {got!r}"
        print(f"[OK] {str(exc)!r:55s} -> {got}")
    print("[OK] All error classifications passed\n")


if __name__ == "__main__":
    test_rate_limit()
    print("All error classification tests completed successfully!")
