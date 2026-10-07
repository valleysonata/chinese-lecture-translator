import sys
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import DEFAULT_CONFIG
from providers.groq_asr import GroqASREngine

# Static method: callable without constructing the engine (no API key needed)
filter_hallucinations = GroqASREngine._filter_hallucinations


def test_filter_cases():
    print("=" * 60)
    print("  PHASE 3.1: Hallucination Filter Regression Tests")
    print("=" * 60)

    prompt = DEFAULT_CONFIG.asr.initial_prompt

    cases = [
        # (label, input transcript, expected output)
        ("Blacklist exact", "本集完", ""),
        ("Blacklist outro set", "謝謝觀看，下次見", ""),
        ("Pure punctuation", "。。。", ""),
        ("Dash filler", "--", ""),
        ("Blacklist embedded", "好，本集完", "好"),
        ("Prompt echo", prompt, ""),
        ("Prompt fragment >=15 chars", "balance factor, node, rotation, pointer, recursion", ""),
        ("Short prompt word kept", "node", "node"),
        ("Normal transcript kept", "這個節點已經不平衡，所以我們需要右旋。",
         "這個節點已經不平衡，所以我們需要右旋"),
        ("Code-switch kept", "這個 node 已經不平衡", "這個 node 已經不平衡"),
        ("Empty input", "", ""),
        ("Whitespace input", "   ", ""),
    ]

    for label, text, expected in cases:
        result = filter_hallucinations(text, prompt)
        assert result == expected, f"[{label}] expected {expected!r}, got {result!r}"
        print(f"[OK] {label}: {text!r} -> {result!r}")

    print("[OK] All hallucination filter cases passed\n")


def test_no_prompt_argument():
    """Filter must work when no prompt is supplied (e.g. prompt=None override)."""
    assert filter_hallucinations("本集完", None) == ""
    assert filter_hallucinations("二叉樹的旋轉", None) == "二叉樹的旋轉"
    print("[OK] Filter works without a prompt argument\n")


if __name__ == "__main__":
    test_filter_cases()
    test_no_prompt_argument()
    print("All hallucination filter tests completed successfully!")
