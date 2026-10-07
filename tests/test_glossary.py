import sys
import tempfile
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.glossary import (
    load_glossary,
    build_asr_prompt,
    build_translation_prompt,
    describe,
    MAX_ASR_PROMPT_CHARS,
)
from providers.groq_translation import SYSTEM_PROMPT


def _write(tmpdir: Path, name: str, content: str | bytes) -> Path:
    p = tmpdir / name
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    return p


def test_load_glossary_parsing():
    print("=" * 60)
    print("  Phase 4: Glossary Loading")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        f = _write(tmp, "ok.txt", "# comment line\nbinary tree\n\nAVL tree\n  balance factor  \nbinary tree\n")
        terms = load_glossary(f)
        assert terms == ["binary tree", "AVL tree", "balance factor"], f"got {terms}"
        print(f"[OK] comments/blanks/dupes stripped: {terms}")

        # Case-insensitive dedupe
        f2 = _write(tmp, "case.txt", "Node\nnode\nNODE\n")
        assert load_glossary(f2) == ["Node"], f"got {load_glossary(f2)}"
        print("[OK] case-insensitive dedupe")

        # BOM tolerated
        f3 = _write(tmp, "bom.txt", "﻿rotation\npointer\n")
        assert load_glossary(f3) == ["rotation", "pointer"]
        print("[OK] UTF-8 BOM handled")

        # GBK fallback for Chinese terms
        f4 = _write(tmp, "gbk.txt", "右旋\n時間複雜度\n".encode("gbk"))
        assert load_glossary(f4) == ["右旋", "時間複雜度"], f"got {load_glossary(f4)}"
        print("[OK] GBK encoding fallback")

        # Missing file raises
        try:
            load_glossary(tmp / "missing.txt")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            print("[OK] missing file raises FileNotFoundError")

    print("[OK] Glossary loading passed\n")


def test_prompt_building():
    print("=" * 60)
    print("  Phase 4: Prompt Injection")
    print("=" * 60)

    base_asr = "binary tree, AVL tree, balance factor."
    terms = ["rotation", "pointer", "recursion", "時間複雜度"]

    # Empty terms -> unchanged
    assert build_asr_prompt(base_asr, []) == base_asr
    assert build_translation_prompt(SYSTEM_PROMPT, []) == SYSTEM_PROMPT
    print("[OK] empty glossary leaves prompts untouched (backward compatible)")

    # ASR prompt: terms appended, bounded
    asr_prompt = build_asr_prompt(base_asr, terms)
    for t in terms:
        assert t in asr_prompt, f"{t} missing from ASR prompt"
    assert len(asr_prompt) <= MAX_ASR_PROMPT_CHARS, f"ASR prompt too long: {len(asr_prompt)}"
    print(f"[OK] ASR prompt bounded at {len(asr_prompt)}/{MAX_ASR_PROMPT_CHARS} chars with all terms")

    # ASR prompt: huge glossary gets truncated but stays bounded and terminology-only
    huge = [f"term-{i:03d}" for i in range(200)]
    bounded = build_asr_prompt(base_asr, huge)
    assert len(bounded) <= MAX_ASR_PROMPT_CHARS, f"bounded prompt too long: {len(bounded)}"
    assert bounded.startswith(base_asr)
    print(f"[OK] 200-term glossary truncated to {len(bounded)} chars")

    # Translation prompt: glossary block appended, original rules preserved
    trans_prompt = build_translation_prompt(SYSTEM_PROMPT, terms)
    assert trans_prompt.startswith(SYSTEM_PROMPT), "original system prompt must be preserved"
    assert "COURSE GLOSSARY" in trans_prompt
    assert "時間複雜度" in trans_prompt
    assert "NEVER invent context" in trans_prompt, "core rules must survive"
    print("[OK] translation prompt keeps core rules + glossary block")

    # describe() never crashes
    assert "none" in describe([])
    assert "4 terms" in describe(terms)
    print("[OK] describe() summary")

    print("[OK] Prompt injection passed\n")


if __name__ == "__main__":
    test_load_glossary_parsing()
    test_prompt_building()
    print("All glossary tests completed successfully!")
