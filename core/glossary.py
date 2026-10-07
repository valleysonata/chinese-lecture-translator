"""
Phase 4: Course glossary loading and prompt injection.

A glossary is a plain-text file with one term per line (English or Chinese),
e.g.:
    # AVL tree lecture terms
    binary tree
    balance factor
    右旋

Terms are injected into:
  1. The Whisper initial_prompt  -> primes ASR vocabulary for code-switching
  2. The translation system prompt -> keeps English renderings stable across
     utterances instead of varying between synonyms

Both injections are terminology lists (no sentence structure) to preserve the
Phase 3.1 anti-prompt-leakage property.
"""
from pathlib import Path
from typing import List, Optional, Union

# Keep the Whisper prompt bounded: long prompts risk sentence-level leakage
MAX_ASR_PROMPT_CHARS = 400


def load_glossary(path: Union[Path, str]) -> List[str]:
    """
    Load glossary terms from a text file: one term per line.
    - Blank lines and lines starting with '#' are skipped
    - Leading/trailing whitespace is stripped
    - Duplicates are removed (case-insensitive, order preserved)
    - Tolerates UTF-8 (with/without BOM), GBK, and Latin-1 encodings
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Glossary file not found: {file_path}")

    raw = file_path.read_bytes()
    text = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("any", raw, 0, 1, f"Cannot decode glossary file: {file_path}")

    terms: List[str] = []
    seen = set()
    for line in text.splitlines():
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def build_asr_prompt(base_prompt: str, terms: List[str]) -> str:
    """Append glossary terms to the Whisper initial_prompt, bounded to MAX_ASR_PROMPT_CHARS."""
    if not terms:
        return base_prompt
    combined = (base_prompt + " " + ", ".join(terms)).strip()
    if len(combined) <= MAX_ASR_PROMPT_CHARS:
        return combined
    # Add terms until the budget is hit so the prompt stays terminology-only and short
    budget = MAX_ASR_PROMPT_CHARS - len(base_prompt) - 1
    kept: List[str] = []
    used = 0
    for term in terms:
        cost = len(term) + (2 if kept else 1)  # ", " separator
        if used + cost > budget:
            break
        kept.append(term)
        used += cost
    if not kept:
        return base_prompt
    return (base_prompt + " " + ", ".join(kept)).strip()


def build_translation_prompt(base_prompt: str, terms: List[str]) -> str:
    """Append a glossary block to the translation system prompt."""
    if not terms:
        return base_prompt
    block = (
        "\n\nCOURSE GLOSSARY (use these exact English renderings for the terms; "
        "do not mention the glossary itself):\n"
        + ", ".join(terms)
    )
    return base_prompt + block


def describe(terms: List[str], sample: int = 5) -> str:
    """Human-readable startup summary of loaded glossary terms."""
    if not terms:
        return "Glossary    : none (default prompts)"
    preview = ", ".join(terms[:sample])
    if len(terms) > sample:
        preview += f", ... (+{len(terms) - sample} more)"
    return f"Glossary    : {len(terms)} terms loaded -> {preview}"
