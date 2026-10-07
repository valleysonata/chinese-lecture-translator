"""
Phase 4: Slide PDF context extraction.

The lecture PDF is the *primary* user-facing context source: its text is
injected into the translation system prompt so the LLM knows the topic and
prefers the terminology used on the slides. The manual glossary
(core/glossary.py) remains an optional, finer-grained override on top.

Injection is bounded (MAX_SLIDES_CHARS) so prompt size — and therefore
latency — stays predictable during a live lecture.
"""
from pathlib import Path
from typing import Tuple, Union

# ~1500 tokens: enough for a slide deck's worth of bullet text without
# noticeably moving translation latency
MAX_SLIDES_CHARS = 6000


class SlidesError(Exception):
    """Raised when a PDF cannot provide usable text (missing, unreadable, scanned)."""


def extract_pdf_text(path: Union[Path, str], max_chars: int = MAX_SLIDES_CHARS) -> Tuple[str, int]:
    """
    Extract text from a lecture PDF (pypdf).

    Returns (text, page_count). Per-page failures are skipped; if no page
    yields text (image-only/scanned PDF) raises SlidesError so the UI can
    tell the user instead of silently injecting nothing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise SlidesError(f"PDF not found: {file_path}")

    try:
        from pypdf import PdfReader
    except ImportError:
        raise SlidesError("pypdf is not installed (py -m pip install pypdf)")

    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        raise SlidesError(f"Cannot read PDF: {e}")

    page_count = len(reader.pages)
    if page_count == 0:
        raise SlidesError("PDF has no pages")

    chunks = []
    for i, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            continue  # unreadable page: skip rather than fail the whole deck
        if text:
            chunks.append(f"[slide {i + 1}]\n{text}")

    combined = "\n\n".join(chunks)
    if not combined:
        raise SlidesError(
            f"No extractable text in {page_count} pages — scanned/image-only PDF?"
        )

    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + "\n[... slides truncated]"
    return combined, page_count


def build_slides_block(text: str) -> str:
    """Wrap extracted slide text as an injectable system-prompt block."""
    return (
        "\n\nLECTURE SLIDES (primary context for this session — prefer the "
        "topic and terminology shown here; do not quote the slides directly):\n"
        + text
    )


def compose_system_prompt(base_prompt: str, slides_text: Union[str, None]) -> str:
    """
    Compose the live translation system prompt.
    `base_prompt` is the default SYSTEM_PROMPT (+ optional glossary block);
    slide text is appended hot — the engine reads the attribute per request,
    so a loaded PDF takes effect on the next translation without a restart.
    """
    if not slides_text:
        return base_prompt
    return base_prompt + build_slides_block(slides_text)
