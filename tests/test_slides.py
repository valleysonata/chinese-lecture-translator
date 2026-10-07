import io
import sys
import logging
import tempfile
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# The corrupt-file test below intentionally feeds pypdf garbage; keep its
# complaint out of the test output.
logging.getLogger("pypdf").setLevel(logging.CRITICAL)

from pypdf import PdfWriter, PdfReader
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject

from core.slides import (
    extract_pdf_text,
    build_slides_block,
    compose_system_prompt,
    SlidesError,
    MAX_SLIDES_CHARS,
)
from providers.groq_translation import SYSTEM_PROMPT


def _make_pdf_bytes(page_texts: list) -> bytes:
    """Build a small PDF fixture in memory: one page per entry.
    An entry of None (or an empty list) produces a text-less blank page."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
            )
            stream = DecodedStreamObject()
            # escape parentheses the naive way; fixtures don't use them
            stream.set_data(
                f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
            )
            page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_pdf_text():
    print("=" * 60)
    print("  Phase 4: Slides PDF Extraction")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Happy path: 3 pages of text
        pdf = tmp / "lecture.pdf"
        pdf.write_bytes(_make_pdf_bytes(["Binary Tree Lecture", "AVL Rotations", "Time Complexity"]))
        text, pages = extract_pdf_text(str(pdf))
        assert pages == 3, f"expected 3 pages, got {pages}"
        assert "Binary Tree Lecture" in text and "Time Complexity" in text
        assert "[slide 1]" in text and "[slide 3]" in text
        assert len(text) <= MAX_SLIDES_CHARS
        print(f"[OK] 3-page extraction ({len(text)} chars, {pages} pages)")

        # Bounding: tiny budget truncates with a visible marker
        small, _ = extract_pdf_text(str(pdf), max_chars=30)
        assert len(small) <= 60, f"bounded text too long: {len(small)}"
        assert "[... slides truncated]" in small
        print(f"[OK] bounded to {len(small)} chars with truncation marker")

        # Missing file
        try:
            extract_pdf_text(tmp / "nope.pdf")
            raise AssertionError("expected SlidesError for missing file")
        except SlidesError as e:
            assert "not found" in str(e)
            print("[OK] missing file raises SlidesError")

        # Blank/scanned PDF (no extractable text anywhere)
        scanned = tmp / "scanned.pdf"
        scanned.write_bytes(_make_pdf_bytes([None, None]))
        try:
            extract_pdf_text(str(scanned))
            raise AssertionError("expected SlidesError for text-less PDF")
        except SlidesError as e:
            assert "scanned" in str(e)
            print("[OK] text-less PDF raises SlidesError with scanned hint")

        # Unreadable file that is not a PDF at all
        broken = tmp / "broken.pdf"
        broken.write_bytes(b"this is definitely not a pdf")
        try:
            extract_pdf_text(str(broken))
            raise AssertionError("expected SlidesError for unreadable file")
        except SlidesError:
            print("[OK] corrupt file raises SlidesError")

    print("[OK] Slides extraction passed\n")


def test_prompt_composition():
    print("=" * 60)
    print("  Phase 4: Slides Prompt Composition")
    print("=" * 60)

    slides_text = "[slide 1]\nBinary Tree Lecture\nAVL rotations"

    # No slides -> base prompt untouched (backward compatible)
    assert compose_system_prompt(SYSTEM_PROMPT, None) == SYSTEM_PROMPT
    print("[OK] no slides leaves the base prompt untouched")

    # With slides -> original rules preserved + block appended
    composed = compose_system_prompt(SYSTEM_PROMPT, slides_text)
    assert composed.startswith(SYSTEM_PROMPT), "core rules must survive"
    assert "LECTURE SLIDES" in composed
    assert "Binary Tree Lecture" in composed
    assert "NEVER invent context" in composed, "core rules must survive"
    print("[OK] slides block appended, core rules preserved")

    # Glossary + slides compose together (glossary first, slides last)
    from core.glossary import build_translation_prompt
    with_glossary = build_translation_prompt(SYSTEM_PROMPT, ["balance factor"])
    both = compose_system_prompt(with_glossary, slides_text)
    assert "COURSE GLOSSARY" in both and "balance factor" in both
    assert "LECTURE SLIDES" in both and "Binary Tree Lecture" in both
    assert both.index("COURSE GLOSSARY") < both.index("LECTURE SLIDES")
    print("[OK] glossary block + slides block coexist in order")

    # Block header itself is bounded when given huge text
    huge = build_slides_block("x" * (MAX_SLIDES_CHARS * 2))
    assert "LECTURE SLIDES" in huge
    print("[OK] slides block builder works")

    print("[OK] Prompt composition passed\n")


if __name__ == "__main__":
    test_extract_pdf_text()
    test_prompt_composition()
    print("All slides tests completed successfully!")
