# AGENTS.md

Guidance for AI agents and contributors working on this repository.

## Project

**Chinese Lecture Interpreter** — a lightweight Windows desktop app that helps STEM
students follow Mandarin-taught lectures in natural English. Live pipeline:
microphone → Silero VAD → Whisper ASR (Mandarin + English code-switching) →
LLM translation → English output.

## Repository Layout

| Path | Purpose |
|---|---|
| `config.py` | Dataclass configs (`AudioConfig`, `VADConfig`, `ASRConfig`, `TranslationConfig`) + paths (`.env` loading, storage dirs) |
| `core/audio_capture.py` | `AudioCapture` — sounddevice mic stream, 16 kHz mono, 32 ms blocks |
| `core/vad.py` | `SileroVADSegmenter` — onset/offset detection, pre-speech buffer, utterance finalization |
| `core/asr.py` | `ASRWorker` — background transcription thread (bounded queue) |
| `core/live_translator.py` | `TranslationWorker` — background translation thread (bounded queue, rolling context) |
| `core/session_log.py` | `SessionLogger` — thread-safe JSONL event logging for live sessions |
| `core/glossary.py` | Phase 4 (optional/advanced): glossary parsing (`load_glossary`) + prompt injection (`apply_glossary`) |
| `core/slides.py` | Phase 4 (primary): bounded PDF text extraction (`extract_pdf_text`) + `LECTURE SLIDES` prompt block (`compose_system_prompt`) |
| `core/pipeline.py` | `LivePipeline` — shared capture→VAD→ASR→translation orchestration (console script + main window app) |
| `core/overlay_window.py` | Phase 5: `SubtitleOverlay` — frameless, topmost, click-through bottom subtitle bar (optional app mode) |
| `providers/base.py` | `ASREngine`, `TranslationEngine`, `ContextEngine` ABCs + `classify_error()` |
| `providers/groq_asr.py` | Groq Whisper engine + `_filter_hallucinations()` |
| `providers/groq_translation.py` | Groq LLM translation engine + `SYSTEM_PROMPT` |
| `scripts/app.py` | Phase 5 launcher: main window UI (Start/Stop, slides drop, feed, subtitle-bar toggle) |
| `scripts/` | Live harnesses (`test_phase1/2/3.py`, `app.py` for Phase 5, `verify_automatic_segmentation.py`) |
| `ChineseLectureInterpreter.spec`, `build_exe.ps1` | Reproducible PyInstaller windowed-exe build; API key stays external (`.env` next to the exe, never bundled) |
| `tests/` | Offline regression tests (no microphone/API required; `test_overlay_ui.py` / `test_app_ui.py` open a window briefly) |
| `storage/audio_chunks/` | Saved WAV segments (gitignored) |
| `storage/session_logs/` | JSONL session logs: transcripts, translations, latencies, drops (gitignored) |

## Current Progress

- [x] **Phase 1: Microphone + Local Silero VAD** — 32 ms chunks, onset debounce,
      300 ms pre-speech buffer, auto-finalize on 800 ms silence or 12 s max speech
- [x] **Phase 2: Mandarin ASR Engine** — Groq `whisper-large-v3`, `language="zh"`,
      `temperature=0`, async worker with bounded queue
- [x] **Phase 3: Fast English Live Translation Layer** — Groq LLM, rolling context
      of the last 3 translations, async worker, never blocks audio/ASR
- [x] **Feat 3.1: Anti-Hallucination & Conservative Meaning Translation**
      - Terminology-only Whisper `initial_prompt` (no sentence-level prompt leakage)
      - Hallucination blacklist + prompt-echo detection + punctuation/filler filtering
      - Translation principle: *"Reconstruct grammar aggressively, reconstruct
        meaning conservatively."* (no invented context, no outros, no finishing trail-offs)
      - `--benchmark-cases` for translation-quality spot checks
- [x] **Feat 3.1.1: Pipeline hardening (this pass)**
      - VAD min-speech enforced on **voiced chunks only** (pre-silence padding no
        longer lets noise bursts through to ASR)
      - Trailing pause trimmed from finalized utterances (fewer Whisper hallucinations)
      - JSONL session logging (`storage/session_logs/`) for after-action review
      - Queue drops counted and surfaced (console warning + summary + log)
      - Single print lock so the VU line and worker callbacks can't garble output
      - Offline regression tests: VAD filtering, hallucination filter, backpressure
- [x] **Feat 3.1.2: Bounded audio queue** — `AudioCapture` queue capped at ~6.1 s
      with drop-oldest backpressure; dropped frames counted in summary + session log
- [x] **Phase 4: Course Context (lecture PDF primary, manual glossary optional)**
      - `core/slides.py`: bounded pypdf extraction (per-page fault tolerance,
        scanned/image-only PDFs reported) → `LECTURE SLIDES` block appended
        after the optional `COURSE GLOSSARY` block; composed at pipeline start
        and hot-swapped when a PDF is loaded mid-session
      - PDF drop zone in `scripts/app.py` (drag-and-drop + browse) and `--slides`
        preload flag
      - `core/glossary.py`: optional advanced input — one-term-per-line file
        (`#` comments, dedupe, UTF-8/GBK tolerant) injected into the Whisper
        `initial_prompt` (bounded to 400 chars, terminology-only) and into the
        translation system prompt as a `COURSE GLOSSARY` block; `--glossary`
        flag on `test_phase3.py` and `app.py`; sample in `glossary.example.txt`
- [x] **Phase 5: PyQt6 Main Window UI (this pass)**
      - `scripts/app.py`: normal draggable/closable window — status row (state,
        mic level, queue depths, drops, avg delay), slides drop zone, card feed
        (ZH first → EN fills in, per-line delay badge, in-place error badges,
        200-card cap); explicit Start/Stop (opens idle; `--autostart` opt-in)
      - Subtitle bar is an optional mode: "Subtitle bar" checkbox (default off,
        `--overlay` to pre-enable) owns the frameless, always-on-top,
        click-through bar in `core/overlay_window.py` (`--clickable` to disable)
      - Modes: live, `--demo` (synthetic, no mic/API), plus `--glossary`,
        `--slides`, `--device`, `--duration`, `--screen`, `--list-devices`
      - Shared `core/pipeline.py` extraction: `LivePipeline.process_once()` +
        `on_event` callback drive both the app and `test_phase3.py`
      - Finish-reason/rate-limit observability: in-place card badges +
        status alerts, `rate_limit_errors` / `truncated_translations` counters
- [ ] **Phase 6: Full 60-90 Minute Endurance Test**

Keep this checklist and the matching one in `README.md` in sync whenever a phase
lands.

## Commit Conventions

Conventional Commits. Format:

```
<type>: <lowercase imperative summary>
```

Optional scope: `fix(vad): ...`

| Type | Use for |
|---|---|
| `feat:` | New user-visible capability or behavior |
| `fix:` | Bug fix |
| `docs:` | Documentation only (README, AGENTS.md) |
| `test:` | Adding or updating tests only |
| `refactor:` | Internal restructure with no behavior change |
| `perf:` | Performance improvement |
| `chore:` | Maintenance: deps, gitignore, tooling, cleanup |

Rules:

1. Lowercase summary after the colon, imperative mood, **no trailing period**.
2. One concern per commit — do not mix a `fix:` with unrelated `feat:` changes.
3. Body may use `-` bullets for detail when the change isn't self-explanatory.
4. Any change that shifts project status must update **both** `README.md` and
   this file's progress checklist in the same commit (usually a `docs:` or the
   feature commit itself).

Examples:

```
feat: add JSONL session logging to phase 3 live pipeline
fix: count actual speech chunks in VAD min-speech filter
test: add regression tests for hallucination filtering
chore: ignore session log files
docs: add AGENTS.md with progress and commit conventions
```

## Testing

Offline (no mic, no API calls) — run these before committing pipeline changes:

```powershell
py tests\test_vad.py
py tests\test_hallucination_filter.py
py tests\test_backpressure.py
py tests\test_audio_backpressure.py
py tests\test_error_classification.py
py tests\test_glossary.py
py tests\test_slides.py               # PDF extraction + prompt composition
py tests\test_overlay_ui.py           # opens a window briefly; no mic/API
py tests\test_app_ui.py               # main window self-check; no mic/API
```

Live / API-dependent:

```powershell
py scripts\test_phase1.py --list-devices   # mic + VAD only
py scripts\test_phase2.py --file <wav>     # ASR only
py scripts\test_phase3.py --benchmark-cases  # translation quality
py scripts\test_phase3.py --duration 120     # full live pipeline (logs to storage/session_logs/)
py scripts\app.py --demo --duration 15       # main window UI smoke test (no mic, no API)
py scripts\app.py --autostart --duration 120 # live window mode (logs to storage/session_logs/)
```

After a live run, review the JSONL session log for `drop` events, latencies, and
translation quality — not just the console output.

## Known Backlog (not yet implemented)

- **Rate-limit backoff**: errors are classified and counted
  (`rate_limit_errors` in summary + `error_type` in JSONL), but there is no
  retry/backoff yet; a 429 currently skips that utterance.
- **Truncation response**: `finish_reason=length` is now visible
  (`[TRUNCATED]` marker + `truncated_translations`), but the fix — raising
  `max_tokens=100` or splitting long segments — is not done.
- **12 s force-split overlap**: mid-sentence splits produce fragments; consider
  overlap or split-at-lowest-probability.
- **Blacklist word boundaries**: phrase replacement can corrupt legitimate speech
  containing a blacklist word (e.g. `收藏`); consider stricter matching.
- **Console loop catch-up**: `test_phase3.py` consumes one 32 ms chunk per
  iteration (the main window / subtitle bar drain up to four); bounded queue
  caps worst-case lag.
- **Settings screen for API key**: the packaged app will need an in-app
  Settings → API Key input for users without a `.env` (not needed for personal
  use).

## Conventions

- Python 3.10+ syntax (`int | None`), type hints on public functions.
- Configuration lives in `config.py` dataclasses; never hardcode model names or
  timings in modules.
- API keys only in `.env` (gitignored) — never commit or print them.
- Scripts reconfigure stdout to UTF-8 and disable Windows QuickEdit.
- Every queue crossing a thread boundary must be bounded, and every rejected
  item must be **counted**, never silently dropped.
