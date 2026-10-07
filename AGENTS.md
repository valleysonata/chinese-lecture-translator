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
| `providers/base.py` | `ASREngine`, `TranslationEngine`, `ContextEngine` ABCs |
| `providers/groq_asr.py` | Groq Whisper engine + `_filter_hallucinations()` |
| `providers/groq_translation.py` | Groq LLM translation engine + `SYSTEM_PROMPT` |
| `scripts/` | Per-phase live test harnesses (`test_phase1/2/3.py`, `verify_automatic_segmentation.py`) |
| `tests/` | Offline regression tests (no microphone required) |
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
- [ ] **Phase 4: Slide Context & Course Glossary Integration**
- [ ] **Phase 5: PyQt6 Transparent Overlay UI**
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
```

Live / API-dependent:

```powershell
py scripts\test_phase1.py --list-devices   # mic + VAD only
py scripts\test_phase2.py --file <wav>     # ASR only
py scripts\test_phase3.py --benchmark-cases  # translation quality
py scripts\test_phase3.py --duration 120     # full live pipeline (logs to storage/session_logs/)
```

After a live run, review the JSONL session log for `drop` events, latencies, and
translation quality — not just the console output.

## Known Backlog (not yet implemented)

- **Audio queue catch-up**: `AudioCapture.audio_queue` is unbounded and the main
  loop consumes one 32 ms chunk per iteration; add stale-audio drop policy so lag
  cannot accumulate during long sessions.
- **Rate-limit (429) handling**: report and recover from Groq TPM limits explicitly;
  drops caused by SDK retries should be visible in metrics.
- **Truncation guard**: check `finish_reason` against `max_tokens=100` so long
  segments are not silently cut mid-sentence.
- **12 s force-split overlap**: mid-sentence splits produce fragments; consider
  overlap or split-at-lowest-probability.
- **Pipeline extraction**: `scripts/test_phase{1,2,3}.py` duplicate the same loop;
  extract a shared `core/pipeline.py` before Phase 4/5.
- **Blacklist word boundaries**: phrase replacement can corrupt legitimate speech
  containing a blacklist word (e.g. `收藏`); consider stricter matching.

## Conventions

- Python 3.10+ syntax (`int | None`), type hints on public functions.
- Configuration lives in `config.py` dataclasses; never hardcode model names or
  timings in modules.
- API keys only in `.env` (gitignored) — never commit or print them.
- Scripts reconfigure stdout to UTF-8 and disable Windows QuickEdit.
- Every queue crossing a thread boundary must be bounded, and every rejected
  item must be **counted**, never silently dropped.
