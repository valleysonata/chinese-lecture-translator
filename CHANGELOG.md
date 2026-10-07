# Changelog

All notable changes to the Chinese Lecture Interpreter.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-10-08

First public beta. Windows x64, packaged as
`ChineseLectureInterpreter-v0.1.0-Windows-x64.zip`.

### Added

- **Main window app** — normal draggable/closable window: press Start and read
  live English translations while the Chinese line appears first, then the
  English fills in below with a per-line delay badge. Errors (rate limits,
  dropped items) show as badges instead of vanishing.
- **Microphone picker + Test mic** — choose the input device in the window and
  press **Test mic** for a live preview (level meter + VAD speech probability)
  so a muted or dead microphone is caught before class starts.
- **Lecture PDF context** — drop today's slides onto the window; extracted
  text is injected into the translation prompt for better terminology.
  Scanned/image-only PDFs are reported and skipped gracefully.
- **Optional subtitle bar** — an always-on-top, click-through caption bar for
  overlaying your slides (off by default; enable with the checkbox or
  `--overlay`).
- **Manual glossary (advanced)** — optional one-term-per-line file injected
  into both ASR and translation prompts (`glossary.example.txt`).
- **Anti-hallucination filtering** — Whisper prompt-echo/blacklist filtering
  and conservative translation ("reconstruct grammar aggressively, reconstruct
  meaning conservatively").
- **Session logs** — every live run writes a JSONL log (transcripts,
  translations, latencies, drops) to `storage\session_logs\` for review.
- **Packaged Windows exe** — PyInstaller build that runs without Python or a
  terminal; the API key stays in an external `.env` (`.env.example` ships
  with the release, your key is never bundled).

### Known limitations

- Groq free-tier rate limits are reported but not retried; the next sentence
  recovers automatically.
- Very long sentences may be truncated at 100 tokens (marked with `…`).
- No in-app Settings screen for the API key yet — it lives in `.env` next to
  the exe.
