# Chinese Lecture Interpreter

Live Mandarin → English translation for lectures. The app listens to the
lecture, transcribes the Chinese speech, and shows natural English in real
time — with your lecture PDF as context for better terminology.

Built for STEM students in Mandarin-taught classes (Windows 10/11, x64).

---

## Quick Start

### 1. Install

**Option A — download the app (recommended):**

1. Download `ChineseLectureInterpreter-v0.1.0-Windows-x64.zip` from the
   [Releases page](../../releases).
2. Extract the zip anywhere (e.g. `Desktop\ChineseLectureInterpreter`).

**Option B — run from source:**

```powershell
py -m pip install -r requirements.txt
```

### 2. Configure your API key (one-time setup)

1. Get a free API key at <https://console.groq.com/keys>.
2. Copy `.env.example` to a new file named `.env` (same folder as
   `ChineseLectureInterpreter.exe`, or the repo root for source runs).
3. Paste your key into it:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

The `.env` file stays on your machine — it is never bundled or shared.

### 3. Open the app

Double-click `ChineseLectureInterpreter.exe` (or `py scripts\app.py` from
source). No terminal needed.

### 4. Drop today's lecture PDF

Drag the lecture slides onto the window (optional but recommended — the
extracted text is used as translation context). Scanned/image-only PDFs
aren't extractable; the app warns and continues without them.

### 5. Start listening

1. Pick your microphone in the **Mic** dropdown and press **Test mic** —
   speak a few words; the level meter should bounce and the `vad` reading
   should rise above ~0.5. If it doesn't move, try another device.
2. Press **Start**. Chinese appears first, English fills in below with a
   delay badge.
3. Tick **Subtitle bar** for a floating, always-on-top caption bar over your
   slides.
4. Close the window to stop. A summary and session log are saved to
   `storage\session_logs\`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| No cards when you speak | Use **Test mic** to verify the selected microphone actually picks up your voice; try another device in the dropdown |
| `START FAILED: GROQ_API_KEY...` | `.env` missing or key not pasted — see step 2 |
| `RATE LIMIT` badges | Groq free-tier limit hit; wait a moment — the next sentence recovers automatically |
| Garbled or missing slides text | Scanned PDF: the app falls back to default prompts; a keyboard-typed glossary (`--glossary`) can still supply terms |
| Want captions over slides | Tick **Subtitle bar** in the window (or launch with `--overlay`) |

---

## Development

Running from source, build instructions, repo layout, progress checklist,
test matrix, and commit conventions: see [AGENTS.md](AGENTS.md).

```powershell
py scripts\app.py                  # main window (press Start)
py scripts\app.py --demo           # UI smoke test, no mic/API
py scripts\app.py --list-devices   # enumerate microphones
.\build_exe.ps1                    # build the Windows .exe
```
