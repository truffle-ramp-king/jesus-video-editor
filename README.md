# Jesus Video Editor

Turn a source video into a tightly cut Bible short. The editor gives Gemini a timestamped transcript, shot timeline and sampled frames, asks it to select the essential story beats, and uses FFmpeg to render those intervals at the original playback speed. You get an MP4 and an editable JSON cut list.

## Install

Install Git, uv, and FFmpeg (including ffprobe). Clone the repository and sync the tested dependencies with Python 3.11:

```sh
git clone https://github.com/truffle-ramp-king/jesus-video-editor.git
cd jesus-video-editor
uv sync --locked --python 3.11
cp .env.example .env
```

Set `GEMINI_API_KEY` and `ELEVENLABS_API_KEY` in the local `.env` file, or export them in your shell. The default planner is `gemini-4-argon` and the default prompt is `tough`. Your Gemini account needs access to the selected model. Planning sends the source audio to ElevenLabs for transcription, then sends the transcript, shot timeline and frame sheets to Google Gemini.

## Edit

```sh
uv run jesus-video-edit input.mp4 --output edited.mp4
```

To run the installed command directly, activate the environment with `source .venv/bin/activate` (PowerShell: `.venv\Scripts\Activate.ps1`). From another directory, use `jesus-video-edit input.mp4 --env-file /path/to/.env --output edited.mp4`. Shell variables take precedence over the selected file.

Each invocation prints the video and run directory. By default, a new plan is saved as `runs/<run-id>/edit-plan.json` alongside the model receipt and analysis artifacts. To give the plan a convenient filename and render alternative versions:

```sh
jesus-video-edit input.mp4 cuts.json --output plain.mp4
jesus-video-edit input.mp4 --from-plan cuts.json --deduplicate --output styled.mp4
```

A saved plan renders locally without API credentials. Plain rendering keeps the source framing and soundtrack. The optional `--deduplicate` applies a 1080×1920 mirror, centered punch-in, color grade and two moving white rectangles at 2.5% opacity. Use `--prompt-style old` to compare the earlier editorial brief or `--duration SECONDS` to request a runtime. See `jesus-video-edit --help` for other options, including the legacy background compositor. The original `trailer-edit` command remains available.

## Test

```sh
uv run pytest -q
uv run ruff check .
```
