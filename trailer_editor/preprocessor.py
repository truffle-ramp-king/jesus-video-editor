"""Deterministic pre-processing: audio extraction, scene detection,
librosa analysis, and ElevenLabs Scribe v2 transcription."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from elevenlabs import ElevenLabs
from scenedetect import ContentDetector, SceneManager, open_video

from trailer_editor.duration_policy import target_duration_range

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DialogueBlock:
    speaker: str
    start: float  # seconds
    end: float
    text: str

@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str


@dataclass
class ShotBoundary:
    start: float
    end: float
    mid: float
    mid_frame: int


@dataclass
class AudioAnalysis:
    rms: np.ndarray  # shape (n_windows,)
    rms_times: np.ndarray  # centre time of each window
    sr: int
    silence_intervals: list[tuple[float, float]]
    peak_intervals: list[tuple[float, float]]


@dataclass
class PreprocessedData:
    video_path: Path
    audio_path: Path
    duration: float
    fps: float
    dialogue: list[DialogueBlock]
    shots: list[ShotBoundary]
    audio: AudioAnalysis
    words: list[Word]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_command(
    cmd: list[str], check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return result


def probe_duration(video_path: Path) -> float:
    r = run_command(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ]
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def has_audio_stream(video_path: Path) -> bool:
    result = run_command(["ffprobe", "-v", "error", "-select_streams", "a:0",
                          "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)])
    return bool(result.stdout.strip())


def _probe_fps(video_path: Path) -> float:
    r = run_command(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json",
            str(video_path),
        ]
    )
    rate_str = json.loads(r.stdout)["streams"][0]["r_frame_rate"]
    num, den = rate_str.split("/")
    return int(num) / int(den)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_audio(video_path: Path, work_dir: Path) -> Path:
    """Extract audio as 16 kHz mono WAV (optimal for Scribe & librosa)."""
    out = work_dir / "audio.wav"
    logger.info("Extracting audio to %s", out)
    run_command([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out),
    ])
    return out


def detect_shots(video_path: Path, threshold: float = 27.0) -> list[ShotBoundary]:
    """Run ContentDetector via pyscenedetect."""
    logger.info("Detecting shot boundaries …")
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video)
    scene_list = sm.get_scene_list(start_in_scene=True)

    fps = video.frame_rate
    shots: list[ShotBoundary] = []
    for (start_f, end_f) in scene_list:
        start_t = start_f.get_seconds()
        end_t = end_f.get_seconds()
        mid_f = (start_f.frame_num + end_f.frame_num) // 2
        mid_t = mid_f / fps
        shots.append(ShotBoundary(start=start_t, end=end_t, mid=mid_t, mid_frame=mid_f))
    logger.info("Detected %d shots.", len(shots))
    return shots


def analyze_audio(audio_path: Path, hop_len: int = 512) -> AudioAnalysis:
    """Compute RMS energy, silence intervals, and peak intervals."""
    logger.info("Analysing audio …")
    y, sr = librosa.load(str(audio_path), sr=None)
    rms = librosa.feature.rms(y=y, hop_length=hop_len)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)

    # Normalise 0-1
    rms_norm = rms / (rms.max() + 1e-9)

    # Silence: below 3 % of max
    silence_mask = rms_norm < 0.03
    silence_intervals = _mask_to_intervals(times, silence_mask)

    # Peaks: above 80 % of max
    peak_mask = rms_norm > 0.80
    peak_intervals = _mask_to_intervals(times, peak_mask)

    logger.info(
        "Audio analysis: %d silence zones, %d peak zones.",
        len(silence_intervals), len(peak_intervals),
    )
    return AudioAnalysis(
        rms=rms_norm,
        rms_times=times,
        sr=sr,
        silence_intervals=silence_intervals,
        peak_intervals=peak_intervals,
    )


def transcribe_scribe(
    audio_path: Path,
    api_key: str,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Transcribe via ElevenLabs Scribe v2 SDK with word timestamps + diarization."""
    logger.info("Calling ElevenLabs Scribe v2 …")

    client = ElevenLabs(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        result = client.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v2",
            diarize=True
        )

    # The SDK returns a TranscribeResult (dict-like object).
    # Extract the raw dict to be resilient across SDK versions.
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif isinstance(result, dict):
        data = result
    else:
        data = dict(result)

    if artifact_path:
        artifact_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    words = data.get("words", [])
    if not words:
        logger.warning("Scribe v2 returned no words — returning empty dialogue.")

    return data

    # Scribe v2 may return words with or without speaker_id.
    # If diarization was requested but speaker_id is missing on words,
    # check for an "utterances" top-level key and fall back gracefully.


def extract_frame(video_path: Path, time_s: float, out_path: Path) -> None:
    """Extract a single JPEG frame at *time_s*."""
    time_float = float(time_s)
    run_command([
        "ffmpeg", "-y",
        "-ss", f"{time_float:.6f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(out_path),
    ])


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _mask_to_intervals(
    times: np.ndarray, mask: np.ndarray
) -> list[tuple[float, float]]:
    """Convert a boolean mask over *times* to a list of (start, end) tuples."""
    intervals: list[tuple[float, float]] = []
    in_region = False
    start = 0.0
    for i, m in enumerate(mask):
        if m and not in_region:
            in_region = True
            start = times[i]
        elif not m and in_region:
            in_region = False
            intervals.append((start, times[i]))
    if in_region:
        intervals.append((start, times[-1]))
    return intervals

def _words_from_dicts(words: list[dict[str, Any]]) -> list[Word]:
    """Convert a list of raw word dictionaries directly into a list of Word objects."""
    return [
        Word(
            start=float(w["start"]),
            end=float(w["end"]),
            text=w["text"],
            speaker=w.get("speaker_id") or "Speaker_0"
        )
        for w in words if w["text"].strip()
    ]


def _blocks_from_words(data: dict[str, Any]) -> list[DialogueBlock]:
    """Group consecutive words sharing the same speaker_id into DialogueBlocks."""
    words: list[dict[str, Any]] = data.get("words", [])
    if not words:
        return []
    utterances = data.get("utterances")
    if utterances and not any(w.get("speaker_id") for w in words):
        return _blocks_from_utterances(utterances)
    blocks: list[DialogueBlock] = []
    cur_speaker = words[0].get("speaker_id") or "Speaker_0"
    cur_start = float(words[0]["start"])
    cur_texts: list[str] = [words[0]["text"]]

    for w in words[1:]:
        spk = w.get("speaker_id") or cur_speaker
        if spk != cur_speaker:
            blocks.append(DialogueBlock(
                speaker=cur_speaker,
                start=cur_start,
                end=float(w["start"]),
                text=" ".join(cur_texts),
            ))
            cur_speaker = spk
            cur_start = float(w["start"])
            cur_texts = [w["text"]]
        else:
            cur_texts.append(w["text"])

    blocks.append(DialogueBlock(
        speaker=cur_speaker,
        start=cur_start,
        end=float(words[-1]["end"]),
        text=" ".join(cur_texts),
    ))
    logger.info("Transcription yielded %d dialogue blocks (from words).", len(blocks))
    return blocks


def _blocks_from_utterances(
    utterances: list[dict[str, Any]],
) -> list[DialogueBlock]:
    """Build DialogueBlocks directly from the utterances list if available."""
    blocks: list[DialogueBlock] = []
    for utt in utterances:
        spk = utt.get("speaker_id", "Speaker_0")
        text = utt.get("text", "")
        # Prefer explicit start/end on the utterance; fall back to words
        if "start" in utt and "end" in utt:
            start = float(utt["start"])
            end = float(utt["end"])
        else:
            inner_words = utt.get("words", [])
            if not inner_words:
                continue
            start = float(inner_words[0]["start"])
            end = float(inner_words[-1]["end"])
            text = text or " ".join(w["text"] for w in inner_words)
        if text.strip():
            blocks.append(DialogueBlock(speaker=spk, start=start, end=end, text=text))
    logger.info("Transcription yielded %d dialogue blocks (from utterances).", len(blocks))
    return blocks


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def preprocess(
    video_path: Path,
    work_dir: Path,
    elevenlabs_key: str,
    *,
    source_duration: float | None = None,
    cached_transcript: Path | None = None,
) -> PreprocessedData:
    """Run every deterministic pre-processing step and return bundled data."""
    duration = source_duration if source_duration is not None else probe_duration(video_path)
    target_duration_range(duration)
    with_audio = has_audio_stream(video_path)
    if with_audio:
        audio_path = extract_audio(video_path, work_dir)
    else:
        # The visual timeline expects a waveform; a silent reference keeps its ruler aligned.
        audio_path = work_dir / "audio.wav"
        run_command(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                     "-t", str(duration), str(audio_path)])
    shots = detect_shots(video_path)
    audio_analysis = analyze_audio(audio_path)
    transcript = work_dir / "transcript.json"
    if cached_transcript is not None:
        data = json.loads(cached_transcript.read_text())
    elif with_audio:
        data = transcribe_scribe(audio_path, elevenlabs_key, transcript)
    else:
        data = {"words": []}
        transcript.write_text(json.dumps(data), encoding="utf-8")
    words = data.get("words",[])
    dialogue = _blocks_from_words(data)
    parsed_words = _words_from_dicts(words)
    fps = _probe_fps(video_path)
    return PreprocessedData(
        video_path=video_path,
        audio_path=audio_path,
        duration=duration,
        fps=fps,
        dialogue=dialogue,
        shots=shots,
        audio=audio_analysis,
        words=parsed_words,
    )
