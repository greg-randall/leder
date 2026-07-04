#!/usr/bin/env python3
"""Local Whisper audio transcription (faster-whisper), GPU-first with CPU fallback.

Replaces MarkItDown's audio path, which transcribes via Google's free Web Speech
API over the network (privacy leak, English-default, mediocre, silently no-ops
without optional deps). This runs faster-whisper locally: it tries CUDA
(float16); if no GPU is available or CUDA init fails, it falls back to CPU (int8)
and prints a loud warning, because the larger models are slow on CPU.

faster-whisper decodes containers (.mp4/.m4a/.wma/.aac/.ogg/.flac) via bundled
PyAV/FFmpeg, so no separate ffmpeg call is needed. Model weights auto-download to
the local cache on first use.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

try:
    from faster_whisper import WhisperModel as _WhisperModel
    _HAS_WHISPER = True
except ImportError:
    _WhisperModel = None  # type: ignore[assignment]
    _HAS_WHISPER = False

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".aac", ".wma")

# Transcription is serialized: one shared model, and serializing avoids GPU
# memory contention under the file-level thread pool (audio files are few).
_model_lock = threading.Lock()


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def get_whisper_model(model_size: str = "medium", device: str = "auto"):
    """Build a faster-whisper model. device: 'auto' | 'cuda' | 'cpu'.

    Returns (model, resolved_device), or (None, None) if faster-whisper is not
    installed. Under 'auto'/'cuda', GPU is tried first; on unavailability or
    failure it falls back to CPU with a prominent warning.
    """
    if not _HAS_WHISPER:
        print("prepare-1: faster-whisper not installed; audio files will be skipped. "
              "Install: pip install faster-whisper", file=sys.stderr)
        return None, None

    want_gpu = device in ("auto", "cuda")
    if want_gpu and _cuda_available():
        try:
            model = _WhisperModel(model_size, device="cuda", compute_type="float16")
            return model, "cuda"
        except Exception as ex:
            print(f"⚠  Whisper: GPU init failed ({type(ex).__name__}: {ex}); "
                  f"falling back to CPU.", file=sys.stderr)
    elif device == "cuda":
        print("⚠  Whisper: device='cuda' requested but no CUDA GPU detected; "
              "falling back to CPU.", file=sys.stderr)

    if want_gpu:
        print(f"⚠  Whisper running on CPU (no usable GPU). The '{model_size}' model is "
              f"slow on CPU (often several x real-time). Set prepare.audio.model to a "
              f"smaller size (e.g. 'small'/'base') or run on a CUDA GPU for speed.",
              file=sys.stderr)
    model = _WhisperModel(model_size, device="cpu", compute_type="int8")
    return model, "cpu"


def convert_audio(inpath: Path, outpath: Path, whisper_model):
    """Transcribe an audio file locally via faster-whisper.

    Returns (ok, size, method, note). ok is False if whisper is unavailable
    (whisper_model is None) or transcription raises.
    """
    inpath = Path(inpath)
    if whisper_model is None:
        return False, 0, "whisper-unavailable", None

    md_path = outpath.with_suffix(".md")
    try:
        with _model_lock:
            segments, info = whisper_model.transcribe(str(inpath))
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as ex:
        print(f"  prepare-1 whisper failed: {inpath.name}: {type(ex).__name__}: {ex}",
              file=sys.stderr)
        return False, 0, "whisper-error", None

    lang = getattr(info, "language", "?")
    dur = getattr(info, "duration", 0.0) or 0.0
    header = (f"# Audio transcript: {inpath.name}\n\n"
              f"**Language:** {lang}  |  **Duration:** {dur:.0f}s  "
              f"|  *Transcribed locally via faster-whisper*\n\n---\n\n")
    body = transcript if transcript else "*(no speech detected)*"
    content = header + body
    md_path.write_text(content, encoding="utf-8")
    note = ("audio transcribed via whisper" if transcript
            else "audio: no speech detected")
    return True, len(content), "whisper", note
