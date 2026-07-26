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

import glob
import os
import site
import subprocess
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


def _nvidia_lib_dirs() -> list:
    """Lib dirs of pip-installed CUDA/cuDNN packages (site-packages/nvidia/*/lib).

    These are NOT on the default dynamic-loader path, so CTranslate2 can't find
    cuDNN at runtime unless we help it (see _preload_nvidia_libs).
    """
    dirs = []
    bases = list(site.getsitepackages())
    try:
        bases.append(site.getusersitepackages())
    except Exception:
        pass
    for base in bases:
        for sub in ("nvidia/cudnn/lib", "nvidia/cublas/lib"):
            d = os.path.join(base, sub)
            if os.path.isdir(d) and d not in dirs:
                dirs.append(d)
    return dirs


def _preload_nvidia_libs() -> None:
    """Make pip-installed cuDNN/cuBLAS discoverable for in-process GPU inference.

    Changing LD_LIBRARY_PATH after the process has started does not affect dlopen,
    but an RTLD_GLOBAL preload does — so GPU transcription works out of the box
    (just `pip install nvidia-cudnn-cu12`) with no environment fiddling.
    """
    import ctypes
    for d in _nvidia_lib_dirs():
        for lib in sorted(glob.glob(os.path.join(d, "*.so*"))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_GPU_CANARY = (
    "import numpy as np;"
    "from faster_whisper import WhisperModel;"
    "m=WhisperModel('tiny',device='cuda',compute_type='float16');"
    "list(m.transcribe(np.zeros(16000,dtype=np.float32))[0]);"
    "print('GPUOK')"
)


def _gpu_works() -> bool:
    """Return True only if GPU transcription actually runs end-to-end.

    A broken CUDA/cuDNN stack (e.g. cuDNN 9 not installed) does not raise a
    catchable exception — it SEGFAULTS mid-transcription and core-dumps the
    process. So we validate the GPU in a throwaway subprocess with a tiny model:
    if the child crashes, the parent survives and falls back to CPU.
    """
    if not _cuda_available():
        return False
    # The canary is a fresh process, so LD_LIBRARY_PATH IS honored at its startup —
    # point it at the pip-installed cuDNN/cuBLAS so the health check is accurate.
    env = dict(os.environ)
    extra = os.pathsep.join(_nvidia_lib_dirs())
    if extra:
        env["LD_LIBRARY_PATH"] = extra + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    try:
        r = subprocess.run([sys.executable, "-c", _GPU_CANARY],
                           capture_output=True, timeout=180, env=env)
        return r.returncode == 0 and b"GPUOK" in r.stdout
    except Exception:
        return False


def get_whisper_model(model_size: str = "medium", device: str = "auto"):
    """Build a faster-whisper model. device: 'auto' | 'cuda' | 'cpu'.

    Returns (model, resolved_device), or (None, None) if faster-whisper is not
    installed. Under 'auto'/'cuda', GPU is used only if a subprocess health check
    confirms GPU transcription actually works; otherwise it falls back to CPU
    with a prominent warning.
    """
    if not _HAS_WHISPER:
        print("prepare-1: faster-whisper not installed; audio files will be skipped. "
              "Install: pip install faster-whisper", file=sys.stderr)
        return None, None

    want_gpu = device in ("auto", "cuda")
    use_gpu = False
    if want_gpu:
        if not _cuda_available():
            if device == "cuda":
                print("⚠  Whisper: device='cuda' requested but no CUDA GPU detected; "
                      "using CPU.", file=sys.stderr)
        elif _gpu_works():
            use_gpu = True
        else:
            print("⚠  Whisper: CUDA GPU present but GPU transcription failed a health "
                  "check (most often missing cuDNN 9) — using CPU to avoid a crash. "
                  "Install cuDNN 9 (e.g. `pip install nvidia-cudnn-cu12`) for GPU speed.",
                  file=sys.stderr)

    if use_gpu:
        try:
            _preload_nvidia_libs()  # make cuDNN/cuBLAS resolvable in this process
            return _WhisperModel(model_size, device="cuda", compute_type="float16"), "cuda"
        except Exception as ex:
            print(f"⚠  Whisper: GPU init failed ({type(ex).__name__}: {ex}); using CPU.",
                  file=sys.stderr)

    if want_gpu:
        print(f"⚠  Whisper running on CPU. The '{model_size}' model is slow on CPU "
              f"(often several x real-time); set a smaller prepare.audio.model or "
              f"install cuDNN 9 for GPU.", file=sys.stderr)
    return _WhisperModel(model_size, device="cpu", compute_type="int8"), "cpu"


def convert_audio(inpath: Path, md_path: Path, whisper_model):
    """Transcribe an audio file locally via faster-whisper.

    Returns (ok, size, method, note). ok is False if whisper is unavailable
    (whisper_model is None) or transcription raises.
    """
    inpath = Path(inpath)
    if whisper_model is None:
        return False, 0, "whisper-unavailable", None
    try:
        with _model_lock:
            segments, info = whisper_model.transcribe(str(inpath))
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as ex:
        print(f"  prepare-1 whisper failed: {inpath.name}: {type(ex).__name__}: {ex}",
              file=sys.stderr)
        return False, 0, "whisper-error", None

    from pipeline.prepare_reflow import reflow_pipeline_text

    lang = getattr(info, "language", "?")
    dur = getattr(info, "duration", 0.0) or 0.0
    header = (f"# Audio transcript: {inpath.name}\n\n"
              f"**Language:** {lang}  |  **Duration:** {dur:.0f}s  "
              f"|  *Transcribed locally via faster-whisper*\n\n---\n\n")
    body = reflow_pipeline_text(transcript) if transcript else "*(no speech detected)*"
    content = header + body
    md_path.write_text(content, encoding="utf-8")
    note = ("audio transcribed via whisper" if transcript
            else "audio: no speech detected")
    return True, len(content), "whisper", note
