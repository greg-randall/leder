"""Tests for local Whisper audio transcription (prepare_audio)."""
from __future__ import annotations

import pipeline.prepare_audio as pa


class _Seg:
    def __init__(self, text):
        self.text = text


class _Info:
    language = "en"
    duration = 12.0


def test_convert_audio_writes_transcript(tmp_path):
    class FakeModel:
        def transcribe(self, path):
            return [_Seg(" Hello "), _Seg("world ")], _Info()

    ok, size, method, note = pa.convert_audio(tmp_path / "a.mp3", tmp_path / "a.md", FakeModel())
    md = (tmp_path / "a.md").read_text()
    assert ok and method == "whisper"
    assert "Hello world" in md
    assert "en" in md
    assert note and "whisper" in note


def test_convert_audio_unavailable(tmp_path):
    ok, size, method, note = pa.convert_audio(tmp_path / "a.mp3", tmp_path / "a.md", None)
    assert not ok and method == "whisper-unavailable" and note is None


def test_convert_audio_no_speech(tmp_path):
    class FakeModel:
        def transcribe(self, path):
            return [], _Info()

    ok, size, method, note = pa.convert_audio(tmp_path / "a.mp3", tmp_path / "a.md", FakeModel())
    md = (tmp_path / "a.md").read_text()
    assert ok and "no speech detected" in md


def test_get_whisper_model_cpu_fallback_warns(monkeypatch, capsys):
    monkeypatch.setattr(pa, "_HAS_WHISPER", True)
    monkeypatch.setattr(pa, "_cuda_available", lambda: False)
    built = {}

    class FakeWM:
        def __init__(self, model, device, compute_type):
            built["device"] = device
            built["compute_type"] = compute_type

    monkeypatch.setattr(pa, "_WhisperModel", FakeWM)
    model, dev = pa.get_whisper_model("medium", "auto")
    assert dev == "cpu" and built["device"] == "cpu" and built["compute_type"] == "int8"
    assert "CPU" in capsys.readouterr().err


def test_get_whisper_model_gpu(monkeypatch):
    monkeypatch.setattr(pa, "_HAS_WHISPER", True)
    monkeypatch.setattr(pa, "_gpu_works", lambda: True)
    monkeypatch.setattr(pa, "_preload_nvidia_libs", lambda: None)

    class FakeWM:
        def __init__(self, model, device, compute_type):
            self.device = device

    monkeypatch.setattr(pa, "_WhisperModel", FakeWM)
    model, dev = pa.get_whisper_model("medium", "auto")
    assert dev == "cuda"


def test_get_whisper_model_gpu_healthcheck_fails_falls_back(monkeypatch, capsys):
    """CUDA present but GPU health check fails (e.g. missing cuDNN) -> CPU + warning."""
    monkeypatch.setattr(pa, "_HAS_WHISPER", True)
    monkeypatch.setattr(pa, "_cuda_available", lambda: True)
    monkeypatch.setattr(pa, "_gpu_works", lambda: False)
    monkeypatch.setattr(pa, "_WhisperModel", lambda *a, **k: object())
    model, dev = pa.get_whisper_model("medium", "auto")
    assert dev == "cpu"
    assert "health check" in capsys.readouterr().err


def test_get_whisper_model_cuda_requested_no_gpu_warns(monkeypatch, capsys):
    monkeypatch.setattr(pa, "_HAS_WHISPER", True)
    monkeypatch.setattr(pa, "_cuda_available", lambda: False)
    monkeypatch.setattr(pa, "_WhisperModel", lambda *a, **k: object())
    model, dev = pa.get_whisper_model("small", "cuda")
    assert dev == "cpu"
    assert "no CUDA GPU" in capsys.readouterr().err


def test_get_whisper_model_missing_dep(monkeypatch):
    monkeypatch.setattr(pa, "_HAS_WHISPER", False)
    model, dev = pa.get_whisper_model()
    assert model is None and dev is None
