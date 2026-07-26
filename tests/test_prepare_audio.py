"""Tests for local Whisper audio transcription (prepare_audio)."""
from __future__ import annotations

import shutil
import subprocess

import pytest

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


def test_convert_audio_reflows_long_transcript(tmp_path):
    """convert_audio calls reflow_pipeline_text on the transcript itself (not
    just via process_file's separate guard) -- a long, unpunctuated single
    segment must come out wrapped, proving that line actually fires rather
    than being a no-op like every existing short-transcript fixture here."""
    long_text = " ".join(["word"] * 100)  # 499 chars, one segment, no punctuation
    assert len(long_text) > 300

    class FakeModel:
        def transcribe(self, path):
            return [_Seg(long_text)], _Info()

    ok, size, method, note = pa.convert_audio(tmp_path / "a.mp3", tmp_path / "a.md", FakeModel())
    md = (tmp_path / "a.md").read_text()
    assert ok

    body = md.split("---\n\n", 1)[1]
    assert "\n" in body
    for line in body.split("\n"):
        assert len(line) <= 120
    # Confirm no words were lost or altered by the wrap.
    assert body.split() == long_text.split()


def test_vocabulary_reaches_transcribe_as_initial_prompt(tmp_path):
    captured = {}

    class FakeSegment:
        text = "hello"

    class FakeModel:
        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return [FakeSegment()], _Info()

    src = tmp_path / "a.wav"
    src.write_bytes(b"fake audio")
    pa.convert_audio(src, tmp_path / "a.md", FakeModel(),
                     vocabulary=["Jerry Carill", "McBride"])

    assert captured.get("initial_prompt") == "Jerry Carill, McBride"


def test_empty_vocabulary_passes_no_initial_prompt(tmp_path):
    captured = {}

    class FakeSegment:
        text = "hello"

    class FakeModel:
        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return [FakeSegment()], _Info()

    src = tmp_path / "a.wav"
    src.write_bytes(b"fake audio")
    pa.convert_audio(src, tmp_path / "a.md", FakeModel())

    assert "initial_prompt" not in captured


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


AUDIO_FIXTURE_TEXT = "The quick brown fox jumps over the lazy dog."


@pytest.fixture(scope="module")
def synthesized_speech_wav(tmp_path_factory):
    """Short offline-synthesized speech clip via espeak-ng -- no network,
    fully reproducible, no licensing concerns (unlike a downloaded sample)."""
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not installed")
    out_dir = tmp_path_factory.mktemp("audio_fixture")
    wav_path = out_dir / "speech.wav"
    subprocess.run(
        ["espeak-ng", "-w", str(wav_path), AUDIO_FIXTURE_TEXT],
        check=True, capture_output=True,
    )
    return wav_path


@pytest.mark.integration
def test_convert_audio_real_transcription(synthesized_speech_wav, tmp_path):
    """End-to-end: real faster-whisper transcription (the pipeline's actual
    default model, per config.yaml's prepare.audio.model) against a real
    synthesized speech clip -- not mocked. First run on a machine without
    the 'medium' model cached will download it (~1.5GB)."""
    try:
        model, device = pa.get_whisper_model("medium", "auto")
    except Exception as ex:
        pytest.skip(f"could not load faster-whisper 'medium' model (no network/cache?): {ex}")
    if model is None:
        pytest.skip("faster-whisper not installed")

    ok, size, method, note = pa.convert_audio(
        synthesized_speech_wav, tmp_path / "out.md", model
    )
    md = (tmp_path / "out.md").read_text()

    assert ok and method == "whisper"
    transcribed = md.lower()
    assert "fox" in transcribed
    assert "dog" in transcribed
