"""Tests for the OCR->vision escalation helper."""
from __future__ import annotations

from unittest import mock

import pipeline.prepare_vision as pv


def test_count_real_words():
    assert pv.count_real_words("") == 0
    assert pv.count_real_words("a 1 2 3 %%%") == 0        # single letters / non-words
    assert pv.count_real_words("the quick brown fox") == 4


def test_needs_vision():
    assert pv.needs_vision("one two three", min_words=5) is True
    assert pv.needs_vision("alpha beta gamma delta epsilon zeta", min_words=5) is False


def test_encode_png_passthrough(tmp_path):
    # 1x1 PNG
    png = tmp_path / "x.png"
    png.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6300010000050001a5f645400000000049454e44ae426082"
    ))
    media, b64 = pv._encode_image(png)
    assert media == "image/png"
    assert isinstance(b64, str) and len(b64) > 0


def test_vision_extract_mocked(tmp_path, monkeypatch):
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    monkeypatch.setattr(pv, "_encode_image", lambda p: ("image/png", "QUJD"))

    class _Msg:
        content = "## Transcription\n(no text)\n\n## Description\nA test image."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    fake_client = mock.Mock()
    fake_client.chat.completions.create.return_value = _Resp()
    monkeypatch.setattr(pv, "_openai_client", lambda: fake_client)

    out = pv.vision_extract(png, model="gpt-4o-mini")
    assert "Transcription" in out
    assert fake_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"
