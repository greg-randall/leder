"""Tests for pipeline/tools/validate_excerpt.py."""
import json
import subprocess
import sys


def _run(*args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "pipeline/tools/validate_excerpt.py", *args],
        capture_output=True, text=True,
    )


def _output(cp: "subprocess.CompletedProcess[str]") -> dict:
    return json.loads(cp.stdout)


def test_exact_match(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("The road commission issued a 68-page notice of violation.")
    cp = _run(str(src), "road commission issued a 68-page notice")
    assert cp.returncode == 0
    out = _output(cp)
    assert out["found"] is True
    assert out["similarity"] == 1.0
    assert "road commission issued a 68-page notice" in out["actual_text"]
    assert out["offset"][0] >= 0 and out["offset"][1] > out["offset"][0]


def test_exact_match_case_insensitive(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("The Road Commission issued a notice.")
    cp = _run(str(src), "road commission issued")
    assert cp.returncode == 0
    out = _output(cp)
    assert out["found"] is True and out["similarity"] == 1.0


def test_fuzzy_match_on_garbled_candidate(tmp_path):
    src = tmp_path / "doc.md"
    # Whisper errors: "road" for "railroad", "fe February" for "February"
    src.write_text(
        "the road commission issued a 68 page notice of violation "
        "for the McBride Was facility on fe February 7th, 2025."
    )
    # Candidate is the LLM's polished version
    cp = _run(str(src), "the Railroad Commission issued a 68-page notice of violation")
    assert cp.returncode == 0
    out = _output(cp)
    assert out["found"] is True
    assert out["similarity"] < 1.0
    assert out["similarity"] >= 0.6
    # The returned actual_text must be from the source (whisper errors preserved)
    assert "road commission" in out["actual_text"].lower() or "fe february" in out["actual_text"].lower()


def test_no_match(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("completely unrelated text about zoning permits")
    cp = _run(str(src), "railroad commission issued a notice")
    assert cp.returncode == 1
    out = _output(cp)
    assert out["found"] is False


def test_file_not_found(tmp_path):
    cp = _run(str(tmp_path / "nonexistent.md"), "some text")
    assert cp.returncode == 1
    out = _output(cp)
    assert out["found"] is False
    assert "error" in out


def test_empty_candidate(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("some content")
    cp = _run(str(src), "")
    assert cp.returncode == 1
    out = _output(cp)
    assert out["found"] is False
    assert "empty" in out.get("error", "")


def test_large_document_handled(tmp_path):
    """Chunk-scoring should complete on a large document, not hang."""
    src = tmp_path / "large.md"
    # Build a document where a needle-adjacent string appears near the end
    filler = "unrelated discussion about city business. " * 800  # ~40K chars
    needle = "the road commission issued a 68 page notice"
    src.write_text(filler + needle)
    cp = _run(str(src), "railroad commission issued 68 page notice")
    assert cp.returncode == 0
    out = _output(cp)
    assert out["found"] is True
