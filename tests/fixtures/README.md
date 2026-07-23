# Test fixtures

## sample_auto_captions.vtt

Real YouTube auto-generated captions, not synthetic. Pulled with `yt-dlp` to
get realistic messy structure for testing `convert_subtitle()` in
`pipeline/prepare_1_convert.py` — specifically its `<c>`-tag stripping and
rolling-caption dedup logic, which synthetic 2-line examples don't stress.

- Source: "Live: NASA News Conference on Moon rocket repairs and the future
  of Artemis program", uploaded by Spaceflight Now
  (https://www.youtube.com/watch?v=nIrg_QEv5rI), auto-generated English
  captions (not manually authored).
- Fetched with:
  `yt-dlp <url> --skip-download --write-auto-sub --sub-lang en --sub-format vtt`
- This file is the first 92 lines of the downloaded `.vtt` (~19 cues,
  roughly the 00:00:26–00:06:02 mark) — a representative excerpt, not the
  full transcript, cut on a clean cue boundary.
- Kept byte-for-byte as downloaded, including single-space " " payload
  lines (blank filler cues) that are meaningfully different from true blank
  lines (cue-block separators) in VTT syntax — don't hand-edit this file's
  whitespace.
