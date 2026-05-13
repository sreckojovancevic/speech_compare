# Voice Compare Tool

A simple Python tool for comparing voices from songs using neural voice embeddings.

The script:
- converts audio to mono 16kHz WAV
- optionally trims audio segments
- isolates vocals using Demucs
- removes most instrumental content
- generates neural voice embeddings
- compares similarity using cosine distance

This is useful for:
- experimental singer comparison
- voice similarity research
- audio forensics experiments
- checking whether two songs may contain the same vocalist

---

# Features

- Supports:
  - MP3
  - WAV
  - FLAC
  - M4A
  - most ffmpeg-compatible formats

- Converts everything to:
  - mono
  - 16kHz PCM WAV

- Vocal isolation with Demucs

- Neural embeddings using Resemblyzer

- Cosine similarity scoring

---

# Requirements

## System packages

Install ffmpeg:

### Ubuntu / Debian

```bash
sudo apt install ffmpeg python3-venv python3-full -y
```

---

# Python Installation

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip

pip install "torch<2.4" "torchaudio<2.4"
pip install "huggingface_hub<0.26"
pip install "speechbrain>=0.5.16,<1.0"

pip install demucs
pip install resemblyzer
pip install librosa numpy soundfile
```

---

# Usage

Basic usage:

```bash
python compare_local.py song1.mp3 song2.mp3
```

Compare only specific sections:

```bash
python compare_local.py song1.mp3 song2.mp3 \
    --start1 30 --end1 60 \
    --start2 45 --end2 75
```

Disable vocal separation:

```bash
python compare_local.py song1.mp3 song2.mp3 --no-separate
```

---

# Example Output

```text
============================================================
  Cosine similarity: 0.9077
  Verdict: Almost certainly the SAME singer.
============================================================
```

---

# Similarity Score Guide

For sung vocals (less reliable than speech):

| Score | Interpretation |
|---|---|
| > 0.80 | Almost certainly same singer |
| 0.70 - 0.80 | Likely same singer |
| 0.55 - 0.70 | Ambiguous |
| 0.40 - 0.55 | Likely different singers |
| < 0.40 | Almost certainly different singers |

---

# Important Notes

This tool is experimental.

Resemblyzer is designed primarily for:
- speech recognition
- speaker embeddings

not heavily processed singing vocals.

Modern music production can affect results:
- autotune
- vocal layering
- reverb
- mastering
- harmonics overlap
- backing vocals
- instrumental bleed

False positives are possible.

For better accuracy:
- use acoustic/live recordings
- use isolated vocals
- compare shorter clean phrases
- avoid choruses with harmonies
- avoid heavy effects

---

# Recommended Workflow

Best results usually come from:

- acoustic/live recordings
- solo vocal sections
- 15–30 second clips
- minimal background instrumentation

---

# License

MIT License

---

# Disclaimer

This tool is intended for:
- research
- experimentation
- educational purposes

It should not be considered a definitive biometric identification system.
