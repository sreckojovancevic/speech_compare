#!/usr/bin/env python3
"""
compare_local.py
----------------
Compare voices in two local audio files (mp3, wav, m4a, flac, etc.)
using Resemblyzer's neural voice embeddings.

USAGE:
    python compare_local.py file1.mp3 file2.mp3
    python compare_local.py file1.mp3 file2.mp3 --start1 30 --end1 60 --start2 45 --end2 75
    python compare_local.py file1.mp3 file2.mp3 --no-separate    # skip vocal isolation

REQUIREMENTS:
    pip install resemblyzer librosa numpy soundfile demucs torch
    plus ffmpeg on your system PATH

INTERPRETING THE SCORE (for SUNG vocals — noisier than speech):
    > 0.80   Almost certainly the same singer.
    0.70-0.80  Likely the same singer.
    0.55-0.70  Ambiguous.
    0.40-0.55  Likely different singers.
    < 0.40   Almost certainly different singers.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return result


def check_tool(name):
    if shutil.which(name) is None:
        sys.exit(f"ERROR: '{name}' is not installed or not on PATH.")


def to_wav(in_path: Path, out_path: Path, start=None, end=None):
    """Convert any audio file to mono 16kHz WAV, optionally trimmed."""
    print(f"\n=== Converting {in_path.name} to WAV ===")
    cmd = ["ffmpeg", "-y", "-i", str(in_path)]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(out_path)]
    run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def isolate_vocals(in_path: Path, work_dir: Path) -> Path:
    print(f"\n=== Isolating vocals from {in_path.name} (takes a minute) ===")
    out_dir = work_dir / "demucs_out"
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-o", str(out_dir),
        str(in_path),
    ]
    run(cmd)
    vocals = list(out_dir.rglob(f"{in_path.stem}/vocals.wav"))
    if not vocals:
        sys.exit(f"ERROR: could not find vocals output for {in_path.name}")
    return vocals[0]


def embed(wav_path: Path):
    from resemblyzer import VoiceEncoder, preprocess_wav
    print(f"\n=== Embedding {wav_path.name} ===")
    wav = preprocess_wav(wav_path)
    encoder = VoiceEncoder()
    return encoder.embed_utterance(wav)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def verdict(score: float) -> str:
    if score > 0.80: return "Almost certainly the SAME singer."
    if score > 0.70: return "Likely the SAME singer."
    if score > 0.55: return "Ambiguous — could be same singer in different style, or different singers."
    if score > 0.40: return "Likely DIFFERENT singers."
    return "Almost certainly DIFFERENT singers."


def process_one(in_file, idx, args, work_dir):
    in_path = Path(in_file).expanduser().resolve()
    if not in_path.exists():
        sys.exit(f"ERROR: file not found: {in_path}")

    start = getattr(args, f"start{idx}")
    end = getattr(args, f"end{idx}")
    wav = to_wav(in_path, work_dir / f"converted_{idx}.wav", start, end)

    final = wav if args.no_separate else isolate_vocals(wav, work_dir / f"sep_{idx}")
    return embed(final)


def main():
    p = argparse.ArgumentParser(description="Compare voices in two local audio files.")
    p.add_argument("file1")
    p.add_argument("file2")
    p.add_argument("--start1", type=float, default=None)
    p.add_argument("--end1",   type=float, default=None)
    p.add_argument("--start2", type=float, default=None)
    p.add_argument("--end2",   type=float, default=None)
    p.add_argument("--no-separate", action="store_true",
                   help="Skip Demucs vocal isolation (faster, much less accurate).")
    p.add_argument("--work-dir", default="./voice_compare_work")
    args = p.parse_args()

    check_tool("ffmpeg")

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {work_dir}")

    emb1 = process_one(args.file1, 1, args, work_dir)
    emb2 = process_one(args.file2, 2, args, work_dir)

    score = cosine(emb1, emb2)

    print("\n" + "=" * 60)
    print(f"  Cosine similarity: {score:.4f}")
    print(f"  Verdict: {verdict(score)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
