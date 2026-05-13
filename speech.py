#!/usr/bin/env python3
"""
compare_ecapa.py
----------------
Compare voices in two local audio files using SpeechBrain's ECAPA-TDNN
speaker verification model. Newer architecture than Resemblyzer.

USAGE:
    python compare_ecapa.py file1.mp3 file2.mp3
    python compare_ecapa.py file1.mp3 file2.mp3 --start1 30 --end1 60 --start2 45 --end2 75
    python compare_ecapa.py file1.mp3 file2.mp3 --no-separate

REQUIREMENTS:
    pip install speechbrain librosa numpy soundfile demucs "torchaudio<2.4"
    plus ffmpeg on PATH

INTERPRETING THE SCORE:
    SpeechBrain returns a similarity score and a binary decision.
    Score is cosine similarity (-1 to 1), threshold is around 0.25 by default
    for speech. For singing the same caveats apply — treat as evidence not proof.
    Compare relative scores across multiple pairs to interpret.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


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
    """Convert any audio file to mono 16kHz WAV (ECAPA expects 16kHz)."""
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
    print(f"\n=== Isolating vocals from {in_path.name} ===")
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


def process_one(in_file, idx, args, work_dir):
    in_path = Path(in_file).expanduser().resolve()
    if not in_path.exists():
        sys.exit(f"ERROR: file not found: {in_path}")

    start = getattr(args, f"start{idx}")
    end = getattr(args, f"end{idx}")
    wav = to_wav(in_path, work_dir / f"converted_{idx}.wav", start, end)
    return wav if args.no_separate else isolate_vocals(wav, work_dir / f"sep_{idx}")


def verdict(score: float) -> str:
    if score > 0.50: return "Likely the SAME singer."
    if score > 0.35: return "Probably same singer (above typical threshold)."
    if score > 0.25: return "Borderline — at the typical decision threshold."
    if score > 0.10: return "Probably different singers."
    return "Likely different singers."


def main():
    p = argparse.ArgumentParser(description="Compare voices with SpeechBrain ECAPA-TDNN.")
    p.add_argument("file1")
    p.add_argument("file2")
    p.add_argument("--start1", type=float, default=None)
    p.add_argument("--end1",   type=float, default=None)
    p.add_argument("--start2", type=float, default=None)
    p.add_argument("--end2",   type=float, default=None)
    p.add_argument("--no-separate", action="store_true")
    p.add_argument("--work-dir", default="./voice_compare_work_ecapa")
    args = p.parse_args()

    check_tool("ffmpeg")

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {work_dir}")

    wav1 = process_one(args.file1, 1, args, work_dir)
    wav2 = process_one(args.file2, 2, args, work_dir)

    print("\n=== Loading SpeechBrain ECAPA-TDNN ===")
    print("(first run downloads ~80MB model)")
    # Import after the heavy preprocessing so user sees download warnings cleanly.
    from speechbrain.inference.speaker import SpeakerRecognition

    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(work_dir / "ecapa_model"),
    )

    print("\n=== Comparing ===")
    score, prediction = verifier.verify_files(str(wav1), str(wav2))
    score_val = float(score.item() if hasattr(score, "item") else score[0])
    pred_val = bool(prediction.item() if hasattr(prediction, "item") else prediction[0])

    print("\n" + "=" * 60)
    print(f"  Cosine similarity: {score_val:.4f}")
    print(f"  Model decision:    {'SAME speaker' if pred_val else 'DIFFERENT speakers'}")
    print(f"  Verdict (singing): {verdict(score_val)}")
    print("=" * 60)
    print("\nNote: ECAPA's default threshold (~0.25) is calibrated for SPEECH.")
    print("Singing tends to produce noisier scores. Run multiple pairs and")
    print("look at relative differences, not absolute numbers.")


if __name__ == "__main__":
    main()
