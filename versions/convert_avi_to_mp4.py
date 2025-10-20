"""Batch convert AVI videos to MP4 using ffmpeg.

This script walks a directory, finds all files with the .avi extension (case-insensitive),
and invokes ffmpeg to transcode them to H.264/AAC MP4 files while preserving the base name.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def find_avi_files(root: Path, recursive: bool = False) -> Iterable[Path]:
    """Yield AVI files under *root*.

    Args:
        root: Directory to search.
        recursive: When True, search subdirectories as well.
    """
    pattern = "**/*.avi" if recursive else "*.avi"
    for path in root.glob(pattern):
        if path.is_file():
            yield path


def build_ffmpeg_command(
    ffmpeg_executable: str,
    source_path: Path,
    destination_path: Path,
    crf: int,
    preset: str,
    audio_bitrate: str,
    overwrite: bool,
) -> list[str]:
    """Return the ffmpeg command that performs the conversion."""
    command = [
        ffmpeg_executable,
        "-y" if overwrite else "-n",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        str(destination_path),
    ]
    return command


def convert_video(
    command: list[str],
    dry_run: bool,
) -> subprocess.CompletedProcess | None:
    """Execute ffmpeg with the provided command.

    Returns the CompletedProcess when executed, otherwise None for dry run.
    """
    if dry_run:
        return None

    completed = subprocess.run(command, check=True)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert AVI files to MP4 using ffmpeg.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="The directory to scan for AVI files (default: current working directory).",
    )
    parser.add_argument(
        "--ffmpeg",
        dest="ffmpeg_path",
        default=None,
        help="Path to the ffmpeg executable. Defaults to looking it up on PATH.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MP4 files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ffmpeg commands without executing them.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Constant Rate Factor for H.264 quality (lower is higher quality). Default: 23.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="H.264 encoding preset. Default: medium.",
    )
    parser.add_argument(
        "--audio-bitrate",
        default="192k",
        help="Audio bitrate for the encoded file. Default: 192k.",
    )
    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.exists():
        raise SystemExit(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    ffmpeg_executable = args.ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg_executable:
        raise SystemExit(
            "ffmpeg executable not found. Please install ffmpeg and ensure it is available on PATH, or provide the path via --ffmpeg."
        )

    avi_files = list(find_avi_files(directory, args.recursive))
    if not avi_files:
        print(f"No AVI files found in {directory}.")
        return

    print(f"Found {len(avi_files)} AVI file(s) in {directory}.")

    for avi_file in avi_files:
        output_path = avi_file.with_suffix(".mp4")
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path}")
            continue

        command = build_ffmpeg_command(
            ffmpeg_executable=ffmpeg_executable,
            source_path=avi_file,
            destination_path=output_path,
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate,
            overwrite=True,  # we handle skipping manually above
        )

        print("\nConverting:")
        print(f"  Source     : {avi_file}")
        print(f"  Destination: {output_path}")
        print(f"  Command    : {' '.join(command)}")

        try:
            convert_video(command, args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"Conversion failed for {avi_file} with exit code {exc.returncode}.")
        else:
            if args.dry_run:
                print("Dry run: skipped execution.")
            else:
                print("Conversion completed successfully.")


if __name__ == "__main__":
    main()
