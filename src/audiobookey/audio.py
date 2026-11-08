"""WAV I/O and ffmpeg assembly: concat, loudnorm, m4b chapters, mp3.

All paths handed to ffmpeg/ffprobe are native Windows paths (e.g.
C:/Users/...); MSYS /c/... paths are never passed to native tools.
"""

from __future__ import annotations

import json
import subprocess
import uuid
import wave
from pathlib import Path

import numpy as np

LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"


class FFmpegError(RuntimeError):
    pass


def _ffmpeg_binary() -> str:
    return "ffmpeg"


def _run_ffmpeg(args: list[str]) -> None:
    cmd = [_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed ({' '.join(args[:6])}...):\n{result.stderr.strip()}"
        )


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write float32 mono samples as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return path


def read_wav_duration(path: str | Path) -> float:
    """Exact duration in seconds from the WAV header."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def read_wav_sample_rate(path: str | Path) -> int:
    with wave.open(str(path), "rb") as w:
        return w.getframerate()


def write_silence(path: str | Path, seconds: float, sample_rate: int) -> Path:
    """Write a silent mono WAV of the given duration."""
    frames = int(round(seconds * sample_rate))
    return write_wav(path, np.zeros(frames, dtype=np.float32), sample_rate)


def resample_wav(
    src: str | Path, dst: str | Path, sample_rate: int, mono: bool = True
) -> Path:
    """Resample (and optionally downmix) a WAV with ffmpeg."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(src)]
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", str(sample_rate), str(dst)]
    _run_ffmpeg(args)
    return dst


def _concat_list_lines(inputs: list[Path]) -> list[str]:
    """One `file '<path>'` line per input, ffmpeg-concat quoting.

    Windows paths are normalized to forward slashes; embedded single
    quotes are escaped as '\\''.
    """
    lines = []
    for p in inputs:
        native = str(p.resolve()).replace("\\", "/")
        lines.append("file '" + native.replace("'", "'\\''") + "'")
    return lines


def _write_concat_list_file(inputs: list[Path], list_file: Path) -> None:
    list_file.write_text("\n".join(_concat_list_lines(inputs)) + "\n", encoding="utf-8")


def concat_wavs(inputs: list[str | Path], out_path: str | Path) -> Path:
    """Concatenate same-format WAVs losslessly with the ffmpeg concat demuxer."""
    inputs = [Path(p) for p in inputs]
    if not inputs:
        raise ValueError("no inputs to concatenate")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.with_name(f"{out_path.stem}-{uuid.uuid4().hex[:8]}.txt")
    _write_concat_list_file(inputs, list_file)
    try:
        _run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy",
             str(out_path)]
        )
    finally:
        list_file.unlink(missing_ok=True)
    return out_path


def copy_wav(src: str | Path, dest: str | Path) -> Path:
    import shutil

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(src), str(dest))
    return dest


def build_ffmetadata(chapters: list[tuple[float, float, str]]) -> str:
    """ffmetadata text with one [CHAPTER] block per (start, end, title)."""
    lines = [";FFMETADATA1"]
    for start, end, title in chapters:
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={int(round(start * 1000))}")
        lines.append(f"END={int(round(end * 1000))}")
        lines.append(f"title={title}")
    return "\n".join(lines) + "\n"


def normalize_wav(
    wav_in: str | Path | list[str | Path] | tuple[str | Path, ...],
    out_path: str | Path,
) -> Path:
    """One loudnorm pass over the whole book into an intermediate WAV.

    ``wav_in`` may be a single WAV or a list of same-format page WAVs; the
    list form feeds an ffmpeg concat-demuxer list file so every page is
    decoded and normalized in one pass and no concatenated book.wav is
    ever written. A one-element list feeds the WAV directly (single-page
    fast path, no list file). The final encode runs this once and ships
    both m4b and mp3 from the single normalized intermediate, instead of
    each encode carrying its own full loudnorm analysis of the book.
    Output is 48 kHz: without it the intermediate would carry loudnorm's
    native 192 kHz (a huge file), and at the engines' 24 kHz source rate
    lame's MPEG-2 branch caps mp3 at 160k, which erases the best-tier
    256k bitrate.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(wav_in, (list, tuple)):
        wavs = [Path(p) for p in wav_in]
        if not wavs:
            raise ValueError("no inputs to normalize")
        if len(wavs) == 1:
            src_args = ["-i", str(wavs[0])]
            list_file = None
        else:
            list_file = out_path.with_name(
                f"{out_path.stem}-{uuid.uuid4().hex[:8]}.txt"
            )
            _write_concat_list_file(wavs, list_file)
            src_args = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
    else:
        src_args = ["-i", str(wav_in)]
        list_file = None
    try:
        _run_ffmpeg(
            [*src_args, "-af", LOUDNORM_FILTER, "-ar", "48000", str(out_path)]
        )
    finally:
        if list_file is not None:
            list_file.unlink(missing_ok=True)
    return out_path


def encode_m4b(
    wav_in: str | Path,
    chapters: list[tuple[float, float, str]],
    out_path: str | Path,
    audio_bitrate: str = "64k",
    loudnorm: bool = True,
) -> Path:
    """Encode the concatenated book WAV to m4b (AAC) with page chapters.

    ``loudnorm=False`` expects an already-normalized input (see
    normalize_wav) and skips the filter; ``-movflags +faststart`` is kept
    either way (browsers hang without it).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_file = out_path.with_suffix(".ffmeta.txt")
    meta_file.write_text(build_ffmetadata(chapters), encoding="utf-8")
    args = [
        "-i", str(wav_in), "-i", str(meta_file),
        "-map", "0:a", "-map_metadata", "1",
        "-c:a", "aac", "-b:a", audio_bitrate,
    ]
    if loudnorm:
        args += ["-af", LOUDNORM_FILTER]
    args += [
        "-movflags", "+faststart",
        "-f", "ipod", str(out_path),
    ]
    try:
        _run_ffmpeg(args)
    finally:
        meta_file.unlink(missing_ok=True)
    return out_path


def encode_mp3(
    wav_in: str | Path,
    out_path: str | Path,
    bitrate: str = "192k",
    loudnorm: bool = True,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(wav_in), "-c:a", "libmp3lame", "-b:a", bitrate]
    if loudnorm:
        args += ["-af", LOUDNORM_FILTER]
    args += [str(out_path)]
    _run_ffmpeg(args)
    return out_path


def encode_m4b_and_mp3(
    wav_in: str | Path | list[str | Path],
    chapters: list[tuple[float, float, str]],
    m4b_path: str | Path,
    mp3_path: str | Path,
    audio_bitrate: str = "64k",
    mp3_bitrate: str = "192k",
    work_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Loudnorm once, then encode both book outputs from that single pass.

    ``wav_in`` may be a single WAV or a list of page WAVs (concatenated
    inside the normalize pass; see normalize_wav -- no book.wav is
    written for the list form). The normalized intermediate lives in
    ``work_dir`` (default: beside the m4b) and is removed once both
    encodes finished or failed.
    """
    m4b_path = Path(m4b_path)
    m4b_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_dir = Path(work_dir) if work_dir is not None else m4b_path.parent
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    intermediate = intermediate_dir / f"{m4b_path.stem}.normalized.wav"
    normalize_wav(wav_in, intermediate)
    try:
        m4b = encode_m4b(
            intermediate, chapters, m4b_path, audio_bitrate, loudnorm=False
        )
        mp3 = encode_mp3(intermediate, mp3_path, mp3_bitrate, loudnorm=False)
    finally:
        intermediate.unlink(missing_ok=True)
    return m4b, mp3


def ffprobe_info(path: str | Path) -> dict:
    """Return parsed ffprobe JSON (format + chapters) for an audio file."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_chapters", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed:\n{result.stderr.strip()}")
    return json.loads(result.stdout)
