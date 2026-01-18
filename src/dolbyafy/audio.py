from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import json
import subprocess
import wave

import numpy as np


def _probe_audio(path: Path) -> Tuple[float, int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path}: {result.stderr.strip()}"
        )

    data = json.loads(result.stdout)
    stream = data["streams"][0]
    duration = float(data["format"]["duration"])
    sample_rate = int(stream["sample_rate"])
    channels = int(stream["channels"])
    return duration, sample_rate, channels


def load_audio(
    path: Path,
    clip_seconds: Optional[float],
    sample_rate: int,
    progress: Optional[Any] = None,
) -> Tuple[np.ndarray, int]:
    duration, src_rate, _ = _probe_audio(path)
    target_rate = sample_rate if sample_rate > 0 else src_rate
    if clip_seconds is not None:
        duration = min(duration, clip_seconds)

    expected_frames = int(duration * target_rate)
    if progress is not None:
        progress.reset(total=max(expected_frames, 1))

    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if clip_seconds is not None:
        cmd += ["-t", str(clip_seconds)]
    cmd += [
        "-ac",
        "2",
        "-ar",
        str(target_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("Failed to spawn ffmpeg for decoding.")

    chunk_frames = max(2048, int(target_rate * 0.5))
    chunk_bytes = chunk_frames * 2 * 2
    buffer = bytearray()
    frames_read = 0

    while True:
        chunk = proc.stdout.read(chunk_bytes)
        if not chunk:
            break
        buffer.extend(chunk)
        frames = len(chunk) // 4
        frames_read += frames
        if progress is not None:
            progress.update(frames)

    stderr = proc.stderr.read().decode().strip()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg decode failed for {path}: {stderr}"
        )

    if progress is not None and progress.total and frames_read != progress.total:
        progress.total = max(frames_read, 1)
        progress.refresh()

    if not buffer:
        raise RuntimeError(f"No audio data decoded from {path}.")

    samples = np.frombuffer(buffer, dtype=np.int16)
    samples = samples.reshape(-1, 2)
    return samples.astype(np.float32) / 32768.0, target_rate


def _moving_average_lowpass(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.copy()
    if window % 2 == 0:
        window += 1

    cumsum = np.cumsum(signal, dtype=np.float32)
    cumsum = np.concatenate(([0.0], cumsum))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    pad_left = window // 2
    pad_right = len(signal) - len(smoothed) - pad_left
    return np.pad(smoothed, (pad_left, pad_right), mode="edge")


def _delayed(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    if delay_samples <= 0:
        return signal
    out = np.zeros_like(signal)
    if delay_samples < len(signal):
        out[delay_samples:] = signal[:-delay_samples]
    return out


def make_surround(
    samples: np.ndarray,
    sample_rate: int,
    rotation_period: float = 18.0,
    intensity: float = 1.0,
    progress: Optional[Any] = None,
) -> np.ndarray:
    if samples.ndim == 1:
        samples = samples[:, None]
    left = samples[:, 0]
    right = samples[:, 1] if samples.shape[1] > 1 else left

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    t = np.arange(len(mid), dtype=np.float32) / float(sample_rate)
    wobble = 0.18 * np.sin(2 * np.pi * t / 7.0) + 0.12 * np.sin(
        2 * np.pi * t / 11.0
    )
    angle = 2 * np.pi * (t / rotation_period) + wobble
    front_angle = angle
    rear_angle = angle + np.pi

    def weight(theta: float, steering: np.ndarray, sharpness: float) -> np.ndarray:
        return np.clip(np.cos(steering - theta), 0.0, 1.0) ** sharpness

    theta_fl = np.deg2rad(30.0)
    theta_fr = np.deg2rad(-30.0)
    theta_sl = np.deg2rad(110.0)
    theta_sr = np.deg2rad(-110.0)

    w_fl = weight(theta_fl, front_angle, 1.6)
    w_fr = weight(theta_fr, front_angle, 1.6)
    w_sl = weight(theta_sl, rear_angle, 1.7)
    w_sr = weight(theta_sr, rear_angle, 1.7)

    pulse = 0.85 + 0.15 * np.sin(2 * np.pi * t / 4.5)
    mid = np.tanh(mid * 1.1)
    side = np.tanh(side * 1.25)

    lfe_window = max(5, int(sample_rate / 120.0))
    bass = _moving_average_lowpass(mid, lfe_window)
    mid_fx = (mid - bass) * pulse

    short = _delayed(mid_fx, int(sample_rate * 0.012))
    medium = _delayed(mid_fx, int(sample_rate * 0.028))
    long = _delayed(mid_fx, int(sample_rate * 0.055))

    front_fx = mid_fx + 0.18 * short
    rear_fx = mid_fx + 0.28 * medium + 0.16 * long

    base_front = mid * 0.6
    base_rear = mid * 0.25

    ch_c_full = mid * 0.85

    lfe = bass * 0.75
    lfe = np.tanh(lfe * 1.1)

    chunk_frames = max(2048, int(sample_rate * 0.5))
    surround = np.empty((len(mid), 6), dtype=np.float32)
    for start in range(0, len(mid), chunk_frames):
        end = min(len(mid), start + chunk_frames)
        ch_fl = base_front[start:end] + (
            front_fx[start:end] + side[start:end] * 0.22
        ) * w_fl[start:end]
        ch_fr = base_front[start:end] + (
            front_fx[start:end] - side[start:end] * 0.22
        ) * w_fr[start:end]
        ch_sl = base_rear[start:end] + (
            rear_fx[start:end] - side[start:end] * 0.15
        ) * w_sl[start:end]
        ch_sr = base_rear[start:end] + (
            rear_fx[start:end] + side[start:end] * 0.15
        ) * w_sr[start:end]
        ch_c = ch_c_full[start:end]
        lfe_chunk = lfe[start:end]

        surround[start:end, 0] = ch_fl
        surround[start:end, 1] = ch_fr
        surround[start:end, 2] = ch_c
        surround[start:end, 3] = lfe_chunk
        surround[start:end, 4] = ch_sl
        surround[start:end, 5] = ch_sr
        if progress is not None:
            progress.update(end - start)

    surround = surround * intensity
    peak = np.max(np.abs(surround))
    if peak > 0:
        surround = surround * (0.96 / peak)
    return surround.astype(np.float32)


def _float_to_pcm(samples: np.ndarray) -> np.ndarray:
    pcm = np.clip(samples, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16)


def _export_ffmpeg(
    samples: np.ndarray,
    sample_rate: int,
    output_path: Path,
    codec: str,
    bitrate: Optional[str] = None,
    progress: Optional[Any] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = samples.shape[1]
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostats",
        "-y",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-c:a",
        codec,
    ]
    if bitrate:
        cmd += ["-b:a", bitrate]
    cmd.append(str(output_path))

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if proc.stdin is None or proc.stderr is None:
        raise RuntimeError(f"Failed to spawn ffmpeg for {codec} encoding.")

    chunk_frames = max(2048, int(sample_rate * 0.5))
    for start in range(0, len(samples), chunk_frames):
        end = min(len(samples), start + chunk_frames)
        pcm = _float_to_pcm(samples[start:end])
        proc.stdin.write(pcm.tobytes())
        if progress is not None:
            progress.update(end - start)

    proc.stdin.close()
    stderr = proc.stderr.read().decode().strip()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg {codec} encode failed for {output_path}: {stderr}"
        )


def matrix_downmix(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    fl, fr, c, lfe, sl, sr = samples.T
    surround_diff = sl - sr
    surround_delayed = _delayed(surround_diff, int(sample_rate * 0.015))

    center = c * 0.707
    lfe_mix = lfe * 0.35

    left = fl + center + lfe_mix + 0.45 * surround_delayed + 0.15 * sl
    right = fr + center + lfe_mix - 0.45 * surround_delayed + 0.15 * sr

    stereo = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo * (0.98 / peak)
    return stereo.astype(np.float32)


def export_surround_wav(
    samples: np.ndarray,
    sample_rate: int,
    output_path: Path,
    progress: Optional[Any] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_frames = max(2048, int(sample_rate * 0.5))
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(samples.shape[1])
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        for start in range(0, len(samples), chunk_frames):
            end = min(len(samples), start + chunk_frames)
            pcm = _float_to_pcm(samples[start:end])
            writer.writeframes(pcm.tobytes())
            if progress is not None:
                progress.update(end - start)


def export_surround_mp3(
    samples: np.ndarray,
    sample_rate: int,
    output_path: Path,
    bitrate: str = "320k",
    progress: Optional[Any] = None,
) -> None:
    stereo = matrix_downmix(samples, sample_rate)
    _export_ffmpeg(
        stereo,
        sample_rate,
        output_path,
        codec="libmp3lame",
        bitrate=bitrate,
        progress=progress,
    )


def export_surround_aac(
    samples: np.ndarray,
    sample_rate: int,
    output_path: Path,
    bitrate: str = "384k",
    progress: Optional[Any] = None,
) -> None:
    _export_ffmpeg(
        samples,
        sample_rate,
        output_path,
        codec="aac",
        bitrate=bitrate,
        progress=progress,
    )


def export_surround_flac(
    samples: np.ndarray,
    sample_rate: int,
    output_path: Path,
    progress: Optional[Any] = None,
) -> None:
    _export_ffmpeg(
        samples,
        sample_rate,
        output_path,
        codec="flac",
        progress=progress,
    )
