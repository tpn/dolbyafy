from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from dolbyafy.audio import (
    export_surround_aac,
    export_surround_flac,
    export_surround_mp3,
    export_surround_wav,
    load_audio,
    make_surround,
)

app = typer.Typer(help="Create swirling surround audio for 5.1 systems.")


class OutputFormat(str, Enum):
    aac = "aac"
    flac = "flac"
    mp3 = "mp3"


@app.callback()
def main() -> None:
    """Dolbyafy command line tools."""


@app.command()
def convert(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Input audio path."
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path. Defaults to '<input>.dolby.5_1.<ext>'.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.aac,
        "--format",
        help="Output format for the main file (mp3 is stereo downmix).",
    ),
    clip: Optional[float] = typer.Option(
        None,
        "--clip",
        help="Trim to the first N seconds before processing.",
    ),
    surround_wav: Optional[Path] = typer.Option(
        None,
        "--surround-wav",
        help="Optional 5.1 WAV output path for true multichannel playback.",
    ),
    sample_rate: int = typer.Option(
        48000, "--sample-rate", "-r", help="Target sample rate."
    ),
    rotation_period: float = typer.Option(
        18.0, "--rotation-period", help="Seconds per full surround rotation."
    ),
    intensity: float = typer.Option(
        1.0, "--intensity", help="Overall effect intensity."
    ),
) -> None:
    if output_path is None:
        suffixes = {
            OutputFormat.aac: ".dolby.5_1.m4a",
            OutputFormat.flac: ".dolby.5_1.flac",
            OutputFormat.mp3: ".dolby.mp3",
        }
        output_path = input_path.with_suffix(suffixes[output_format])

    with tqdm(unit="frame", desc="Loading audio") as progress:
        samples, sample_rate = load_audio(
            input_path, clip, sample_rate, progress=progress
        )

    with tqdm(total=len(samples), unit="frame", desc="Designing surround") as progress:
        surround = make_surround(
            samples,
            sample_rate,
            rotation_period=rotation_period,
            intensity=intensity,
            progress=progress,
        )

    if surround_wav is not None:
        with tqdm(
            total=len(surround), unit="frame", desc="Writing 5.1 WAV"
        ) as progress:
            export_surround_wav(
                surround, sample_rate, surround_wav, progress=progress
            )

    if output_format is OutputFormat.aac:
        with tqdm(
            total=len(surround), unit="frame", desc="Writing AAC (5.1)"
        ) as progress:
            export_surround_aac(
                surround, sample_rate, output_path, progress=progress
            )
    elif output_format is OutputFormat.flac:
        with tqdm(
            total=len(surround), unit="frame", desc="Writing FLAC (5.1)"
        ) as progress:
            export_surround_flac(
                surround, sample_rate, output_path, progress=progress
            )
    else:
        with tqdm(
            total=len(surround), unit="frame", desc="Writing MP3 (stereo)"
        ) as progress:
            export_surround_mp3(
                surround, sample_rate, output_path, progress=progress
            )
    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
