from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from dolbyafy.audio import (
    export_surround_mp3,
    export_surround_wav,
    load_audio,
    make_surround,
)

app = typer.Typer(
    help="Create swirling surround MP3s (matrix-encoded for 5.1 systems)."
)


@app.callback()
def main() -> None:
    """Dolbyafy command line tools."""


@app.command()
def convert(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Input MP3 path."
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output MP3 path. Defaults to '<input>.dolby.mp3'.",
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
    output_path = (
        output_path
        if output_path is not None
        else input_path.with_suffix(".dolby.mp3")
    )

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

    with tqdm(total=len(surround), unit="frame", desc="Writing MP3") as progress:
        export_surround_mp3(surround, sample_rate, output_path, progress=progress)
    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
