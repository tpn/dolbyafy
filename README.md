# dolbyafy

Swirl a stereo track around a 5.1 setup with a rotating surround field, while
keeping center and subwoofer channels steady. Outputs true multichannel files
by default, with a stereo downmix option when you need MP3.

## Features

- Dynamic 5.1 surround motion with a smooth, rotating pan field.
- Center and LFE stay anchored for punchy dialogue/bass stability.
- Exports 5.1 WAV, 5.1 AAC (m4a), 5.1 FLAC, or stereo MP3 downmix.
- Frame-level progress bars for long conversions.

## Install

This is a Python package, but it depends on system `ffmpeg`/`ffprobe` for
decoding and encoding. That means it is not a pure pip install.

```
pip install -e .
```

Install ffmpeg with one of:

- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS (brew): `brew install ffmpeg`
- Conda: `conda install -c conda-forge ffmpeg`

If you prefer conda/mamba:

```
mamba env create -f environment.yml
mamba activate dolbyafy
```

## Quickstart

```
dolbyafy convert samples/dark-house-24s-46s.mp3
```

Outputs by default:

- `samples/dark-house-24s-46s.dolby.5_1.wav`
- `samples/dark-house-24s-46s.dolby.5_1.m4a`

## Examples

Change rotation speed and intensity:

```
dolbyafy convert samples/dark-house-24s-46s.mp3 \
  --rotation-period 14 \
  --intensity 1.2
```

Create a 5.1 FLAC instead of AAC:

```
dolbyafy convert samples/dark-house-24s-46s.mp3 --format flac
```

Create a stereo MP3 downmix:

```
dolbyafy convert samples/dark-house-24s-46s.mp3 --format mp3
```

Write the 5.1 WAV to a custom path:

```
dolbyafy convert samples/dark-house-24s-46s.mp3 \
  --surround-wav samples/ref.5_1.wav
```

## Output Notes

- MP3 does not support 5.1 with `libmp3lame`, so `--format mp3` is a stereo
  downmix that may sound static on surround systems.
- For full multichannel playback, use the default AAC or FLAC output, or the
  5.1 WAV.

## Tests

Install dev dependencies if needed:

```
pip install -e ".[dev]"
```

## Development

Auto-format and lint:

```
black .
ruff check .
```

Enable pre-commit hooks:

```
pre-commit install
```

```
pytest
```
