# Examples

## Reference sample

The repo includes a short reference clip:

```
samples/dark-house-24s-46s.mp3
```

Generate the default 5.1 outputs:

```
dolbyafy convert samples/dark-house-24s-46s.mp3
```

## Alternate formats

5.1 FLAC:

```
dolbyafy convert samples/dark-house-24s-46s.mp3 --format flac
```

Stereo MP3 downmix:

```
dolbyafy convert samples/dark-house-24s-46s.mp3 --format mp3
```
