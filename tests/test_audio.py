import numpy as np

from dolbyafy.audio import make_surround, matrix_downmix


def test_make_surround_shape_and_finite() -> None:
    sample_rate = 48000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    tone = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    stereo = np.stack([tone, tone], axis=1)

    surround = make_surround(stereo, sample_rate)

    assert surround.shape == (sample_rate, 6)
    assert np.isfinite(surround).all()
    assert np.max(np.abs(surround)) > 0.0


def test_make_surround_zero_input() -> None:
    sample_rate = 48000
    stereo = np.zeros((2048, 2), dtype=np.float32)

    surround = make_surround(stereo, sample_rate)

    assert np.allclose(surround, 0.0)


def test_matrix_downmix_shape() -> None:
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((4096, 6)).astype(np.float32) * 0.1

    stereo = matrix_downmix(samples, sample_rate=48000)

    assert stereo.shape == (4096, 2)
    assert np.isfinite(stereo).all()
