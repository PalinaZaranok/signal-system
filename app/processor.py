import numpy as np


def _window(n: int, name: str) -> np.ndarray:
    if name == 'hann':
        return np.hanning(n)
    if name == 'hamming':
        return np.hamming(n)
    if name == 'blackman':
        return np.blackman(n)
    if name == 'rect':
        return np.ones(n)
    raise ValueError(name)


def compute_spectrum(samples: np.ndarray, fs: float, window: str = 'hann'):
    """Возвращает (freqs, magnitudes) — амплитудный спектр (пиковая амплитуда)."""
    n = len(samples)
    if n < 4:
        return np.zeros(0), np.zeros(0)
    w = _window(n, window)
    #x = np.asarray(samples, dtype=np.float64) * w
    x = np.asarray(samples, dtype=np.float64) - np.mean(samples)
    spec = np.fft.rfft(x)
    mag = 2.0 * np.abs(spec) / w.sum()   # нормировка на коэффициент окна
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mag