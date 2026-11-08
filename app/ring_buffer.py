import numpy as np


class RingBuffer:
    """Кольцевой буфер для отсчётов (float32)."""

    def __init__(self, size: int):
        self.size = int(size)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.pos = 0
        self.count = 0

    def clear(self):
        self.pos = 0
        self.count = 0

    def write(self, data):
        data = np.asarray(data, dtype=np.float32)
        n = len(data)
        if n == 0:
            return
        if n >= self.size:
            self.buf[:] = data[-self.size:]
            self.pos = 0
            self.count = self.size
            return
        end = self.pos + n
        if end <= self.size:
            self.buf[self.pos:end] = data
        else:
            first = self.size - self.pos
            self.buf[self.pos:] = data[:first]
            self.buf[:end - self.size] = data[first:]
        self.pos = end % self.size
        self.count = min(self.count + n, self.size)

    def last(self, n: int) -> np.ndarray:
        n = int(min(n, self.count))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        end = self.pos
        start = (end - n) % self.size
        if start < end:
            return self.buf[start:end].copy()
        return np.concatenate((self.buf[start:], self.buf[:end]))