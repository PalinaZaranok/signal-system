"""Источники данных: Mock и Serial"""
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

try:
    import serial
except ImportError:
    serial = None

from protocol import build_packet


class MockSource(QThread):
    """Генерирует реальные байтовые пакеты протокола с таймингом."""

    data_ready = pyqtSignal(bytes)
    error = pyqtSignal(str)

    def __init__(self, fs=80_000.0, packet_size=128, signals=None):
        super().__init__()
        self.fs = float(fs)
        self.packet_size = int(packet_size)
        # (частота Гц, амплитуда в отсчётах АЦП)
        self.signals = signals or [(1_000.0, 900.0), (3_500.0, 600.0), (7_200.0, 350.0)]
        self.noise_amplitude = 80.0
        self._running = False
        self._pid = 0
        self._phase_t = 0.0

    def stop(self):
        self._running = False
        self.wait(1500)

    def run(self):
        self._running = True
        period = self.packet_size / self.fs
        next_t = time.monotonic()
        while self._running:
            samples = self._generate()
            pkt = build_packet(self._pid, samples)
            self._pid = (self._pid + 1) & 0xFF
            self.data_ready.emit(pkt)

            next_t += period
            dt = next_t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.monotonic()  # отстаём — догоняем без сна

    def _generate(self) -> np.ndarray:
        n = self.packet_size
        t = self._phase_t + np.arange(n) / self.fs
        self._phase_t = t[-1] + 1.0 / self.fs
        y = np.full(n, 2048.0)
        for f, a in self.signals:
            y += a * np.sin(2 * np.pi * f * t)
        y += np.random.normal(0, self.noise_amplitude, n)
        y = np.clip(y, 0, 4095).astype(np.uint16)
        return y


class SerialSource(QThread):
    data_ready = pyqtSignal(bytes)
    error = pyqtSignal(str)

    def __init__(self, port: str, baud: int):
        super().__init__()
        self.port = port
        self.baud = int(baud)
        self._running = False
        self._ser = None

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        if serial is None:
            self.error.emit("Установите pyserial: pip install pyserial")
            return
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
            self._ser.reset_input_buffer()
        except Exception as e:
            self.error.emit(f"Не удалось открыть {self.port}: {e}")
            return
        self._running = True
        try:
            while self._running:
                try:
                    data = self._ser.read(8192)
                except Exception as e:
                    self.error.emit(str(e))
                    break
                if data:
                    self.data_ready.emit(data)
        finally:
            try:
                self._ser.close()
            except Exception:
                pass