"""Протокол обмена (см. docs/protocol.md)."""
import struct
import numpy as np

SYNC0 = 0xAA
SYNC1 = 0x55
HEADER = 4     # sync0 + sync1 + id + n_samples
CRC = 2
MAX_SAMPLES = 255


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build_packet(packet_id: int, samples: np.ndarray) -> bytes:
    """Собирает пакет по протоколу. samples — uint16."""
    arr = np.asarray(samples, dtype=np.uint16)
    n = len(arr)
    if not 0 < n <= MAX_SAMPLES:
        raise ValueError(f"n_samples must be in 1..{MAX_SAMPLES}")
    body = struct.pack('<BB', packet_id & 0xFF, n) + arr.astype('<u2').tobytes()
    crc = crc16_ccitt(body)
    return bytes([SYNC0, SYNC1]) + body + struct.pack('<H', crc)


class PacketParser:
    """Потоковый парсер: на вход байты, на выход список np.uint16-массивов."""

    def __init__(self):
        self.buf = bytearray()
        self.last_id = None
        self.received = 0
        self.dropped = 0
        self.crc_errors = 0

    def reset(self):
        self.buf.clear()
        self.last_id = None
        self.received = 0
        self.dropped = 0
        self.crc_errors = 0

    def feed(self, data: bytes):
        self.buf.extend(data)
        out = []
        while True:
            i = self.buf.find(b'\xAA\x55')
            if i < 0:
                # сохраняем последний байт (0xAA может быть началом sync)
                if len(self.buf) > 1:
                    del self.buf[:-1]
                return out
            if i:
                del self.buf[:i]
            if len(self.buf) < HEADER:
                return out
            n = self.buf[3]
            if n == 0:
                del self.buf[:2]
                self.crc_errors += 1
                continue
            total = HEADER + 2 * n + CRC
            if len(self.buf) < total:
                return out
            frame = bytes(self.buf[:total])
            body = frame[2:HEADER + 2 * n]
            crc_rx = struct.unpack('<H', frame[HEADER + 2 * n:])[0]
            if crc16_ccitt(body) != crc_rx:
                del self.buf[:2]
                self.crc_errors += 1
                continue
            pid = frame[2]
            if self.last_id is not None:
                d = (pid - self.last_id) & 0xFF
                if 1 < d < 128:
                    self.dropped += d - 1
            self.last_id = pid
            self.received += 1
            out.append(np.frombuffer(body[2:], dtype='<u2').astype(np.uint16))
            del self.buf[:total]