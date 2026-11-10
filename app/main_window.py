import time
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

import serial.tools.list_ports

from ring_buffer import RingBuffer
from processor import compute_spectrum
from sources import MockSource, SerialSource
from protocol import PacketParser


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Analyzer")
        self.resize(1280, 820)

        # состояние
        self.source = None
        self.parser = PacketParser()
        self.buffer = RingBuffer(1_000_000)
        self.total_samples = 0
        self.last_update = time.monotonic()
        self.updates_in_sec = 0
        self.fps = 0.0

        self._build_ui()

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_plots)
        self.refresh_timer.start(33)   # ~30 Гц

        self.refresh_ports()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(320)
        form = QtWidgets.QFormLayout(panel)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Test", "Serial"])
        form.addRow("Режим:", self.mode_combo)

        # порт
        self.port_combo = QtWidgets.QComboBox()
        form.addRow("COM-порт:", self.port_combo)
        self.refresh_btn = QtWidgets.QPushButton("Обновить порты")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        form.addRow(self.refresh_btn)

        # baud
        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.addItems(["115200", "460800", "921600",
                                  "1000000", "2000000", "3000000"])
        self.baud_combo.setCurrentText("2000000")
        form.addRow("Baudrate:", self.baud_combo)

        # Fs
        self.fs_spin = QtWidgets.QDoubleSpinBox()
        self.fs_spin.setRange(100, 500_000)
        self.fs_spin.setDecimals(0)
        self.fs_spin.setValue(80_000)
        self.fs_spin.setSuffix(" Гц")
        form.addRow("Fs (Гц):", self.fs_spin)

        # размер пакета (Mock)
        self.pkt_spin = QtWidgets.QSpinBox()
        self.pkt_spin.setRange(16, 255)
        self.pkt_spin.setValue(128)
        form.addRow("Пакет (Test):", self.pkt_spin)

        # размер окна FFT
        self.fft_size_combo = QtWidgets.QComboBox()
        for n in (256, 512, 1024, 2048, 4096, 8192, 16384):
            self.fft_size_combo.addItem(str(n), n)
        self.fft_size_combo.setCurrentText("2048")
        form.addRow("Окно FFT:", self.fft_size_combo)

        # окно
        #self.window_combo = QtWidgets.QComboBox()
        #self.window_combo.addItems(["hann", "hamming", "blackman", "rect"])
        #form.addRow("Оконная функция:", self.window_combo)

        # окно времени
        self.time_window_spin = QtWidgets.QDoubleSpinBox()
        self.time_window_spin.setRange(0.001, 2.0)
        self.time_window_spin.setDecimals(4)
        self.time_window_spin.setValue(0.02)
        self.time_window_spin.setSuffix(" с")
        form.addRow("Окно времени:", self.time_window_spin)

        # сетка
        self.grid_check = QtWidgets.QCheckBox("Сетка")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self._apply_grid)
        form.addRow(self.grid_check)

        # автомасштаб Y
        self.autoscale_check = QtWidgets.QCheckBox("Автомасштаб Y")
        self.autoscale_check.setChecked(True)
        form.addRow(self.autoscale_check)

        # кнопки
        self.start_btn = QtWidgets.QPushButton("Старт")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QtWidgets.QPushButton("Стоп")
        self.stop_btn.clicked.connect(self.stop)
        self.reset_btn = QtWidgets.QPushButton("Сброс")
        self.reset_btn.clicked.connect(self.reset)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.reset_btn)
        form.addRow(row)

        # статистика
        self.stats_label = QtWidgets.QLabel("—")
        self.stats_label.setWordWrap(True)
        form.addRow("Статистика:", self.stats_label)

        root.addWidget(panel)

        # ---------- графики ----------
        plots = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plots)
        pg.setConfigOptions(antialias=True, background='k', foreground='w')

        self.time_plot = pg.PlotWidget(title="Временная область U(t)")
        self.time_plot.setLabel('left', 'Амплитуда, отсчёты АЦП')
        self.time_plot.setLabel('bottom', 'Время', units='s')
        self.time_curve = self.time_plot.plot(pen=pg.mkPen('c', width=1))

        self.freq_plot = pg.PlotWidget(title="Спектр A(f)")
        self.freq_plot.setLabel('left', 'Амплитуда')
        self.freq_plot.setLabel('bottom', 'Частота', units='Hz')
        self.freq_curve = self.freq_plot.plot(pen=pg.mkPen('y', width=1))
        self._apply_grid()

        pv.addWidget(self.time_plot)
        pv.addWidget(self.freq_plot)
        root.addWidget(plots, 1)

    def _apply_grid(self):
        on = self.grid_check.isChecked()
        for w in (self.time_plot, self.freq_plot):
            w.showGrid(x=on, y=on, alpha=0.3)

    # ------------------------------------------------------------- actions
    def refresh_ports(self):
        self.port_combo.clear()
        for p in serial.tools.list_ports.comports():
            self.port_combo.addItem(f"{p.device} — {p.description}", p.device)

    def start(self):
        if self.source is not None:
            return
        self.parser.reset()
        self.buffer.clear()
        self.total_samples = 0
        self.time_curve.setData([], [])
        self.freq_curve.setData([], [])

        if self.mode_combo.currentIndex() == 0:  # Mock
            self.source = MockSource(
                fs=self.fs_spin.value(),
                packet_size=self.pkt_spin.value(),
            )
        else:                                    # Serial
            if self.port_combo.count() == 0:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Нет COM-портов")
                return
            self.source = SerialSource(
                self.port_combo.currentData(),
                int(self.baud_combo.currentText()),
            )
        self.source.data_ready.connect(self._on_data)
        self.source.error.connect(self._on_error)
        self.source.start()

    def stop(self):
        if self.source is None:
            return
        self.source.stop()
        self.source = None

    def reset(self):
        self.stop()
        self.parser.reset()
        self.buffer.clear()
        self.total_samples = 0
        self.time_curve.setData([], [])
        self.freq_curve.setData([], [])
        self.stats_label.setText("—")

    # --------------------------------------------------------------- slots
    def _on_data(self, data: bytes):
        packets = self.parser.feed(data)
        for s in packets:
            self.buffer.write(s.astype(np.float32))
            self.total_samples += len(s)

    def _on_error(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Ошибка источника", msg)
        self.stop()

    # --------------------------------------------------------- обновление
    def _refresh_plots(self):
        if self.source is None:
            return
        fs = self.fs_spin.value()

        # временная область
        n_t = int(fs * self.time_window_spin.value())
        n_t = max(64, min(n_t, 100_000))
        tv = self.buffer.last(n_t)
        if tv.size > 1:
            t = (np.arange(tv.size) - tv.size + 1) / fs
            self.time_curve.setData(t, tv)
            if self.autoscale_check.isChecked():
                self.time_plot.enableAutoRange()

        # спектр
        n_fft = int(self.fft_size_combo.currentData())
        if self.buffer.count >= n_fft:
            sv = self.buffer.last(n_fft)
            f, mag = compute_spectrum(sv, fs, window="hann")
            self.freq_curve.setData(f, mag)
            if self.autoscale_check.isChecked():
                self.freq_plot.enableAutoRange()

        # статистика раз в секунду
        self.updates_in_sec += 1
        now = time.monotonic()
        if now - self.last_update >= 1.0:
            self.fps = self.updates_in_sec / (now - self.last_update)
            self.updates_in_sec = 0
            self.last_update = now
            self.stats_label.setText(
                f"Пакетов: {self.parser.received}\n"
                f"Потеряно: {self.parser.dropped}\n"
                f"CRC-ошибок: {self.parser.crc_errors}\n"
                f"Отсчётов: {self.total_samples}\n"
                f"GUI FPS: {self.fps:.1f}"
            )

    def closeEvent(self, ev):
        self.stop()
        super().closeEvent(ev)