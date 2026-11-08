import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyqtgraph as pg
from PyQt5 import QtWidgets

from main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, background='k', foreground='w')
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()