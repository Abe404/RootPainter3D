"""
Copyright (C) 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
import numpy as np


class ContrastSlider(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal(int, int)

    def __init__(self, presets):
        super().__init__()
        # min HU, max HU, brightness %
        self.presets = presets
        preset = self.presets[list(self.presets.keys())[0]] 
        self.hu_range = [preset[0], preset[1]]
        self.min_value = preset[0]
        self.max_value = preset[1]
        self.brightness_value = round(preset[2])
        self.initUI()

    def update_range(self, img):
        min_hu = np.min(img)
        max_hu = np.max(img)
        self.hu_range = [min_hu, max_hu]
        self.min_slider.setMinimum(self.hu_range[0])
        self.min_slider.setMaximum(self.hu_range[1])
        self.max_slider.setMinimum(self.hu_range[0])
        self.max_slider.setMaximum(self.hu_range[1])
        self.min_value_label.setText(str(self.min_value) + ' HU')
        self.max_value_label.setText(str(self.max_value) + ' HU')

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.debounce = QtCore.QTimer()
        self.debounce.setInterval(5)
        self.debounce.setSingleShot(True)
        self.debounce.timeout.connect(self.debounced)

        self.min_slider_container = QtWidgets.QWidget()
        self.min_slider_layout = QtWidgets.QHBoxLayout()
        self.min_slider_layout.addWidget(QtWidgets.QLabel("Min:"))
        self.min_slider_container.setLayout(self.min_slider_layout)
        self.min_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_slider.setMinimum(self.hu_range[0])
        self.min_slider.setMaximum(self.hu_range[1])
        self.min_slider.setValue(self.min_value)
        self.min_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.min_slider.setTickInterval((round((self.hu_range[1] - self.hu_range[0]) / 100)))
        self.min_slider.setFixedWidth(300)
        self.min_slider.valueChanged.connect(self.value_changed)
        self.min_slider_layout.addWidget(self.min_slider)
        self.min_value_label = QtWidgets.QLabel(str(self.min_value) + ' HU')
        self.min_slider_layout.addWidget(self.min_value_label)
        self.layout.addWidget(self.min_slider_container)

        self.max_slider_container = QtWidgets.QWidget()
        self.max_slider_layout = QtWidgets.QHBoxLayout()
        self.max_slider_layout.addWidget(QtWidgets.QLabel("Max:"))
        self.max_slider_container.setLayout(self.max_slider_layout)
        self.max_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_slider.setMinimum(self.hu_range[0])
        self.max_slider.setMaximum(self.hu_range[1])
        self.max_slider.setValue(self.max_value)
        self.max_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.min_slider.setTickInterval((round((self.hu_range[1] - self.hu_range[0]) / 100)))
        self.max_slider.setFixedWidth(300)
        self.max_slider.valueChanged.connect(self.value_changed)
        self.max_slider_layout.addWidget(self.max_slider)
        self.max_value_label = QtWidgets.QLabel(str(self.max_value) + ' HU')
        self.max_slider_layout.addWidget(self.max_value_label)
        self.layout.addWidget(self.max_slider_container)


        self.brightness_slider_container = QtWidgets.QWidget()
        self.brightness_slider_layout = QtWidgets.QHBoxLayout()
        self.brightness_slider_layout.addWidget(QtWidgets.QLabel("Brightness:"))
        self.brightness_slider_container.setLayout(self.brightness_slider_layout)
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setMinimum(0)
        self.brightness_slider.setMaximum(300)
        self.brightness_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.brightness_slider.setTickInterval(1)
        self.brightness_slider.setFixedWidth(300)
        self.brightness_slider.setValue(self.brightness_value)
        self.brightness_slider.valueChanged.connect(self.value_changed)
        self.brightness_slider_layout.addWidget(self.brightness_slider)
        self.brightness_value_label = QtWidgets.QLabel(str(self.brightness_value) + ' %')
        self.brightness_slider_layout.addWidget(self.brightness_value_label)
        self.layout.addWidget(self.brightness_slider_container)

        self.setLayout(self.layout)
        self.setWindowTitle("Contrast Settings")
        self.setFixedWidth(525)

        self.preset_container = QtWidgets.QWidget()
        self.preset_layout = QtWidgets.QHBoxLayout()
        self.preset_container.setLayout(self.preset_layout)
        self.preset_layout.addWidget(QtWidgets.QLabel("Select preset:"))

        presets_combo = QtWidgets.QComboBox(self)
        self.presets_combo = presets_combo
        for preset_name in self.presets:
            presets_combo.addItem(preset_name)
        presets_combo.activated[str].connect(self.preset_selected)
        self.preset_layout.addWidget(presets_combo)
        self.layout.addWidget(self.preset_container)

    def preset_selected(self, preset_name):
        preset = self.presets[preset_name]
        self.min_slider.setValue(preset[0])
        self.max_slider.setValue(preset[1])
        self.brightness_slider.setValue(preset[2])
        self.presets_combo.setCurrentText(preset_name)
        self.value_changed()

    def value_changed(self):
        self.min_value = self.min_slider.value()
        self.min_value_label.setText(str(self.min_value) + ' HU')
        self.max_value = self.max_slider.value()
        self.max_value_label.setText(str(self.max_value) + ' HU')
        self.brightness_value = self.brightness_slider.value()
        self.brightness_value_label.setText(str(self.brightness_value) + ' %')
        self.debounce.start()

    def debounced(self):
        self.changed.emit(self.min_value, self.max_value)
