"""
Navigate through slices in 3D image

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
# Too many instance attributes
# pylint: disable=R0902

from PyQt5 import QtWidgets
from PyQt5 import QtCore

class SliceNav(QtWidgets.QWidget):
    """ Navigate through slices in 3D image """
    changed = QtCore.pyqtSignal()

    def __init__(self, min_slice=0, max_slice=100):
        super().__init__()
        self.min_slice_idx = min_slice
        self.max_slice_idx = max_slice
        self.slice_idx = self.min_slice_idx
        self.init_ui()

    def update_range(self, new_image, mode):
        """ update range of slices based on shape of input image
            and view mode """
        if mode == 'axial':
            slice_count = new_image.shape[0]
        elif mode == 'coronal':
            slice_count = new_image.shape[1]
        elif mode == 'sagittal':
            slice_count = new_image.shape[2]
        else: 
            raise Exception(f"Unhandled mode:{mode}")

        self.max_slice_idx = slice_count - 1
        self.slider.setMaximum(self.max_slice_idx)

        # default to central slice.
        self.slice_idx = self.max_slice_idx // 2

        if self.slice_idx > self.max_slice_idx:
            self.slice_idx = self.max_slice_idx
            self.slider.setValue(self.slice_idx)

        self.slider.setValue(self.slice_idx)
        self.update_text()

    def update_text(self):
        """ Update label text. Can be useful when slice changes """
        # self.value_label.setText(f"{self.slice_idx+1}/{self.max_slice_idx+1}")
        self.value_label.setText(f"{self.slice_idx+1}")

    def init_ui(self):
        """ Create UI elements """
        self.layout = QtWidgets.QVBoxLayout()
        # self.label = QtWidgets.QLabel("Axial Slice")
        # self.label.setAlignment(QtCore.Qt.AlignCenter)
        # self.layout.addWidget(self.label)

        self.debounce = QtCore.QTimer()
        self.debounce.setInterval(5)
        self.debounce.setSingleShot(True)
        self.debounce.timeout.connect(self.debounced)

        self.slider_container = QtWidgets.QWidget()
        self.slider_layout = QtWidgets.QVBoxLayout()
        # self.slider_layout.addWidget(QtWidgets.QLabel("Axial"))
        self.slider_container.setLayout(self.slider_layout)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slider.setMinimum(self.min_slice_idx)
        self.slider.setMaximum(self.max_slice_idx)
        self.slider.setValue(self.slice_idx)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        #self.slider.setFixedHeight(600)
        self.slider.valueChanged.connect(self.value_changed)
        self.slider_layout.addWidget(self.slider)
        self.value_label = QtWidgets.QLabel()
        self.update_text()
        #self.value_label.setText(f"{self.slice_idx+1}/{self.max_slice_idx+1}")
        self.value_label.setText(f"{self.slice_idx+1}")

        self.slider_layout.addWidget(self.value_label)
        self.layout.addWidget(self.slider_container)
        self.setLayout(self.layout)
        self.setFixedWidth(65)
        self.setWindowTitle("Axial Position")

    def value_changed(self):
        """ when value changed update slice index"""
        self.slice_idx = self.slider.value()
        self.update_text()
        self.debounce.start()

    def up_slice(self):
        self.slice_idx = self.slice_idx + 1
        if self.slice_idx > self.max_slice_idx:
            self.slice_idx = self.max_slice_idx
        self.slider.setValue(self.slice_idx)
        self.update_text()
        self.debounce.start()

    def down_slice(self):
        self.slice_idx = self.slice_idx - 1
        if self.slice_idx < 1:
            self.slice_idx = 1
        self.slider.setValue(self.slice_idx)
        self.update_text()
        self.debounce.start()


    def debounced(self):
        """ trigger event only so often """
        self.changed.emit()
