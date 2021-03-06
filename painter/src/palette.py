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


Palette: Provides a way to add, edit and remove classes
"""

from PyQt5 import QtWidgets
from PyQt5 import QtCore

class BrushEditWidget(QtWidgets.QWidget):
    """
    Provide a way for a user to edit the name and colour of a brush
    """
    
    changed = QtCore.pyqtSignal()
    removed = QtCore.pyqtSignal()

    def __init__(self, name, show_remove):
        super().__init__()
        self.name = name
        self.initUI(show_remove)

    def initUI(self, show_remove):
        # Provide user with a way to edit the brush name
        self.layout = QtWidgets.QHBoxLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setText(self.name)
        self.name_edit.textChanged.connect(self.text_changed)
        self.layout.addWidget(self.name_edit)
    
        if show_remove:
            self.remove_btn = QtWidgets.QPushButton('Remove')
            self.remove_btn.clicked.connect(self.removed.emit)
            self.layout.addWidget(self.remove_btn)

        self.setLayout(self.layout)

    def text_changed(self):
        new_text = self.name_edit.text()
        self.name = new_text
        self.changed.emit()


class PaletteEditWidget(QtWidgets.QWidget):
    """ Add, edit and remove brushes """

    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.brush_widgets = []
        self.initUI()


    def initUI(self):
        label = QtWidgets.QLabel()
        label.setText("Classes:")
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(label)

        # Use a container for the brushes so the add brush widget can go after.
        self.brushes_container = QtWidgets.QWidget()
        self.brushes_layout = QtWidgets.QVBoxLayout()
        self.brushes_container.setLayout(self.brushes_layout)
        self.layout.addWidget(self.brushes_container)

        # Default brush
        self.add_brush('foreground', show_remove=False)

        # btn to add brushes
        self.add_brush_btn = QtWidgets.QPushButton('Add class')
        self.add_brush_btn.clicked.connect(self.add_brush)
        self.layout.addWidget(self.add_brush_btn)

    def get_new_name(self):
        return f"class_{len(self.brush_widgets) + 1}"

    def add_brush(self, name=None, show_remove=True):
        if not name:
            name = self.get_new_name()

        brush = BrushEditWidget(name, show_remove)
        self.brush_widgets.append(brush)

        brush.removed.connect(self.remove_brush)
        self.brushes_layout.addWidget(brush)
        self.changed.emit()

    def remove_brush(self):
        brush = self.sender()
        self.brush_widgets.remove(brush)
        self.brushes_layout.removeWidget(brush)
        self.changed.emit()

    def get_brush_data(self):
        """ Used for saving the class names to JSON file """
        return [b.name for b in self.brush_widgets]
