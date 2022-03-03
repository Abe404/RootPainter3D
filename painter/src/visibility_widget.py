"""
Show visibility status of segmentation, image and annotation.

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

# pylint: disable=E0611, C0111, C0111, R0903, I1101
from PyQt5 import QtWidgets

class VisibilityWidget(QtWidgets.QWidget):

    def __init__(self, layout_class, parent, show_guide=False):
        super().__init__()
        self.initUI(layout_class, parent, show_guide)

    def initUI(self, layout_class, parent, show_guide):
        # container goes full width to allow contents to be center aligned within it.
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        # left, top, right, bottom
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        container_layout = layout_class()
        self.setLayout(container_layout)
        container_layout.setContentsMargins(0, 0, 0, 0)

        seg_checkbox = QtWidgets.QCheckBox("Segmentation (S)")
        container_layout.addWidget(seg_checkbox)

        annot_checkbox = QtWidgets.QCheckBox("Annotation (A)")
        container_layout.addWidget(annot_checkbox)

        im_checkbox = QtWidgets.QCheckBox("Image (I)")
        container_layout.addWidget(im_checkbox)

        outline_checkbox = QtWidgets.QCheckBox("Outline (T)")
        container_layout.addWidget(outline_checkbox)

        # the guide image is optional and only enabled for some projects
        if show_guide:
            guide_image_checkbox = QtWidgets.QCheckBox("Guide (H)")
            container_layout.addWidget(guide_image_checkbox)
            guide_image_checkbox.setChecked(False)
            self.guide_image_checkbox = guide_image_checkbox
            self.guide_image_checkbox.stateChanged.connect(parent.guide_checkbox_change)

        seg_checkbox.setChecked(False)
        annot_checkbox.setChecked(True)
        im_checkbox.setChecked(True)
        outline_checkbox.setChecked(False)

  

        self.seg_checkbox = seg_checkbox
        self.annot_checkbox = annot_checkbox
        self.im_checkbox = im_checkbox
        self.outline_checkbox = outline_checkbox

        # connecting events to parent - tight coupling is OK if that's all we want.
        self.seg_checkbox.stateChanged.connect(parent.seg_checkbox_change)
        self.annot_checkbox.stateChanged.connect(parent.annot_checkbox_change)
        self.im_checkbox.stateChanged.connect(parent.im_checkbox_change)
        self.outline_checkbox.stateChanged.connect(parent.outline_checkbox_change)
 