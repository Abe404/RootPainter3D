"""
Copyright (C) 2022 Abraham George Smith

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

#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914
import os


from PyQt5 import QtWidgets
from PyQt5 import QtCore
import numpy as np
from base_extract import BaseExtractWidget
import im_utils

image_props_headers = ['fname',
                       'image_volume_voxels',
                       'fg_volume_voxels', 
                       'bg_volume_voxels', 
                       'fg_volume_percent',
                       'bg_volume_percent']


def get_image_props(seg_dir, fname, _):
    # warning some images might have different axis order or might need flipping
    # we ignore that here.
    seg_im = im_utils.load_seg(os.path.join(seg_dir, fname))
    seg_im = seg_im.astype(bool).astype(int)
    total_volume_voxels = seg_im.size
    fg_volume_voxels = np.sum(seg_im)
    bg_volume_voxels = total_volume_voxels - fg_volume_voxels
    fg_volume_percent = (fg_volume_voxels/total_volume_voxels) * 100
    bg_volume_percent = (bg_volume_voxels/total_volume_voxels) * 100

    return [fname, total_volume_voxels, fg_volume_voxels, 
            bg_volume_voxels, fg_volume_percent, bg_volume_percent]


class ExtractSegImagePropsWidget(BaseExtractWidget):
    def __init__(self):
        super().__init__(
            "Segmentation Properites",
            image_props_headers,
            get_image_props)

        info_label = QtWidgets.QLabel()
        info_label.setFixedWidth(600)
        info_label.setText("Extracted properties include image volume, foreground volume and background volume.")
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label)
