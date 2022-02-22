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

def segment_full_image(root_painter):
    # send instruction to segment the image.
    root_painter.send_instruction('segment', {
        "dataset_dir": root_painter.dataset_dir,
        "seg_dir": root_painter.seg_dir,
        "file_names": [root_painter.fname],
        "message_dir": root_painter.message_dir,
        "model_dir": root_painter.model_dir,
        "classes": root_painter.classes # used for saving segmentation output to correct directories
    })

