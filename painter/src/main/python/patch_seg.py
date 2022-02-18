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

import os
import time
import numpy as np
from PyQt5 import QtCore

from tcp_utils import request_patch_seg
from instructions import fix_instruction_paths


class SegmentPatchThread(QtCore.QThread):


    def __init__(self,  x, y, z, root_painter):
         super().__init__()
         self.x = x
         self.y = y
         self.z = z
         self.root_painter = root_painter

    def run(self):

        x = self.x
        y = self.y
        z = self.z
        root_painter = self.root_painter

        run_start = time.time()
        z = z + 1 # we dont segment the current slice. It gets confusing.
        # keep track of the region the user wanted to update. We dont allow changes outside of this region.
        z_valid_min = z
        z_valid_max = z_valid_min + root_painter.output_shape[0]

        # Move the centroid of the predicted region down by half the output depth
        # That way the whole output has a chance to be in the valid region.
        z = z + (root_painter.output_shape[0] // 2)

        # clip so it's at least output//2. This should allow the edge of the image to 
        # be resegmented.
        z = max((root_painter.output_shape[0] // 2), z)
        y = max((root_painter.output_shape[1] // 2), y)
        x = max((root_painter.output_shape[2] // 2), x)

        # image dimensions
        _, d, h, w = root_painter.annot_data.shape

        # also make sure we don't try to segment too far out on the other side
        z = min(d - (root_painter.output_shape[0] // 2), z)
        y = min(h - (root_painter.output_shape[1] // 2), y)
        x = min(w - (root_painter.output_shape[2] // 2), x)

        # and not more than the image shape
        # as only centroid is supplied, we must calculate left and right.
        z_start =  z - (root_painter.input_shape[0] // 2)
        z_end = z_start + root_painter.input_shape[0]
        y_start = y - (root_painter.input_shape[1] // 2)
        y_end = y_start + root_painter.input_shape[1]
        x_start = x - (root_painter.input_shape[2] // 2)
        x_end = x_start + root_painter.input_shape[2]

        start_time = time.time()
        content = {
            "file_name": root_painter.bounded_fname,
            "dataset_dir": os.path.join(root_painter.proj_location, 'bounded_images'),
            "model_dir": root_painter.model_dir,
            "z_start": z_start, "z_end": z_end,
            "y_start": y_start, "y_end": y_end,
            "x_start": x_start, "x_end": x_end,
            "patch_annot_dir": os.path.join(root_painter.proj_location, 'patch', 'annotation'),
            "classes": root_painter.classes
        }
        content = fix_instruction_paths(content, root_painter.sync_dir)
        # And then create an annotation for the patch input region
        # NOTE: annot_data shape is [2, d, h, w]
        #       First dimension is for bg (0) and fg (1)

        # if annotation goes outside the image, then we must pad it.
        pad_z_start = min(0, z_start) * -1
        pad_y_start = min(0, y_start) * -1
        pad_x_start = min(0, x_start) * -1 
        
        annot_patch = root_painter.annot_data[:,
                                              max(0, z_start):z_end,
                                              max(0, y_start):y_end,
                                              max(0, x_start):x_end]
        annot_patch = annot_patch.astype(bool)
        pad_z_end = root_painter.input_shape[0] - (annot_patch.shape[1] + pad_z_start)
        pad_y_end = root_painter.input_shape[1] - (pad_y_start + annot_patch.shape[2])
        pad_x_end = root_painter.input_shape[2] - (pad_x_start + annot_patch.shape[3])

        if sum([pad_z_start, pad_z_end, pad_y_start, pad_y_end, pad_x_start, pad_x_end]):
            annot_patch = np.pad(annot_patch, 
                                [(0, 0), # dont add channels
                                (pad_z_start, pad_z_end),
                                (pad_y_start, pad_y_end),
                                (pad_x_start, pad_x_end)],
                                mode='constant')

        patch_seg = request_patch_seg(annot_patch, content,
                                      root_painter.server_ip,
                                      root_painter.server_port)

        # TODO: I know 17 from memory. Compute '17' by 
        #       comparing input and output size
        seg_start_x = x_start + 17
        seg_start_y = y_start + 17
        seg_start_z = z_start + 17
        # first copy the seg data.
        new_seg_data = np.array(root_painter.seg_data)
        # update based on new patch data
        new_seg_data[
            seg_start_z:seg_start_z+patch_seg.shape[0],
            seg_start_y:seg_start_y+patch_seg.shape[1],
            seg_start_x:seg_start_x+patch_seg.shape[2]
        ] = patch_seg
        # now only update the displayed seg data based on the region the user intended to update
        root_painter.seg_data[z_valid_min:z_valid_max] = new_seg_data[z_valid_min:z_valid_max]
        for v in root_painter.viewers:
            if v.isVisible():
                v.update_seg_slice()
                v.update_outline()
        print('patch seg duration = ', time.time() - start_time)