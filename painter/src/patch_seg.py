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

from logging import root
import os
import time
import numpy as np
from functools import partial
from PyQt5 import QtCore
from enum import Enum


from tcp_utils import request_patch_seg, establish_connection
from instructions import fix_instruction_paths


class SegmentPatchThread(QtCore.QThread):

    complete = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, annot_patch, content, server_ip, server_port):
         super().__init__()
         self.annot_patch = annot_patch
         self.content = content
         self.server_ip = server_ip
         self.server_port = server_port

    def run(self):
        # if running without data, just establish connection
        if self.annot_patch is None:
            establish_connection(self.server_ip, self.server_port)
        else:
            patch_seg = request_patch_seg(self.annot_patch,
                                          self.content)
            self.complete.emit(patch_seg)


class SegState(Enum):
    IDLE = 1
    SEGMENTING = 2 

class PatchSegmentor():
    """ 
    Holds the reference to root_painter
    and a reference to a QThread object that
    handles the server request to get the segmented patch.
    Then updates the root_painter segmentation data and 
    and triggers re-render.
    """

    def __init__(self, root_painter):
        self.root_painter = root_painter
        self.state = SegState.IDLE
        self.seg_patch_thread = SegmentPatchThread(None, None,
                                                   root_painter.server_ip,
                                                   root_painter.server_port)
        self.seg_patch_thread.start()


    def patch_received(self, patch_seg, fname):
        self.state = SegState.IDLE
        # only update if the image is still the same
        if self.root_painter.fname == fname:
            # TODO: I know 17 from memory. Compute '17' by 
            #       comparing input and output size
            seg_start_x = self.x_start + 17
            seg_start_y = self.y_start + 17
            seg_start_z = self.z_start + 17
            # first copy the seg data.
            new_seg_data = np.array(self.root_painter.seg_data)
            # update based on new patch data
            new_seg_data[
                seg_start_z:seg_start_z+patch_seg.shape[0],
                seg_start_y:seg_start_y+patch_seg.shape[1],
                seg_start_x:seg_start_x+patch_seg.shape[2]
            ] = patch_seg
            # now only update the displayed seg data based on the region the user intended to update
            self.root_painter.seg_data[
                self.z_valid_min:self.z_valid_max] = new_seg_data[self.z_valid_min:self.z_valid_max]
            for v in self.root_painter.viewers:
                if v.isVisible():
                    v.update_seg_slice()
                    v.update_outline()
            print('patch seg duration = ', time.time() - self.start_time)


    def segment_patch(self, centroid_x, centroid_y, z):
        """
        Prepare the data from the annotation and passes only the essentials.
        to a thread that uses a TCP SSL socket to send the patch details to the server
        and get back the patch segmentation.

        When the segmentation is received from the server, the patch_received event handler
        will be invoked.
        """
        if self.state == SegState.IDLE:
            self.state = SegState.SEGMENTING
            self.start_time = time.time()
            x = centroid_x
            y = centroid_y
            z = z
            root_painter = self.root_painter
            z = z + 1 # we dont segment the current slice. It gets confusing.
            # keep track of the region the user wanted to update. We dont allow changes outside of this region.
            self.z_valid_min = z
            self.z_valid_max = self.z_valid_min + root_painter.output_shape[0]

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
            self.z_start =  z - (root_painter.input_shape[0] // 2)
            z_end = self.z_start + root_painter.input_shape[0]
            self.y_start = y - (root_painter.input_shape[1] // 2)
            y_end = self.y_start + root_painter.input_shape[1]
            self.x_start = x - (root_painter.input_shape[2] // 2)
            x_end = self.x_start + root_painter.input_shape[2]

            content = {
                # required to know which image to load
                "file_name": root_painter.fname, 
                "dataset_dir": root_painter.dataset_dir,
                # required to know where to load the model from
                "model_dir": root_painter.model_dir,
                # which region of the image to segment
                "z_start": self.z_start, "z_end": z_end,
                "y_start": self.y_start, "y_end": y_end,
                "x_start": self.x_start, "x_end": x_end,
                # which classes to use with this model
                "classes": root_painter.classes
            }
            content = fix_instruction_paths(content, root_painter.sync_dir)
            # And then create an annotation for the patch input region
            # NOTE: annot_data shape is [2, d, h, w]
            #       First dimension is for bg (0) and fg (1)

            # if annotation goes outside the image, then we must pad it.
            pad_z_start = min(0, self.z_start) * -1
            pad_y_start = min(0, self.y_start) * -1
            pad_x_start = min(0, self.x_start) * -1 
            
            annot_patch = root_painter.annot_data[:,
                                                max(0, self.z_start):z_end,
                                                max(0, self.y_start):y_end,
                                                max(0, self.x_start):x_end]
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
            self.seg_patch_thread = SegmentPatchThread(annot_patch, content,
                                                    root_painter.server_ip,
                                                    root_painter.server_port)

            self.seg_patch_thread.complete.connect(partial(self.patch_received, fname=self.root_painter.fname))
            self.seg_patch_thread.start()