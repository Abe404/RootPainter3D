"""
Copyright (C) 2023 Abraham George Smith

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
import numpy as np

from trainer import Trainer
from startup import startup_setup


def test_segmentation_001():
    from model_utils import segment_3d, random_model, pad_then_segment_3d
    image_shape = (72, 412, 412)  # depth, height, width, (z, y, x)
    image = np.random.random(image_shape)
    in_patch_shape = (64, 256, 256)
    out_patch_shape = (64-34, 256-34, 256-34)
    batch_size = 2
    classes = ['structure_of_interest']
    cnn = random_model(classes)
    pred_maps = pad_then_segment_3d(
        cnn, image, batch_size,
        in_patch_shape[1], out_patch_shape[1],
        in_patch_shape[0], out_patch_shape[0])
    assert len(pred_maps) == len(classes)
    assert pred_maps[0].shape == image.shape


def test_validation():
    # implement a very simple validation function.
    # it will go through all images in the validation set and get their dice.
    pass 
