""" Tests for U-Net module

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

import random
import torch
from unet3d import UNet3D
from model_utils import get_device


def test_segment_mixed_sizes():
    """ the u-net network should be able to segment different sized input
        and always return an output that is exactly 34 voxels smaller on each dimension
        due to valid padding in the network. So that is 17 voxels on each edge of the input
        patch that is removed in the output. """
    # 3 channels to enable optional annotation as input.
    model = UNet3D(num_classes=1, im_channels=1) 
    device = get_device()
    model.to(device)
    random_tries = 60 
    for _ in range(random_tries):
        d = random.randint(1, 300)
        h = random.randint(1, 300)
        w = random.randint(1, 300)
        try:
            inputs = torch.zeros((1, 1, d, h, w)).to(device)
            outputs = model(inputs)
        except Exception as _e:
            print(_e, 'When trying with', d, h, w)
            assert not _e

        for idx in [2,3,4]:
            debug_str = f'\noutput_shape: {outputs.shape}, \ninput_shape: {inputs.shape}. \n'
            shape_diff = outputs.shape[idx] - inputs.shape[idx]
            debug_str += f'Wrong at index {idx} '
            debug_str += f'(output {outputs.shape[idx]}, input {inputs.shape[idx]}) '
            debug_str += f'as diff should be -34 but is {shape_diff}'
            assert shape_diff == -34, debug_str
