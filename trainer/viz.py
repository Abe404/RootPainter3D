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


import os
from skimage.io import imsave
import numpy as np
import torch

def save_patches_image(patches, patch_names, out_folder):

    tiles = []
    largest_h = max(p.shape[1] for p in patches)
    largest_w = max(p.shape[2] for p in patches)

    for p in patches:
        mid = p.shape[0] //2

        tile = p[mid]

        if torch.is_tensor(tile):
            tile = tile.detach().cpu().numpy()
        tile = tile.astype(float)

        # 0-1 scale
        tile -= np.min(tile)
        tile /= np.max(tile)

        # pad to largest so they can all be shown together
        to_pad_h = largest_h - p.shape[1]
        h_pad_start = to_pad_h // 2
        h_pad_end = to_pad_h - h_pad_start

        to_pad_w = largest_w - p.shape[2]
        w_pad_start = to_pad_w // 2
        w_pad_end = to_pad_w - w_pad_start
         
        tile = np.pad(tile, ((h_pad_start, h_pad_end), (w_pad_start, w_pad_end)))
        tiles.append(tile)

    out_im = np.hstack(tiles)
    num_frames = len(os.listdir(out_folder))
    out_path = f'{out_folder}/{str(num_frames).zfill(4)}_'
    out_path += '_'.join(patch_names) + '.jpg'
    print('saving', out_path)
    imsave(out_path, out_im)
    return num_frames
