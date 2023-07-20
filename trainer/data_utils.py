
"""
Form a batch

Copyright (C) 2021-2023 Abraham George Smith

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
import torch
import im_utils

def pad_patches_to_dimension(patches, max_d, max_h, max_w):
    patches_padded = []
    for patch in patches:
        if torch.is_tensor(patch):
            patch = patch.numpy()
        new_im_patch, _was_padded = im_utils.maybe_pad_image_to_pad_size(
            patch, (max_d, max_h, max_w))
        patches_padded.append(new_im_patch)
    return np.array(patches_padded)


def collate_fn(batch):
    num_items = len(batch)
    im_patches = []
    batch_fgs = []
    batch_bgs = []
    batch_segs = []
    batch_classes = []
    ignore_masks = []
    for i in range(num_items):
        item = batch[i]
        im_patches.append(item[0])
        batch_fgs.append(item[1])
        batch_bgs.append(item[2])
        ignore_masks.append(item[3])
        batch_segs.append(item[4])
        batch_classes.append(item[5])

    max_d = max(i.shape[0] for i in im_patches)
    max_h = max(i.shape[1] for i in im_patches)
    max_w = max(i.shape[2] for i in im_patches)
    
    im_patches = pad_patches_to_dimension(im_patches, max_d, max_h, max_w)

    batch_fgs_padded = []
    # for the list of fg annotations for each item in the batch
    for fgs in batch_fgs:
        batch_fgs_padded.append(pad_patches_to_dimension(fgs, max_d, max_h, max_w))

    batch_bgs_padded = []
    # for the list of bg annotations for each item in the batch
    for bgs in batch_bgs:
        batch_bgs_padded.append(pad_patches_to_dimension(bgs, max_d, max_h, max_w))

    ignore_masks_padded = pad_patches_to_dimension(ignore_masks,
                                                   max_d-34, max_h-34, max_w-34)

    return (im_patches, batch_fgs_padded, batch_bgs_padded,
            ignore_masks_padded, batch_segs, batch_classes)
