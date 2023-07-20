
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

    max_d = max(i.shape[1] for i in im_patches)
    max_h = max(i.shape[2] for i in im_patches)
    max_w = max(i.shape[3] for i in im_patches)
    
    im_patches_padded = []
    for im_patch in im_patches:
        old_patch = im_patch[0]
        if torch.is_tensor(old_patch):
            old_patch = old_patch.numpy()
        new_im_patch, _was_padded = im_utils.maybe_pad_image_to_pad_size(
            old_patch, (max_d, max_h, max_w))
        # add channel dimension back as it is expected.
        im_patches_padded.append(np.expand_dims(new_im_patch, 0)) 
    im_patches = np.array(im_patches_padded)

    batch_fgs_padded = []
    for fgs in batch_fgs:
        new_fgs = []
        for fg_patch in fgs:
            if torch.is_tensor(fg_patch):
                fg_patch = fg_patch.numpy()
            new_fg_patch, _was_padded = im_utils.maybe_pad_image_to_pad_size(
                fg_patch, (max_d, max_h, max_w))
            new_fgs.append(new_fg_patch)
        # append all fg patches for this image to the list for the batch
        batch_fgs_padded.append(new_fgs) 

    batch_bgs_padded = []
    for bgs in batch_bgs:
        new_bgs = []
        for bg_patch in bgs:
            if torch.is_tensor(bg_patch):
                bg_patch = bg_patch.numpy()
            new_bg_patch, _was_padded = im_utils.maybe_pad_image_to_pad_size(
                bg_patch, (max_d, max_h, max_w))
            new_bgs.append(new_bg_patch)
        # append all bg patches for this image to the list for the batch
        batch_bgs_padded.append(new_bgs) 

    im_patches = np.array(im_patches_padded)
    return im_patches, batch_fgs_padded, batch_bgs_padded, ignore_masks, batch_segs, batch_classes
