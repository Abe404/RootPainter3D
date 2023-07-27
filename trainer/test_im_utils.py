"""
Test im_utils.py

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
import im_utils

def test_get_random_patch_even_fg():
    """ test that get_random_patch handles returns approximately correct fg/bg ratio
    """

    segs = [None]
    # the image will be padded with 17 on each edge when loaded
    # (to allow all regions of the image to be predicted at the boundary)

    im_shape = (100, 120, 140) # depth, height, width
    image = np.zeros(im_shape)
    image = np.pad(image, ((17,17), (17,17), (17, 17)), mode='constant')
    
    # annotation is padded equivalently
    annot = np.zeros([2] + list(im_shape))

    # set half the image to be foreground i.e bottom half
    annot[1][im_shape[0]//2:, :, :] = 1
    # the bg in this reason should be 0
    annot[0][im_shape[0]//2:, :, :] = 0

    # the bg in the top half should be 1
    annot[0][:im_shape[0]//2, :, :] = 1

    im_percent_fg = (np.sum(annot[1]) / annot[1].size) * 100
    
    # pad the annotation to be consistent with the image.
    annot = np.pad(annot, ((0, 0), (17,17), (17,17), (17, 17)), mode='constant')

    annots = [annot]
    fname = 'fake_data'
    
    in_w = 36 + (16*4)
    force_fg = False

    total_fg = 0
    total_bg = 0
    total_vox = 0

    for _ in range(1220):
        (annot_patches,
         _seg_patches,
         im_patch)  = im_utils.get_random_patch_3d(annots, segs,
                                                   image, fname,
                                                   force_fg,
                                                   in_w, in_w)
        # in this case where the patch size is smaller than the image,
        # all the patches should have the same size as the in_d, in_w
        assert im_patch.shape == (in_d, in_w, in_w)
        assert len(annot_patches) == 1
        assert annot_patches[0].shape == (2, in_w, in_w, in_w)
        annot_patch = annot_patches[0][:, 17:-17, 17:-17, 17:-17]
        total_vox += annot_patch[0].size
        total_bg += np.sum(annot_patch[0])
        total_fg += np.sum(annot_patch[0])

    total_fg_percent = (total_fg/total_vox) * 100
    total_bg_percent = (total_bg/total_vox) * 100

    # should be close to 50% for both image and sampled patch average
    assert im_percent_fg == 50, im_percent_fg
    assert abs(total_bg_percent - 50) < 3, total_bg_percent
    assert abs(total_fg_percent - 50) < 3, total_fg_percent

    

def test_get_random_patch_big():
    """ test that get_random_patch handles patch sizes larger than image
    """

    segs = [None]
    # the image will be padded with 17 on each edge when loaded
    # (to allow all regions of the image to be predicted at the boundary)

    im_shape = (100, 120, 140) # depth, height, width
    image = np.random.random(im_shape)
    image = np.pad(image, ((17,17), (17,17), (17, 17)), mode='constant')
    annot = np.zeros([2] + list(im_shape))
    annot[0] = np.ones(im_shape) # all bg (empty)
    # pad the annotation to be consistent with the image.
    annot = np.pad(annot, ((0, 0), (17,17), (17,17), (17, 17)), mode='constant')

    annots = [annot]
    fname = 'fake_data'
    
    # in this case where the patch size is bigger than the image,
    # in that specific dimension, patch should be same size as image
    # for other dimensions, it should have same as patch size

    in_d = 300
    in_w = 40
    force_fg = False

    for _ in range(20):
        (annot_patches,
         _seg_patches,
         im_patch)  = im_utils.get_random_patch_3d(annots, segs,
                                                   image, fname,
                                                   force_fg,
                                                   in_d, in_w)
        assert im_patch.shape == (image.shape[0], in_w, in_w)
        assert len(annot_patches) == 1
        assert annot_patches[0].shape == (2, image.shape[0], in_w, in_w)

    # in the case where the patch is bigger in all directions, just return the image
    in_d = 300
    in_w = 400
    force_fg = False
  
    for _ in range(20):
        (annot_patches,
         _seg_patches,
         im_patch)  = im_utils.get_random_patch_3d(annots, segs,
                                                   image, fname,
                                                   force_fg,
                                                   in_d, in_w)
        assert im_patch.shape == image.shape
        assert len(annot_patches) == 1
        assert annot_patches[0].shape == (2, image.shape[0], image.shape[1], image.shape[2])
        assert np.array_equal(im_patch, image)
