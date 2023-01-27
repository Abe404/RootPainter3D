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
    from model_utils import segment_3d, random_model

    image_shape = (72, 412, 412)  # depth, height, width, (z, y, x)
    image = np.random.random(image_shape)
    in_patch_shape = (64, 256, 256)
    out_patch_shape = (64-34, 256-34, 256-34)
    batch_size = 2
    cnn = random_model(classes=['structure_of_interest'])
    print('input image shape', image.shape)
    # test segmentation can be generated with the same shape as an input image.
    pred_maps = segment_3d(cnn, image, batch_size, in_patch_shape, out_patch_shape)
    print('pred maps len = ', len(pred_maps))
    print('pred maps[0].shape = ', pred_maps[0].shape)


def test_batch_loss_handles_overlapping_patches():
    """ A problem was identified where metrics were returned
        twice for overlapping regions (voxels), leading
        to an incorrect number of metrics.

        get_batch_loss should return the metrics
        accurately for patches that overlap.

        it is necessary to have overlapping patches
        as the patch size is fixed and some images 
        do not divide evenly by the patch size """

    from loss import get_batch_loss
    batch_size = 4
    image_shape = (73, 438, 561) # selected arbitrarily
    batch_shape = 
    # create fg annotation
    fg_patches = np.zeros((4, 
    # TODO: create bg annotation 
    # TODO: create network outputs 

    # TODO: compute expected metrics (tps, tns, fps, fns) given
    #       the output and annotation

    seg_patches = None # we are not interested in this functionality right now
    project_classes = ['structure_of_interest']
    batch_classes = ['structure_of_interest']


    """
    outputs shape torch.Size([4, 2, 18, 258, 258])
    fg patchs len 1
    fg patchs [0] len 1
    fg patch [0][0] shape torch.Size([52, 292, 292])
    """

    (loss, tps, tns, fps, fns) = get_batch_loss(
         outputs, fg_patches, bg_patches, 
         ignore_masks, seg_patches,
         batch_classes, project_classes,
         compute_loss=False)

    # Check that the metrics (tps, tns, fps, fns) returned
    # correspond to the expected metrics given the:
    #  - network outputs
    #  - fg annotation
    #  - bg annotation
    assert tps == expected_tps
    assert fps == expected_fps
    assert tns == expected_tns
    assert fns == expected_fns





