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
import torch

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
    batch_size = 2

    # selected arbitrarily, with depth slightly bigger than 1 patch
    image_depth = 18+7
    image_height = 258
    image_width = 258

    image_shape = (image_depth, image_height, image_width) 

    batch_out_depth = 18
    batch_out_width = 258
    channel_count = 2 # implies single class output.
    # output size logged from network output.
    batch_out_shape = (batch_size, channel_count,
                       batch_out_depth, batch_out_width, batch_out_width) 

    network_padding = 17 # amount that network output is reduced on each side due to padding
    batch_in_depth = 18 + (network_padding*2)
    batch_in_width = 258 + (network_padding*2)

    assert image_depth % batch_out_depth > 0, ("check that image depth is "
        "not a multiple of batch shape, to test overlapping patch metrics computation.")

    # we need to create annotation for the image, to allow computation of metrics
    fg_patch = np.zeros((batch_in_depth, batch_in_width, batch_in_width))
    fg_patches = [[fg_patch], [fg_patch]] # for each instance, for each class
    bg_patch = np.ones((batch_in_depth, batch_in_width, batch_in_width))
    bg_patches = [[bg_patch], [bg_patch]] # for each instance, for each class

    # move to torch tensor as tested loss function requires this.
    # FIXME split apart the tested function so metrics computation is seperate and not on GPU.
    for i in range(len(fg_patches)):
        fg_patches[i][0] = torch.tensor(fg_patches[i][0])
    for i in range(len(bg_patches)):
        bg_patches[i][0] = torch.tensor(bg_patches[i][0])

    network_output = np.zeros(batch_out_shape)
    # network predicts 1 for background.
    network_output[:, 0] = np.ones((batch_out_depth, batch_out_width, batch_out_width))

    network_output = torch.tensor(network_output).cuda()
   
    # will implement later to get test to pass
    ignore_masks = None 

    # Compute expected metrics (tps, tns, fps, fns) given
    # the output and annotation

    # in this simplified case, the output is all 0.
    # there is also no foreground and all background. 
    expected_fp = 0
    expected_fn = 0
    expected_tn = np.prod(image_shape)
    expected_tp = 0

    seg_patches = None # we are not interested in this functionality right now
    project_classes = ['structure_of_interest']
    # for each instance in batch, have list of classes present for that instance.
    batch_classes = batch_size * [['structure_of_interest']]
    

    (loss, tps, tns, fps, fns) = get_batch_loss(
         network_output, fg_patches, bg_patches, 
         ignore_masks, seg_patches,
         batch_classes, project_classes,
         compute_loss=False)

    assert len(tps) == len(fg_patches) # corresponds to total number of patches/instances

    assert np.sum(tns) > 0, ('network predicted 0 tns but should predict many tns as '
                             'everything was background in the output and the '
                             ' specified annotation')

    assert np.sum(tps) == expected_tp
    assert np.sum(fps) == expected_fp
    assert np.sum(tns) == expected_tn 
    assert np.sum(fns) == expected_fn

