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
from metrics import Metrics
from model_utils import random_model, pad_then_segment_3d
from loss import get_batch_loss


def test_segmentation_001():
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


def test_batch_loss_handles_overlapping_patches():
    """ A problem was identified where metrics were returned
        twice for overlapping regions (voxels), leading
        to an incorrect number of metrics.

        get_batch_loss should return the metrics
        accurately for patches that overlap.

        it is necessary to have overlapping patches
        as the patch size is fixed and some images 
        do not divide evenly by the patch size """

    batch_size = 2

    # selected arbitrarily, with depth slightly bigger than 1 patch
    image_depth = 18+3
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
    for i, fg_patch in enumerate(fg_patches):
        fg_patches[i][0] = torch.tensor(fg_patch[0])

    for i, bg_patch in enumerate(bg_patches):
        bg_patches[i][0] = torch.tensor(bg_patch[0])

    network_output = np.zeros(batch_out_shape)
    # network predicts 1 for background.
    network_output[:, 0] = np.ones((batch_out_depth, batch_out_width, batch_out_width))

    network_output = torch.tensor(network_output).cuda()
    
    # ignore masks define which regions should be ignore for each instance in the batch
    ignore_masks = []
    # first instance can have nothing ignore.
    ignore_masks.append(np.zeros((batch_out_depth, batch_out_width, batch_out_width)))
    # second instance has overlap with the first instance, so some should
    # be 'ignored' i.e not computed in the metrics computation.
    ignore_mask2 = np.zeros((batch_out_depth, batch_out_width, batch_out_width))

    # first 18 slices of the depth should be ignored
    # as we know from the defined image shape and batch shapes
    # that these will be overlapping.
    
    # image_depth = 21
    # batch_depth = 18
    # So that means after the first patch we have information for 18 out of 21 slices
    # that means we use the second patch to get the remaining information 
    # which is 21-18 slices (as we have information for the 18).
    # That means the second patch should only give us information for 21-18=3 slices.
    # so we should ignore all of the second patch except for 3 slices
    # as the second patch is 18 slices, we want to ignore 18-3 = 15 slices.
    # giving us just the last 3 slices to get information for all 21 slices.
    ignore_mask2[0:15] = 1 

    ignore_masks.append(ignore_mask2)

    # Compute expected metrics (tps, tns, fps, fns) given
    # the output and annotation

    # in this simplified case, the output is all 0.
    # there is also no foreground and all background. 
    expected_metrics = Metrics(fp=0, fn=0, tn=np.prod(image_shape), tp=0)

    seg_patches = None # we are not interested in this functionality right now
    project_classes = ['structure_of_interest']
    # for each instance in batch, have list of classes present for that instance.
    batch_classes = batch_size * [['structure_of_interest']]
    
    (_loss, metrics_list) = get_batch_loss(
         network_output, fg_patches, bg_patches, 
         ignore_masks, seg_patches,
         batch_classes, project_classes,
         compute_loss=False)

    assert len(metrics_list) == len(fg_patches) # corresponds to total number of patches/instances

    batch_metrics = Metrics.sum(metrics_list)
    assert batch_metrics.tn > 0, ('network predicted 0 tns '
        'but should predict many tns as everything was background in the '
        'output and the specified annotation')
    assert batch_metrics == expected_metrics
