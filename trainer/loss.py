"""
Copyright (C) 2019 Abraham George Smith

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
# torch has no sum member
# pylint: disable=E1101

import torch
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
import numpy as np


def dice_loss(predictions, labels):
    """ based on loss function from V-Net paper """
    softmaxed = softmax(predictions, 1)
    predictions = softmaxed[:, 1, :]  # just the root probability.
    labels = labels.float()
    preds = predictions.contiguous().view(-1)
    labels = labels.view(-1)
    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))


def combined_loss(predictions, labels):
    """ mix of dice and cross entropy """
    # if they are bigger than 1 you get a strange gpu error
    # without a stack track so you will have no idea why.
    assert torch.max(labels) <= 1
    if torch.sum(labels) > 0:
        return (dice_loss(predictions, labels) +
                (0.3 * cross_entropy(predictions, labels)))
    # When no roots use only cross entropy
    # as dice is undefined.
    return 0.3 * cross_entropy(predictions, labels)


def get_batch_loss(outputs, batch_fg_tiles, batch_bg_tiles,
                   batch_classes, project_classes,
                   compute_loss):
    """

        outputs - predictions from neural network (not softmaxed)
        batch_fg_tiles - list of tiles, each tile is binary map of foreground annotation
        batch_bg_tiles - list of tiles, each tile is binary map of background annotation
        compute_loss - can be false if only metrics are required.

        returns
            batch_loss - loss used to update the network
            tps - true positives for batch
            tns - true negatives for batch
            fps - false positives for batch
            fns - false negatives for batch
            defined_total - number of pixels with annotation defined.
    """

    defined_total = 0
    class_losses = [] # loss for each class

    instance_tps = [0 for _ in range(outputs.shape[0])]
    instance_tns = list(instance_tps)
    instance_fps = list(instance_tps)
    instance_fns = list(instance_tps)


    for unique_class in project_classes:

        # for each class we need to get a tensor with shape
        # [batch_size, 2 (bg,fg), h, w]

        # fg,bg pairs related to this class for each im_tile in the batch
        class_outputs = []

        # The fg tiles (ground truth)
        fg_tiles = []
        masks = [] # and regions of the image that were annotated.
        
        # go through each instance in the batch.
        for im_idx in range(outputs.shape[0]):

            # go through each class for this instance.
            for i, classname in enumerate(batch_classes[im_idx]):
                
                # if the classname is the class we are interested in
                if classname == unique_class:

                    # foregorund and background channels
                    fg_tile = batch_fg_tiles[im_idx][i][17:-17,17:-17,17:-17]
                    bg_tile = batch_bg_tiles[im_idx][i][17:-17,17:-17,17:-17]
                    mask = fg_tile + bg_tile
                    class_idx = project_classes.index(classname) * 2 # posiion in output.
                    class_output = outputs[im_idx][class_idx:class_idx+2]
                    mask = mask.cuda()
                    fg_tile = fg_tile.cuda()

                    # I want to get tps, tns, fps and fns 
                    # from fg_tile, mask, class_outputs
                    softmaxed = softmax(class_output, 0)
                    fg_prob = softmaxed[1]
                    fg_prob = fg_prob * mask
                    class_pred = fg_prob > 0.5   
                    class_pred = class_pred[mask > 0]
                    fg = fg_tile[mask > 0]
                    
                    instance_tps[im_idx] += torch.sum((fg == 1) * (class_pred == 1)).cpu().numpy()
                    instance_tns[im_idx] += torch.sum((fg == 0) * (class_pred == 0)).cpu().numpy()
                    instance_fps[im_idx] += torch.sum((fg == 0) * (class_pred == 1)).cpu().numpy()
                    instance_fns[im_idx] += torch.sum((fg == 1) * (class_pred == 0)).cpu().numpy()

                    masks.append(mask)
                    fg_tiles.append(fg_tile)
                    class_outputs.append(class_output)

        if not len(fg_tiles):
            continue
        
        if compute_loss:
            fg_tiles = torch.stack(fg_tiles).cuda()
            masks = torch.stack(masks).cuda()
            class_outputs = torch.stack(class_outputs)
            softmaxed = softmax(class_outputs, 1)
            # just the foreground probability.
            foreground_probs = softmaxed[:, 1]
            # remove any of the predictions for which we don't have ground truth
            # Set outputs to 0 where annotation undefined so that
            # The network can predict whatever it wants without any penalty.
            class_outputs[:, 0] *= masks
            class_outputs[:, 1] *= masks
            class_loss = combined_loss(class_outputs, fg_tiles)
            class_losses.append(class_loss)
    if compute_loss:
        return torch.mean(torch.stack(class_losses)), instance_tps, instance_tns, instance_fps, instance_fns
    return None, instance_tps, instance_tns, instance_fps, instance_fns
