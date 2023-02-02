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
from torch.nn.functional import l1_loss
import numpy as np

from metrics import Metrics, metrics_from_binary_masks

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



# FIXME too much going on in this function. Split the metrics computation into another function.
# And avoid for loops where possible.
def get_batch_loss(outputs, batch_fg_patches, batch_bg_patches,
                   batch_ignore_masks, batch_seg_patches,
                   batch_classes, project_classes,
                   compute_loss):

    
    assert (isinstance(batch_classes, list) and isinstance(batch_classes[0], list)), (
            f"assert batch_class is a list of lists, batch_classes: {batch_classes}")


    if batch_ignore_masks is not None:
        for i, m in enumerate(batch_ignore_masks):
            assert isinstance(m, np.ndarray), (
                    f"masks should be numpy arrays, mask type: {type(m)}")
            assert (len((m.shape)) == 3), (
                "ignore masks should have 3 dimensions to match mask"
                f"mask {i} shape: {m.shape}")
            assert m.shape == outputs[0][0].shape, (f"ignore mask shape {m.shape}"
                f" should match output shape of {outputs[0][0].shape}")
    """
        outputs - predictions from neural network (not softmaxed)
        batch_fg_patches - list of patches, each patch is binary map of foreground annotation
        batch_bg_patches - list of patches, each patch is binary map of background annotation
        compute_loss - can be false if only metrics are required.

        returns
            batch_loss - loss used to update the network
            tps - true positives for batch
            tns - true negatives for batch
            fps - false positives for batch
            fns - false negatives for batch
    """

    class_losses = [] # loss for each class
    
    instances_metrics = [Metrics() for _ in range(outputs.shape[0])]

    for unique_class in project_classes:

        # FIXME: Why do we have seperate channels for fg and bg.
        #        when this could be represented by a single fg channel?

        # for each class we need to get a tensor with shape
        # [batch_size, 2 (bg,fg), h, w]

        # fg,bg pairs related to this class for each im_patch in the batch
        class_outputs = []

        # The fg patches (ground truth)
        fg_patches = []
        masks = [] # and regions of the image that were annotated.
        seg_patches = []
        seg_class_outputs = []
        
        # go through each instance in the batch.
        for im_idx in range(outputs.shape[0]):

            # go through each class for this instance.
            for i, classname in enumerate(batch_classes[im_idx]):
                
                # if the classname is the class we are interested in
                if classname == unique_class:
                    
                    # FIXME: where does this 17 come from? It's connected to the specific
                    # network so perhaps we should
                    # load it from some properties/setting from the network class / file?

                    # foregorund and background channels
                    fg_patch = batch_fg_patches[im_idx][i][17:-17,17:-17,17:-17]
                    bg_patch = batch_bg_patches[im_idx][i][17:-17,17:-17,17:-17]

                    if batch_ignore_masks is not None:
                        ignore_mask = batch_ignore_masks[im_idx] # some regions are ignored, as they overlap with other patches
                    else:
                        ignore_mask = np.zeros(bg_patch.shape) # dont ignore anything unless mask is specified.

                    mask = fg_patch + bg_patch # all locations where annotation is defined in the annotation masks
                    class_idx = project_classes.index(classname) * 2 # posiion in output.

                    # FIXME have single channel output for each class
                    # Right now there is two output channels per class, 
                    # with one for bg and one for fg
                    class_output = outputs[im_idx][class_idx:class_idx+2] 

                    mask = mask.cuda()
                    fg_patch = fg_patch.cuda()
                    
                    # I want to get tps, tns, fps and fns 
                    # from fg_patch, mask, class_outputs
                    softmaxed = softmax(class_output, 0)
                    fg_prob = softmaxed[1]


                    assert fg_prob.shape == mask.shape, (
                        f"fg_prob shape {fg_prob.shape} and mask shape {mask.shape}"
                        f"should be equal")

                    # ignore (set to 0) any regions of the predicted where annotation is not defined
                    fg_prob = fg_prob * mask 
                    class_pred = fg_prob > 0.5

                    if ignore_mask is not None:
                        # FIXME consider cost of moving this to GPU
                        # perhaps just do everything on CPU
                        ignore_mask = torch.tensor(ignore_mask).cuda()
                        # ignore things in the ignore mask to avoid duplicate metrics
                        mask[ignore_mask > 0] = 0 
                    
                    class_pred = class_pred[mask > 0]
                    fg = fg_patch[mask > 0]
                    # compute metrics based on agreement between predictions and labels.
                    instances_metrics[im_idx] += metrics_from_binary_masks(class_pred, fg)
                    masks.append(mask)

                    fg_patches.append(fg_patch)
                    class_outputs.append(class_output)
                    seg_patch = None

                    if batch_seg_patches is not None:
                        seg_patch = batch_seg_patches[im_idx][i]
                    if seg_patch is not None:
                        seg_patch = torch.from_numpy(seg_patch[17:-17,17:-17,17:-17]).cuda()
                        # for this purpose we ensure segmentation never disagrees with annotation.
                        seg_patch[fg_patch] = 1
                        seg_patch[bg_patch] = 0
                        seg_patches.append(seg_patch)
                        seg_class_outputs.append(class_output)
             

        if not len(fg_patches):
            continue
        
        if compute_loss:
            fg_patches = torch.stack(fg_patches).cuda()
            masks = torch.stack(masks).cuda()
            class_outputs = torch.stack(class_outputs)
                        
            if len(seg_class_outputs):
                seg_class_outputs = torch.stack(seg_class_outputs)
                seg_class_outputs_softmaxed = softmax(seg_class_outputs, 1)
                seg_foreground_probs = seg_class_outputs_softmaxed[:, 1]
                seg_patches = torch.stack(seg_patches).long()
                # See loss functions in https://arxiv.org/pdf/1912.02911.pdfhttps://arxiv.org/pdf/1912.02911.pdf
                class_seg_loss = l1_loss(seg_foreground_probs, seg_patches, reduction='mean') # MAE
                class_losses.append(class_seg_loss)

            # remove any of the predictions for which we don't have ground truth
            # Set outputs to 0 where annotation undefined so that
            # The network can predict whatever it wants without any penalty.
            class_outputs[:, 0] *= masks
            class_outputs[:, 1] *= masks
            class_loss = combined_loss(class_outputs, fg_patches)
            class_losses.append(class_loss)

    if compute_loss:
        return torch.mean(torch.stack(class_losses)), instances_metrics
    # FIXME: create a metrics object from a metrics class and return here.
    #        bugs are being caused by just assuming these are in a specific order.
    return None, instances_metrics



