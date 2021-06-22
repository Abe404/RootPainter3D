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
import torch
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy


def combined_loss(predictions, labels):
    """ combine CE and dice """
    loss_sum = 0
    if torch.sum(labels[:, 1]):
        loss_sum += dice_loss(predictions, labels[:, 1])
    loss_sum += 0.3 * cross_entropy(predictions, labels[:, 1])
    return loss_sum


def dice_loss(predictions, labels):
    """ soft dice to help handle imbalanced classes """
    softmaxed = softmax(predictions, 1)
    predictions = softmaxed[:, 1, :]  # just the root probability.
    labels = labels.float()
    preds = predictions.contiguous().view(-1)
    labels = labels.view(-1)
    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    dice = ((2 * intersection) / (union))
    return 1 - dice
