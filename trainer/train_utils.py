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
from patch_seg import handle_patch_update_in_epoch_step
from model_utils import debug_memory
from loss import get_batch_loss


def epoch(model, classes, loader, batch_size,
          optimizer, patch_update_enabled=False,
          step_callback=None, stop_fn=None):
    """ One training epoch """
    assert type(classes) == list, f'classes should be list, classes:{classes}'
    model.train()

    epoch_items_metrics = []  
    loss_sum = 0
    for step, (batch_im_patches, batch_fg_patches,
               batch_bg_patches, batch_ignore_masks,
               batch_seg_patches, batch_classes) in enumerate(loader):

        batch_im_patches = torch.from_numpy(np.array(batch_im_patches)).cuda()
        optimizer.zero_grad()
       
        if patch_update_enabled:
            batch_im_patches = handle_patch_update_in_epoch_step(batch_im_patches, 'train')

        outputs = model(batch_im_patches)

        for im_fg_patches in batch_fg_patches:
            for class_fg_patch in im_fg_patches:
                shape_str = f'output: {outputs.shape[2:]}, fg: {class_fg_patch.shape}'
                cropped_annot_shape = [a-34 for a in list(class_fg_patch.shape)]
                assert list(outputs.shape[2:]) == cropped_annot_shape, shape_str
        
        (batch_loss, batch_items_metrics) = get_batch_loss(
             outputs, batch_fg_patches, batch_bg_patches, 
             batch_ignore_masks, batch_seg_patches,
             batch_classes, classes,
             compute_loss=True)

        epoch_items_metrics += batch_items_metrics 
        loss_sum += batch_loss.item() # float
        batch_loss.backward()
        optimizer.step()
    
        total_fg = ''
        for b in batch_items_metrics:
            total_fg += f',{b.total_true()}'
        total_fg = total_fg[1:] # remove first ','
        
        debug_memory('train epoch step')

        # clear the output to remove residual text before printing again
        print("\r                                           " 
              "                                             ",
              end='', flush=True)

        # https://github.com/googlecolab/colabtools/issues/166
        print(f"\rTraining: {(step+1) * batch_size}/"
              f"{len(loader.dataset)} "
              f" loss={round(batch_loss.item(), 3)}, fg={total_fg}",
              end='', flush=True)
        if step_callback:
            step_callback() # could update training parameter
        if stop_fn and stop_fn(): # in this context we consider validation part of training.
            return None # understood as training stopping early
    return epoch_items_metrics
