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


import time
import copy

import numpy as np
import torch

from patch_seg import handle_patch_update_in_epoch_step
from model_utils import debug_memory
from loss import get_batch_loss
from viz import save_patches_image

def train_epoch(model, classes, loader, batch_size,
          optimizer, patch_update_enabled=False,
          step_callback=None, stop_fn=None,
          debug_dir=None):
    """ One training epoch """
    assert isinstance(classes, list), f'classes should be list, classes:{classes}'
    model.train()
    torch.set_grad_enabled(True)

    epoch_start = time.time()
    epoch_items_metrics = []  
    loss_sum = 0
    for step, (batch_im_patches, batch_fg_patches,
               batch_bg_patches, batch_ignore_masks,
               batch_seg_patches, batch_classes) in enumerate(loader):

        batch_im_patches = np.expand_dims(batch_im_patches, 1) # add channel dimension
        batch_im_patches = torch.from_numpy(np.array(batch_im_patches)).cuda()
        optimizer.zero_grad()
       
        if patch_update_enabled:
            batch_im_patches = handle_patch_update_in_epoch_step(batch_im_patches, 'train')

        outputs = model(batch_im_patches)

        for im_fg_patches in batch_fg_patches:
            for class_fg_patch in im_fg_patches:

                cropped_annot_shape = [a-34 for a in list(class_fg_patch.shape)]

                shape_str = f'output: {outputs.shape[2:]}'
                shape_str += f'expected output (to match cropped annot): {cropped_annot_shape}'
                shape_str += f'fg annot shape: {class_fg_patch.shape}'

                assert list(outputs.shape[2:]) == cropped_annot_shape, shape_str

        if debug_dir:
            num_saved = save_patches_image(
                [batch_im_patches[0][0].cpu().numpy(),
                batch_fg_patches[0][0],
                outputs[0][1]],
                ['im', 'fg', 'pred'],
                debug_dir)
            print('saved', num_saved, 'debug images')

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


        # https://github.com/googlecolab/colabtools/issues/166
        print(f"\rTraining: {(step+1) * batch_size}/"
              f"{len(loader.dataset)} "
              f" loss={round(batch_loss.item(), 3)}, fg={total_fg}"
              "                        ",
              end='', flush=True)
        if step_callback:
            step_callback() # could update training parameter
        if stop_fn and stop_fn(): # in this context we consider validation part of training.
            return None # understood as training stopping early

    duration = round(time.time() - epoch_start, 3)
    print('')
    print('Training epoch duration', duration, 'time per instance',
          round((time.time() - epoch_start) / len(epoch_items_metrics), 3))

    return epoch_items_metrics




def val_epoch(model, project_classes, dataset, val_patch_refs,
              patch_update_enabled=False,
              step_callback=None, stop_fn=None):
    """
    Compute the metrics for a given
        model, annotation directory and dataset (image directory).
    """
    debug_memory('val epoch start')
    epoch_start = time.time()
    torch.set_grad_enabled(False)
    fnames = {r.annot_fname for r in val_patch_refs}
    epoch_items_metrics = []
    step = 0

    val_cnn = copy.deepcopy(model)
    val_cnn.half()
    for fname in fnames:
        # We assume image has same extension as annotation.
        # Is this always the case?
        image = dataset.load_im(fname)
        annots, classes = dataset.get_annots_for_image(fname)
        refs = [r for r in val_patch_refs if r.annot_fname == fname]

        for ref in refs:
            (im_patch, fg_patches,
             bg_patches, _segs) = dataset.get_patch_from_image(image, annots, ref)            
            ignore_mask = ref.ignore_mask
            
            if step_callback:
                step_callback()

            batch_im_patches = torch.from_numpy(np.array([im_patch])).cuda()

            if patch_update_enabled:
                batch_im_patches = handle_patch_update_in_epoch_step(batch_im_patches, mode='val')

            batch_im_patches = batch_im_patches.half()
            outputs = val_cnn(batch_im_patches)
            (_, batch_items_metrics) = get_batch_loss(
                outputs,
                [fg_patches],
                [bg_patches],
                [ignore_mask],
                None,
                [classes],
                project_classes,
                compute_loss=False)

            epoch_items_metrics += batch_items_metrics
            
            if stop_fn and stop_fn():
                return None # a way to stop validation quickly if user specifies

            debug_memory('val epoch step')
            # https://github.com/googlecolab/colabtools/issues/166
            print(f"\rValidation: {(step+1)}/{len(val_patch_refs)}", end='', flush=True)
            step += 1

    duration = round(time.time() - epoch_start, 3)
    print('')
    print('Validation epoch duration', duration, 'time per instance',
           round((time.time() - epoch_start) / len(epoch_items_metrics), 3))
    return epoch_items_metrics

