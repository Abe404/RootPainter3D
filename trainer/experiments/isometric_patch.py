"""
Experiment with isometric-patch (Patch having same size in all dimensions).
How does this compare to the previous shape which had less size in the depth dimension.

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

import os
import sys
import inspect
import time
import math

import torch
from torch.utils.data import DataLoader

# allow imports from parent folder
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


import datasets 
import data_utils
import model_utils
import train_utils
from metrics import Metrics


sync_dir = os.path.join(os.getcwd(), 'test_rp_sync')
datasets_dir = os.path.join(sync_dir, 'datasets')
total_seg_dataset_dir = os.path.join(datasets_dir, 'total_seg')
subset_dir_annots = os.path.join(total_seg_dataset_dir, '100_random_annots')
liver_annot_train_dir = os.path.join(subset_dir_annots, 'liver', 'train')
subset_dir_images = os.path.join(total_seg_dataset_dir, '100_random_images')

num_workers = 12
# in_w = 36 + (4*16) # patch size of 100
learning_rate = 0.01
momentum = 0.99


def test_isometric_patch():
    """ Isometric patch should not drop performance substantially """
    print('test_isometric_patch()')
    batch_size = 4
    classes = ['liver']
    dataset_len = len(os.listdir(liver_annot_train_dir))
    print('dataset_len', dataset_len)


    def train_attempt(tries, epoch_lim, in_w, out_w, in_d, out_d):
        epoch_dices = [] # list of (epoch, dice)
        train_annot_dirs = [liver_annot_train_dir] # for liver
        dataset = datasets.RPDataset(train_annot_dirs,
                                     train_seg_dirs=[None] * len(train_annot_dirs),
                                     dataset_dir=subset_dir_images,
                                     in_w=in_w,
                                     out_w=out_w,
                                     in_d=in_d,
                                     out_d=out_d,
                                     mode=datasets.Modes.TRAIN,
                                     patch_refs=None,
                                     use_seg_in_training=False,
                                     length=dataset_len)
        def fg_fn():
            # always force foreground in item for this test
            # as otherways convergence may be slow.
            return True
        dataset.should_force_fg = fg_fn

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=data_utils.collate_fn,
                            num_workers=num_workers,
                            drop_last=False, pin_memory=True)
        t = 0
        while t < tries:
            model = model_utils.random_model(classes)
            optim = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=momentum, nesterov=True)
            e = 0
            t += 1
            while e < epoch_lim:
                start_time = time.time()
                train_result = train_utils.train_epoch(model,
                     classes,
                     loader,
                     batch_size,
                     optimizer=optim,
                     step_callback=None,
                     stop_fn=None,
                     # debug_dir='frames')
                     debug_dir=None)
                assert train_result
                print('')
                print(f'Train epoch {e + 1} complete in',
                      round(time.time() - start_time, 1), 'seconds')
                train_metrics = Metrics.sum(train_result)
                epoch_dices.append((e, train_metrics.dice()))
                print('Train dice:', train_metrics.dice(),
                      'FG predicted', train_metrics.total_pred(),
                      'FG true', train_metrics.total_true(),
                      'FG true mean', train_metrics.true_mean(),
                      'FG pred mean', train_metrics.pred_mean())
                e += 1
                
                # if we hit nan then restart (convergence failed)
                if math.isnan(train_metrics.dice()):
                    model = model_utils.random_model(classes)
                    optim = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=momentum, nesterov=True)
                    e = 0
                    t += 1
                    if t >= tries:
                        return epoch_dices

        return epoch_dices

    # patch sizes are largest possible without error in 48GB GPU.
    in_w = 36 + (17*16) 
    conv_dices = train_attempt(tries=10, epoch_lim=10,
                               in_w=in_w, out_w=in_w-34,
                               in_d=52, out_d=52-34)
     
    print('conv_dices = ', conv_dices)
    in_w = 36 + (7*16) 
    iso_dices = train_attempt(tries=10, epoch_lim=10,
                              in_w=in_w, out_w=in_w-34,
                              in_d=in_w, out_d=in_w-34)
    print('iso_dices = ', iso_dices)
