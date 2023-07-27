"""
Experiment comparing warmup (gradually increasing learning rate)
to fixed learning rate. 

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


from warmup_scheduler import GradualWarmupScheduler
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
in_w = 36 + (7*16) 
out_w = in_w - 34
in_d = in_w
out_d = out_w

learning_rate = 0.01
momentum = 0.99


def test_warmup():
    """ train with wamrup should offer more stability (less nans) """
    print('test_warmup()')
    batch_size = 4
    classes = ['liver']
    dice_target = 0.3
    epoch_limit = 10 
    dataset_len = len(os.listdir(liver_annot_train_dir))
    print('dataset_len', dataset_len)

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

    fails = 0
    tries = 10

    for t in range(tries):
        model = model_utils.random_model(classes)
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, nesterov=True)
        for e in range(epoch_limit):
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
            print(f'Train epoch {e + 1} complete in', round(time.time() - start_time, 1), 'seconds')
            train_metrics = Metrics.sum(train_result)
            print('Train dice:', train_metrics.dice(),
                  'FG predicted', train_metrics.total_pred(),
                  'FG true', train_metrics.total_true(),
                  'FG true mean', train_metrics.true_mean(),
                  'FG pred mean', train_metrics.pred_mean())

        if train_metrics.dice() < dice_target or math.isnan(train_metrics.dice()):
            fails += 1
        print('conv fails', fails, 'out of ', t, 'tries')

    print('Conventional fails:', fails)
    warmup_fails = 0

    for t in range(tries):
        # now we try the same but with scheduler enabled.
        model = model_utils.random_model(classes)
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, nesterov=True)
        # this zero gradient update is needed to avoid a warning message
        # casued by scheduler, issue #8.
        optim.zero_grad()
        optim.step()

        scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=8)
     
        for e in range(epoch_limit):
            start_time = time.time()
            scheduler_warmup.step()
            print('start_epoch', e, ' lr', optim.param_groups[0]['lr'])
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
            print(f'Train epoch {e + 1} complete in', round(time.time() - start_time, 1), 'seconds')
            train_metrics = Metrics.sum(train_result)
            print('Train dice:', train_metrics.dice(),
                  'FG predicted', train_metrics.total_pred(),
                  'FG true', train_metrics.total_true(),
                  'FG true mean', train_metrics.true_mean(),
                  'FG pred mean', train_metrics.pred_mean())

        if train_metrics.dice() < dice_target or math.isnan(train_metrics.dice()):
            warmup_fails += 1
        print('warmup fails', warmup_fails, 'out of ', t, 'tries')

    print('tries each', tries, 'warmup fails', warmup_fails, 'vs', fails, 'standard fails')
