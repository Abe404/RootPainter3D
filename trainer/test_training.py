"""
Test that training works without error. 

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
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import nibabel as nib

import im_utils
from datasets import RPDataset
import datasets
import model_utils
from test_utils import dl_dir_from_zip
import data_utils
import train_utils
from metrics import Metrics



# sync directory for use with tests
sync_dir = os.path.join(os.getcwd(), 'test_rp_sync')
annot_dir = os.path.join(sync_dir, 'projects', 'total_seg_test', 'annotations')
datasets_dir = os.path.join(sync_dir, 'datasets')
total_seg_dataset_dir = os.path.join(datasets_dir, 'total_seg')

subset_dir_images = os.path.join(total_seg_dataset_dir, '50_random_images')
subset_dir_annots = os.path.join(total_seg_dataset_dir, '50_random_annots')

annot_train_dir = os.path.join(subset_dir_annots, 'liver', 'train')
annot_val_dir = os.path.join(subset_dir_annots, 'liver', 'val')

timeout_ms = 20000


def convert_seg_to_annot(in_fpath):
    """ load the seg file from `in_fpath` 
        convert to rp annot format (numpy) and return """
    seg = im_utils.load_image(in_fpath)
    assert len(seg.shape) == 3, 'should be 3d binary mask, shape: ' + str(seg.shape)
    seg_bg = seg == 0
    seg_fg = seg > 0
    annot = np.zeros([2] + list(seg.shape))
    annot[0] = seg_bg
    annot[1] = seg_fg
    return annot


def prep_random_50(dataset_dir):
    """ take random 50 images from total segmentor dataset and put them in a folder
    """
    print('Creating datasets')
    all_dirs = os.listdir(dataset_dir)
    all_dirs = [d for d in all_dirs if '50_random' not in d]
    all_dirs = [d for d in all_dirs if os.path.isdir(os.path.join(dataset_dir, d))]

    subset_size = 50
    sampled_dirs = random.sample(all_dirs, subset_size)

    os.makedirs(subset_dir_images)
    os.makedirs(subset_dir_annots)
    os.makedirs(annot_train_dir)
    os.makedirs(annot_val_dir)

    for i, d in enumerate(sampled_dirs):
        imfpath = os.path.join(dataset_dir, d, 'ct.nii.gz')
        out_im_fpath = os.path.join(subset_dir_images, d  + '_ct.nii.gz')
        shutil.copyfile(imfpath, out_im_fpath) # images are good to go, no modification required.

        in_annot_fpath = os.path.join(dataset_dir, d, 'segmentations', 'liver.nii.gz')
       
        # copy first 80% to train
        if i <= (subset_size * 0.8):
            out_annot_fpath = os.path.join(annot_train_dir, d  + '_ct.nii.gz')
        else:
            # and the last 20% to validation
            out_annot_fpath = os.path.join(annot_val_dir, d  + '_ct.nii.gz')

        # convert segmentations (total seg format) to rp3d annotations
        annot = convert_seg_to_annot(in_annot_fpath)
        annot = nib.Nifti1Image(annot, np.eye(4))
        annot.to_filename(out_annot_fpath)


def setup_function():
    """ download and prepare files required for the tests """
    print('running setup')
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    if not os.path.isdir(total_seg_dataset_dir):
        total_seg_url = 'https://zenodo.org/record/6802614/files/Totalsegmentator_dataset.zip'
        dl_dir_from_zip(total_seg_url, datasets_dir)
    if not os.path.isdir(subset_dir_images):
        prep_random_50(total_seg_dataset_dir)


def test_training():
    """ test training can run one epoch without error """
    in_w = 36 + (3*16)
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    num_workers = 12
    batch_size = 6
    classes = ['liver']

    train_annot_dirs = [annot_train_dir] # for liver

    dataset = RPDataset(train_annot_dirs,
                        train_seg_dirs=[None] * len(train_annot_dirs),
                        dataset_dir=subset_dir_images,
                        in_w=in_w,
                        out_w=out_w,
                        in_d=in_d,
                        out_d=out_d,
                        mode=datasets.Modes.TRAIN,
                        patch_refs=None,
                        use_seg_in_training=False,
                        length=batch_size*4)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=data_utils.collate_fn,
                        num_workers=num_workers,
                        drop_last=False, pin_memory=True)

    model = model_utils.random_model(classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.99, nesterov=True)

    start_time = time.time()

    train_result = train_utils.train_epoch(model,
                                           classes,
                                           loader,
                                           batch_size,
                                           optimizer=optimizer,
                                           step_callback=None,
                                           stop_fn=None)
    assert train_result
    print('')
    print('Train epoch complete in', round(time.time() - start_time, 1), 'seconds')
    # pass - epoch runs without error.



def test_validation():
    """ test validation epoch completes without error """
    in_w = 36 + (6*16)
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    classes = ['liver']

    val_annot_dirs = [annot_val_dir] # for liver

    # should be some files in the annot dir for this test to work
    assert os.path.isdir(val_annot_dirs[0])
    assert os.listdir(val_annot_dirs[0])

    patch_refs = im_utils.get_val_patch_refs(
        val_annot_dirs,
        [],
        out_shape=(out_d, out_w, out_w))
 

    dataset = RPDataset(val_annot_dirs,
                        None, # train_seg_dirs
                        dataset_dir=subset_dir_images,
                        # only specifying w and d as h is always same as w
                        in_w=in_w,
                        out_w=out_w,
                        in_d=in_d,
                        out_d=out_d,
                        mode=datasets.Modes.VAL, 
                        patch_refs=patch_refs)

    model = model_utils.random_model(classes)

    print('Running validation on', len(patch_refs), 'patches from',
          len(os.listdir(val_annot_dirs[0])), 'images.')

    assert len(patch_refs) >= len(os.listdir(val_annot_dirs[0])), (
        f"Should be at least as many patch_refs ({len(patch_refs)}) "
        f" as annotation files {os.listdir(val_annot_dirs[0])}")

    val_result = train_utils.val_epoch(model,
                                       classes,
                                       dataset,
                                       patch_refs,
                                       step_callback=None,
                                       stop_fn=None)
    assert val_result is not None
    assert len(val_result) == len(patch_refs)


def test_training_converges():
    """ test training can get to a model with training dice of 0.6 """
    in_w = 36 + (3*16)
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    num_workers = 12
    batch_size = 6
    classes = ['liver']

    train_annot_dirs = [annot_train_dir] # for liver

    dataset = RPDataset(train_annot_dirs,
                        train_seg_dirs=[None] * len(train_annot_dirs),
                        dataset_dir=subset_dir_images,
                        in_w=in_w,
                        out_w=out_w,
                        in_d=in_d,
                        out_d=out_d,
                        mode=datasets.Modes.TRAIN,
                        patch_refs=None,
                        use_seg_in_training=False,
                        length=batch_size*64)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=data_utils.collate_fn,
                        num_workers=num_workers,
                        drop_last=False, pin_memory=True)

    model = model_utils.random_model(classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.99, nesterov=True)
    for _ in range(10):
        start_time = time.time()
        train_result = train_utils.train_epoch(model,
                                               classes,
                                               loader,
                                               batch_size,
                                               optimizer=optimizer,
                                               step_callback=None,
                                               stop_fn=None)
        assert train_result
        print('')
        print('Train epoch complete in', round(time.time() - start_time, 1), 'seconds')
        train_metrics = Metrics.sum(train_result)
        if train_metrics.dice() > 0.6:
            return # test passes.
        print('Metrics', train_metrics.__str__(to_use=['dice']))
    raise Exception('Dice did not get to 0.6 in 10 epochs')


def test_training_converges_on_validation():
    """ test training can get to a model with validation dice of 0.6 """
    in_w = 36 + (3*16)
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    num_workers = 12
    batch_size = 6
    classes = ['liver']

    train_annot_dirs = [annot_train_dir] # for liver
    val_annot_dirs = [annot_val_dir] # for liver

    dataset = RPDataset(train_annot_dirs,
                        train_seg_dirs=[None] * len(train_annot_dirs),
                        dataset_dir=subset_dir_images,
                        in_w=in_w,
                        out_w=out_w,
                        in_d=in_d,
                        out_d=out_d,
                        mode=datasets.Modes.TRAIN,
                        patch_refs=None,
                        use_seg_in_training=False,
                        length=batch_size*64)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=data_utils.collate_fn,
                        num_workers=num_workers,
                        drop_last=False, pin_memory=True)

    model = model_utils.random_model(classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.99, nesterov=True)


    # should be some files in the annot dir for this test to work
    assert os.path.isdir(val_annot_dirs[0])
    assert os.listdir(val_annot_dirs[0])

    patch_refs = im_utils.get_val_patch_refs(
        val_annot_dirs,
        [],
        out_shape=(out_d, out_w, out_w))
 
    val_dataset = RPDataset(val_annot_dirs,
                            None, # train_seg_dirs
                            dataset_dir=subset_dir_images,
                            # only specifying w and d as h is always same as w
                            in_w=in_w,
                            out_w=out_w,
                            in_d=in_d,
                            out_d=out_d,
                            mode=datasets.Modes.VAL, 
                            patch_refs=patch_refs)

    for _ in range(40):
        start_time = time.time()
        train_result = train_utils.train_epoch(model,
                                               classes,
                                               loader,
                                               batch_size,
                                               optimizer=optimizer,
                                               step_callback=None,
                                               stop_fn=None)
        assert train_result
        print('')
        print('Train epoch complete in', round(time.time() - start_time, 1), 'seconds')
        train_metrics = Metrics.sum(train_result)
        print('Train metrics dice', train_metrics.dice())
        val_result = train_utils.val_epoch(model,
                                           classes,
                                           val_dataset,
                                           patch_refs,
                                           step_callback=None,
                                           stop_fn=None)
        val_metrics = Metrics.sum(val_result)
        print('val metrics dice', val_metrics.dice())
        if val_metrics.dice() > 0.6:
            return # test passes.
    raise Exception('Validation Dice did not get to 0.6 in 40 epochs')





