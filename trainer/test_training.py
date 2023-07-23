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
from data_utils import collate_fn


# sync directory for use with tests
sync_dir = os.path.join(os.getcwd(), 'test_rp_sync')
annot_dir = os.path.join(sync_dir, 'projects', 'total_seg_test', 'annotations')
datasets_dir = os.path.join(sync_dir, 'datasets')
total_seg_dataset_dir = os.path.join(datasets_dir, 'total_seg')

subset_dir_images = os.path.join(total_seg_dataset_dir, '100_random_images')
subset_dir_annots = os.path.join(total_seg_dataset_dir, '100_random_annots')

liver_annot_train_dir = os.path.join(subset_dir_annots, 'liver', 'train')
liver_annot_val_dir = os.path.join(subset_dir_annots, 'liver', 'val')

spleen_annot_train_dir = os.path.join(subset_dir_annots, 'spleen', 'train')
spleen_annot_val_dir = os.path.join(subset_dir_annots, 'spleen', 'val')
partial_spleen_annot_val_dir = os.path.join(subset_dir_annots, 'spleen_partial', 'val')

    
num_workers = 12
in_w = 36 + (6*16) # patch size of 116
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


def prep_random_100(dataset_dir):
    """ take random 100 images from total segmentor dataset and put them in a folder
    """
    print('Creating datasets')
    all_dirs = os.listdir(dataset_dir)
    all_dirs = [d for d in all_dirs if '100_random' not in d]
    all_dirs = [d for d in all_dirs if os.path.isdir(os.path.join(dataset_dir, d))]

    subset_size = 100
    sampled_dirs = random.sample(all_dirs, subset_size)

    os.makedirs(subset_dir_images)
    os.makedirs(subset_dir_annots)
    os.makedirs(liver_annot_train_dir)
    os.makedirs(liver_annot_val_dir)

    os.makedirs(spleen_annot_train_dir)
    os.makedirs(spleen_annot_val_dir)
    os.makedirs(partial_spleen_annot_val_dir)

    for i, d in enumerate(sampled_dirs):
        imfpath = os.path.join(dataset_dir, d, 'ct.nii.gz')
        out_im_fpath = os.path.join(subset_dir_images, d  + '_ct.nii.gz')
        shutil.copyfile(imfpath, out_im_fpath) # images are good to go, no modification required.

        liver_in_annot_fpath = os.path.join(dataset_dir, d, 'segmentations', 'liver.nii.gz')
        spleen_in_annot_fpath = os.path.join(dataset_dir, d, 'segmentations', 'spleen.nii.gz')
       
        # copy first 80% to train
        if i <= (subset_size * 0.8):
            liver_out_annot_fpath = os.path.join(liver_annot_train_dir, d  + '_ct.nii.gz')
            spleen_out_annot_fpath = os.path.join(spleen_annot_train_dir, d  + '_ct.nii.gz')
            partial_spleen_out_annot_fpath = None # we dont need train for this test.
        else:
            # and the last 20% to validation
            liver_out_annot_fpath = os.path.join(liver_annot_val_dir, d  + '_ct.nii.gz')
            spleen_out_annot_fpath = os.path.join(spleen_annot_val_dir, d  + '_ct.nii.gz')
            partial_spleen_out_annot_fpath = os.path.join(partial_spleen_annot_val_dir,
                                                          d + '_ct.nii.gz')

        # convert segmentations (total seg format) to rp3d annotations
        liver_annot = convert_seg_to_annot(liver_in_annot_fpath)
        liver_annot = nib.Nifti1Image(liver_annot, np.eye(4))
        liver_annot.to_filename(liver_out_annot_fpath)

        spleen_annot = convert_seg_to_annot(spleen_in_annot_fpath)
        spleen_annot = nib.Nifti1Image(spleen_annot, np.eye(4))
        spleen_annot.to_filename(spleen_out_annot_fpath)
        
        # only copy have of the spleens to the partial dataset.
        if i % 2 == 0 and partial_spleen_out_annot_fpath is not None:
            spleen_annot.to_filename(partial_spleen_out_annot_fpath)
            


def setup_function():
    """ download and prepare files required for the tests """
    print('running setup')
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    if not os.path.isdir(total_seg_dataset_dir):
        total_seg_url = 'https://zenodo.org/record/6802614/files/Totalsegmentator_dataset.zip'
        dl_dir_from_zip(total_seg_url, datasets_dir)
    if not os.path.isdir(subset_dir_images):
        prep_random_100(total_seg_dataset_dir)


def test_training():
    """ test training can run one epoch without error """
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    batch_size = 4
    classes = ['liver']

    train_annot_dirs = [liver_annot_train_dir] # for liver

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
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    classes = ['liver']

    val_annot_dirs = [liver_annot_val_dir] # for liver

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
    """ test training can get to a model with training dice of 0.4 """
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    batch_size = 4
    classes = ['liver']

    train_annot_dirs = [liver_annot_train_dir] # for liver

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
        if train_metrics.dice() > 0.4:
            return # test passes.
        print('Metrics', train_metrics.__str__(to_use=['dice']))
    raise Exception('Dice did not get to 0.4 in 10 epochs')


def test_training_converges_on_validation():
    """ test training can get to a model with validation dice of 0.4 """
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    batch_size = 4
    classes = ['liver']

    train_annot_dirs = [liver_annot_train_dir] # for liver
    val_annot_dirs = [liver_annot_val_dir] # for liver

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
        if val_metrics.dice() > 0.4:
            return # test passes.
    raise Exception('Validation Dice did not get to 0.4 in 40 epochs')


def test_multiclass_validation():
    """ test validation does not throw exception when multiple classes used.
        Dont train - only validate the initial random model """
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    classes = ['liver', 'spleen']

    val_annot_dirs = [liver_annot_val_dir, spleen_annot_val_dir]

    model = model_utils.random_model(classes)

    # should be some files in the annot dir for this test to work
    assert os.path.isdir(val_annot_dirs[0])
    assert os.listdir(val_annot_dirs[0])
    assert os.path.isdir(val_annot_dirs[1])
    assert os.listdir(val_annot_dirs[1])

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

    val_result = train_utils.val_epoch(model,
                                       classes,
                                       val_dataset,
                                       patch_refs,
                                       step_callback=None,
                                       stop_fn=None)
    val_metrics = Metrics.sum(val_result)
    print('val metrics dice', val_metrics.dice())
    assert val_metrics.dice() > 0.000001



def test_multiclass_validation_missing_annotations():
    """ test validation does not throw exception when multiple classes used.
        Dont train - only validate the initial random model

        This test uses a folder that only contains validation
        annotations for some of the images for the spleen dataset.
        
        The aim is to reproduce a reported bug:
        https://github.com/Abe404/RootPainter3D/issues/32
    """
    out_w = in_w - 34
    in_d = 52
    out_d = 18
    classes = ['liver', 'partial_spleen']

    val_annot_dirs = [liver_annot_val_dir, partial_spleen_annot_val_dir]

    model = model_utils.random_model(classes)

    # should be some files in the annot dir for this test to work
    assert os.path.isdir(val_annot_dirs[0])
    assert os.listdir(val_annot_dirs[0])
    assert os.path.isdir(val_annot_dirs[1])
    assert os.listdir(val_annot_dirs[1])

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

    val_result = train_utils.val_epoch(model,
                                       classes,
                                       val_dataset,
                                       patch_refs,
                                       step_callback=None,
                                       stop_fn=None)
    val_metrics = Metrics.sum(val_result)
    print('val metrics dice', val_metrics.dice())
    assert val_metrics.dice() > 0.000001


def test_get_val_patch_refs():
    """ get val patch refs should return a
        list of patches for each folder in the annotation directory.
    """

    out_w = in_w - 34
    out_d = 18
    out_shape = (out_d, out_w, out_w)
    prev_patch_refs = []

    annot_dirs = [liver_annot_val_dir, partial_spleen_annot_val_dir]
    patch_refs = im_utils.get_val_patch_refs(annot_dirs, prev_patch_refs, out_shape)
    
    # the patch refs should contain items for both directories.
    patch_ref_dirs = set(p.annot_dir for p in patch_refs)

    for d in annot_dirs:
        assert d in patch_ref_dirs

    # Each of the patch refs should refer to actual files on disk
    for p in patch_refs:
        fpath = os.path.join(p.annot_dir, p.annot_fname)
        assert os.path.isfile(fpath)


    # Each of the files on disk should have a patch ref
    all_fpaths = []
    for d in annot_dirs:
        fnames = os.listdir(d)
        for f in fnames:
            all_fpaths.append(os.path.join(d, f))
    patch_ref_fpaths = [p.annot_fpath() for p in patch_refs]

    for fpath in all_fpaths:
        assert fpath in patch_ref_fpaths



def test_training_patch_size_bigger_than_image():
    """ test that training does not error when the patch size
        for the neural network is bigger than the images. """
    larger_in_w = 36 + (11*16)
    out_w = larger_in_w - 34
    in_d = 52
    out_d = 18
    batch_size = 1
    classes = ['liver']

    train_annot_dirs = [liver_annot_train_dir] # for liver

    # we will try training on images that are mostly smaller than the patch width.
    """
    This check allows us to confirm that the dataset used for testing has some images
    that are smaller than the patch size. We don't need to run it every time.
    bigger_patch = 0
    fnames = os.listdir(subset_dir_images)
    for f in fnames:
        fpath = os.path.join(subset_dir_images, f)
        im = im_utils.load_image(fpath)
        if (im.shape[0] < in_d or im.shape[1] < larger_in_w or im.shape[2] < larger_in_w):
            bigger_patch += 1
    assert bigger_patch > len(fnames) // 2, f'Only {bigger_patch} in {len(fnames)}'
    """

    dataset = RPDataset(train_annot_dirs,
                        train_seg_dirs=[None] * len(train_annot_dirs),
                        dataset_dir=subset_dir_images,
                        in_w=larger_in_w,
                        out_w=out_w,
                        in_d=in_d,
                        out_d=out_d,
                        mode=datasets.Modes.TRAIN,
                        patch_refs=None,
                        use_seg_in_training=False,
                        length=batch_size*10)

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

    print('Train epoch complete in', round(time.time() - start_time, 1), 'seconds')
    train_metrics = Metrics.sum(train_result)
    assert train_metrics.dice() > 0.000001


def test_validation_patch_size_bigger_than_image():
    """ test that validation does not error when the patch size
        for the neural network is bigger than the images. """
    larger_in_w = 36 + (11*16)
    out_w = larger_in_w - 34
    in_d = 52
    out_d = 18
    classes = ['liver']
    val_annot_dirs = [liver_annot_val_dir]

    model = model_utils.random_model(classes)

    patch_refs = im_utils.get_val_patch_refs(
        val_annot_dirs,
        [],
        out_shape=(out_d, out_w, out_w))
 
    val_dataset = RPDataset(val_annot_dirs,
                            None, # train_seg_dirs
                            dataset_dir=subset_dir_images,
                            # only specifying w and d as h is always same as w
                            in_w=larger_in_w,
                            out_w=out_w,
                            in_d=in_d,
                            out_d=out_d,
                            mode=datasets.Modes.VAL, 
                            patch_refs=patch_refs)

    val_result = train_utils.val_epoch(model,
                                       classes,
                                       val_dataset,
                                       patch_refs,
                                       step_callback=None,
                                       stop_fn=None)
    val_metrics = Metrics.sum(val_result)
    assert val_metrics.dice() > 0.000001


def test_collate_pad_image_patches_to_largest():
    """ test that the collate_fn will pad the 
        items to the largest dimensions found.

    """
    mock_items = []
    mock_items.append([
        np.zeros((52, 253, 64)), # im_patch 
        [np.zeros(torch.Size([52, 253, 64]))], # fg patches
        [np.zeros(torch.Size([52, 253, 64]))], # bg patches
        np.zeros((18, 253-34, 64-34)),# ignore_mask shape 
        None, # seg 
        ['liver'], # class
    ])
    mock_items.append([
        np.zeros((52, 192, 271)),# im_patch 
        [np.zeros(torch.Size([52, 192, 271]))], # fg patches
        [np.zeros(torch.Size([52, 192, 271]))], # bg patches
        np.zeros((18, 192-34, 271-34)),# ignore_mask shape 
        None, # seg 
        ['liver'], # class
    ])
    mock_items.append([
        np.zeros((52, 253, 113)),# im_patch 
        [np.zeros(torch.Size([52, 253, 113]))], # fg patches
        [np.zeros(torch.Size([52, 253, 113]))], # bg patches
        np.zeros((18, 253-34, 113-34)),# ignore_mask shape 
        None, # seg 
        ['liver'], # class
    ])
    collated = collate_fn(mock_items)

    (patches, batch_fgs, batch_bgs,
     ignore_masks, _batch_segs, _batch_classes) = collated

    assert len(patches) == len(mock_items)
    assert len(batch_fgs) == len(mock_items)

    expected_im_shape = (52, 253, 271)
    for p, fg, bg, ig in zip(patches, batch_fgs, batch_bgs, ignore_masks):
        assert p.shape == expected_im_shape
        assert fg[0].shape == expected_im_shape
        assert bg[0].shape == expected_im_shape
        # ignore mask should be size of output
        # as ignore mask is only relevant to predicted region.
        assert ig.shape == (expected_im_shape[0] - 34,
                            expected_im_shape[1] - 34,
                            expected_im_shape[2] - 34)

    expected_annot_shape = (52, 253, 271)
    for a in batch_fgs + batch_bgs:
        assert a[0].shape == expected_annot_shape
