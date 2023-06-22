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
import numpy as np
import torch
import time
import im_utils
from torch.utils.data import DataLoader
from datasets import RPDataset
import nibabel as nib
import model_utils

# sync directory for use with tests
sync_dir = os.path.join(os.getcwd(), 'test_rp_sync')
annot_dir = os.path.join(sync_dir, 'projects', 'total_seg_test', 'annotations')
datasets_dir = os.path.join(sync_dir, 'datasets')
total_seg_dataset_dir = os.path.join(datasets_dir, 'total_seg')

subset_dir_images = os.path.join(total_seg_dataset_dir, '50_random_images')
subset_dir_annots = os.path.join(total_seg_dataset_dir, '50_random_annots')

annot_train_dir = os.path.join(subset_dir_annots, 'liver', 'train')

timeout_ms = 20000


def convert_seg_to_annot(in_fpath):
    seg = im_utils.load_image(in_fpath)
    assert len(seg.shape) == 3, 'should be 3d binary mask, shape: ' + str(seg.shape)
    seg_bg = seg == 0
    seg_fg = seg > 0
    annot = np.zeros([2] + list(seg.shape))
    annot[0] = seg_bg
    annot[1] = seg_fg
    return annot


def prep_random_50(dataset_dir):
    # take random 50 images from total segmentor dataset and put them in a folder
    import random
    import shutil
   
    if not os.path.isdir(subset_dir_images):
        print('creating subsets')
        all_dirs = os.listdir(dataset_dir)
        all_dirs = [d for d in all_dirs if '50_random' not in d]
        all_dirs = [d for d in all_dirs if os.path.isdir(os.path.join(dataset_dir, d))]
        sampled_dirs = random.sample(all_dirs, 50)

        os.makedirs(subset_dir_images)
        os.makedirs(subset_dir_annots)
        os.makedirs(annot_train_dir)

        for d in sampled_dirs:
            imfpath = os.path.join(dataset_dir, d, 'ct.nii.gz')
            out_im_fpath = os.path.join(subset_dir_images, d  + '_ct.nii.gz')
            im = im_utils.load_image(imfpath)
            img = nib.Nifti1Image(im, np.eye(4))
            img.to_filename(out_im_fpath)

            in_annot_fpath = os.path.join(dataset_dir, d, 'segmentations', 'liver.nii.gz')
            out_annot_fpath = os.path.join(annot_train_dir, d  + '_ct.nii.gz')

            # convert segmentations (total seg format) to rp3d annotations
            annot = convert_seg_to_annot(in_annot_fpath)
            assert im.shape == annot[0].shape
            annot = nib.Nifti1Image(annot, np.eye(4))
            annot.to_filename(out_annot_fpath)
    
def setup_function():
    import urllib.request
    import zipfile
    import shutil
    from test_utils import dl_dir_from_zip
    print('running setup')
    # prepare training dataset
    #if not os.path.isdir(datasets_dir):
    #    os.makedirs(datasets_dir)
    #total_seg_url = 'https://zenodo.org/record/6802614/files/Totalsegmentator_dataset.zip'
    #dl_dir_from_zip(total_seg_url, dataset_dir)
    prep_random_50(total_seg_dataset_dir)


def test_training():
    import data_utils
    import train_utils
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
                        mode='train',
                        patch_refs=None,
                        use_seg_in_training=False,
                        length=120)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=data_utils.collate_fn,
                        num_workers=num_workers,
                        drop_last=False, pin_memory=True)

    model = model_utils.random_model(classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.99, nesterov=True)

    start_time = time.time()

    train_result = train_utils.epoch(model,
                                     classes,
                                     loader,
                                     batch_size,
                                     optimizer=optimizer,
                                     step_callback=None,
                                     stop_fn=None)
    print('')
    print('Train epoch complete in', round(time.time() - start_time, 1), 'seconds')
    # pass - epoch runs without error.
