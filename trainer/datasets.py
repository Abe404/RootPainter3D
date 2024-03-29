"""
Copyright (C) 2019, 2020, 2023 Abraham George Smith

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

import random
import os
from pathlib import Path
from enum import Enum

import torch
import numpy as np
from skimage import img_as_float32
from torch.utils.data import Dataset

from file_utils import ls
from im_utils import load_train_image_and_annot
import im_utils

class Modes(Enum):
    TRAIN = 1
    VAL = 2

class RPDataset(Dataset):
    def __init__(self, annot_dirs, train_seg_dirs, dataset_dir, in_w, out_w,
                 in_d, out_d, mode, patch_refs=None,
                 use_seg_in_training=True, length=None):
        """
        in_w and out_w are the patch size in pixels

        target_classes is a list of the possible output classes
            the position in the list is the index (target) to be predicted by
            the network in the output.
            The value of the elmenent is the rgba (int, int, int) used to draw this
            class in the annotation.

            When the data is 3D the raw channels (for each class)
            are saved and the RGB values are not necessary.
        """
        assert mode in Modes
        self.mode = mode
        self.in_w = in_w
        self.out_w = out_w
        self.in_d = in_d
        self.out_d = out_d
        self.annot_dirs = annot_dirs
        self.train_seg_dirs = train_seg_dirs
        self.dataset_dir = dataset_dir
        assert (patch_refs is None) or (length is None) and (length or patch_refs)
        # if patch_refs are defined then these will be used.
        self.patch_refs = patch_refs
        # other wise length will return the number of items
        self.length = length
        self.use_seg = use_seg_in_training

    def __len__(self):
        if self.mode == Modes.VAL:
            return len(self.patch_refs)
        if self.patch_refs is not None:
            return len(self.patch_refs)
        return self.length

    def __getitem__(self, i):
        if self.mode == Modes.VAL:
            return self.get_val_item(self.patch_refs[i])
        return self.get_train_item()

    def get_train_item(self):
        return self.get_train_item_3d()


    def should_force_fg(self):
        num_annots = len(ls(self.annot_dirs[0])) # estimate num annotations from first class 
        # start at 90% force fg and go down to 0 by the time 90 images are annotated.
        force_fg_prob = max(0, (90-(num_annots)) / 100) 
        force_fg = force_fg_prob > random.random()
        return force_fg


    def get_train_item_3d(self):
        force_fg = self.should_force_fg()
        (image, annots, segs, classes, fname) = load_train_image_and_annot(self.dataset_dir,
                                                                           self.train_seg_dirs,
                                                                           self.annot_dirs,
                                                                           self.use_seg,
                                                                           force_fg)

        annot_patches, seg_patches, im_patch = im_utils.get_random_patch_3d(annots, segs,
                                                                            image,
                                                                            fname, force_fg,
                                                                            self.in_d, self.in_w)
        im_patch = img_as_float32(im_patch)
        im_patch = im_utils.normalize_patch(im_patch)
        # ensure image is still 32 bit after normalisation.
        im_patch = im_patch.astype(np.float32)
        # need list of foregrounds and masks for all patches.
        foregrounds = []
        backgrounds = []
        # ignore_masks prevent coordinates from being added to the metrics computation twice.
        # They tell us which region of the image prediction has already been stored in the metrics
        # and thus should not be added to the metrics again.
        ignore_mask = None
        for annot_patch in annot_patches:
            #annot patch shape is  (2, 18, 194, 194)
            foreground = np.array(annot_patch)[1]
            background = np.array(annot_patch)[0]
            foreground = foreground.astype(np.int64)
            foreground = torch.from_numpy(foreground)
            foregrounds.append(foreground)
            background = background.astype(np.int64)
            background = torch.from_numpy(background)
            backgrounds.append(background)

            # mask is same for all annotations so just return one.
            im_d = im_patch.shape[0]
            im_h = im_patch.shape[1]
            im_w = im_patch.shape[2]
            # output is 34 less than input. igmore mask only concerns output
            ignore_mask = np.zeros((im_d - 34, im_h - 34, im_w - 34), dtype=np.uint8)

            shape_str = f'fname: {fname}, im_patch.shape: {im_patch.shape}, fg: {foreground.shape}'
            assert im_patch.shape == foreground.shape, shape_str

        im_patch = im_patch.astype(np.float32)
        
        assert len(backgrounds) == len(seg_patches) == len(foregrounds) == len(classes)
    
        return im_patch, foregrounds, backgrounds, ignore_mask, seg_patches, classes
       
    def get_val_item(self, patch_ref):
        return self.get_patch_from_ref_3d(patch_ref)

    def load_im(self, fname):
        image_path = os.path.join(self.dataset_dir, fname)
        # image could have nrrd extension
        if not os.path.isfile(image_path):
            image_path = image_path.replace('.nii.gz', '.nrrd')
        image = im_utils.load_with_retry(im_utils.load_image, image_path)
        # FiXME: Consider moving padding to the GPU. See:
        # https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad3d.html#torch.nn.ReflectionPad3d
        # pad so seg will be size of input image
        image = np.pad(image, ((17, 17), (17, 17), (17, 17)), mode='constant')
        return image

    def get_patch_from_ref_3d(self, patch_ref):
        """ return image patch, annotation patch and ignore mask
            for a given file name and location specified
            in x,y,z relative to the full image annotation """

        # TODO: One concern is that we could end up with a lot of these patch_refs. 
        #       is adding the ignore_mask going to introduce significant memory usage?
        #       please investigate.
        image = self.load_im(patch_ref.annot_fname)
        annots, classes = self.get_annots_for_image(patch_ref.annot_fname)

        (im_patch, foregrounds,
         backgrounds, segs) = self.get_patch_from_image(image, annots, patch_ref)            

        return (im_patch, foregrounds, backgrounds,
                patch_ref.ignore_mask, segs, classes)


    def get_patch_from_image(self, image, annots, patch_ref):
        annot_patches = []
        for annot in annots: 
            annot_patch = annot[:,
                                patch_ref.z:patch_ref.z+self.in_d,
                                patch_ref.y:patch_ref.y+self.in_w,
                                patch_ref.x:patch_ref.x+self.in_w]
            annot_patches.append(annot_patch)

        im_patch = image[patch_ref.z:patch_ref.z + self.in_d,
                        patch_ref.y:patch_ref.y + self.in_w,
                        patch_ref.x:patch_ref.x + self.in_w]

        # patch will either be the specified in_d/in_w or the image dimension.
        # whichever is smaller.
        expected_patch_d = min(image.shape[0], self.in_d)
        expected_patch_h = min(image.shape[1], self.in_w)
        expected_patch_w = min(image.shape[2], self.in_w)
        expected_shape = (expected_patch_d, expected_patch_h, expected_patch_w)

        assert im_patch.shape == expected_shape, (
            f" shape is {im_patch.shape} but expected shape is {expected_shape}")

        assert annot_patch.shape[1:] == expected_shape, (
            f" annot.shape[1:] is {annot_patch.shape}, and "
            f" should be {expected_shape}")

        foregrounds = []
        backgrounds = []
        for annot_patch in annot_patches:
            foreground = np.array(annot_patch)[1]
            background = np.array(annot_patch)[0]
            foreground = foreground.astype(np.int64)
            foregrounds.append(foreground)
            background = background.astype(np.int64)
            backgrounds.append(background)

        im_patch = img_as_float32(im_patch)
        im_patch = im_utils.normalize_patch(im_patch)
        im_patch = im_patch.astype(np.float32)
        im_patch = np.expand_dims(im_patch, axis=0)
        segs = [None] * len(backgrounds)

        return im_patch, foregrounds, backgrounds, segs  
 
    def get_annots_for_image(self, annot_fname): 
        classes = []
        annots = []
        for annot_dir in self.annot_dirs:
            annot_path = os.path.join(annot_dir, annot_fname)
            if os.path.isfile(annot_path):
                annot = im_utils.load_with_retry(im_utils.load_image, annot_path)
                classes.append(Path(annot_dir).parts[-2])
                
                # pad to provide annotation at same size as input image.
                annot = np.pad(annot, ((0, 0), (17, 17), (17, 17), (17, 17)), mode='constant')
                annots.append(annot)
        return annots, classes

