"""
Copyright (C) 2019, 2020 Abraham George Smith

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
import math
import os
from pathlib import Path

import torch
import numpy as np
from skimage import img_as_float32
from torch.utils.data import Dataset

from im_utils import load_train_image_and_annot
import im_utils

def rnd():
    """ Give higher than random chance to select the edges """
    return max(0, min(1, (1.2 * random.random()) - 0.1))


class RPDataset(Dataset):
    def __init__(self, annot_dirs, train_seg_dirs, dataset_dir, in_w, out_w,
                 in_d, out_d, mode, tile_refs=None,
                 use_seg_in_training=True, length=None):
        """
        in_w and out_w are the tile size in pixels

        target_classes is a list of the possible output classes
            the position in the list is the index (target) to be predicted by
            the network in the output.
            The value of the elmenent is the rgba (int, int, int) used to draw this
            class in the annotation.

            When the data is 3D the raw channels (for each class)
            are saved and the RGB values are not necessary.
        """
        self.mode = mode
        self.in_w = in_w
        self.out_w = out_w
        self.in_d = in_d
        self.out_d = out_d
        self.annot_dirs = annot_dirs
        self.train_seg_dirs = train_seg_dirs
        self.dataset_dir = dataset_dir
        assert (tile_refs is None) or (length is None) and (length or tile_refs)
        # if tile_refs are defined then these will be used.
        self.tile_refs = tile_refs
        # other wise length will return the number of items
        self.length = length
        self.use_seg = use_seg_in_training

    def __len__(self):
        if self.mode == 'val':
            return len(self.tile_refs)
        if self.tile_refs is not None:
            return len(self.tile_refs)
        return self.length

    def __getitem__(self, i):
        if self.mode == 'val':
            return self.get_val_item(self.tile_refs[i])
        if self.tile_refs is not None:
            return self.get_train_item(self.tile_refs)
        return self.get_train_item()

    def get_train_item(self, tile_ref=None):
        return self.get_train_item_3d(tile_ref)

    def get_random_tile_3d(self, annots, segs, image, fname):
        # this will find something eventually as we know
        # all annotation contain labels somewhere

        # Limits for possible sampling locations from image (based on size of image)
        depth_lim = image.shape[0] - self.in_d
        bottom_lim = image.shape[1] - self.in_w
        right_lim = image.shape[2] - self.in_w

        attempts = 0 
        warn_after_attempts = 100
        
        while True:
            attempts += 1
            x_in = math.floor(rnd() * right_lim)
            y_in = math.floor(rnd() * bottom_lim)
            z_in = math.floor(rnd() * depth_lim)

            annot_tiles = []
            seg_tiles = []
            for seg, annot in zip(segs, annots):
                # Get the corresponding region of the annotation after network crop
                annot_tiles.append(annot[:,
                                         z_in:z_in+self.in_d,
                                         y_in:y_in+self.in_w,
                                         x_in:x_in+self.in_w])
                if seg is None:
                    seg_tiles.append(None)
                else:
                    seg_tiles.append(seg[z_in:z_in+self.in_d,
                                         y_in:y_in+self.in_w,
                                         x_in:x_in+self.in_w])



            # we only want annotations with defiend regions in the output area.
            # Otherwise we will have nothing to update the loss.
            if np.any([np.any(a) for a in annot_tiles]):
                # ok we have some annotation for this
                # part of the image so let's return the patch.
                im_tile = image[z_in:z_in+self.in_d,
                                y_in:y_in+self.in_w,
                                x_in:x_in+self.in_w]

                return annot_tiles, seg_tiles, im_tile
            if attempts > warn_after_attempts:
                print(f'Warning {attempts} attempts to get random patch from {fname}')
                warn_after_attempts *= 10
    

    def get_train_item_3d(self, tile_ref):
        # When tile_ref is specified we use these coordinates to get
        # the input tile. Otherwise we will sample randomly
        if tile_ref:
            raise Exception('not using these')
            im_tile, foregrounds, backgrounds, classes = self.get_tile_from_ref_3d(tile_ref)
            # For now just return the tile. We plan to add augmentation here.
            return im_tile, foregrounds, backgrounds, classes

        (image, annots, segs, classes, fname) = load_train_image_and_annot(self.dataset_dir,
                                                                           self.train_seg_dirs,
                                                                           self.annot_dirs,
                                                                           self.use_seg)
        annot_tiles, seg_tiles, im_tile = self.get_random_tile_3d(annots, segs, image, fname)

        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        # ensure image is still 32 bit after normalisation.
        im_tile = im_tile.astype(np.float32)
        # need list of foregrounds and masks for all tiles.
        foregrounds = []
        backgrounds = []
        for annot_tile in annot_tiles:
            #annot tile shape is  (2, 18, 194, 194)
            foreground = np.array(annot_tile)[1]
            background = np.array(annot_tile)[0]
            foreground = foreground.astype(np.int64)
            foreground = torch.from_numpy(foreground)
            foregrounds.append(foreground)
            background = background.astype(np.int64)
            background = torch.from_numpy(background)
            backgrounds.append(background)

        im_tile = im_tile.astype(np.float32)
        
        # add dimension for input channel
        im_tile = np.expand_dims(im_tile, axis=0)
        assert len(backgrounds) == len(seg_tiles)
        return im_tile, foregrounds, backgrounds, seg_tiles, classes
       
    def get_val_item(self, tile_ref):
        return self.get_tile_from_ref_3d(tile_ref)

    def get_tile_from_ref_3d(self, tile_ref):
        """ return image tile, annotation tile and mask
            for a given file name ans location specified
            in x,y,z relative to the annotation """

        fname, (tile_x, tile_y, tile_z), _, _ = tile_ref
        image_path = os.path.join(self.dataset_dir, fname)
        # image could have nrrd extension
        if not os.path.isfile(image_path):
            image_path = image_path.replace('.nii.gz', '.nrrd')
        image = im_utils.load_with_retry(im_utils.load_image, image_path)
        #  needs to be swapped to channels first and rotated etc
        # to be consistent with everything else.
        # todo: consider removing this soon.
        image = np.rot90(image, k=3)
        image = np.moveaxis(image, -1, 0) # depth moved to beginning
        # reverse lr and ud
        image = image[::-1, :, ::-1]

        # pad so seg will be size of input image
        image = np.pad(image, ((17, 17), (17, 17), (17, 17)), mode='constant')

        classes = []
        foregrounds = []
        backgrounds = []
        annot_tiles = []

        for annot_dir in self.annot_dirs:
            annot_path = os.path.join(annot_dir, fname)

            annot = im_utils.load_with_retry(im_utils.load_image, annot_path)
            classes.append(Path(annot_dir).parts[-2])
            
            # pad to provide annotation at same size as input image.
            annot = np.pad(annot, ((0, 0), (17, 17), (17, 17), (17, 17)), mode='constant')
            # The x, y and z are in reference to the annotation tile before padding.
            annot_tile = annot[:,
                               tile_z:tile_z+self.in_d,
                               tile_y:tile_y+self.in_w,
                               tile_x:tile_x+self.in_w]

            assert annot_tile.shape[1:] == (self.in_d, self.in_w, self.in_w), (
                f" annot is {annot_tile.shape}, and "
                f"should be ({self.in_d},{self.in_w},{self.in_w})")

            #else: # not auto-complete
            #    if os.path.isfile(annot_path):
            #        # The x, y and z are in reference to the annotation tile before padding.
            #        annot_tile = annot[:,
            #                           tile_z:tile_z+self.out_d,
            #                           tile_y:tile_y+self.out_w,
            #                           tile_x:tile_x+self.out_w]

            annot_tiles.append(annot_tile)

        im_tile = image[tile_z:tile_z + self.in_d,
                        tile_y:tile_y + self.in_w,
                        tile_x:tile_x + self.in_w]
 
        assert im_tile.shape == (self.in_d, self.in_w, self.in_w), (
            f" shape is {im_tile.shape}")
        
        for annot_tile in annot_tiles:
            foreground = np.array(annot_tile)[1]
            background = np.array(annot_tile)[0]
            foreground = foreground.astype(np.int64)
            foreground = torch.from_numpy(foreground)
            foregrounds.append(foreground)
            background = background.astype(np.int64)
            background = torch.from_numpy(background)
            backgrounds.append(background)

        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        im_tile = im_tile.astype(np.float32)
        im_tile = np.expand_dims(im_tile, axis=0)
        segs = [None] * len(backgrounds)
        return im_tile, foregrounds, backgrounds, segs, classes
