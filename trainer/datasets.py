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

import torch
import numpy as np
from skimage import img_as_float32
from torch.utils.data import Dataset

from file_utils import get_crop_start

from im_utils import load_train_image_and_annot
import im_utils

def rnd():
    """ Give higher than random chance to select the edges """
    return max(0, min(1, (1.2 * random.random()) - 0.1))


class RPDataset(Dataset):
    def __init__(self, annot_dirs, dataset_dir, in_w, out_w,
                 in_d, out_d, mode, tile_refs=None, length=None):
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
        self.dataset_dir = dataset_dir
        assert (tile_refs is None) or (length is None) and (length or tile_refs)
        # if tile_refs are defined then these will be used.
        self.tile_refs = tile_refs
        # other wise length will return the number of items
        self.length = length

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

    def get_random_tile_3d(self, annots, image, fname):
        # this will find something eventually as we know
        # all annotation contain labels somewhere

        # Limits for possible sampling locations from image (based on size of image)
        depth_lim = image.shape[0] - self.in_d
        bottom_lim = image.shape[1] - self.in_w
        right_lim = image.shape[2] - self.in_w

        (x_crop_start,
         y_crop_start,
         z_crop_start) = get_crop_start(fname)

        pad_d = (self.in_d - self.out_d) // 2
        pad_h = (self.in_w - self.out_w) // 2
        pad_w = (self.in_w - self.out_w) // 2
    
        attempts = 0 
        warn_after_attempts = 100

        
        while True:
            attempts += 1
            x_in = math.floor(rnd() * right_lim)
            y_in = math.floor(rnd() * bottom_lim)
            z_in = math.floor(rnd() * depth_lim)

            # remove the padding from the coords to get the respective annotation region
            # We include the minimum padded mount in the annotation region
            x_in_annot = x_in - (x_crop_start - pad_w)
            y_in_annot = y_in - (y_crop_start - pad_h)
            z_in_annot = z_in - (z_crop_start - pad_d)

            annot_tile_centers = []

            for annot in annots:
                # Get the corresponding region of the annotation after network crop
                annot_tile_centers.append(annot[:,
                                                max(0, z_in_annot):z_in_annot+self.out_d,
                                                max(0, y_in_annot):y_in_annot+self.out_w,
                                                max(0, x_in_annot):x_in_annot+self.out_w])

            # we only want annotations with defiend regions in the output area.
            # Otherwise we will have nothing to update the loss.
            if np.any([np.any(a) for a in annot_tile_centers]):
                # ok we have some annotation for this
                # part of the image so let's return the patch.
                im_tile = image[z_in:z_in+self.in_d,
                                y_in:y_in+self.in_w,
                                x_in:x_in+self.in_w]
                # return annot tile with the full crop,
                # to allow crop post augment
                return annot_tile_centers, im_tile
            if attempts > warn_after_attempts:
                print(f'Warning {attempts} attempts to get random patch from {fname}')
                warn_after_attempts *= 10
    

    def get_train_item_3d(self, tile_ref):
        # When tile_ref is specified we use these coordinates to get
        # the input tile. Otherwise we will sample randomly
        if tile_ref:
            im_tile, foregrounds, backgrounds, classes = self.get_tile_from_ref_3d(tile_ref)
            # For now just return the tile. We plan to add augmentation here.
            return im_tile, foregrounds, backgrounds, classes

        (image, annots, classes, fname) = load_train_image_and_annot(self.dataset_dir,
                                                                     self.annot_dirs)

        annot_tiles, im_tile = self.get_random_tile_3d(annots, image, fname)

        # if the annotation is not big enough then pad it out. 
        if  annot_tiles[0].shape[1:] != (self.out_d, self.out_w, self.out_w):
            for i, annot_tile in enumerate(annot_tiles):
                annot_tiles[i] = self.get_padded_annot(fname, annot_tile)
            
        assert annot_tiles[0].shape[1:] == (self.out_d, self.out_w, self.out_w), (
            f" annot is {annots[0].shape}")

        assert im_tile.shape == (self.in_d, self.in_w, self.in_w), (
            f" shape is {im_tile.shape}")

        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        # ensure image is still 32 bit after normalisation.
        im_tile = im_tile.astype(np.float32)
        # need list of foregrounds and masks for all tiles.
        foregrounds = []
        backgrounds = []
        for annot_tile in annot_tiles:
            print('annot tile shape is ', annot_tile.shape)
            print('does this match the assumptions of the next lines?')
            exit()
            foreground = np.array(annot_tile)[:, :, 0]
            background = np.array(annot_tile)[:, :, 1]
            foreground = foreground.astype(np.int64)
            foreground = torch.from_numpy(foreground)
            foregrounds.append(foreground)
            background = background.astype(np.int64)
            background = torch.from_numpy(background)
            backgrounds.append(background)
        im_tile = im_tile.astype(np.float32)
        im_tile = np.moveaxis(im_tile, -1, 0)
        return im_tile, foregrounds, backgrounds, classes
       
    def get_padded_annot(self, fname, annot_tile):
        fname_parts = fname.replace('.nii.gz', '').split('_')
        # This padding information allows us to convert between bounded image
        # and annotation coordinates
        x_crop_start = int(fname_parts[-8])
        y_crop_start = int(fname_parts[-5])
        z_crop_start = int(fname_parts[-2])

        pad_d = (self.in_d - self.out_d) // 2
        pad_h = (self.in_w - self.out_w) // 2
        pad_w = (self.in_w - self.out_w) // 2

        annot_tile_zeros = np.zeros((annot_tile.shape[0], self.out_d, self.out_w, self.out_w))
        x_start = (x_crop_start - pad_w)
        y_start = (y_crop_start - pad_h)
        z_start = (z_crop_start - pad_d)
        annot_tile_zeros[:,
                         z_start:z_start+annot_tile.shape[1],
                         y_start:y_start+annot_tile.shape[2], 
                         x_start:x_start+annot_tile.shape[3]] = annot_tile
        annot_tile = annot_tile_zeros
        return annot_tile

    def get_val_item(self, tile_ref):
        return self.get_tile_from_ref_3d(tile_ref)

    def get_tile_from_ref_3d(self, tile_ref):
        """ return image tile, annotation tile and mask
            for a given file name ans location specified
            in x,y,z relative to the annotation """

        fname, (tile_x, tile_y, tile_z), _, _ = tile_ref
        image_path = os.path.join(self.dataset_dir, fname)
        image = im_utils.load_with_retry(im_utils.load_image, image_path)

        annot_tiles = []
        for annot_dir in self.annot_dirs:
            annot_path = os.path.join(annot_dir, fname)
            annot = im_utils.load_with_retry(im_utils.load_image, annot_path)
            # The x, y and z are in reference to the annotation tile before padding.
            annot_tile = annot[:,
                               tile_z:tile_z+self.out_d,
                               tile_y:tile_y+self.out_w,
                               tile_x:tile_x+self.out_w]

            annot_tile = self.get_padded_annot(fname, annot_tile)

            assert annot_tile.shape[1:] == (self.out_d, self.out_w, self.out_w), (
                f"annot tile shape is {annot_tile.shape[1:]}")
            annot_tiles.append(annot_tile)
 
        foregrounds = []
        backgrounds = []
        for annot_tile in annot_tiles:
            foreground = np.array(annot_tile)[:, :, 0]
            background = np.array(annot_tile)[:, :, 1]
            foreground = foreground.astype(np.int64)
            foreground = torch.from_numpy(foreground)
            foregrounds.append(foreground)
            background = background.astype(np.int64)
            background = torch.from_numpy(background)
            backgrounds.append(background)

        im_tile = image[tile_z:tile_z + self.in_d,
                        tile_y:tile_y + self.in_w,
                        tile_x:tile_x + self.in_w]
           
        assert im_tile.shape == (self.in_d, self.in_w, self.in_w), (
            f" shape is {im_tile.shape}")

        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        # ensure image is still 32 bit after normalisation.
        im_tile = im_tile.astype(np.float32)
        mask = annot_tile[0] + annot_tile[1]
        mask[mask > 1] = 1
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        im_tile = torch.from_numpy(np.expand_dims(im_tile, axis=0))
        annot_tile = torch.from_numpy(annot_tile).long()
        classes = [os.path.basename(d) for d in self.annot_dirs]
        return im_tile, foregrounds, backgrounds, classes
