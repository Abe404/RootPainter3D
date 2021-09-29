"""
Copyright (C) 2020 Abraham George Smith

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

# pylint: disable=C0111,E1102,C0103,W0703,W0511,E1136
import os
import glob
import shutil
import time
from functools import partial
from math import ceil
import random
import numpy as np
import skimage.util as skim_util
from skimage.exposure import rescale_intensity
from skimage.io import imread
import nibabel as nib
from file_utils import ls
import nrrd
from pathlib import Path


def is_image(fname):
    """ extensions that have been tested with so far """
    extensions = {".jpg", ".png", ".jpeg", '.tif', '.tiff'}
    fname_ext = os.path.splitext(fname)[1].lower()
    return (fname_ext in extensions or fname.endswith('.nii.gz') or 
            fname.endswith('.npy') or fname.endswith('.nrrd'))

def normalize_tile(tile):
    if np.min(tile) < np.max(tile):
        tile = rescale_intensity(tile, out_range=(0, 1))
    else:
        # a single value tile is very rare but it has been claimed
        # that it may occur in X-ray background regions.
        # set to be 0 to ensure it is within a similar range to the normalized tiles.
        tile *= 0 
    assert np.min(tile) >= 0, f"tile min {np.min(tile)}"
    assert np.max(tile) <= 1, f"tile max {np.max(tile)}"
    return tile


def reconstruct_from_tiles(tiles, coords, output_shape):
    image = np.zeros(output_shape)
    for tile, (x_coord, y_coord, z_coord) in zip(tiles, coords):
        image[z_coord:z_coord+tile.shape[0],
              y_coord:y_coord+tile.shape[1],
              x_coord:x_coord+tile.shape[2]] = tile
    return image


def load_with_retry(load_fn, fpath):
    max_attempts = 60
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        # file systems are unpredictable.
        # We may have problems reading the file.
        # try-catch to avoid this.
        # (just try again)
        try:
            image = load_fn(fpath)
            return image
        except Exception as e:
            print('load_with_retry', fpath, 'exception', e)
            # This could be due to an empty annotation saved by the user.
            # Which happens rarely due to deleting all labels in an
            # existing annotation and is not a problem.
            # give it some time and try again.
            time.sleep(0.1)
    if attempts == max_attempts:
        raise Exception('Could not load. Too many retries')


def load_train_image_and_annot(dataset_dir, train_annot_dirs):
    """
    returns
        image (np.array) - image data
        annots (list(np.array)) - annotations associated with fname
        classes (list(string)) - classes for each annot,
                                 taken from annot directory name
        fname - file name
    """

    def load_random(train_annot_dirs, dataset_dir, _):
        # This might take ages, profile and optimize
        fnames = []
            # each annotation corresponds to an individual class.
        all_classes = []
        all_dirs = []
        for train_annot_dir in train_annot_dirs:
            annot_fnames = ls(train_annot_dir)
            fnames += annot_fnames
            # Assuming class name is in annotation path
            # i.e annotations/{class_name}/train/annot1.png,annot2.png..
            class_name = Path(train_annot_dir).parts[-2]
            class_name = Path(train_annot_dir).parts[-2]
            all_classes += [class_name] * len(annot_fnames)
            all_dirs += [train_annot_dir] * len(annot_fnames)

        fname = random.sample(fnames, 1)[0]

        # triggers retry if assertion fails
        assert is_image(fname), f'{fname} is not a valid image'

        # annots and classes associated with fname
        indices = [i for i, f in enumerate(fnames) if f == fname]
        classes = [all_classes[i] for i in indices]
        annot_dirs = [all_dirs[i] for i in indices]
        annots = []

        for annot_dir in annot_dirs:
            annot_path = os.path.join(annot_dir, fname)
            annot = imread(annot_path).astype(bool)
            # Why would we have annotations without content?
            assert np.sum(annot) > 0
            annots.append(annot)

        # it's possible the image has a different extenstion
        # so use glob to get it
        image_path_part = os.path.join(dataset_dir,
                                       os.path.splitext(fname)[0])
        image_path = glob.glob(image_path_part + '.*')[0]
        image = load_image(image_path)
        assert image.shape[2] == 3 # should be RGB

        # also return fname for debugging purposes.
        return image, annots, classes, fname

    load_random = partial(load_random, dataset_dir, train_annot_dirs)
    return load_with_retry(load_random, None)

def pad_3d(image, width, depth, mode='reflect', constant_values=0):
    pad_shape = [(depth, depth), (width, width), (width, width)]
    if len(image.shape) == 4:
        # assume channels first for 4 dimensional data.
        # don't pad channels
        pad_shape = [(0, 0)] + pad_shape
    if mode == 'reflect':
        return skim_util.pad(image, pad_shape, mode)
    return skim_util.pad(image, pad_shape, mode=mode,
                         constant_values=constant_values)


def get_val_tile_refs(annot_dirs, prev_tile_refs, out_shape):
    """
    Get tile info which covers all annotated regions of the annotation dataset.
    The list must be structured such that an index can be used to refer to each example
    so that it can be used with a dataloader.

    returns tile_refs (list)
        Each element of tile_refs is a list that includes:
            * image file name (string) - for loading the image from disk during validation
            * coord (x int, y int) - for addressing the location within the padded image
            * annot_mtime (int)
                The image annotation may get updated by the user at any time.
                We can use the mtime to check for this.
                If the annotation has changed then we need to retrieve tile
                coords for this image again. The reason for this is that we
                only want tile coords with annotations in. The user may have added or removed
                annotation in part of an image. This could mean a different set of coords (or
                not) should be returned for this image.

    Parameter prev_tile_refs is used for comparing both file names and mtime.

    The annot_dir folder should be checked for any new files (not in
    prev_tile_refs) or files with an mtime different from prev_tile_refs. For these
    file, the image should be loaded and new tile_refs should be retrieved. For all
    other images the tile_refs from prev_tile_refs can be used.
    """
    tile_refs = []

    # TODO, change this so we have a list of all annot names and their corresopnding classes.
    #  make a large list of classes that corresponds to all the file names. Ive done this elsewhere.

    # This might take ages, profile and optimize
    cur_annot_fnames = []
    # each annotation corresponds to an individual class.
    all_classes = []
    all_dirs = []
    for annot_dir in annot_dirs:
        annot_fnames = ls(annot_dir)
        cur_annot_fnames += annot_fnames
        # Assuming class name is in annotation path
        # i.e annotations/{class_name}/train/annot1.png,annot2.png..
        class_name = Path(annot_fnames).parts[-2]
        class_name = Path(annot_fnames).parts[-2]
        all_classes += [class_name] * len(annot_fnames)
        all_dirs += [annot_dir] * len(annot_fnames)
    
    prev_annot_fnames = [r[0] for r in prev_tile_refs]
    all_annot_fnames = set(cur_annot_fnames + prev_annot_fnames)

    for annot_dir, annot_fname in zip(all_dirs, all_annot_fnames):
        # get existing coord refs for this image
        prev_refs = [r for r in prev_tile_refs if r[0] == annot_fname]
        prev_mtimes = [r[2] for r in prev_tile_refs if r[0] == annot_fname]
        need_new_refs = False
        # if no refs for this image then check again
        if not prev_refs:
            need_new_refs = True
        else:
            annot_path = os.path.join(annot_dir, annot_fname)
            # if the file no longer exists then we do need new refs
            # surprisingly this did happen, I presume the file list was somehow out of date
            # and the removal of the file was only detected when trying to read it.
            if not os.path.isfile(annot_path):
                need_new_refs = True
            else:
                # otherwise check the modified time of the refs against the file.
                prev_mtime = prev_mtimes[0]
                cur_mtime = os.path.getmtime(os.path.join(annot_dir, annot_fname))

                # if file has been updated then get new refs
                if cur_mtime > prev_mtime:
                    need_new_refs = True
        if need_new_refs:
            new_file_refs = get_val_tile_refs_for_annot_3d(annot_dir, annot_fname, out_shape)
            tile_refs += new_file_refs
        else:
            tile_refs += prev_refs
    return tile_refs


def get_val_tile_refs_for_annot_3d(annot_dir, annot_fname, out_shape):
    """
    Each element of tile_refs is a list that includes:
        * image file name (string) - for loading the image from disk during validation
        * coord (x int, y int) - for addressing the location within the padded image
        * annot_mtime (int)
        * cached performance for this tile with previous (current best) model.
          Initialized to None but otherwise [tp, fp, tn, fn]
    """
    annot_path = os.path.join(annot_dir, annot_fname)
    if not os.path.isfile(annot_path):
        return []
    annot = load_image(annot_path)
    new_file_refs = []
    annot_shape = annot.shape[1:]
    coords = get_coords_3d(annot_shape, out_tile_shape=out_shape)

    mtime = os.path.getmtime(annot_path)
    for (x, y, z) in coords:
        annot_tile = annot[:, z:z+out_shape[0], y:y+out_shape[1], x:x+out_shape[2]]
        # we only want to validate on annotation tiles
        # which have annotation information.
        if np.any(annot_tile):
            # fname, [x, y, z], mtime, prev model metrics i.e [tp, tn, fp, fn] or None
            new_file_refs.append([annot_fname, [x, y, z], mtime, None])
    return new_file_refs


def get_coords_3d(annot_shape, out_tile_shape):
    """ Get the coordinates relative to the output image for the 
        validation routine. These coordinates will lead to patches
        which cover the image with minimum overlap (assuming fixed size patch) """

    assert len(annot_shape) == 3, str(annot_shape) # d, h, w
    
    depth_count = ceil(annot_shape[0] / out_tile_shape[0])
    vertical_count = ceil(annot_shape[1] / out_tile_shape[1])
    horizontal_count = ceil(annot_shape[2] / out_tile_shape[2])

    # first split the image based on the tiles that fit
    z_coords = [d*out_tile_shape[0] for d in range(depth_count-1)] # z is depth
    y_coords = [v*out_tile_shape[1] for v in range(vertical_count-1)]
    x_coords = [h*out_tile_shape[2] for h in range(horizontal_count-1)]

    # The last row and column of tiles might not fit
    # (Might go outside the image)
    # so get the tile positiion by subtracting tile size from the
    # edge of the image.
    lower_z = annot_shape[0] - out_tile_shape[0]
    bottom_y = annot_shape[1] - out_tile_shape[1]
    right_x = annot_shape[2] - out_tile_shape[2]

    z_coords.append(max(0, lower_z))
    y_coords.append(max(0, bottom_y))
    x_coords.append(max(0, right_x))

    # because its a cuboid get all combinations of x, y and z
    tile_coords = [(x, y, z) for x in x_coords for y in y_coords for z in z_coords]
    return tile_coords


def save_then_move(out_path, seg):
    """ need to save in a temp folder first and
        then move to the segmentation folder after saving
        this is because scripts are monitoring the segmentation folder
        and the file saving takes time..
        We don't want the scripts that monitor the segmentation
        folder to try loading the file half way through saving
        as this causes errors. Thus we save and then rename.
    """
    fname = os.path.basename(out_path)
    temp_path = os.path.join('/tmp', fname)
    if out_path.endswith('.nii.gz'):
        img = nib.Nifti1Image(seg, np.eye(4))
        img.to_filename(temp_path)
    elif out_path.endswith('.npy'):
        np.save(temp_path, seg)
    else:
        raise Exception(f'Unhandled {out_path}')
    shutil.move(temp_path, out_path)


def load_image(image_path):
    if image_path.endswith('.npy'):
        image = np.load(image_path, mmap_mode='c')
    elif image_path.endswith('.nrrd'):
        image, _ = nrrd.read(image_path)
    elif image_path.endswith('.nii.gz'):
        # We don't currently use them during training but it's useful to be
        # able to load nifty files directory to give the user
        # more convenient segmentation options.
        image = nib.load(image_path)
        image = np.array(image.dataobj)
    return image
