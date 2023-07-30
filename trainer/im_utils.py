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
import math
import shutil
import time
from functools import partial
from math import ceil
import traceback
import random
from pathlib import Path

import numpy as np
from skimage.exposure import rescale_intensity
import nibabel as nib
import nrrd

from file_utils import ls
from patch_ref import PatchRef


def get_random_patch_3d(annots, segs, image, fname,
                        force_fg, in_d, in_w):
    """ return a patch with random location with specified size
        from the supplied annots, segs, and image.
        If force_fg is true then make sure the patch contains foreground.
    """
    def rnd():
        """ Give higher than random chance to select the edges """
        return max(0, min(1, (1.2 * random.random()) - 0.1))

    def annot_patch_has_fg(annot):
        return np.any(annot[1][17:-17,17:-17,17:-17])

    # Limits for possible sampling locations from image (based on size of image)
    depth_lim = image.shape[0] - min(in_d, image.shape[0])
    bottom_lim = image.shape[1] - min(in_w, image.shape[1])
    right_lim = image.shape[2] - min(in_w, image.shape[2])

    attempts = 0 
    warn_after_attempts = 1000
    
    while True:
        attempts += 1
        x_in = math.floor(rnd() * right_lim)
        y_in = math.floor(rnd() * bottom_lim)
        z_in = math.floor(rnd() * depth_lim)

        annot_patches = []
        seg_patches = []
        for seg, annot in zip(segs, annots):
            # Get the corresponding region of the annotation after network crop
            annot_patches.append(annot[:,
                                       z_in:z_in+in_d,
                                       y_in:y_in+in_w,
                                       x_in:x_in+in_w])
            if seg is None:
                seg_patches.append(None)
            else:
                seg_patches.append(seg[z_in:z_in+in_d,
                                       y_in:y_in+in_w,
                                       x_in:x_in+in_w])

        # we only want annotations with defiend regions in the output area.
        # Otherwise we will have nothing to update the loss.
        if np.any([np.any(a) for a in annot_patches]):
            # if force fg is true then make sure fg is defined.
            if not force_fg or np.any([annot_patch_has_fg(a) for a in annot_patches]):
                # ok we have some annotation for this
                # part of the image so let's return the patch.
                im_patch = image[z_in:z_in+in_d,
                                 y_in:y_in+in_w,
                                 x_in:x_in+in_w]

                return annot_patches, seg_patches, im_patch
        if attempts > warn_after_attempts:
            print(f'Warning {attempts} attempts to get random patch from {fname}')
            warn_after_attempts *= 10


def maybe_pad_image_to_pad_size(image, in_patch_shape):
    # if the image is smaller than the patch size then pad it to be the same as the patch.
    padded_for_patch = False
    patch_pad_z = 0
    patch_pad_y = 0
    patch_pad_x = 0

    if image.shape[0] < in_patch_shape[0]:
        padded_for_patch = True
        patch_pad_z = in_patch_shape[0] - image.shape[0]

    if image.shape[1] < in_patch_shape[1]:
        padded_for_patch = True
        patch_pad_y = in_patch_shape[1] - image.shape[1]

    if image.shape[2] < in_patch_shape[2]:
        padded_for_patch = True
        patch_pad_x = in_patch_shape[2] - image.shape[2]

    if padded_for_patch:
        padded_image = np.zeros((
            image.shape[0] + patch_pad_z,
            image.shape[1] + patch_pad_y,
            image.shape[2] + patch_pad_x),
            dtype=image.dtype)
        padded_image[:image.shape[0], :image.shape[1], :image.shape[2]] = image
        image = padded_image  
    return image, padded_for_patch


def is_image(fname):
    """ extensions that have been tested with so far """
    extensions = {".jpg", ".png", ".jpeg", '.tif', '.tiff'}
    fname_ext = os.path.splitext(fname)[1].lower()
    return (fname_ext in extensions or fname.endswith('.nii.gz') or 
            fname.endswith('.npy') or fname.endswith('.nrrd'))

def normalize_patch(patch):
    if np.min(patch) < np.max(patch):
        patch = rescale_intensity(patch, out_range=(0, 1))
    else:
        # a single value patch is very rare but it has been claimed
        # that it may occur in X-ray background regions.
        # set to be 0 to ensure it is within a similar range to the normalized patches.
        patch *= 0 
    assert np.min(patch) >= 0, f"patch min {np.min(patch)}"
    assert np.max(patch) <= 1, f"patch max {np.max(patch)}"
    return patch


def reconstruct_from_patches(patches, coords, output_shape):
    image = np.zeros(output_shape)
    # reverse patches and coords because in validation we dont
    # overwrite predictions for coords earlier in the list.
    # We instead tell patch-refs/coords later in the list to ignore
    # already predicted regions.
    patches.reverse()
    coords.reverse()
    for patch, (x_coord, y_coord, z_coord) in zip(patches, coords):
        image[z_coord:z_coord+patch.shape[0],
              y_coord:y_coord+patch.shape[1],
              x_coord:x_coord+patch.shape[2]] = patch
    return image


def load_with_retry(load_fn, fpath):
    max_attempts = 200
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
            print('load_with_retry', fpath, 'exception', e, traceback.format_exc())
            # This could be due to an empty annotation saved by the user.
            # Which happens rarely due to deleting all labels in an
            # existing annotation and is not a problem.
            # give it some time and try again.
            time.sleep(0.1)
    if attempts == max_attempts:
        raise Exception('Could not load. Too many retries')



def load_image_and_annot_for_seg(dataset_dir, train_annot_dirs, fname):
    """
    Load image and annotation to enable segmentation.

    returns
        image (np.array) - image data
        annots (list(np.array)) - annotations associated with fname
        classes (list(string)) - classes for each annot,
                                 taken from annot directory name
        fname - file name
    """
    def load_fname(train_annot_dirs, dataset_dir, fname):
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
            all_classes += [class_name] * len(annot_fnames)
            all_dirs += [train_annot_dir] * len(annot_fnames)

        # triggers retry if assertion fails
        assert is_image(fname), f'{fname} is not a valid image'

        # annots and classes associated with fname
        indices = [i for i, f in enumerate(fnames) if f == fname]
        classes = [all_classes[i] for i in indices]
        annot_dirs = [all_dirs[i] for i in indices]
        annots = []

        for annot_dir in annot_dirs:
            annot_fpath = os.path.join(annot_dir, fname)
            annot = load_image(annot_fpath)
            # Why would we have annotations without content?
            assert np.sum(annot) > 0
            annot = np.pad(annot, ((0, 0), (17,17), (17,17), (17, 17)), mode='constant')
            annots.append(annot)

        # it's possible the image has a different extenstion
        # so use glob to get it
        fname_no_ext = fname.replace('.nii.gz', '').replace('.nrrd', '')
        image_path_part = os.path.join(dataset_dir, fname_no_ext)
        image_path = glob.glob(image_path_part + '.*')[0]
        image = load_image(image_path)
        # also return fname for debugging purposes.
        return image, annots, classes, fname

    load_fname = partial(load_fname, train_annot_dirs, dataset_dir)
    return load_with_retry(load_fname, fname)



def load_train_image_and_annot(dataset_dir, train_seg_dirs, train_annot_dirs, use_seg,
                               force_fg):
    """
    returns
        image (np.array) - image data
        annots (list(np.array)) - annotations associated with fname
        segs (list(np.array)) - segmentations associated with fname
        classes (list(string)) - classes for each annot,
                                 taken from annot directory name
        fname - file name
    """

    def load_random(train_annot_dirs, train_seg_dirs, dataset_dir, _):
        # This might take ages, profile and optimize
        fnames = []
        # each annotation corresponds to an individual class.
        all_classes = []
        all_annot_dirs = []
        all_seg_dirs = []

        # puts None values at the end. 
        train_seg_dirs =  sorted(train_seg_dirs, key=lambda x: (x is None, x))
        train_annot_dirs = sorted(train_annot_dirs)

        assert len(train_seg_dirs) == len(train_annot_dirs)
    
        for train_seg_dir, train_annot_dir in zip(train_seg_dirs, train_annot_dirs):
            annot_fnames = ls(train_annot_dir)
            fnames += annot_fnames
            # Assuming class name is in annotation path
            # i.e annotations/{class_name}/train/annot1.png,annot2.png..
            class_name = Path(train_annot_dir).parts[-2]
            all_classes += [class_name] * len(annot_fnames)
            all_annot_dirs += [train_annot_dir] * len(annot_fnames)
            all_seg_dirs += [train_seg_dir] * len(annot_fnames)
        
        assert fnames, 'should be at least one fname'

       
        tries = 0
        annots = None
        # retry - because we may have force_fg=True and an annotation without fg
        while not annots:
            if tries >= 100:
                raise Exception(f'Tried to find an annotation {tries} times, '
                                'perhaps none have foreground?')
            tries += 1
            fname = random.sample(fnames, 1)[0]
            
            # triggers retry if assertion fails
            assert is_image(fname), f'{fname} is not a valid image'

            # annots and classes associated with fname
            indices = [i for i, f in enumerate(fnames) if f == fname]

            possible_classes = [all_classes[i] for i in indices]
            annot_dirs = [all_annot_dirs[i] for i in indices]
            seg_dirs = [all_seg_dirs[i] for i in indices]

            classes = []
            annots = []
            segs = []

            # for each of the possible classes.
            for class_name, annot_dir, seg_dir in zip(possible_classes, annot_dirs, seg_dirs): 
                annot_fpath = os.path.join(annot_dir, fname)
                annot = load_image(annot_fpath).astype(int)

                # Why would we have annotations without content?
                assert np.any(annot)
               
                # if fg is forced then only add the annotations where fg is present
                if not force_fg or (force_fg and np.any(annot[1])):

                    annot = np.pad(annot, ((0, 0), (17,17), (17,17), (17, 17)), mode='constant')
                    annots.append(annot)
                    classes.append(class_name)

                    if use_seg:
                        seg_path = os.path.join(seg_dir, fname)

                    if use_seg and os.path.isfile(seg_path):
                        seg_path = os.path.join(seg_dir, fname)
                        seg = load_image(seg_path)
                        seg = np.pad(seg, ((17,17), (17,17), (17, 17)), mode='constant')
                    else:
                        seg = None

                    segs.append(seg)

                else:
                    # print('no foreground for ', fname)
                    pass 

        # it's possible the image has a different extenstion
        # so use glob to get it
        fname_no_ext = fname.replace('.nii.gz', '').replace('.nrrd', '')
        image_path_part = os.path.join(dataset_dir, fname_no_ext)
        image_path = glob.glob(image_path_part + '.*')[0]
        image = load_image(image_path)
        # images are no longer padded on disk
        image = np.pad(image, ((17,17), (17,17), (17, 17)), mode='constant')


        assert image.shape == annots[0][0].shape, (f'Image shape {image.shape} '
                f'should match annots[0][0].shape {annots[0][0].shape}. '
                ' perhaps there is a dimensions mismatch?'
                ' Dataset images and annotations should be (Depth, Height, Width).')
        
        assert len(annots) == len(classes)
        # also return fname for debugging purposes.
        return image, annots, segs, classes, fname

    load_random = partial(load_random, train_annot_dirs, train_seg_dirs, dataset_dir)

    return load_with_retry(load_random, None)

def pad_3d(image, width, depth, mode='reflect', constant_values=0):
    pad_shape = [(depth, depth), (width, width), (width, width)]
    if len(image.shape) == 4:
        # assume channels first for 4 dimensional data.
        # don't pad channels
        pad_shape = [(0, 0)] + pad_shape
    if mode == 'reflect':
        return np.pad(image, pad_shape, mode)
    return np.pad(image, pad_shape, mode=mode,
                  constant_values=constant_values)


def get_val_patch_refs(annot_dirs, prev_patch_refs, out_shape):
    """
    Get patch info which covers all annotated regions of the annotation dataset.
    The list must be structured such that an index can be used to refer to each example
    so that it can be used with a dataloader.

    Parameter prev_patch_refs is used for comparing both file names and mtime.

    The annot_dir folder should be checked for any new files (not in
    prev_patch_refs) or files with an mtime different from prev_patch_refs. For these
    file, the image should be loaded and new patch_refs should be retrieved. For all
    other images the patch_refs from prev_patch_refs can be used.
    """
    patch_refs = []

    # TODO, change this so we have a list of all annot names and their corresopnding classes.
    #  make a large list of classes that corresponds to all the file names. Ive done this elsewhere.

    # This might take ages, profile and optimize


    # create a list of annotation file paths to check
    # this should include all current annotation files on disk
    # and any in the prev_patch_refs (as these need removing)

    annot_fpaths_to_check = []

    for annot_dir in annot_dirs:
        annot_fnames = ls(annot_dir)
        if annot_fnames:
            for a in annot_fnames:
                annot_fpaths_to_check.append(os.path.join(annot_dir, a))
    
    for prev in prev_patch_refs:
        annot_fpaths_to_check.append(prev.annot_fpath())

    # no need to check the same file twice if it is in both
    # the current file system and the prev refs
    annot_fpaths_to_check = set(annot_fpaths_to_check)

    for annot_fpath in annot_fpaths_to_check:
        # get existing coord refs for this image
        prev_refs = [r for r in prev_patch_refs if r.annot_fpath() == annot_fpath]
        prev_mtimes = [r.mtime for r in prev_refs]
        need_new_refs = False
        # if no refs for this image then check again
        if not prev_refs:
            need_new_refs = True
        else:
            # if the file no longer exists then we do need new refs
            # surprisingly this did happen, I presume the file list was somehow out of date
            # and the removal of the file was only detected when trying to read it.
            if not os.path.isfile(annot_fpath):
                need_new_refs = True
            else:
                # otherwise check the modified time of the refs against the file.
                prev_mtime = prev_mtimes[0]
                cur_mtime = os.path.getmtime(annot_fpath)

                # if file has been updated then get new refs
                if cur_mtime > prev_mtime:
                    need_new_refs = True
        if need_new_refs:
            new_file_refs = get_val_patch_refs_for_annot_3d(annot_fpath, out_shape)
            patch_refs += new_file_refs
        else:
            patch_refs += prev_refs
    return patch_refs


def get_val_patch_refs_for_annot_3d(annot_fpath, out_shape):
    if not os.path.isfile(annot_fpath):
        return []
    annot = load_image(annot_fpath)
    new_file_refs = []
    annot_shape = annot.shape[1:]
    coords = get_coords_3d(annot_shape, out_patch_shape=out_shape)
    mtime = os.path.getmtime(annot_fpath)

    # which regions to ignore because they already exist in another patch
    full_ignore_mask = np.zeros(list(annot.shape)[1:]) 
    for (x, y, z) in coords:
        annot_patch = annot[:, z:z+out_shape[0], y:y+out_shape[1], x:x+out_shape[2]]
        ignore_mask = np.array(full_ignore_mask[z:z+out_shape[0],
                                                y:y+out_shape[1],
                                                x:x+out_shape[2]])

        # we only want to validate on annotation patches
        # which have annotation information.
        if np.any(annot_patch):
            # fname, [x, y, z], mtime, prev model metrics i.e [tp, tn, fp, fn] or None
            
            new_ref = PatchRef(annot_dir=os.path.dirname(annot_fpath),
                               annot_fname=os.path.basename(annot_fpath),
                               x=x, y=y, z=z, mtime=mtime,
                               ignore_mask=ignore_mask)
            new_file_refs.append(new_ref)
            # this region should get ignored in future patches from this image.
            full_ignore_mask[z:z+out_shape[0], y:y+out_shape[1], x:x+out_shape[2]] = 1

    return new_file_refs


def get_coords_3d(annot_shape, out_patch_shape):
    """ Get the coordinates relative to the output image for the 
        validation routine. These coordinates will lead to patches
        which cover the image with minimum overlap (assuming fixed size patch) """

    assert len(annot_shape) == 3, str(annot_shape) # d, h, w
    
    depth_count = ceil(annot_shape[0] / out_patch_shape[0])
    vertical_count = ceil(annot_shape[1] / out_patch_shape[1])
    horizontal_count = ceil(annot_shape[2] / out_patch_shape[2])

    # first split the image based on the patches that fit
    z_coords = [d*out_patch_shape[0] for d in range(depth_count-1)] # z is depth
    y_coords = [v*out_patch_shape[1] for v in range(vertical_count-1)]
    x_coords = [h*out_patch_shape[2] for h in range(horizontal_count-1)]

    # The last row and column of patches might not fit
    # (Might go outside the image)
    # so get the patch positiion by subtracting patch size from the
    # edge of the image.
    lower_z = annot_shape[0] - out_patch_shape[0]
    bottom_y = annot_shape[1] - out_patch_shape[1]
    right_x = annot_shape[2] - out_patch_shape[2]

    z_coords.append(max(0, lower_z))
    y_coords.append(max(0, bottom_y))
    x_coords.append(max(0, right_x))

    # because its a cuboid get all combinations of x, y and z
    patch_coords = [(x, y, z) for x in x_coords for y in y_coords for z in z_coords]
    return patch_coords



def save_then_move(out_path, seg):
    """ need to save in a temp folder first and
        then move to the segmentation folder after saving
        this is because scripts are monitoring the segmentation folder
        and the file saving takes time..
        We don't want the scripts that monitor the segmentation
        folder to try loading the file half way through saving
        as this causes errors. Thus we save and then rename.
    """
    raise Exception('Depracated')
    fname = os.path.basename(out_path)
    token = str(time.time()) # add token to avoid resaving over files wiith the same name
    temp_path = os.path.join('/tmp', token + fname)
    if out_path.endswith('.nii.gz'):
        img = nib.Nifti1Image(seg, np.eye(4))
        img.to_filename(temp_path)
    elif out_path.endswith('.npy'):
        np.save(temp_path, seg)
    else:
        raise Exception(f'Unhandled {out_path}')
    shutil.copy(temp_path, out_path)
    os.remove(temp_path)


def save(out_path, seg):
    if out_path.endswith('.nii.gz'):
        img = nib.Nifti1Image(seg, np.eye(4))
        img.to_filename(out_path)
    else:
        raise Exception(f'Unhandled {out_path}')


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
