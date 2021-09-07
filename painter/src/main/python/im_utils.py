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

# pylint: disable=C0111, W0511
# pylint: disable=E0401 # import error

import os
import warnings
import glob
import sys

import numpy as np
from skimage import color
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.transform import resize
from skimage.measure import label
from skimage.morphology import binary_dilation, remove_small_holes
from PIL import Image
from PyQt5 import QtGui
import qimage2ndarray
import nibabel as nib
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import flood
import nrrd


def is_image(fname):
    extensions = {".jpg", ".png", ".jpeg", '.tif', '.tiff', '.npy', 'gz', 'nrrd'}
    return any(fname.lower().endswith(ext) for ext in extensions)


def load_image(image_path):
    if image_path.endswith('.npy'):
        return np.load(image_path, mmap_mode='c')
    if image_path.endswith('.nii.gz'):
        image = nib.load(image_path)
        image = np.array(image.dataobj)
        image = np.rot90(image, k=3)
        image = np.moveaxis(image, -1, 0) # depth moved to beginning
        # reverse lr and ud
        image = image[::-1, :, ::-1]
    if image_path.endswith('.nrrd'):
        image, header = nrrd.read(image_path)
        image = np.rot90(image, k=3)
        image = np.moveaxis(image, -1, 0) # depth moved to beginning
        # reverse lr and ud
        image = image[::-1, :, ::-1]
    return image


def load_annot(annot_path, img_data_shape):
    """ pad the annotation with zeros and return """
    # The  path will have an ending like this
    #   -14  -13    -11      -8 -7   -5 -4 -3 -2 -1 
    # x_231_y_222_z_111_pad_x_29_30_y_59_60_z_17_17
    annot_image = nib.load(annot_path)
    annot_data = np.array(annot_image.dataobj)
    name = os.path.basename(annot_path)
    name = name.replace('.nii.gz', '')
    name = name.replace('.nrrd', '')
    parts = name.split('_')  
    x = int(parts[-15])
    y = int(parts[-13])
    z = int(parts[-11])
    annot = np.zeros([2] + list(img_data_shape), dtype=np.int8)
    annot[:, z:z+annot_data.shape[1],
             y:y+annot_data.shape[2],
             x:x+annot_data.shape[3]] = annot_data
    return annot


def load_seg(seg_path, img_data):
    """ pad the segmentation with zeros and return """
    # The seg path will have an ending like this
    #   -14  -13    -11      -8 -7   -5 -4 -3 -2 -1 
    # x_231_y_222_z_111_pad_x_29_30_y_59_60_z_17_17
    seg_image = nib.load(seg_path)
    seg_data = np.array(seg_image.dataobj)
    name = os.path.basename(seg_path)
    name = name.replace('.nii.gz', '')
    name = name.replace('.nrrd', '')

    parts = name.split('_')  
    x = int(parts[-15])
    y = int(parts[-13])
    z = int(parts[-11])
    
    # This issue may be related to file system issues.
    assert  len(seg_data.shape) == 3, f"seg shape is {seg_data.shape} for {seg_path}"
    
    seg_depth = seg_data.shape[0]
    seg_height = seg_data.shape[1]
    seg_width = seg_data.shape[2]
    
    # We use -1 to indicate that a region is outside of the bounding box
    seg = np.ones(img_data.shape, dtype=np.int8) * -1

    seg[z:z+seg_data.shape[0],
        y:y+seg_data.shape[1],
        x:x+seg_data.shape[2]] = seg_data

    return seg, (z, y, x, seg_depth, seg_height, seg_width)


def norm_slice(img, min_v, max_v, brightness_percent):
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    bright_v = (brightness_percent / 100)
    img[img < min_v] = min_v
    img[img > max_v] = max_v
    img -= min_v
    img /= (max_v - min_v)
    img *= bright_v
    img[img > 1] = 1.0
    img *= 255
    return img


def annot_slice_to_pixmap(slice_np):
    """ convert slice from the numpy annotation data
        to a PyQt5 pixmap object """
    # for now fg and bg colors are hard coded.
    # later we plan to let the user specify these in the user interface.
    np_rgb = np.zeros((slice_np.shape[1], slice_np.shape[2], 4))
    np_rgb[:, :, 1] = slice_np[0] * 255 # green is bg
    np_rgb[:, :, 0] = slice_np[1] * 255 # red is fg
    np_rgb[:, :, 3] = np.sum(slice_np, axis=0) * 180 # alpha is defined
    q_image = qimage2ndarray.array2qimage(np_rgb)
    return QtGui.QPixmap.fromImage(q_image)


def get_outline_pixmap(seg_slice, annot_slice):
    seg_map = (seg_slice > 0).astype(np.int)
    annot_plus = (annot_slice[1] > 0).astype(np.int)
    annot_minus = (annot_slice[0] > 0).astype(np.int)

    # remove anything where seg is less than 0 as this is outside of the box
    seg_minus = (seg_slice < 0).astype(np.int)
    mask = ((((seg_map + annot_plus) - annot_minus) - seg_minus) > 0)
    dilated = binary_dilation(mask)
    outline = dilated.astype(np.int) - mask.astype(np.int)
    np_rgb = np.zeros((outline.shape[0], outline.shape[1], 4))
    np_rgb[outline > 0] = [255, 255, 0, 180]
    q_image = qimage2ndarray.array2qimage(np_rgb)
    return QtGui.QPixmap.fromImage(q_image)


def seg_slice_to_pixmap(slice_np):
    """ convert slice from the numpy segmentation data
        to a PyQt5 pixmap object """
    np_rgb = np.zeros((slice_np.shape[0], slice_np.shape[1], 4))
    np_rgb[slice_np > 0] = [0, 255, 255, 180]
    # we use -1 to indicate that the voxel is outside the bounding box
    np_rgb[slice_np == -1] = [255, 0, 0, 60]
    q_image = qimage2ndarray.array2qimage(np_rgb)
    return QtGui.QPixmap.fromImage(q_image)

def get_slice(volume, slice_idx, mode):
    if mode == 'axial':
        if len(volume.shape) > 3:
            slice_idx = (volume.shape[1] - slice_idx) - 1
            # if more than 3 presume first is channel dimension
            slice_data = volume[:, slice_idx, :, :]
        else:
            slice_idx = (volume.shape[0] - slice_idx) - 1
            slice_data = volume[slice_idx, :, :]
    elif mode == 'coronal':
        raise Exception("not yet implemented")
        if len(volume.shape) > 3:
            # if more than 3 presume first is channel dimension
            slice_data = volume[:, :, :, slice_idx]
        else:
            slice_data = volume[:, slice_idx, :]
    elif mode == 'sagittal':
        if len(volume.shape) > 3:
            # if more than 3 presume first is channel dimension
            slice_data = volume[:, :, :, slice_idx]
        else:
            slice_data = volume[:, :, slice_idx]
    else:
        raise Exception(f"Unhandled slice mode: {mode}")
    return slice_data


def store_annot_slice(annot_pixmap, annot_data, slice_idx, mode):
    """
    Update .annot_data at slice_idx
    so the values for fg and bg correspond to annot_pixmap)
    """
    slice_rgb_np = np.array(qimage2ndarray.rgb_view(annot_pixmap.toImage()))
    fg = slice_rgb_np[:, :, 0] > 0
    bg = slice_rgb_np[:, :, 1] > 0
    if mode == 'axial': 
        slice_idx = (annot_data.shape[1] - slice_idx) - 1
        annot_data[0, slice_idx] = bg
        annot_data[1, slice_idx] = fg
    elif mode == 'coronal':
        raise Exception("not yet implemented")
        annot_data[0, :, slice_idx, :] = bg
        annot_data[1, :, slice_idx, :] = fg
    elif mode == 'sagittal':
        #slice_idx = (annot_data.shape[3] - slice_idx) - 1
        annot_data[0, :, :, slice_idx] = bg
        annot_data[1, :, :, slice_idx] = fg
    else:
        raise Exception(f"Unhandled slice mode: {mode}")
    return annot_data


def get_num_regions(seg_data, annot_data):
    seg_map = (seg_data > 0).astype(np.int)
    annot_plus = (annot_data[1] > 0).astype(np.int)
    annot_minus = (annot_data[0] > 0).astype(np.int)
    # remove anything where seg is less than 0 as this is outside of the box
    corrected = (((seg_map + annot_plus) - annot_minus) > 0)
    labelled = label(corrected, connectivity=2)
    return len(np.unique(labelled)) - 1 # don't consider background a region.

def restrict_to_region_containing_point(seg_data, annot_data, x, y, z):
    # restrict corrected structure to only the selected
    # connected region found at x,y,z
    # also remove small holes.
    seg_map = (seg_data > 0).astype(np.int)
    annot_plus = (annot_data[1] > 0).astype(np.int)
    annot_minus = (annot_data[0] > 0).astype(np.int)

    # remove anything where seg is less than 0 as this is outside of the box
    corrected = (((seg_map + annot_plus) - annot_minus) > 0)
    labelled = label(corrected, connectivity=2)
    selected_label = labelled[z, y, x]
    if selected_label == 0:
        error = "Selected region was background. Select a foreground region to keep."
        return annot_data, 0, 0, error
    selected_component = labelled == selected_label
    labelled[selected_component] = -1
    disconnected_regions = labelled > 0
    removed_count =  np.unique(labelled[disconnected_regions])
    # Then update the annotation so that this region is now background.
    annot_data[0][disconnected_regions] = 1
    annot_data[1][disconnected_regions] = 0
    removed_count = len(np.unique(labelled[disconnected_regions]))
    # removing small holes
    # this was taking too long so we restrict to the object of interest.
    coords = np.where(selected_component > 0)
    min_z = np.min(coords[0])
    max_z = np.max(coords[0])
    min_y = np.min(coords[1])
    max_y = np.max(coords[1])
    min_x = np.min(coords[2])
    max_x = np.max(coords[2])
    # remove_small_holes
    roi_annot_plus = (annot_data[1, min_z: max_z, min_y:max_y, min_x:max_x] > 0).astype(np.int)
    roi_annot_minus = (annot_data[0,  min_z: max_z, min_y:max_y, min_x:max_x] > 0).astype(np.int)
    # remove anything where seg is less than 0 as this is outside of the box
    roi_corrected = (((seg_map[min_z: max_z, min_y:max_y, min_x:max_x] + roi_annot_plus) - roi_annot_minus) > 0)
    roi_corrected_no_holes = binary_fill_holes(roi_corrected).astype(np.int)
    roi_extra_fg = roi_corrected_no_holes - roi_corrected
    holes_removed = len(np.unique(label(roi_extra_fg))) - 1
    # Set the extra foreground from remove small holes to foreground in the annotation.
    annot_data[0, min_z: max_z, min_y:max_y, min_x:max_x][roi_extra_fg > 0] = 0
    annot_data[1, min_z: max_z, min_y:max_y, min_x:max_x][roi_extra_fg > 0] = 1
    return annot_data, removed_count, holes_removed, False



def fill_annot(annot_pixmap):
    # convert the current annot slice to numpy
    image = annot_pixmap.toImage()
    rgb_np = np.array(qimage2ndarray.rgb_view(image))
    fg_mask = rgb_np[:, :, 0]
    fg_mask = binary_fill_holes(fg_mask)
    fg_mask = fg_mask.astype(np.int)
    np_rgba = np.zeros((rgb_np.shape[0], rgb_np.shape[1], 4))
    # set fg annotation to be new fg mask
    # set bg annotation to be opposite of the fg mask
    np_rgba[:, :, :][fg_mask < 1] = [0, 255, 0, 180]
    np_rgba[:, :, :][fg_mask > 0] = [255, 0, 0, 180]

    q_image = qimage2ndarray.array2qimage(np_rgba)
    return QtGui.QPixmap.fromImage(q_image)