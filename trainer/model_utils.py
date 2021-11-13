""" Utilities for working with the U-Net models 
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

# pylint: disable=C0111, R0913, R0914, W0511
import os
import time
import glob
import math
import numpy as np
import torch
import copy
from skimage.io import imread
from skimage import img_as_float32
import im_utils
from unet3d import UNet3D
from metrics import get_metrics
from file_utils import ls
from torch.nn.functional import softmax

cached_model = None
cached_model_path = None
use_fake_cnn = False

def fake_cnn(tiles_for_gpu):
    """ Useful debug function for checking tile layout etc """
    output = []
    for t in tiles_for_gpu:
        v = t[0, 17:-17, 17:-17, 17:-17].data.cpu().numpy()
        v_mean = np.mean(v)
        output.append((v > v_mean).astype(np.int8))
    return np.array(output)
 
def get_in_w_out_w_pairs():
    # matching pairs of input/output sizes for specific unet used
    # 52 to 228 in incrememnts of 16 (sorted large to small)
    in_w_list = sorted([52 + (x*16) for x in range(12)], reverse=True)
    
    # output always 34 less than input
    out_w_list = [x - 34 for x in in_w_list]
    return list(zip(in_w_list, out_w_list))

def get_in_w_out_w_for_memory(num_classes):
    # search for appropriate input size for GPU
    # in_w, out_w = get_in_w_out_w_for_memory(num_classes)
    net = UNet3D(im_channels=1, out_channels=num_classes*2).cuda()
    net = torch.nn.DataParallel(net)
    for in_w, out_w in get_in_w_out_w_pairs():
        torch.cuda.empty_cache()
        try:
            #                      b, c,  d,  h,    w    
            input_data = np.zeros((4, 1, 52, in_w, in_w))
            output = net(torch.from_numpy(input_data).cuda().float())
            del input_data
            del output
            torch.cuda.empty_cache()
            return in_w, out_w
        except Exception as e:
            if 'out of memory' in str(e):
                print(in_w, out_w, 'too big')
    raise Exception('Could not find patch small enough for available GPU memory')


def get_latest_model_paths(model_dir, k):
    fnames = ls(model_dir)
    fnames = sorted(fnames)[-k:]
    fpaths = [os.path.join(model_dir, f) for f in fnames]
    return fpaths


def load_model(model_path, num_classes):
    global cached_model
    global cached_model_path
    
    # using cache can save up to half a second per segmentation with network drives
    if model_path == cached_model_path:
        return copy.deepcopy(cached_model)

    model = UNet3D(im_channels=1, out_channels=num_classes*2)
    try:
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model)
    # pylint: disable=broad-except, bare-except
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    if not use_fake_cnn:
        model.cuda()
    # store in cache as most frequest model is laoded often
    cached_model_path = model_path
    cached_model = model
    return copy.deepcopy(model)


def random_model(num_classes):
    # num out channels is twice number of channels
    # as we have a positive and negative output for each structure.
    model = UNet3D(im_channels=1, out_channels=num_classes*2)
    model = torch.nn.DataParallel(model)
    if not use_fake_cnn: 
        model.cuda()
    return model

def create_first_model_with_random_weights(model_dir, num_classes):
    # used when no model was specified on project creation.
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_' + str(int(round(time.time()))) + '.pkl'
    model = random_model(num_classes)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    if not use_fake_cnn: 
        model.cuda()
    return model


def get_prev_model(model_dir, num_classes):
    prev_path = get_latest_model_paths(model_dir, k=1)[0]
    prev_model = load_model(prev_path, num_classes)
    return prev_model, prev_path

def get_class_metrics(get_val_annots, get_seg, classes) -> list:
    """
    Segment the validation images and
    return metrics for each of the classes.
    """
    class_metrics = [{'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'class': c} for c in classes]
    classes_rgb = [c[1][:3] for c in classes]

    # for each image
    for fname, annot in get_val_annots():
        assert annot.dtype == np.ubyte, str(annot.dtype)

        # remove parts where annotation is not defined e.g alhpa=0
        a_channel = annot[:, :, 3]
        y_defined = (a_channel > 0).astype(np.int).reshape(-1)

        # load sed, returns a channel for each class
        seg = get_seg(fname)

        # for each class
        for i, class_rgb in enumerate(classes_rgb):
            y_true = im_utils.get_class_map(annot, class_rgb)
            y_pred = seg == i
            assert y_true.shape == y_pred.shape, str(y_true.shape) + str(y_pred.shape)

            # only compute metrics on regions where annotation is defined.
            y_true = y_true.reshape(-1)[y_defined > 0]
            y_pred = y_pred.reshape(-1)[y_defined > 0]
            class_metrics[i]['tp'] += np.sum(np.logical_and(y_pred == 1,
                                                            y_true == 1))
            class_metrics[i]['tn'] += np.sum(np.logical_and(y_pred == 0,
                                                            y_true == 0))
            class_metrics[i]['fp'] += np.sum(np.logical_and(y_pred == 1,
                                                            y_true == 0))
            class_metrics[i]['fn'] += np.sum(np.logical_and(y_pred == 0,
                                                            y_true == 1))
    for i, metric in enumerate(class_metrics):
        class_metrics[i] = get_metrics(metric['tp'], metric['fp'],
                                       metric['tn'], metric['fn'])
    return class_metrics


def get_val_metrics(cnn, val_annot_dir, dataset_dir, in_w, out_w, batch_size, classes):
    # This is no longer used.
    start = time.time()
    fnames = ls(val_annot_dir)
    fnames = [a for a in fnames if im_utils.is_photo(a)]

    def get_seg(fname):
        image_path_part = os.path.join(dataset_dir, os.path.splitext(fname)[0])
        image_path = glob.glob(image_path_part + '.*')[0]
        image = im_utils.load_image(image_path)
        predicted = segment(cnn, image, batch_size, in_w, out_w)

        # Need to convert to predicted class.
        predicted = np.argmax(predicted, 0)
        return predicted

    def get_val_annots():
        for fname in fnames:
            annot_path = os.path.join(val_annot_dir,
                                      os.path.splitext(fname)[0] + '.png')
            annot = imread(annot_path)
            annot = np.array(annot)
            yield [fname, annot]

    print('Validation duration', time.time() - start)
    return get_class_metrics(get_val_annots, get_seg, classes)

def save_if_better(model_dir, cur_model, prev_model_path, cur_dice, prev_dice):
    # convert the nans as they don't work in comparison
    if math.isnan(cur_dice):
        cur_dice = 0
    if math.isnan(prev_dice):
        prev_dice = 0
    print('Validation: prev dice', str(round(prev_dice, 5)).ljust(7, '0'),
          'cur dice', str(round(cur_dice, 5)).ljust(7, '0'))
    if cur_dice > prev_dice:
        save_model(model_dir, cur_model, prev_model_path)
        return True
    return False

def save_model(model_dir, cur_model, prev_model_path):
    prev_model_fname = os.path.basename(prev_model_path)
    prev_model_num = int(prev_model_fname.split('_')[0])
    model_num = prev_model_num + 1
    now = int(round(time.time()))
    model_name = str(model_num).zfill(6) + '_' + str(now) + '.pkl'
    model_path = os.path.join(model_dir, model_name)
    print('saving', model_path, time.strftime('%H:%M:%S', time.localtime(now)))
    torch.save(cur_model.state_dict(), model_path)


def ensemble_segment_3d(model_paths, image, fname, batch_size, in_w, out_w, in_d,
                        out_d, num_classes, bounded, threshold=0.5, aug=True):
    """ Average predictions from each model specified in model_paths """
    t = time.time()
    input_image_shape = image.shape
    cnn = load_model(model_paths[0], num_classes)
    in_patch_shape = (in_d, in_w, in_w)
    out_patch_shape = (out_d, out_w, out_w)

    depth_diff = in_patch_shape[0] - out_patch_shape[0]
    height_diff = in_patch_shape[1] - out_patch_shape[1]
    width_diff = in_patch_shape[2] - out_patch_shape[2]

    if not bounded:
        # pad so seg will be size of input image
        image = im_utils.pad_3d(image, width_diff//2, depth_diff//2,
                                mode='reflect', constant_values=0)
    seg = segment_3d(cnn, image, batch_size, in_patch_shape, out_patch_shape)

    if not bounded:
        assert seg.shape == input_image_shape
    """
    end of fname is constructed like this
    the indices e.g -14 are inserted here for convenience
    f"_x_{box['x'] (-14) }_y_{box['y'] (-13) }_z_{box['z'] (-11) }_pad_"
    f"x_{x_pad_start (-8) }_{x_pad_end (-7) }"
    f"y_{y_pad_start (-5) }_{y_pad_end (-4 )}"
    f"z_{z_pad_start (-2) }_{z_pad_end (-1) }.nii.gz")
    """
    if bounded: 
        fname_parts = fname.replace('.nii.gz', '').split('_')
        x_crop_start = int(fname_parts[-8])
        x_crop_end = int(fname_parts[-7])
        y_crop_start = int(fname_parts[-5])
        y_crop_end = int(fname_parts[-4])
        z_crop_start = int(fname_parts[-2])
        z_crop_end = int(fname_parts[-1])

        # The output of the cnn is already cropped during inference.
        # subtract this default cropping from each of the crop sizes
        z_crop_start -= depth_diff // 2
        z_crop_end -= depth_diff // 2

        y_crop_start -= height_diff // 2
        y_crop_end -= height_diff // 2

        x_crop_start -= width_diff // 2
        x_crop_end -= width_diff // 2

        seg = seg[z_crop_start:seg.shape[0] - z_crop_end,
                  y_crop_start:seg.shape[1] - y_crop_end,
                  x_crop_start:seg.shape[2] - x_crop_end]

    print('time to segment image with ensemble segment', time.time() - t)
    return seg


def segment_3d(cnn, image, batch_size, in_tile_shape, out_tile_shape):
    """
    in_tile_shape and out_tile_shape are (depth, height, width)
    """
    # Return prediction for each pixel in the image
    # The cnn will give a the output as channels where
    # each channel corresponds to a specific class 'probability'
    # don't need channel dimension
    # make sure the width, height and depth is at least as big as the tile.
    assert len(image.shape) == 3, str(image.shape)
    assert image.shape[0] >= in_tile_shape[0], f"{image.shape[0]},{in_tile_shape[0]}"
    assert image.shape[1] >= in_tile_shape[1], f"{image.shape[1]},{in_tile_shape[1]}"
    assert image.shape[2] >= in_tile_shape[2], f"{image.shape[2]},{in_tile_shape[2]}"

    depth_diff = in_tile_shape[0] - out_tile_shape[0]
    width_diff = in_tile_shape[1] - out_tile_shape[1]
    
    out_im_shape = (image.shape[0] - depth_diff,
                    image.shape[1] - width_diff,
                    image.shape[2] - width_diff)

    coords = im_utils.get_coords_3d(out_im_shape, out_tile_shape)
    coord_idx = 0
    # segmentation for the full image
    # assign once we get number of classes from the cnn output shape.
    seg = np.zeros(out_im_shape, dtype=np.int8)
    while coord_idx < len(coords):
        tiles_to_process = []
        coords_to_process = []
        for _ in range(batch_size):
            if coord_idx < len(coords):
                coord = coords[coord_idx]
                x_coord, y_coord, z_coord = coord
                tile = image[z_coord:z_coord+in_tile_shape[0],
                             y_coord:y_coord+in_tile_shape[1],
                             x_coord:x_coord+in_tile_shape[2]]

                # need to add channel dimension for GPU processing.
                tile = np.expand_dims(tile, axis=0)

                assert tile.shape[1] == in_tile_shape[0], str(tile.shape)
                assert tile.shape[2] == in_tile_shape[1], str(tile.shape)
                assert tile.shape[3] == in_tile_shape[2], str(tile.shape)

                tile = img_as_float32(tile)
                tile = im_utils.normalize_tile(tile)
                coord_idx += 1
                tiles_to_process.append(tile) # need channel dimension
                coords_to_process.append(coord)

        tiles_to_process = np.array(tiles_to_process)
        tiles_for_gpu = torch.from_numpy(tiles_to_process)
        if use_fake_cnn:
            pred_np = fake_cnn(tiles_for_gpu)
        else:
            tiles_for_gpu = tiles_for_gpu.cuda()
            tile_predictions = cnn(tiles_for_gpu)
            tile_predictions = softmax(tile_predictions, 1)[:, 1, :] # just foreground
            tile_predictions = (tile_predictions > 0.5).type(torch.int8)
            pred_np = tile_predictions.data.cpu().numpy()
        out_tiles = pred_np.reshape(([len(tiles_for_gpu)] + list(out_tile_shape)))
        # add the predictions from the gpu to the output segmentation
        # use their correspond coordinates
        for tile, (x_coord, y_coord, z_coord) in zip(out_tiles, coords_to_process):
            seg[z_coord:z_coord+tile.shape[0],
                y_coord:y_coord+tile.shape[1],
                x_coord:x_coord+tile.shape[2]] = tile
    return seg
