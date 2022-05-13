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
import math
import traceback
import numpy as np
import torch
import copy
from skimage import img_as_float32
import im_utils
from unet3d import UNet3D
from file_utils import ls
from torch.nn.functional import softmax
import torch.nn.functional as F
from loss import get_batch_loss

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
    # 36 to 228 in incrememnts of 16 (sorted large to small)
    in_w_list = sorted([36 + (x*16) for x in range(15)], reverse=True)
    
    # output always 34 less than input
    out_w_list = [x - 34 for x in in_w_list]
    return list(zip(in_w_list, out_w_list))


def allocate_net(in_w, out_w, num_classes):
    net = UNet3D(im_channels=3, num_classes=num_classes).cuda()
    net = torch.nn.DataParallel(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01,
                                momentum=0.99, nesterov=True)
    net.train()
    for _ in range(3):
    
        #                      b, c,  d,  h,    w    
        input_data = np.zeros((4, 3, 52, in_w, in_w))
        optimizer.zero_grad()
        outputs = net(torch.from_numpy(input_data).cuda().float())
        batch_fg_tiles = torch.ones(4, num_classes, 52, in_w, in_w).long().cuda()
        batch_bg_tiles = torch.zeros(4, num_classes, 52, in_w, in_w).long().cuda()
        batch_fg_tiles[:, 0, 0] = 0
        batch_bg_tiles[:, 0, 0] = 1
        batch_classes = []
        for _ in range(4):
            batch_classes.append([f'c_{c}' for c in range(num_classes)])
        (batch_loss, _, _,
         _, _) = get_batch_loss(
             outputs, batch_fg_tiles, batch_bg_tiles,
             [[None for c in range(num_classes)] for t in batch_fg_tiles], # segmentation excluded from loss for now.
             batch_classes,
             [f'c_{c}' for c in range(num_classes)],
             compute_loss=True)
        batch_loss.backward()
        optimizer.step()
        del batch_loss
        del batch_fg_tiles
        del batch_bg_tiles
        del outputs
        del input_data


def get_in_w_out_w_for_memory(num_classes):
    print('computing largest patch size for GPU')
    # search for appropriate input size for GPU
    # in_w, out_w = get_in_w_out_w_for_memory(num_classes)
    # try to train a network and see which patch size fits on the gpu.
    pairs = get_in_w_out_w_pairs()
    for i, (in_w, out_w) in enumerate(pairs):
        torch.cuda.empty_cache()
        try:
            allocate_net(in_w, out_w, num_classes)
            torch.cuda.empty_cache()
            print(in_w, out_w, 'ok')
            print('using', pairs[i+1], 'to be safe') # return the next smallest to be safe
            return pairs[i+1] # return the next smallest to be safe
        except Exception as e:
            if 'out of memory' in str(e):
                print(in_w, out_w, 'too big')
            else:
                print(e, traceback.format_exc())
    raise Exception('Could not find patch small enough for available GPU memory')


def get_latest_model_paths(model_dir, k):
    fnames = ls(model_dir)
    fnames = sorted(fnames)[-k:]
    fpaths = [os.path.join(model_dir, f) for f in fnames]
    return fpaths


def load_model(model_path, classes):
    global cached_model
    global cached_model_path
    
    # using cache can save up to half a second per segmentation with network drives
    if model_path == cached_model_path:
        return copy.deepcopy(cached_model)

    # two channels as one is input image and another is some of the fg and bg annotation
    # each non-empty channel in the annotation is included with 50% chance.
    # - fg and bg will go in as seprate channels 
    #  so channels are [image, fg_annot, bg_annot]
    model = UNet3D(num_classes=len(classes), im_channels=3) # 3 channels as annotation may be used as input.
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


def random_model(classes):
    # num out channels is twice number of channels
    # as we have a positive and negative output for each structure.
    # disabled for now as auto-complete feature is stalled.
    #model = UNet3D(classes, im_channels=3)
    model = UNet3D(num_classes=len(classes), im_channels=3) # 3 channels to enable optional annotation as input.
    model = torch.nn.DataParallel(model)
    if not use_fake_cnn: 
        model.cuda()
    return model

def create_first_model_with_random_weights(model_dir, classes):
    # used when no model was specified on project creation.
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_' + str(int(round(time.time()))) + '.pkl'
    model = random_model(classes)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    if not use_fake_cnn: 
        model.cuda()
    return model


def get_prev_model(model_dir, classes):
    prev_path = get_latest_model_paths(model_dir, k=1)[0]
    prev_model = load_model(prev_path, classes)
    return prev_model, prev_path


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
                        out_d, classes):
    """ Average predictions from each model specified in model_paths """
    t = time.time()
    input_image_shape = image.shape
    cnn = load_model(model_paths[0], classes)
    in_patch_shape = (in_d, in_w, in_w)
    out_patch_shape = (out_d, out_w, out_w)

    depth_diff = in_patch_shape[0] - out_patch_shape[0]
    height_diff = in_patch_shape[1] - out_patch_shape[1]
    width_diff = in_patch_shape[2] - out_patch_shape[2]


    print('input image shape (before pad)= ', image.shape)
    # pad so seg will be size of input image
    image = im_utils.pad_3d(image, width_diff//2, depth_diff//2,
                            mode='reflect', constant_values=0)

    # segment returns a series of prediction maps. one for each class.
    print('input image shape (after pad)= ', image.shape)
    pred_maps = segment_3d(cnn, image, batch_size, in_patch_shape, out_patch_shape)
    print('pred maps[0].shape = ', pred_maps[0].shape)
    assert pred_maps[0].shape == input_image_shape
    print('time to segment image', time.time() - t)
    return pred_maps


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

    original_shape = image.shape

    # if the image is smaller than the patch size then pad it to be the same as the patch.
    padded_for_patch = False
    patch_pad_z = 0
    patch_pad_y = 0
    patch_pad_x = 0

    if image.shape[0] < in_tile_shape[0]:
        padded_for_patch = True
        patch_pad_z = in_tile_shape[0] - image.shape[0]

    if image.shape[1] < in_tile_shape[1]:
        padded_for_patch = True
        patch_pad_y = in_tile_shape[1] - image.shape[1]

    if image.shape[2] < in_tile_shape[2]:
        padded_for_patch = True
        patch_pad_x = in_tile_shape[2] - image.shape[2]

    if padded_for_patch:
        padded_image = np.zeros((
            image.shape[0] + patch_pad_z,
            image.shape[1] + patch_pad_y,
            image.shape[2] + patch_pad_x),
            dtype=image.dtype)
        padded_image[:image.shape[0], :image.shape[1], :image.shape[2]] = image
        image = padded_image  

    depth_diff = in_tile_shape[0] - out_tile_shape[0]
    width_diff = in_tile_shape[1] - out_tile_shape[1]
    
    out_im_shape = (image.shape[0] - depth_diff,
                    image.shape[1] - width_diff,
                    image.shape[2] - width_diff)

    coords = im_utils.get_coords_3d(out_im_shape, out_tile_shape)
    coord_idx = 0
    class_output_tiles = None # list of tiles for each class

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
                tile = img_as_float32(tile)
                coord_idx += 1
                tiles_to_process.append(tile) # need channel dimension
                coords_to_process.append(coord)

        tiles_to_process = np.array(tiles_to_process)
        tiles_for_gpu = torch.from_numpy(tiles_to_process)

        tiles_for_gpu = tiles_for_gpu.cuda().float()
        # TODO: consider use of detach. 
        # I might want to move to cpu later to speed up the next few operations.
        # I added .detach().cpu() to prevent a memory error.
        # pad with zeros for the annotation input channels
        # l,r, l,r, but from end to start     w  w  h  h  d  d, c, c, b, b
        tiles_for_gpu = F.pad(tiles_for_gpu, (0, 0, 0, 0, 0, 0, 0, 2), 'constant', 0)
        # tiles shape after padding torch.Size([4, 3, 52, 228, 228])
        outputs = cnn(tiles_for_gpu).detach().cpu()
        # bg channel index for each class in network output.
        class_idxs = [x * 2 for x in range(outputs.shape[1] // 2)]
        
        if class_output_tiles is None:
            class_output_tiles = [[] for _ in class_idxs]

        for i, class_idx in enumerate(class_idxs):
            class_output = outputs[:, class_idx:class_idx+2]
            # class_output : (batch_size, bg/fg, depth, height, width)
            softmaxed = softmax(class_output, 1) 
            foreground_probs = softmaxed[:, 1]  # just the foreground probability.
            predicted = foreground_probs > 0.5
            predicted = predicted.int()
            pred_np = predicted.data.cpu().numpy()
            for out_tile in pred_np:
                class_output_tiles[i].append(out_tile)

    class_pred_maps = []
    for i, output_tiles in enumerate(class_output_tiles):
        # reconstruct for each class
        reconstructed = im_utils.reconstruct_from_tiles(output_tiles,
                                                        coords, out_im_shape)
        if padded_for_patch:
            # go back to the original shape before padding.
            reconstructed = reconstructed[:original_shape[0], :original_shape[1], :original_shape[2]]
        class_pred_maps.append(reconstructed)

    return class_pred_maps
