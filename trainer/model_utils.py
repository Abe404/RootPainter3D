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
import copy
from inspect import currentframe, getframeinfo

import torch
from torch.nn.functional import softmax
import torch.nn.functional as F
import numpy as np

from skimage import img_as_float32
import im_utils
from unet3d import UNet3D
from file_utils import ls
from loss import get_batch_loss

cached_model = None
cached_model_path = None
use_fake_cnn = False

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device('cuda')

device = get_device()


# to enable this, put MEM_DEBUG=True in the command before invoking python.
mem_debug_enabled = ('MEM_DEBUG' in os.environ)

def debug_memory(message=''):
    if mem_debug_enabled:
        frameinfo = getframeinfo(currentframe())
        print(frameinfo.filename, frameinfo.lineno, message)

        print("torch.cuda.memory_allocated: "
                f"{torch.cuda.memory_allocated(0)/1024/1024/1024:%f}GB")
        print("torch.cuda.memory_reserved: "
                f"{torch.cuda.memory_reserved(0)/1024/1024/1024:%f}GB")
        print("torch.cuda.max_memory_reserved: "
                f"{torch.cuda.max_memory_reserved(0)/1024/1024/1024:%f}GB")


def get_in_w_and_out_w_for_image(im, in_w, out_w):
    """ the input image may be smaller than the default 
        patch size, so find a patch size where the output patch
        size will fit inside the image """

    # FIXME: See above comment - Is using variable patch sizes 
    #        reliable, considering how the network was trained?
    _im_depth, im_height, im_width = im.shape

    if out_w < im_width and out_w < im_height:
        return in_w, out_w
    
    for valid_in_w, valid_out_w in get_in_w_out_w_pairs():
        if valid_out_w < im_width and valid_out_w < im_height and valid_out_w < out_w:
            return valid_in_w, valid_out_w

    raise Exception('cannot find patch size small enough for image with shape' + str(im.shape))




def get_in_w_out_w_pairs():
    # matching pairs of input/output sizes for specific unet used
    # 36 to 228 in incrememnts of 16 (sorted large to small)
    #in_w_list = sorted([36 + (x*16) for x in range(20)], reverse=True)
    in_w_list = sorted([36 + (x*16) for x in range(10)], reverse=True)
    # output always 34 less than input
    out_w_list = [x - 34 for x in in_w_list]
    return list(zip(in_w_list, out_w_list))

def allocate_net(in_w, num_classes):

    channels = 1 # change to 3 for auto-complete
    net = UNet3D(im_channels=channels, num_classes=num_classes).cuda()
    net = torch.nn.DataParallel(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01,
                                momentum=0.99, nesterov=True)
    net.train()
    for _ in range(3):
    
        #                      b, c,  d,  h,    w    
        input_data = np.zeros((4, channels, 52, in_w, in_w))
        optimizer.zero_grad()
        outputs = net(torch.from_numpy(input_data).cuda().float())
        batch_fg_patches = torch.ones(4, num_classes, 52, in_w, in_w).long().cuda()
        batch_bg_patches = torch.zeros(4, num_classes, 52, in_w, in_w).long().cuda()
        batch_fg_patches[:, 0, 0] = 0
        batch_bg_patches[:, 0, 0] = 1
        batch_classes = []
        for _ in range(4):
            batch_classes.append([f'c_{c}' for c in range(num_classes)])
        (batch_loss, _) = get_batch_loss(
             outputs, batch_fg_patches, batch_bg_patches, None,
             # segmentation excluded from loss for now.
             [[None for c in range(num_classes)] for t in batch_fg_patches], 
             batch_classes,
             [f'c_{c}' for c in range(num_classes)],
             compute_loss=True)
        batch_loss.backward()
        optimizer.step()
        del batch_loss
        del batch_fg_patches
        del batch_bg_patches
        del outputs
        del input_data


def get_in_w_out_w_for_memory(num_classes):
    print('computing largest patch size for GPU, num class = ', num_classes)
    # search for appropriate input size for GPU
    # in_w, out_w = get_in_w_out_w_for_memory(num_classes)
    # try to train a network and see which patch size fits on the gpu.
    pairs = get_in_w_out_w_pairs()
    for i, (in_w, out_w) in enumerate(pairs):
        torch.cuda.empty_cache()
        try:
            allocate_net(in_w, num_classes)
            torch.cuda.empty_cache()
            print(in_w, out_w, 'ok')
            print('Using', pairs[i+1], 'to be safe') # return next smallest to be safe
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
    # 3 channels as annotation may be used as input.
    model = UNet3D(num_classes=len(classes), im_channels=1) 

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
    # 3 channels to enable optional annotation as input.
    model = UNet3D(num_classes=len(classes), im_channels=1) 
    model = torch.nn.DataParallel(model)
    if not use_fake_cnn: 
        model.to(device)
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


def load_model_then_segment_3d(model_paths, image, batch_size,
                               in_w, out_w, in_d, out_d, classes):
    cnn = load_model(model_paths[0], classes)
    return pad_then_segment_3d(cnn, image, batch_size,
                               in_w, out_w, in_d, out_d)

def pad_then_segment_3d(cnn, image, batch_size, in_w, out_w, in_d, out_d):
    t = time.time()
    input_image_shape = image.shape
    in_patch_shape = (in_d, in_w, in_w)
    out_patch_shape = (out_d, out_w, out_w)

    # pad so seg will be size of input image
    image = np.pad(image, ((17, 17), (17, 17), (17, 17)), mode='constant')

    # segment returns a series of prediction maps. one for each class.
    pred_maps = segment_3d(cnn, image, batch_size, in_patch_shape, out_patch_shape)

    assert pred_maps[0].shape == input_image_shape, (
        f'pred_maps[0].shape: {pred_maps[0].shape}, '
        f'input_image_shape: {input_image_shape}')

    print('Time to segment image', time.time() - t)
    return pred_maps


def segment_3d(cnn, image, batch_size, in_patch_shape,
               out_patch_shape, auto_complete_enabled=False):
    """
    in_patch_shape and out_patch_shape are (depth, height, width)
    """
    # Return prediction for each pixel in the image
    # The cnn will give a the output as channels where
    # each channel corresponds to a specific class 'probability'
    # don't need channel dimension
    # make sure the width, height and depth is at least as big as the patch.
    assert len(image.shape) == 3, str(image.shape)

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

    depth_diff = in_patch_shape[0] - out_patch_shape[0]
    width_diff = in_patch_shape[1] - out_patch_shape[1]
    
    out_im_shape = (image.shape[0] - depth_diff,
                    image.shape[1] - width_diff,
                    image.shape[2] - width_diff)

    coords = im_utils.get_coords_3d(out_im_shape, out_patch_shape)
    coord_idx = 0
    class_output_patches = None # list of patches for each class

    while coord_idx < len(coords):
        patches_to_process = []
        coords_to_process = []
        for _ in range(batch_size):
            if coord_idx < len(coords):
                coord = coords[coord_idx]
                x_coord, y_coord, z_coord = coord
                patch = image[z_coord:z_coord+in_patch_shape[0],
                             y_coord:y_coord+in_patch_shape[1],
                             x_coord:x_coord+in_patch_shape[2]]

                # need to add channel dimension for GPU processing.
                patch = np.expand_dims(patch, axis=0)
                
                assert patch.shape[1] == in_patch_shape[0], str(patch.shape)
                assert patch.shape[2] == in_patch_shape[1], str(patch.shape)
                assert patch.shape[3] == in_patch_shape[2], str(patch.shape)

                patch = img_as_float32(patch)
                patch = im_utils.normalize_patch(patch)
                patch = img_as_float32(patch)
                coord_idx += 1
                patches_to_process.append(patch) # need channel dimension
                coords_to_process.append(coord)

        patches_to_process = np.array(patches_to_process)
        patches_for_gpu = torch.from_numpy(patches_to_process)

        patches_for_gpu = patches_for_gpu.cuda().float()
        # TODO: consider use of detach. 
        # I might want to move to cpu later to speed up the next few operations.
        # I added .detach().cpu() to prevent a memory error.
        # pad with zeros for the annotation input channels
        # l,r, l,r, but from end to start     w  w  h  h  d  d, c, c, b, b
        if auto_complete_enabled:
            # add channels for annotation if auto_complete enabled
            patches_for_gpu = F.pad(patches_for_gpu, (0, 0, 0, 0, 0, 0, 0, 2), 'constant', 0)

        # patches shape after padding torch.Size([4, 3, 52, 228, 228])
        outputs = cnn(patches_for_gpu).detach().cpu()

        # bg channel index for each class in network output.
        class_idxs = [x * 2 for x in range(outputs.shape[1] // 2)]
        
        if class_output_patches is None:
            class_output_patches = [[] for _ in class_idxs]

        for i, class_idx in enumerate(class_idxs):
            class_output = outputs[:, class_idx:class_idx+2]
            # class_output : (batch_size, bg/fg, depth, height, width)
            softmaxed = softmax(class_output, 1) 
            foreground_probs = softmaxed[:, 1]  # just the foreground probability.
            predicted = foreground_probs > 0.5
            predicted = predicted.int()
            pred_np = predicted.data.cpu().numpy()
            for out_patch in pred_np:
                class_output_patches[i].append(out_patch)

    class_pred_maps = []
    for i, output_patches in enumerate(class_output_patches):
        # reconstruct for each class
        reconstructed = im_utils.reconstruct_from_patches(output_patches,
                                                        coords, out_im_shape)
        if padded_for_patch:
            # go back to the original shape before padding.
            # what ever we added on to make it as big as the patch size
            # now take that away.
            reconstructed = reconstructed[:reconstructed.shape[0] - patch_pad_z,
                                          :reconstructed.shape[1] - patch_pad_y,
                                          :reconstructed.shape[2] - patch_pad_x]
        class_pred_maps.append(reconstructed)

    return class_pred_maps


def add_config_shape(config, in_w, out_w):
    new_config = copy.deepcopy(config)
    num_classes = len(config['classes'])
    if in_w is None:
        in_w, out_w = get_in_w_out_w_for_memory(num_classes)
        print('found input width of', in_w, 'and output width of', out_w)
    new_config['in_w'] = in_w
    new_config['out_w'] = out_w
    new_config['in_d'] = 52
    new_config['out_d'] = 18
    return new_config
