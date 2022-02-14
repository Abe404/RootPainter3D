""" Segment just one patch
Copyright (C) 2021 Abraham George Smith

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
import time
import torch

from skimage import img_as_float32
import numpy as np
from torch.nn.functional import softmax

from startup import add_config_shape
import im_utils
from model_utils import load_model, get_latest_model_paths
from scp_utils import scp_transfer

cached_image = None
cached_image_fname = None
cached_model = None
cached_model_path = None
 

def segment_patch(segment_config):
    global cached_image
    global cached_image_fname
    global cached_model
    global cached_model_path
 

    # in_dir, seg_dir, fname, model_paths,
    # in_w, out_w, in_d, out_d, class_name
    """
    Segment {file_names} from {dataset_dir} using {model_paths}
    and save to {seg_dir}.

    If model paths are not specified then use
    the latest model in {model_dir}.

    If no models are in {model_dir} then create a
    random weights model and use that.

    TODO: model saving is a counter-intuitve side effect,
    re-think project creation process to avoid this
    """

    fname = segment_config['file_name']
    patch_annot_fname = segment_config['patch_annot_fname']
    dataset_dir = segment_config['dataset_dir']

    model_dir = segment_config['model_dir']
    seg_dir = segment_config['seg_dir']

    patch_annot_dir = segment_config['patch_annot_dir']
    classes = segment_config['classes']
   
    # Where do we take the patch from? (coord dont include padding)
    x_start = segment_config['x_start']
    y_start = segment_config['y_start']
    z_start = segment_config['z_start']
    x_end = segment_config['x_end']
    y_end = segment_config['y_end']
    z_end = segment_config['z_end']

    client_ip = segment_config['client_ip']
    client_scp_path = segment_config['client_scp_path']
    client_username = segment_config['client_username']

    model_path = get_latest_model_paths(model_dir, 1)[0]
    fpath = os.path.join(dataset_dir, fname)

    if not os.path.isfile(fpath):
        print('Cannot load ', fpath, 'file does not exist')
        return

    # TODO: Get annot_dirs later - for now leave empty, as using single class only
    if fname == cached_image_fname:
        image = cached_image
    else:
        # load bounded/padded image for this patch
        (image, _, _, _) = im_utils.load_image_and_annot_for_seg(dataset_dir,
                                                                            [],
                                                                            fname)
        cached_image = image
        cached_image_fname = fname

    segment_config = add_config_shape(segment_config)
    # we know that the loaded image is padded so consider this when extracting the patch
    pad_z = (segment_config['in_d'] - segment_config['out_d']) // 2
    pad_y = (segment_config['in_w'] - segment_config['out_w']) // 2
    pad_x = pad_y

    # take the relevant region as specified in the instrction
    im_patch = image[z_start+pad_z:z_end+pad_z,
                     y_start+pad_y:y_end+pad_y,
                     x_start+pad_y:x_end+pad_x]

    # now normalise the tile (as this is done for all input to the network
    im_patch = im_utils.normalize_tile(img_as_float32(im_patch))
    model_input = torch.cuda.FloatTensor(1, 3, im_patch.shape[0], im_patch.shape[1], im_patch.shape[2])
    model_input[0, 0] = torch.from_numpy(im_patch.astype(np.float32)).cuda()

    # load the annotation, as this must also be used as input to the network (for optimal results)
    annot_fpath = os.path.join(patch_annot_dir, patch_annot_fname)
    annot_patch = im_utils.load_image(annot_fpath)
    bg_patch = annot_patch[0]
    fg_patch = annot_patch[1]
    model_input[0, 1] = torch.from_numpy(fg_patch).cuda()
    model_input[0, 2] = torch.from_numpy(bg_patch).cuda()

    # Use a cached model for improved performance
    if model_path == cached_model_path:
        cnn = cached_model
    else:
        cnn = load_model(model_path, classes).half()
        cached_model = cnn
        cached_model_path = model_path
    outputs = cnn(model_input.half())
    a = outputs.shape
    # bg channel index for each class in network output.
    class_idxs = [x * 2 for x in range(outputs.shape[1] // 2)]
    class_output_patches = [[] for _ in class_idxs]

    for i, class_idx in enumerate(class_idxs):
        # class_output = (batch_size, bg/fg, depth, height, width)
        class_output = outputs[:, class_idx:class_idx+2]
        softmaxed = softmax(class_output, 1) 
        foreground_probs = softmaxed[:, 1]  # just the foreground probability.
        predicted = foreground_probs > 0.5
        predicted = predicted.type(torch.cuda.ByteTensor)
        pred_np = predicted.data.detach().cpu().numpy()
        for out_tile in pred_np:
            class_output_patches[i].append(out_tile)

    seg_fname = patch_annot_fname.replace('.npy', '.npz')
    out_path = os.path.join(seg_dir, seg_fname)
    # For now only the first class will be segmented 
    seg = class_output_patches[0][0]
    # send segmented region to client
    scp_transfer(seg, seg_fname, client_ip, client_scp_path, client_username)
    # delete the annotation patch. dont want to use up all the disk.
    os.remove(annot_fpath)
