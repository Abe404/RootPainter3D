""" Segment just one patch
Copyright (C) 2021 Abraham George Smith
Copyright (C) 2022 Abraham George Smith

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
import socket
import ssl
import numpy as np
import zlib
import json
from pathlib import Path
import threading


import rp_annot as rpa
from skimage import img_as_float32
import numpy as np
from torch.nn.functional import softmax
from instructions import fix_config_paths

from model_utils import add_config_shape
import im_utils
from model_utils import load_model, get_latest_model_paths
from scp_utils import scp_transfer

cached_image = None
cached_image_fname = None
cached_model = None
cached_model_path = None
 

def start_server(sync_dir, ip, port):


    def create_server_socket(sync_dir, ip, port):
        server_cert = os.path.join(Path.home(), 'root_painter_server.public_key')
        server_key = os.path.join(Path.home(), 'root_painter_server.private_key')
        client_certs = os.path.join(Path.home(), 'root_painter_client.public_key')

        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_cert_chain(certfile=server_cert, keyfile=server_key)
        context.load_verify_locations(cafile=client_certs)

        # CUDA can be slow the first time it is used. We run this here so user interation is not
        # delayed by waiting for CUDA.
        print('starting CUDA..')
        _ = torch.from_numpy(np.zeros((100))).cuda()



        # Create a TCP/IP socket
        # Listen for incoming connections
        sock = socket.socket()
        print(f'Listening on {ip} port {port}')
        sock.bind((ip, port))
        sock.listen(1)
        while True:
            conn, client_address = sock.accept()
            conn = context.wrap_socket(conn, server_side=True)
            print("SSL established. Peer: {}".format(conn.getpeercert()))
            print('connection from', client_address)
            annot = None
            buf = b''
            while True:
                data = conn.recv(16)
                if data == b'end':
                    shape = np.frombuffer(buf[:16], dtype='int32')
                    annot1d = rpa.decompress(buf[16:], np.prod(shape))
                    annot = annot1d.reshape(shape)
                    buf = b''
                elif data == b'cfg':
                    segment_config = json.loads(buf.decode())
                    segment_config = fix_config_paths(sync_dir, segment_config)
                    segment_patch(segment_config, annot, conn)
                    buf = b''
                else:
                    buf += data
    

    server_thread = threading.Thread(target=create_server_socket, args=(sync_dir, ip, port))
    server_thread.start() # it doesn't finish.



def segment_patch(segment_config, annot_patch, conn):
    global cached_image
    global cached_image_fname
    global cached_model
    global cached_model_path

    start = time.time()
    fname = segment_config['file_name']
    dataset_dir = segment_config['dataset_dir']
    model_dir = segment_config['model_dir']
    classes = segment_config['classes']
    # Where do we take the patch from? (coord dont include padding)
    x_start = segment_config['x_start']
    y_start = segment_config['y_start']
    z_start = segment_config['z_start']
    x_end = segment_config['x_end']
    y_end = segment_config['y_end']
    z_end = segment_config['z_end']
    model_path = get_latest_model_paths(model_dir, 1)[0]
    fpath = os.path.join(dataset_dir, fname)

    if not os.path.isfile(fpath):
        print('Cannot load ', fpath, 'file does not exist')
        return

    # TODO: Get annot_dirs later - for now leave empty, as using single class only
    if fname == cached_image_fname:
        image = cached_image
    else:
        # load image for this patch
        (image, _, _, _) = im_utils.load_image_and_annot_for_seg(dataset_dir,
                                                                            [],
                                                                            fname)
        # loaded image is not longer pre-padded so padding must be assigned.
        image = im_utils.pad_3d(image, 17, 17, mode='reflect', constant_values=0)
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

    # now normalise the patch (as this is done for all input to the network
    im_patch = im_utils.normalize_patch(img_as_float32(im_patch))
    model_input = torch.cuda.FloatTensor(1, 3, im_patch.shape[0], im_patch.shape[1], im_patch.shape[2])
    model_input[0, 0] = torch.from_numpy(im_patch.astype(np.float32)).cuda()

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
        for out_patch in pred_np:
            class_output_patches[i].append(out_patch)
    # For now only the first class will be segmented 
    seg = class_output_patches[0][0]
    # send segmented region to client
    seg = seg.astype(bool)
    compressed = rpa.compress(seg.reshape(-1))
    compressed = zlib.compress(compressed)
    conn.sendall(compressed)
    conn.sendall(b'end')
    print('time to segment patch and transfer', time.time() - start)




def handle_patch_update_in_epoch_step(batch_im_patches, mode):
    assert mode in ['train', 'val'], f'unexptected mode {mode}'
    # padd channels to allow annotation input (or not)
    # l,r, l,r, but from end to start    w  w  h  h  d  d, c, c, b, b
    model_input = F.pad(batch_im_patches, (0, 0, 0, 0, 0, 0, 0, 2), 'constant', 0)

    # model_input[:, 0] is the input image
    # model_input[:, 1] is fg
    # model_input[:, 2] is bg
    if mode == 'train':
        for i, (fg_patches, bg_patches) in enumerate(zip(batch_fg_patches, batch_bg_patches)):
            # if it's trianing then with 50% chance 
            # add the annotations to the model input
            # Validation should not have access to the annotations.
            if random.random() > 0.5:
                # go through fg patches and bg_patches for each batch item
                # in this case we know there is always 1 bg and 1 fg patch.
                # at random add the annotation slice
                for slice_idx in range(fg_patches[0].shape[0]):
                    if torch.any(fg_patches[0][slice_idx]) or torch.any(bg_patches[0][slice_idx]):
                        # each slice with annotation is included with 50 percent probability.
                        # This allows the network to learn how to use the annotation to improve predictions
                        if random.random() > 0.5: 
                            model_input[i, 1, slice_idx] = fg_patches[0][slice_idx]
                            model_input[i, 2, slice_idx] = bg_patches[0][slice_idx]
    return model_input






