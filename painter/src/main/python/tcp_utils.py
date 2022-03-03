"""
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
import socket
import time
import ssl
import rp_annot as rpa
import numpy as np
import json
import os
from pathlib import Path
import zlib

conn = None # global connection object to be re-used.

def establish_connection(ip, port):
    global conn
    if conn is None:
        print('Establish connection')
        conn = False # prevent running again while establishing
        server_cert = os.path.join(Path.home(), 'root_painter_server.public_key')
        client_cert = os.path.join(Path.home(), 'root_painter_client.public_key')
        client_key = os.path.join(Path.home(), 'root_painter_client.private_key')
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=server_cert)
        context.load_cert_chain(certfile=client_cert, keyfile=client_key)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn = context.wrap_socket(sock, server_side=False, server_hostname=ip)
        conn.connect((ip, port))
        log_cert = False
        if log_cert:
            print(f"SSL established. Peer cert: {conn.getpeercert()}'")

def request_patch_seg(annot_patch, segment_config):
    global conn
    annot_shape = np.array(annot_patch.shape)
    shape_bytes = annot_shape.tobytes()
    annot_patch_1d = annot_patch.reshape(-1)
    message = shape_bytes + rpa.compress(annot_patch_1d)
    conn.sendall(message)
    conn.sendall(b'end')
    message = json.dumps(segment_config).encode()
    conn.sendall(message)
    conn.sendall(b'cfg')
    seg_buffer = b'' # segmentation received from server.
    seg_shape = annot_shape[1:]
    # 17 is known for the current architecture. 
    # At some point in the future, we will probably want to accomodate 
    # different differences between input and output.
    seg_shape[0] = seg_shape[0] - (17*2)
    seg_shape[1] = seg_shape[1] - (17*2)
    seg_shape[2] = seg_shape[2] - (17*2)
    
    while True:
        data = conn.recv(16)
        # server sends 'end' when message over.
        if b'end' == data:
            seg_buffer = zlib.decompress(seg_buffer)
            seg_1d = rpa.decompress(seg_buffer, np.prod(seg_shape))
            seg_buffer = b'' # segmentation received from server.
            return seg_1d.reshape(seg_shape).astype(int)
        else:
            seg_buffer += data
    # we could call conn.close() to close the connection but instead we will leave it open.
