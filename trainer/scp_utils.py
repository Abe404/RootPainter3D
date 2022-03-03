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


from paramiko import SSHClient
import paramiko 
from scp import SCPClient
import os
import numpy as np

scp_client = None

def scp_transfer(seg, seg_fname, remote_ip, remote_uname):
    global scp_client 
    if not scp_client:
        ssh = SSHClient()
        ssh.load_system_host_keys()
        # be careful with this - private network only
        # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(remote_ip, username=remote_uname)
        # SCP Client takes a paramiko transport as an argument
        scp_client = SCPClient(ssh.get_transport())
    tmp_out = os.path.join('tmp', seg_fname)
    np.savez_compressed(tmp_out, seg=seg)
    scp_client.put(tmp_out, 'rp_scp_in')
