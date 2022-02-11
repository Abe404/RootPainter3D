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
import os
import time

def ls(dir_path):
    """
    list directory with
    retry as there may be temporary issues with a mounted network drive. 
    hidden files are not returned
    """
    retries = 100
    for _ in range(retries):
        try:
            fnames = os.listdir(dir_path)
            # Don't show hidden files
            # These can happen due to issues like file system 
            # synchonisation technology. RootPainter doesn't use them anywhere
            fnames = [f for f in fnames if f[0] != '.']
            return fnames
        except Exception as ex:
            print(f'exception listing file names from {dir_path}')
            print(ex)
            time.sleep(1)
    raise Exception(f'Cannot list file names from {dir_path} after {retries} retries')
        

def get_crop_start(fname):
    fname_parts = fname.replace('.nii.gz', '').split('_')

    # This padding information allows us to convert between bounded image
    # and annotation coordinates
    x_crop_start = int(fname_parts[-8])
    y_crop_start = int(fname_parts[-5])
    z_crop_start = int(fname_parts[-2])
    return x_crop_start, y_crop_start, z_crop_start