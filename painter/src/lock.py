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
from PyQt5 import QtWidgets
import os
import getpass

def create_lock_file(proj_dir, fname):
    """
    Create a lock file for the current user and the image they are annotating.
    """
    user_lock_dir = os.path.join(proj_dir, 'lock_files', getpass.getuser())
    if not os.path.isdir(user_lock_dir):
        os.makedirs(user_lock_dir)
    fpath = os.path.join(user_lock_dir, fname)
    f = open(fpath, 'w+')
    print('', file=f)
    f.close()

def delete_lock_files_for_current_user(proj_dir):
    """ get the current user
        and delete any lock files for this user
    """
    user_lock_dir = os.path.join(proj_dir, 'lock_files', getpass.getuser())
    if not os.path.isdir(user_lock_dir):
        os.makedirs(user_lock_dir)
    fnames = os.listdir(user_lock_dir)
    for f in fnames:
        os.remove(os.path.join(user_lock_dir, f))

def get_lock_file_path(proj_dir, fname):
    users = os.listdir(os.path.join(proj_dir,  'lock_files'))
    for u in users:
        fpath = os.path.join(proj_dir, 'lock_files', u, fname)
        if os.path.isfile(fpath):
            return u, fpath
    return False

def show_locked_message(proj_dir, fname):
    username = getpass.getuser()
    msg = QtWidgets.QMessageBox()
    uname, fpath = get_lock_file_path(proj_dir, fname)
    msg.setText(f"A lock file exists indicating"
                f' that {uname} is editing {fname}. \n'
                f" {fname} will be skipped and you will be shown the next image. \n"
                " If you wish to edit the file anyway, you can delete the "
                f"lock file located at {fpath}.")
    msg.show()
    return msg # need to return it so it doesn't go out of scope and disapear