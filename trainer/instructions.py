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

import os

def fix_config_paths(sync_dir, old_config):
    """ get paths relative to local machine """
    new_config = {}
    for k, v in old_config.items():
        if k in ['file_names',
                 'file_name']:
            # names dont need a path appending
            new_config[k] = v
        elif k == 'classes':
            # classes should not be altered
            new_config[k] = v
        elif isinstance(v, list):
            # if its a list fix each string in the list.
            new_list = []
            for e in v:
                new_val = e.replace('\\', '/')
                new_val = os.path.join(sync_dir,
                                       os.path.normpath(new_val))
                new_list.append(new_val)
            new_config[k] = new_list
        elif isinstance(v, str):
            v = v.replace('\\', '/')
            new_config[k] = os.path.join(sync_dir,
                                         os.path.normpath(v))
        else:
            new_config[k] = v
    return new_config

