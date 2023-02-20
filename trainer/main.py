"""
Human in the loop deep learning segmentation for biological images

Copyright (C) 2020 Abraham George Smith
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
from pathlib import Path
import os
import re
import json
import argparse
from trainer import Trainer
from startup import startup_setup

def parse_patchsize(text):
    """
    Parse the patchsize argument.
    It is expected to be two integers, e.g. "164 130"
    Ignores surrounding and separating characters, e.g. "(164,130)".
    """
    # Find all integers in string
    p = re.compile(r"\d+")
    result = p.findall(text or "")

    # Only accept results where there are exactly two integers
    if len(result) != 2:
        return (None, None)

    result = map(int, result)
    return tuple(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--syncdir',
                        help=('location of directory where data is'
                              ' synced between the client and server'))
    parser.add_argument('--maxworkers',
                    type=int,
                    default=12,
                    help=('maximum number of workers used for the dataloader'))

    parser.add_argument('--epochlength',
                        type=int,
                        help=('Number of instances to be used in each training epoch. '
                              'Primarily used as a manual override for debugging purposes.'))


    parser.add_argument('--patchsize',
                        type=str,
                        help=('Manually assign patch size. '
                              'Primarily used as a manual override for debugging purposes.'))

    settings_path = os.path.join(Path.home(), 'root_painter_settings.json')
   
    settings = None
    
    args = parser.parse_args()
    (in_w, out_w) = parse_patchsize(args.patchsize)
    
    if args.syncdir:
        sync_dir = args.syncdir
        startup_setup(settings_path, sync_dir=sync_dir)
    else:
        startup_setup(settings_path, sync_dir=None)
        settings = json.load(open(settings_path, 'r'))
        sync_dir = Path(settings['sync_dir'])
        
    if settings and 'auto_complete' in settings and settings['auto_complete']:
        ip = settings['server_ip']
        port = settings['server_port']
        trainer = Trainer(sync_dir, ip, port, max_workers=args.maxworkers,
                          epoch_length=args.epochlength,
                          in_w=in_w,
                          out_w=out_w)
    else:
        trainer = Trainer(sync_dir,
                          max_workers=args.maxworkers,
                          epoch_length=args.epochlength,
                          in_w=in_w,
                          out_w=out_w)

    trainer.main_loop()
