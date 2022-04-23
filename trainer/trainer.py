"""
Handle training and segmentation for a specific project

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
# pylint: disable=W0511, E1136, C0111, R0902, R0914, W0703, R0913, R0915
# W0511 is TODO
import os
import time
import warnings
import traceback
from pathlib import Path
import json
import sys
from datetime import datetime
import copy
import random


import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loss import get_batch_loss
from instructions import fix_config_paths

from datasets import RPDataset
from metrics import get_metrics, get_metrics_str, get_metric_csv_row
from model_utils import ensemble_segment_3d
from model_utils import create_first_model_with_random_weights
from model_utils import random_model
import model_utils
from model_utils import save_if_better, save_model
from metrics import metrics_from_val_tile_refs
import data_utils
import patch_seg

from im_utils import is_image, load_image, save_then_move
import im_utils
from file_utils import ls
from startup import add_config_shape


class Trainer():

    def __init__(self, sync_dir, ip=None, port=None):
        self.sync_dir = sync_dir
        self.ip = ip
        self.port = port
        self.patch_update_enabled = ip and port

        self.instruction_dir = os.path.join(self.sync_dir, 'instructions')
        self.training = False
        self.running = False
        self.train_set = None
        # Can be set by instructions.
        self.train_config = None
        self.model = None
        self.first_loop = True
        # TODO: derrive both batch_size and input_patch size based on available GPU memory
        self.batch_size = 4 
        self.optimizer = None
        self.val_tile_refs = []
        # used to check for updates
        self.annot_mtimes = []
        self.msg_dir = None
        self.epochs_without_progress = 0
        self.training_restart = False
        self.restart_best_val_dice = 0

        # approx 30 minutes
        self.max_epochs_without_progress = 60
        # These can be trigged by data sent from client
        self.valid_instructions = [self.start_training,
                                   self.segment,
                                   self.stop_training,
                                   self.segment_patch]

    def main_loop(self, on_epoch_end=None):
        print('Started main loop. Checking for instructions in',
              self.instruction_dir)
        if self.patch_update_enabled:
            print('start patch seg server')
            patch_seg.start_server(self.sync_dir, self.ip, self.port) # direct socket connection
            print('after start server')

        self.running = True
        while self.running:
            self.check_for_instructions()
            if self.training:
                # can take a while so checks for
                # new instructions are also made inside
                self.val_tile_refs = self.get_new_val_tiles_refs()
                if self.val_tile_refs:
                    # selected as it leads to training taking around 5x validation time
                    train_epoch_length = max(64, 2 * len(self.val_tile_refs))
                else:
                    train_epoch_length = 128
                epoch_result = self.one_epoch(self.model, 'train', length=train_epoch_length)
                if epoch_result:
                    (tps, fps, tns, fns) = epoch_result
                    train_m = get_metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
                    self.log_metrics('train', train_m)
                    print(get_metrics_str(train_m, to_use=['dice', 'precision', 'recall']))
            if self.training:
                self.validation()
                if on_epoch_end:
                    on_epoch_end()
            else:
                self.first_loop = True
                time.sleep(1.0)


    def check_for_instructions(self):
        for fname in ls(self.instruction_dir):
            print('found instruction', fname)
            if self.execute_instruction(fname):
                os.remove(os.path.join(self.instruction_dir, fname))

    def execute_instruction(self, fname):
        fpath = os.path.join(self.instruction_dir, fname)
        name = fname.rpartition('_')[0] # remove hash
        if name in [i.__name__ for i in self.valid_instructions]:
            print('execute_instruction', name)
            try:
                with open(fpath, 'r') as json_file:
                    contents = json_file.read()
                    config = fix_config_paths(self.sync_dir, json.loads(contents))
                    getattr(self, name)(config)
            except Exception as e:
                tb = traceback.format_exc()
                print('Exception parsing instruction', e, tb)
                return False
        else:
            #TODO put in a log and display error to the user.
            raise Exception(f"unhandled instruction {name})")
        return True

    def stop_training(self, _):
        if self.training:
            self.training = False
            self.epochs_without_progress = 0
            message = 'Training stopped'
            self.write_message(message)
            self.log(message)

    def start_training(self, config):
        if self.training_restart:
            self.training_restart = False
            self.training = False

        if not self.training:
            self.train_config = config
            
            classes = ['annotations']
            if 'classes' in self.train_config:
                classes = self.train_config['classes']
            else:
                self.train_config['classes'] = classes

            self.val_tile_refs = [] # dont want to cache these between projects
            self.epochs_without_progress = 0
            self.msg_dir = self.train_config['message_dir']
            model_dir = self.train_config['model_dir']
            classes = self.train_config['classes']
            self.train_config = add_config_shape(self.train_config)

            model_paths = model_utils.get_latest_model_paths(model_dir, 1)
            if model_paths:
                self.model = model_utils.load_model(model_paths[0], classes)
            else:
                self.model = create_first_model_with_random_weights(model_dir, classes)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01,
                                             momentum=0.99, nesterov=True)

            self.model.train()
            self.training = True


    def reset_progress_if_annots_changed(self):
        train_annot_dirs = self.train_config['train_annot_dirs']
        val_annot_dirs = self.train_config['val_annot_dirs']
        new_annot_mtimes = []
        for annot_dir in train_annot_dirs + val_annot_dirs:
            for fname in ls(annot_dir):
                fpath = os.path.join(annot_dir, fname)
                new_annot_mtimes.append(os.path.getmtime(fpath))
        new_annot_mtimes = sorted(new_annot_mtimes)
        if new_annot_mtimes != self.annot_mtimes:
            print('reset epochs without progress as annotations have changed')
            self.epochs_without_progress = 0
        self.annot_mtimes = new_annot_mtimes

    def write_message(self, message):
        """ write a message for the user (client) """
        Path(os.path.join(self.msg_dir, message)).touch()


    def train_annotation_exists(self):
        train_annot_dirs = self.train_config['train_annot_dirs']
        found_train_annot = False
        for d in train_annot_dirs:
            if [is_image(a) for a in ls(d)]:
                found_train_annot = True
        return found_train_annot 

    def one_epoch(self, model, mode='train', val_tile_refs=None, length=None):
        if not self.train_annotation_exists():
            # no training until data ready
            return False

        if self.first_loop:
            self.first_loop = False
            self.write_message('Training started')
            self.log('Starting Training')

        if mode == 'val':
            dataset = RPDataset(self.train_config['val_annot_dirs'],
                                None,
                                self.train_config['dataset_dir'],
                                self.train_config['in_w'],
                                self.train_config['out_w'],
                                self.train_config['in_d'],
                                self.train_config['out_d'],
                                'val',
                                val_tile_refs)
            torch.set_grad_enabled(False)
            loader = DataLoader(dataset, self.batch_size * 2, shuffle=True,
                                collate_fn=data_utils.collate_fn,
                                num_workers=16, drop_last=False, pin_memory=True)
        elif mode == 'train':
            dataset = RPDataset(self.train_config['train_annot_dirs'],
                                self.train_config['train_seg_dirs'],
                                self.train_config['dataset_dir'],
                                self.train_config['in_w'],
                                self.train_config['out_w'],
                                self.train_config['in_d'],
                                self.train_config['out_d'],
                                'train',
                                val_tile_refs,
                                length=length)
            torch.set_grad_enabled(True)
            loader = DataLoader(dataset, self.batch_size, shuffle=False,
                                collate_fn=data_utils.collate_fn,
                                num_workers=16,
                                drop_last=False, pin_memory=True)
            model.train()
        else:
            raise Exception(f"Invalid mode: {mode}")

        epoch_start = time.time()

        tps = []
        fps = []
        tns = []
        fns = []
        loss_sum = 0
        for step, (batch_im_tiles, batch_fg_tiles,
                   batch_bg_tiles, batch_seg_tiles, batch_classes) in enumerate(loader):

            self.check_for_instructions()
            batch_im_tiles = torch.from_numpy(np.array(batch_im_tiles)).cuda()
            self.optimizer.zero_grad()
        
            # padd channels to allow annotation input (or not)
            # l,r, l,r, but from end to start    w  w  h  h  d  d, c, c, b, b
            model_input = F.pad(batch_im_tiles, (0, 0, 0, 0, 0, 0, 0, 2), 'constant', 0)
    
            # model_input[:, 0] is the input image
            # model_input[:, 1] is fg
            # model_input[:, 2] is bg
            if self.patch_update_enabled and mode == 'train':
                for i, (fg_tiles, bg_tiles) in enumerate(zip(batch_fg_tiles, batch_bg_tiles)):
                    # if it's trianing then with 50% chance 
                    # add the annotations to the model input
                    # Validation should not have access to the annotations.
                    if random.random() > 0.5:
                        # go through fg tiles and bg_tiles for each batch item
                        # in this case we know there is always 1 bg and 1 fg tile.
                        # at random add the annotation slice
                        for slice_idx in range(fg_tiles[0].shape[0]):
                            if torch.any(fg_tiles[0][slice_idx]) or torch.any(bg_tiles[0][slice_idx]):
                                # each slice with annotation is included with 50 percent probability.
                                # This allows the network to learn how to use the annotation to improve predictions
                                if random.random() > 0.5: 
                                    model_input[i, 1, slice_idx] = fg_tiles[0][slice_idx]
                                    model_input[i, 2, slice_idx] = bg_tiles[0][slice_idx]

            outputs = model(model_input)

            (batch_loss, batch_tps, batch_tns,
             batch_fps, batch_fns) = get_batch_loss(
                 outputs, batch_fg_tiles, batch_bg_tiles, batch_seg_tiles,
                 #outputs, batch_fg_tiles, batch_bg_tiles,
                 batch_classes, self.train_config['classes'],
                 compute_loss=(mode=='train'))

            tps += batch_tps
            fps += batch_fps
            tns += batch_tns
            fns += batch_fns

            if mode == 'train':
                loss_sum += batch_loss.item() # float
                batch_loss.backward()
                self.optimizer.step()

            if mode == 'train':
                sys.stdout.write(f"{mode} {(step+1) * self.batch_size}/"
                                 f"{len(loader.dataset)} "
                                 f" loss={round(batch_loss.item(), 3)} \r")
                sys.stdout.flush()

            self.check_for_instructions() # could update training parameter
            if not self.training: # in this context we consider validation part of training.
                return

        duration = round(time.time() - epoch_start, 3)
        print(f'{mode} epoch duration', duration,
              'time per instance', round((time.time() - epoch_start) / len(tps), 3))

        return [tps, fps, tns, fns]

    def assign_metrics_to_refs(self, tps, fps, tns, fns):
        # now go through and assign the errors to the appropriate tile refs.
        for i, (tp, fp, tn, fn) in enumerate(zip(tps, fps, tns, fns)):
            # go through the val tile refs to find the equivalent tile ref
            self.val_tile_refs[i][3] = [tp, fp, tn, fn]

    def get_new_val_tiles_refs(self):
        return im_utils.get_val_tile_refs(self.train_config['val_annot_dirs'],
                                          copy.deepcopy(self.val_tile_refs),
                                          out_shape=(self.train_config['out_d'],
                                                     self.train_config['out_w'],
                                                     self.train_config['out_w']))

    def log_metrics(self, name, metrics):
        fname = datetime.today().strftime('%Y-%m-%d')
        fname += f'_{name}.csv'
        fpath = os.path.join(self.train_config['log_dir'], fname)
        if not os.path.isfile(fpath):
            # write headers if file didn't exist
            print('date_time,true_positives,false_positives,true_negatives,'
                  'false_negatives,precision,recall,dice',
                  file=open(fpath, 'w+'))
        with open(fpath, 'a+') as log_file:
            log_file.write(get_metric_csv_row(metrics))
            log_file.flush()

    def validation(self):
        """ Get validation set loss for current model and previous model.
            log those metrics and update the model if the
            current model is better than the previous model.
            Also stop training if the current model hasnt
            beat the previous model for {max_epochs}
        """
        model_dir = self.train_config['model_dir']
        prev_model, prev_path = model_utils.get_prev_model(model_dir,
                                                           self.train_config['classes'])
        self.val_tile_refs = self.get_new_val_tiles_refs()

        if not self.val_tile_refs:
            # if we don't yet have any validation data
            # but we are training then just save the model
            # Assuming we will get better than random weights
            # by the time the user gets to the second / third image
            save_model(model_dir, self.model, prev_path)
            was_saved = True
        else:
            # for current model get errors for all tiles in the validation set.
            epoch_result = self.one_epoch(copy.deepcopy(self.model), 'val', self.val_tile_refs)
            if not epoch_result:
                # if we didn't get anything back then it means the
                # dataset did not contain any annotations so no need
                # to proceed with validation.
                return 
            (tps, fps, tns, fns) = epoch_result
        
            cur_m = get_metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
            self.log_metrics('cur_val', cur_m)

            prev_m = self.get_prev_model_metrics(prev_model)
            self.log_metrics('prev_val', prev_m)
            was_saved = save_if_better(model_dir, self.model, prev_path,
                                       cur_m['dice'], prev_m['dice'])
            if was_saved:
                # update the cache to use metrics from current model
                self.assign_metrics_to_refs(tps, fps, tns, fns)

        if was_saved:
            self.epochs_without_progress = 0
        else:
            self.epochs_without_progress += 1

        # if we are doing a restart (from random weights) then lets consider the
        # performance improvements local to that restart, 
        # rather than the best model from all starts.
        if self.training_restart: 
            # we know that data doesn't change during a restart 
            # (because this would cause the restart to stop and typical training to resumse)
            # so we keep track of the best dice so far for this specific restart, and extend
            # training if we beat it, i.e set epochs_without_progress to 0
            if cur_m['dice'] > self.restart_best_val_dice:
                print('local restart dice improvement from',
                      round(self.restart_best_val_dice, 4), 'to', round(cur_m['dice'], 4))
                self.restart_best_val_dice = cur_m['dice']
                self.epochs_without_progress = 0

        self.reset_progress_if_annots_changed()

        message = (f'Training {self.epochs_without_progress}'
                   f' of max {self.max_epochs_without_progress}'
                   ' epochs without progress')
        self.write_message(message)
        if self.epochs_without_progress >= self.max_epochs_without_progress:
            message = (f'Training finished as {self.epochs_without_progress}'
                       ' epochs without progress')
            print(message)
            self.log(message)
            self.training = False
            self.write_message(message)
            self.restart_training()
    
    def restart_training(self):
        print('Restarting training from scratch')
        self.write_message('Restarting training from scratch')
        self.log('Restarting training from scratch')
        self.training_restart = True
        self.restart_best_val_dice = 0 # need to beat this or restart will stop after 60 epochs
        self.val_tile_refs = [] # dont want to cache these
        self.epochs_without_progress = 0
        self.model = random_model(self.train_config['classes'])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01,
                                         momentum=0.99, nesterov=True)
        self.model.train()
        self.training = True

    def log(self, message):
        with open(os.path.join(self.sync_dir, 'server_log.txt'), 'a+') as log_file:
            log_file.write(f"{datetime.now()}|{time.time()}|{message}\n")
            log_file.flush()

    def segment(self, segment_config):
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
        in_dir = segment_config['dataset_dir']
        seg_dir = segment_config['seg_dir']

        segment_config = add_config_shape(segment_config)
        classes = segment_config['classes']
        if "file_names" in segment_config:
            fnames = segment_config['file_names']
            assert isinstance(fnames, list), type(fnames)
        else:
            # default to using all files in the directory if file_names is not specified.
            fnames = ls(in_dir)
        # if model paths not specified use latest.
        if "model_paths" in segment_config:
            model_paths = segment_config['model_paths']
        else:
            model_dir = segment_config['model_dir']
            model_paths = model_utils.get_latest_model_paths(model_dir, 1)
            # if latest is not found then create a model with random weights
            # and use that.
            if not model_paths:
                create_first_model_with_random_weights(model_dir, classes)
                model_paths = model_utils.get_latest_model_paths(model_dir, 1)
        if "overwrite" in segment_config:
            overwrite = segment_config['overwrite']
        else:
            overwrite = False

        start = time.time()

        for fname in fnames:
            self.segment_file(in_dir, seg_dir, fname,
                              model_paths,
                              in_w=segment_config['in_w'],
                              out_w=segment_config['out_w'],
                              in_d=segment_config['in_d'],
                              out_d=segment_config['out_d'],
                              classes=classes,
                              sync_save=len(fnames) == 1,
                              overwrite=overwrite)

        duration = time.time() - start
        print(f'Seconds to segment {len(fnames)} images: ', round(duration, 3))


    def get_prev_model_metrics(self, prev_model, use_cache=True):
        # for previous model get errors for all tiles which do not yet have metrics
        refs_to_compute = []

        if use_cache:
            # for each val tile
            for t in self.val_tile_refs:
                if t[3] is None:
                    refs_to_compute.append(t)
        else:
            refs_to_compute = self.val_tile_refs

        print('computing prev model metrics for ', len(refs_to_compute),
              'out of', len(self.val_tile_refs))
        # if it is missing metrics then add it to refs_to_compute
        # then compute the errors for these tile refs
        if refs_to_compute:
            
            (tps, fps, tns, fns) = self.one_epoch(prev_model, 'val', refs_to_compute)
            assert len(tps) == len(fps) == len(tns) == len(fns) == len(refs_to_compute)

            # now go through and assign the errors to the appropriate tile refs.
            for tp, fp, tn, fn, computed_ref in zip(tps, fps, tns, fns, refs_to_compute):
                # go through the val tile refs to find the equivalent tile ref
                for i, ref in enumerate(self.val_tile_refs):
                    if ref[0] == computed_ref[0] and ref[1] == computed_ref[1]:
                        if use_cache:
                            assert self.val_tile_refs[i][3] is None, self.val_tile_refs[i][3]
                            assert ref[3] is None
                        ref[3] = [tp, fp, tn, fn]
                        assert self.val_tile_refs[i][3] is not None

        prev_m = metrics_from_val_tile_refs(self.val_tile_refs)
        return prev_m

    def segment_patch(self, segment_config):
        patch_seg.segment_patch(segment_config)

    def segment_file(self, in_dir, seg_dir, fname, model_paths,
                     in_w, out_w, in_d, out_d, classes, sync_save, overwrite=False):

        # segmentations are always saved as .nii.gz
        out_paths = []
        if len(classes) > 1:
            for c in classes:
                out_paths.append(os.path.join(seg_dir, c, fname))
        else:
            # segment to nifty as they don't get loaded repeatedly in training.
            out_paths = [os.path.join(seg_dir, fname)]

        if not overwrite and all([os.path.isfile(out_path) for out_path in out_paths]):
            print(f'Skip because found existing segmentation files for {fname}')
            return

        fpath = os.path.join(in_dir, fname)

        if not os.path.isfile(fpath):
            raise Exception(f'Cannot segment as missing file {fpath}')
        try:
            im = load_image(fpath)
            # TODO: Consider removing thie soon
            im = np.rot90(im, k=3)
            im = np.moveaxis(im, -1, 0) # depth moved to beginning
            # reverse lr and ud
            im = im[::-1, :, ::-1]
        except Exception as e:
            # Could be temporary issues reading the image.
            # its ok just skip it.
            print('Exception loading', fpath, e)
            return
        seg_start = time.time()
        print('segment image, input shape = ', im.shape, datetime.now())
        segmented = ensemble_segment_3d(model_paths, im, fname, self.batch_size,
                                        in_w, out_w, in_d, out_d, classes)
        print(f'ensemble segment {fname}, dur', round(time.time() - seg_start, 2))
        
        for seg, outpath in zip(segmented, out_paths):
            # catch warnings as low contrast is ok here.
            with warnings.catch_warnings():
                # create a version with alpha channel
                warnings.simplefilter("ignore")
                if sync_save:
                    # other wise do sync because we don't want to delete the segment
                    # instruction too early.
                    outpath = outpath.replace('.nrrd', '.nii.gz')
                    save_then_move(outpath, seg)
                else:
                    # TODO find a cleaner way to do this.
                    # if more than one file then optimize speed over stability.
                    x = threading.Thread(target=save_then_move,
                                         args=(outpath, segmented))
                    x.start()
