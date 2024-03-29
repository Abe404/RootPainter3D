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
from datetime import datetime
import copy
import multiprocessing

import torch
from torch.utils.data import DataLoader

from metrics import Metrics
from instructions import fix_config_paths
from datasets import RPDataset
import datasets

import model_utils
from model_utils import load_model_then_segment_3d
from model_utils import add_config_shape
from model_utils import create_first_model_with_random_weights
from model_utils import random_model
from model_utils import save_if_better, save_model
from model_utils import debug_memory

import data_utils
from im_utils import is_image, load_image, save
import im_utils
from file_utils import ls
import train_utils

metrics_to_print = ['dice', 'precision', 'recall', 'total_true', 'total_pred']

class Trainer():
    def __init__(self, sync_dir, ip=None, port=None, max_workers=12, epoch_length=None):
        self.sync_dir = sync_dir
        self.ip = ip
        self.port = port
        self.patch_update_enabled = ip and port
        # if this is not specified, epoch_length will be calculated automatically
        # This can be specified as a manual override for debugging purposes (for example).
        self.epoch_length = epoch_length 
        self.instruction_dir = os.path.join(self.sync_dir, 'instructions')
        self.training = False
        self.running = False
        self.train_set = None
        # Can be set by instructions.
        self.train_config = None
        self.model = None
        self.first_loop = True
        self.batch_size = 4
        self.num_workers = min(multiprocessing.cpu_count(), max_workers)
        print(self.num_workers, 'workers will be assigned for data loaders')

        self.optimizer = None
        self.val_patch_refs = []
        # used to check for updates
        self.annot_mtimes = []
        self.msg_dir = None
        self.epochs_without_progress = 0
        self.restarts_enabled = False
        self.training_restart = False # is training current in a restart period
        self.restart_best_val_dice = 0
        
        self.in_w = None
        self.out_w = None


        self.use_seg_in_training = False

        # approx 30 minutes
        self.max_epochs_without_progress = 60
        # These can be trigged by data sent from client
        self.valid_instructions = [self.start_training,
                                   self.segment,
                                   self.stop_training]
                                   #self.segment_patch]

    def get_train_epoch_length(self):
        if self.epoch_length: # manual override
            train_epoch_length = self.epoch_length
        elif self.val_patch_refs:
            # selected as it leads to training taking around 5x validation time
            train_epoch_length = max(64, 2 * len(self.val_patch_refs))
        else:
            train_epoch_length = 128
        return train_epoch_length

    def main_loop(self, on_epoch_end=None):
        print('Started main loop. Checking for instructions in',
              self.instruction_dir)
        if self.patch_update_enabled:
            raise Exception('implementation temporarily removed')
            # print('start patch seg server')
            # patch_seg.start_server(self.sync_dir, self.ip, self.port) # direct socket connection
            # print('after start server')

        self.running = True
        while self.running:
            self.check_for_instructions()
            if self.training:
                # can take a while so checks for
                # new instructions are also made inside
                self.val_patch_refs = self.get_new_val_patches_refs()
                train_result = self.train_epoch(self.model, length=self.get_train_epoch_length())
                if train_result:
                    train_metrics = Metrics.sum(train_result)
                    self.log_metrics('train', train_metrics)
                    print(train_metrics.__str__(to_use=metrics_to_print))
            if self.training:
                self.validation()
                if on_epoch_end:
                    on_epoch_end()
            else:
                self.first_loop = True
                time.sleep(1.0)

    def fix_config_paths(self, old_config):
        """ get paths relative to local machine """
        new_config = {}
        for k, v in old_config.items():
            if k == 'file_names':
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
                    new_val = os.path.join(self.sync_dir,
                                           os.path.normpath(new_val))
                    new_list.append(new_val)
                new_config[k] = new_list
            elif isinstance(v, str):
                v = v.replace('\\', '/')
                new_config[k] = os.path.join(self.sync_dir,
                                             os.path.normpath(v))
            else:
                new_config[k] = v
        return new_config

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
                with open(fpath, 'r', encoding='utf-8') as json_file:
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

            self.val_patch_refs = [] # dont want to cache these between projects
            self.epochs_without_progress = 0
            self.msg_dir = self.train_config['message_dir']
            model_dir = self.train_config['model_dir']
            classes = self.train_config['classes']
            self.train_config = add_config_shape(self.train_config, self.in_w, self.out_w)
            self.in_w = self.train_config['in_w']
            self.out_w = self.train_config['out_w']

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


    def val_epoch(self, model, val_patch_refs):
        """
        Compute the metrics for a given
            model, annotation directory and dataset (image directory).
        """
        dataset = RPDataset(self.train_config['val_annot_dirs'],
                            None, # train_seg_dirs
                            self.train_config['dataset_dir'],
                            # only specifying w and d as h is always same as w
                            self.train_config['in_w'],
                            self.train_config['out_w'],
                            self.train_config['in_d'],
                            self.train_config['out_d'],
                            datasets.Modes.VAL,
                            val_patch_refs)


        return train_utils.val_epoch(model,
                                     self.train_config['classes'],
                                     dataset,
                                     val_patch_refs,
                                     self.patch_update_enabled,
                                     self.epoch_step_callback,
                                     self.epoch_stop_fn)

    def epoch_stop_fn(self):
        return not self.training
    def epoch_step_callback(self):
        self.check_for_instructions()

    def train_epoch(self, model, length=None):
        if not self.train_annotation_exists():
            # no training until data ready
            return False

        debug_memory('train epoch start')
        if self.first_loop:
            self.first_loop = False
            self.write_message('Training started')
            self.log('Starting Training')

        dataset = RPDataset(self.train_config['train_annot_dirs'],
                            self.train_config['train_seg_dirs'],
                            self.train_config['dataset_dir'],
                            self.train_config['in_w'],
                            self.train_config['out_w'],
                            self.train_config['in_d'],
                            self.train_config['out_d'],
                            datasets.Modes.TRAIN,
                            None,
                            self.use_seg_in_training,
                            length=length)
        torch.set_grad_enabled(True)
        loader = DataLoader(dataset, self.batch_size, shuffle=False,
                            collate_fn=data_utils.collate_fn,
                            num_workers=self.num_workers,
                            drop_last=False, pin_memory=True)

        epoch_items_metrics = train_utils.train_epoch(
            model, self.train_config['classes'], loader,
            self.batch_size, self.optimizer, self.patch_update_enabled,
            self.epoch_step_callback, self.epoch_stop_fn)


        return epoch_items_metrics

    def assign_metrics_to_refs(self, metrics_list):
        # now go through and assign the errors to the appropriate patch refs.
        for i, metrics_for_ref in enumerate(metrics_list):
            # go through the val patch refs to find the equivalent patch ref
            self.val_patch_refs[i].metrics = metrics_for_ref

    def get_new_val_patches_refs(self):
        return im_utils.get_val_patch_refs(self.train_config['val_annot_dirs'],
                                           copy.deepcopy(self.val_patch_refs),
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
                  file=open(fpath, 'w+', encoding='utf-8'))
        with open(fpath, 'a+', encoding='utf-8') as log_file:
            log_file.write(metrics.csv_row())
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
        


        # this could include some of the prevous patches refs 
        # (if annotation has not changed etc)
        self.val_patch_refs = self.get_new_val_patches_refs()




        if not self.val_patch_refs:
            # if we don't yet have any validation data
            # but we are training then just save the model
            # Assuming we will get better than random weights
            # by the time the user gets to the second / third image
            save_model(model_dir, self.model, prev_path)
            was_saved = True
        else:
            # for current model get errors for all patches in the validation set.
            cur_val_items_metrics = self.val_epoch(copy.deepcopy(self.model),
                                                   self.val_patch_refs)
            if not cur_val_items_metrics:
                # if we didn't get anything back then it means the
                # dataset did not contain any annotations so no need
                # to proceed with validation.
                return 

            cur_val_metrics = Metrics.sum(cur_val_items_metrics)
            self.log_metrics('cur_val', cur_val_metrics)

            print('Current Model Validation:', cur_val_metrics.__str__(to_use=metrics_to_print))

            # uses current val_patch_refs, where computation is required only.
            prev_val_metrics = self.get_prev_model_metrics(prev_model)
            self.log_metrics('prev_val', prev_val_metrics)

            was_saved = save_if_better(model_dir, self.model, prev_path,
                                       cur_val_metrics.dice(),
                                       prev_val_metrics.dice())
            if was_saved:
                # if it was saved then cur_model will become prev_model so
                # update the cache to use metrics from current model
                # assign metrics to refs
                for i, metrics_for_ref in enumerate(cur_val_items_metrics):
                    self.val_patch_refs[i].metrics = metrics_for_ref

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
            if cur_val_metrics.dice() > self.restart_best_val_dice:
                print('local restart dice improvement from',
                      round(self.restart_best_val_dice, 4), 'to',
                      round(cur_val_metrics.dice(), 4))
                self.restart_best_val_dice = cur_val_metrics.dice()
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
            if self.restarts_enabled:
                self.restart_training()
    
    def restart_training(self):
        print('Restarting training from scratch')
        self.write_message('Restarting training from scratch')
        self.log('Restarting training from scratch')
        self.training_restart = True
        self.restart_best_val_dice = 0 # need to beat this or restart will stop after 60 epochs
        self.val_patch_refs = [] # dont want to cache these
        self.epochs_without_progress = 0
        # FIXME: should train_config be a dataclass?
        self.model = random_model(self.train_config['classes'])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01,
                                         momentum=0.99, nesterov=True)
        self.model.train()
        self.training = True

    def log(self, message):
        log_fpath = os.path.join(self.sync_dir, 'server_log.txt')
        with open(log_fpath, 'a+', encoding='utf-8') as log_file:
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

        segment_config = add_config_shape(segment_config, self.in_w, self.out_w)
        self.in_w = segment_config['in_w']
        self.out_w = segment_config['out_w']
        
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
                              overwrite=overwrite)

        duration = time.time() - start
        print(f'Seconds to segment {len(fnames)} images: ', round(duration, 3))


    def get_prev_model_metrics(self, prev_model, use_cache=True):
        # for previous model get errors for all patches which do not yet have metrics
        refs_to_compute = []

        if use_cache:
            for ref in self.val_patch_refs:
                if not ref.has_metrics():
                    refs_to_compute.append(ref)
        else:
            refs_to_compute = self.val_patch_refs

        print('computing prev model metrics for ', len(refs_to_compute),
              'out of', len(self.val_patch_refs))
        # if it is missing metrics then add it to refs_to_compute
        # then compute the errors for these patch refs
        if refs_to_compute:
            
            val_metrics = self.val_epoch(prev_model, refs_to_compute)
            assert len(val_metrics) == len(refs_to_compute)

            # now go through and assign the errors to the appropriate patch refs.
            for val_metric, computed_ref in zip(val_metrics, refs_to_compute):
                # go through the val patch refs to find the equivalent patch ref
                for i, ref in enumerate(self.val_patch_refs):
                    if ref.is_same_region_as(computed_ref):
                        if use_cache:
                            # it was already found that this ref needed computing
                            # confirm that it is indeed missing metrics
                            assert not self.val_patch_refs[i].has_metrics(), (
                                self.val_patch_refs[i].metrics)
                            assert not ref.has_metrics()
                        ref.metrics = val_metric 
                        assert self.val_patch_refs[i].has_metrics()

        prev_m = Metrics.sum([r.metrics for r in self.val_patch_refs])
        return prev_m

    #def segment_patch(self, segment_config):
    #    raise Exception('feature disabled')
    #    # patch_seg.segment_patch(segment_config)
        
    def segment_file(self, in_dir, seg_dir, fname, model_paths,
                     in_w, out_w, in_d, out_d, classes, overwrite=False):

        # segmentations are always saved as .nii.gz
        out_paths = []
        if len(classes) > 1:
            for c in classes:
                # output to nii.gz regardless of input format.
                out_fname = fname.replace('.nrrd', '.nii.gz') 
                out_paths.append(os.path.join(seg_dir, c, out_fname))
        else:
            # segment to nifty as they don't get loaded repeatedly in training.
            out_paths = [os.path.join(seg_dir, fname)]

        if not overwrite and all(os.path.isfile(out_path) for out_path in out_paths):
            print(f'Skip because found existing segmentation files for {fname}')
            return

        fpath = os.path.join(in_dir, fname)

        if not os.path.isfile(fpath):
            raise Exception(f'Cannot segment as missing file {fpath}')
        try:
            im = load_image(fpath)
        except Exception as e:
            # Could be temporary issues reading the image.
            # its ok just skip it.
            print('Exception loading', fpath, e)
            return
        seg_start = time.time()
        print('segment image, input shape = ', im.shape, datetime.now())

        seg_in_w, seg_out_w = model_utils.get_in_w_and_out_w_for_image(im, in_w, out_w) 
        segmented = load_model_then_segment_3d(model_paths, im, self.batch_size,
                                               seg_in_w, seg_out_w, in_d, out_d, classes)

        print(f'segment {fname}, dur', round(time.time() - seg_start, 2))
        
        for seg, outpath in zip(segmented, out_paths):
            # if the output folder doesn't exist then create it (class specific directory)
            out_dir = os.path.split(outpath)[0]
            if not os.path.exists(out_dir):
                print('making directory', out_dir)
                os.makedirs(out_dir)
            
            # catch warnings as low contrast is ok here.
            with warnings.catch_warnings():
                # create a version with alpha channel
                warnings.simplefilter("ignore")
                outpath = outpath.replace('.nrrd', '.nii.gz')
                save(outpath, seg)
