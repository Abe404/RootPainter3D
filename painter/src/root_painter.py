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

# pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914
# too many statements
# pylint: disable=R0915
import sys
import os
from pathlib import PurePath
import json
from functools import partial
import traceback
from datetime import datetime
import time

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
import nibabel as nib

from skimage.io import use_plugin
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

from view_state import ViewState
from about import AboutWindow, LicenseWindow, ShortcutWindow
from create_project import CreateProjectWidget
from im_viewer import ImViewer, ImViewerWindow
from nav import NavWidget
from file_utils import penultimate_fname_with_segmentation
from file_utils import get_annot_path
from file_utils import maybe_save_annotation_3d
from instructions import send_instruction
from contrast_slider import ContrastSlider
import im_utils
import menus
from segment import segment_full_image
from lock import create_lock_file, delete_lock_files_for_current_user, get_lock_file_path, show_locked_message


use_plugin("pil")

class RootPainter(QtWidgets.QMainWindow):

    closed = QtCore.pyqtSignal()

    def __init__(self, sync_dir, contrast_presets, server_ip=None, server_port=None):
        super().__init__()
        self.sync_dir = sync_dir
        self.instruction_dir = sync_dir / 'instructions'
        self.send_instruction = partial(send_instruction,
                                        instruction_dir=self.instruction_dir,
                                        sync_dir=sync_dir)
        self.contrast_presets = contrast_presets
        self.view_state = ViewState.LOADING_SEG
        self.auto_complete_enabled = server_ip and server_port # aka patch_update
        # for scp communication from server to client.
        self.server_ip = server_ip
        self.server_port = server_port

        self.tracking = False
        self.seg_mtime = None
        self.im_width = None
        self.im_height = None
        self.annot_data = None
        self.seg_data = None

        # for patch segment, useful for knowing how much annotation to send to the server.
        self.input_shape = (52, 228, 228)
        self.output_shape = (18, 194, 194)

        self.lines_to_log = []
        self.log_debounce = QtCore.QTimer()
        self.log_debounce.setInterval(500)
        self.log_debounce.setSingleShot(True)
        self.log_debounce.timeout.connect(self.log_debounced)

        self.initUI()

    def initUI(self):
        if len(sys.argv) < 2:
            self.init_missing_project_ui()
            return

        fname = sys.argv[1]
        if os.path.splitext(fname)[1] == '.seg_proj':
            proj_file_path = os.path.abspath(sys.argv[1])
            self.open_project(proj_file_path)
        else:
            # only warn if -psn not in the args. -psn is in the args when
            # user opened app in a normal way by clicking on the Application icon.
            if not '-psn' in sys.argv[1]:
                QtWidgets.QMessageBox.about(
                    self,
                    'Error',
                    f"{sys.argv[1]} ' is not a valid "
                    "segmentation project (.seg_proj) file")
            self.init_missing_project_ui()

    def get_train_annot_dir(self):
        # taking into account the current class.
        if len(self.classes) > 1:
            return self.proj_location / 'annotations' / self.cur_class / 'train'
        return self.proj_location / 'annotations' / 'train'

    def get_val_annot_dir(self):
        # taking into account the current class.
        if len(self.classes) > 1:
            return self.proj_location / 'annotations' / self.cur_class / 'val'
        return self.proj_location / 'annotations' / 'val'

    def open_project(self, proj_file_path):
        # extract json
        with open(proj_file_path, 'r') as json_file:
            settings = json.load(json_file)
            self.dataset_dir = self.sync_dir / 'datasets' / PurePath(settings['dataset'])
            
            if 'guide_image_dir' in settings:
                self.guide_image_dir = self.sync_dir / 'datasets' / PurePath(settings['guide_image_dir'])

            self.proj_location = self.sync_dir / PurePath(settings['location'])
            self.image_fnames = settings['file_names']
            self.seg_dir = self.proj_location / 'segmentations'
            self.log_dir = self.proj_location / 'logs'

            self.train_seg_dirs = []
            self.train_annot_dirs = []
            self.val_annot_dirs = []
            # if going with a single class or old style settings
            # then use old style project structure with single train and val
            # folder, without the class name being specified
            if "classes" in settings and len(settings['classes']) > 1:
                # if more than one class is present then create train and val folders for each class
                self.classes = settings['classes']
                self.cur_class = self.classes[0]
                for c in self.classes:
                    self.train_annot_dirs.append(self.proj_location / 'annotations' / c / 'train')
                    self.val_annot_dirs.append(self.proj_location / 'annotations' / c / 'val')
                    self.train_seg_dirs.append(self.proj_location / 'train_segmentations' / c )
            else:         
                self.classes = ['annotations'] # default class for single class project.
                self.cur_class = self.classes[0]
                self.train_annot_dirs = [self.proj_location / 'annotations' / 'train']
                self.val_annot_dirs = [self.proj_location / 'annotations' / 'val']
                self.train_seg_dirs = [self.proj_location / 'train_segmentations']

            self.model_dir = self.proj_location / 'models'
            self.message_dir = self.proj_location / 'messages'

            # If there are any segmentations which have already been saved
            # then go through the segmentations in the order specified
            # by self.image_fnames
            # and set fname (current image) to be the penultimate image with a segmentation
            # The client will always segment the image after the one being viewed. So we don't
            # show the last image segmented until the user navigates to it, or we will constantly
            # move forwards through the dataset without annotating images 
            # (if we simply close and re-open the client)
            last_with_seg = penultimate_fname_with_segmentation(self.image_fnames, self.seg_dir)
            if last_with_seg:
                fname = last_with_seg
            else:
                fname = self.image_fnames[0]

            # set first image from project to be current image
            self.image_path = os.path.join(self.dataset_dir, fname)
            self.update_window_title()
            self.annot_path = get_annot_path(fname, self.get_train_annot_dir(),
                                             self.get_val_annot_dir())
            self.init_active_project_ui()

            self.track_changes()

    def log_debounced(self):
        """ write to log file only so often to avoid lag """
        with open(os.path.join(self.log_dir, 'client.csv'), 'a+') as log_file:
            while self.lines_to_log:
                line = self.lines_to_log[0]
                log_file.write(line)
                self.lines_to_log = self.lines_to_log[1:]

    def log(self, message):
        self.lines_to_log.append(f"{datetime.now()}|{time.time()}|{message}\n")
        self.log_debounce.start() # write after 1 second

    def update_file(self, fpath):
        """ Invoked when the file to view has been changed by the user.
            Show image file and it's associated annotation and segmentation """
        # save annotation for current file before changing to new file.
        self.log(f'update_file_start,fname:{os.path.basename(fpath)},view_state:{self.view_state}')

        delete_lock_files_for_current_user(self.proj_location) 
        lock_file_path = get_lock_file_path(self.proj_location, os.path.basename(fpath))
        if lock_file_path:
            self.msg = show_locked_message(self.proj_location, os.path.basename(fpath))
            # if a file is locked then show a warning to the user
            self.nav.update_to_next_image()
            return

        create_lock_file(self.proj_location, os.path.basename(fpath)) 
        self.tracking = False # take a break from tracking until we get the next image.
        
        if self.view_state == ViewState.ANNOTATING:
            self.save_annotation()
        self.fname = os.path.basename(fpath)
        self.image_path = os.path.join(self.dataset_dir, self.fname)
        self.img_data = im_utils.load_image(self.image_path) 
        # if a guide image directory is specified - TODO: Consider removing guide image functionality if it isn't used frequently
        if hasattr(self, 'guide_image_dir'):
            guide_image_path = os.path.join(os.path.join(self.guide_image_dir, self.fname))
            # and a guide image is available for the current image.
            if os.path.isfile(guide_image_path):
                self.guide_img_data = im_utils.load_image(guide_image_path)
            else:
                pass
                # no guide image found for guide_image_path - it's optional anyway
        else:
            pass
            # no guide image directory found - it's optional anyway

        self.update_annot_and_seg()

        self.contrast_slider.update_range(self.img_data)
        self.update_window_title()

        # only segment if a segmentation is missing.
        if not os.path.isfile(self.get_seg_path()):
            print('no segmentation found at', self.get_seg_path())
            segment_full_image(self, self.fname) # segment current image.
        
        # segment the next image also.
        cur_im_idx = self.image_fnames.index(self.fname)
        if cur_im_idx < len(self.image_fnames):    
            next_im_fname = self.image_fnames[cur_im_idx+1]
            num_segmentations = len(self.get_all_seg_paths())
            # dont segment the next image early on in training (first 10 images).
            # We don't want the user to have to correct segmentations from old models at this point in training.
            if num_segmentations > 10 and not os.path.isfile(self.get_seg_path(next_im_fname)):
                segment_full_image(self, next_im_fname) 


        self.log(f'update_file_end,fname:{os.path.basename(fpath)},view_state:{self.view_state}')

    def navigate_to_top_of_structure(self, roi_connected_structure):
        # Used to speed things up when we first open an image, as this is where a user 
        # should start making the corrections.
        for i in range(roi_connected_structure.shape[0]):
            if np.any(roi_connected_structure[i]):
                slice_idx = (roi_connected_structure.shape[0] - i) - 1
                self.axial_viewer.slice_nav.slider.setValue(slice_idx)
                self.axial_viewer.slice_nav.value_changed()
                break


    def update_annot_and_seg(self):
        self.annot_path = None
        self.view_state = ViewState.LOADING_SEG 
        if self.fname:
            self.annot_path = get_annot_path(self.fname,
                                             self.get_train_annot_dir(),
                                             self.get_val_annot_dir())
        
        if self.annot_path and os.path.isfile(self.annot_path):
            self.annot_data = im_utils.load_annot(self.annot_path, self.img_data.shape)
        else:
            # otherwise create empty annotation array
            # if we are working with 3D data (npy file) and the
            # file hasn't been found then create an empty array to be
            # used for storing the annotation information.
            # channel for bg (0) and fg (1)
            self.annot_data = np.zeros([2] + list(self.img_data.shape))

        if self.fname and os.path.isfile(self.get_seg_path()):
            print('load seg')
            self.log(f'load_seg,fname:{os.path.basename(self.get_seg_path())}')
            self.seg_data = im_utils.load_seg(self.get_seg_path())
            self.view_state = ViewState.ANNOTATING
            self.update_segmentation()
        else:
            # it should come later
            self.seg_data = None
           
        for v in self.viewers:
            v.update_image()
            v.update_cursor()
            # hide the segmentation if we don't have it
            if self.seg_data is None and v.seg_visible:
                # show seg in order to show the loading message
                v.show_hide_seg()

        """
        if self.seg_data is not None:
            # assign bg annotations to regions that are not the largest
            import cc3d
            if not np.any(self.annot_data):
                print('run cc3d')
                labels_out = cc3d.connected_components(self.seg_data) # 26-connected
                voxel_counts = cc3d.statistics(labels_out)['voxel_counts']
                to_keep_idx = voxel_counts[1:].argmax()
                if len(voxel_counts) < 30:
                    for label_idx in range(len(voxel_counts)):
                        print(label_idx, 'of', len(voxel_counts))
                        if label_idx == to_keep_idx:
                            biggest_region_mask = labels_out == (to_keep_idx + 1)
                            roi_corrected_no_holes = binary_fill_holes(biggest_region_mask).astype(np.int)
                            roi_extra_fg = roi_corrected_no_holes - biggest_region_mask
                            self.annot_data[1][roi_extra_fg > 0] = 1 # set fg regions to remove holes.
                        else:
                            fg_region_label = label_idx + 1
                            mask = labels_out == fg_region_label
                            self.annot_data[0][mask] = 1 # set bg regions to remove extra components
                    self.update_viewer_annot_slice() # update view so next view change doesnt update the annot with previously displayed annot.
                    print('navigate to top')
                    self.navigate_to_top_of_structure(roi_corrected_no_holes)
                    print('done navigating to top')
                else:
                    print('Did not automatically correct disconnected regions as there are over 30') # it takes too long.
            else:
                print('found annot so not automatically remoing small holes and regions')
        """

    def update_class(self, class_name):
        # Save current annotation (if it exists) before moving on
        self.save_annotation()
        self.cur_class = class_name
        self.annot_path = get_annot_path(self.fname,
                                         self.get_train_annot_dir(),
                                         self.get_val_annot_dir())
        for v in self.viewers:
            v.scene.history = []
            v.scene.redo_list = []
        self.update_annot_and_seg()

    def get_seg_path(self, fname=None):
        if fname is None:
            fname = self.fname
        seg_fname = fname.replace('.nrrd', '.nii.gz')  # @TODO: How to handle this with .nii files?
        # just seg path for current class.
        if hasattr(self, 'classes') and len(self.classes) > 1:
            return os.path.join(self.seg_dir,
                                self.cur_class,
                                seg_fname)
        return os.path.join(self.seg_dir, seg_fname)


    def get_train_seg_path(self, fname=None):
        if fname is None:
            fname = self.fname
        seg_fname = fname.replace('.nrrd', '.nii.gz')  # @TODO: How to handle this with .nii files?
        if hasattr(self, 'classes') and len(self.classes) > 1:
            return os.path.join(self.proj_location,
                                'train_segmentations',
                                self.cur_class,
                                seg_fname)
        return os.path.join(self.proj_location, 'train_segmentations', seg_fname)


    def get_all_seg_paths(self):
        if hasattr(self, 'classes') and len(self.classes) > 1:
            spaths = []
            for c in self.classes:
                spaths.append(os.path.join(self.seg_dir,
                                self.cur_class,
                                self.fname))
            return spaths
        return [os.path.join(self.seg_dir, self.fname)]

    def update_segmentation(self):
        # if seg file is present then load.
        if os.path.isfile(self.get_seg_path()):
            self.seg_mtime = os.path.getmtime(self.get_seg_path())
            self.nav.next_image_button.setText('Save && Next >')
            self.nav.next_image_button.setEnabled(True)
        else:
            self.seg_mtime = None
            self.nav.next_image_button.setEnabled(False)
            self.nav.next_image_button.setText('Loading Segmentation...')

    def set_seg_loading(self):
        """ Transition to loading segmentation """
        self.seg_mtime = None
        self.nav.next_image_button.setEnabled(False)
        self.nav.next_image_button.setText('Loading Segmentation...')
        self.view_state = ViewState.LOADING_SEG
        for v in self.viewers:
            v.scene.last_x = None
            v.scene.last_y = None
            if not v.seg_visible:
                # show seg in order to show the loading message
                v.show_hide_seg()

    def show_open_project_widget(self):
        options = QtWidgets.QFileDialog.Options()
        default_loc = self.sync_dir / 'projects'
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load project file",
            str(default_loc),
            "Segmentation project file (*.seg_proj)",
            options=options)

        if file_path:
            self.open_project(file_path)

    def show_create_project_widget(self):
        print("Open the create project widget..")
        self.create_project_widget = CreateProjectWidget(self.sync_dir)
        self.create_project_widget.show()
        self.create_project_widget.created.connect(self.open_project)

    def init_missing_project_ui(self):
        ## Create project menu
        # project has not yet been selected or created
        # need to open minimal interface which allows users
        # to open or create a project.

        menu_bar = self.menuBar()
        self.menu_bar = menu_bar
        self.menu_bar.clear()
        self.project_menu = menu_bar.addMenu("Project")

        # Open project
        self.open_project_action = QtWidgets.QAction(QtGui.QIcon(""), "Open project", self)
        self.open_project_action.setShortcut("Ctrl+O")

        self.project_menu.addAction(self.open_project_action)
        self.open_project_action.triggered.connect(self.show_open_project_widget)

        # Create project
        self.create_project_action = QtWidgets.QAction(QtGui.QIcon(""), "Create project", self)
        self.create_project_action.setShortcut("Ctrl+C")
        self.project_menu.addAction(self.create_project_action)
        self.create_project_action.triggered.connect(self.show_create_project_widget)

        menus.add_help_menu(self, menu_bar)
        menus.add_extras_menu(self, menu_bar)

        # Add project btns to open window (so it shows something useful)
        project_btn_widget = QtWidgets.QWidget()
        self.setCentralWidget(project_btn_widget)

        layout = QtWidgets.QHBoxLayout()
        project_btn_widget.setLayout(layout)
        open_project_btn = QtWidgets.QPushButton('Open existing project')
        open_project_btn.clicked.connect(self.show_open_project_widget)
        layout.addWidget(open_project_btn)

        create_project_btn = QtWidgets.QPushButton('Create new project')
        create_project_btn.clicked.connect(self.show_create_project_widget)
        layout.addWidget(create_project_btn)

        self.setWindowTitle("RootPainter3D - Not approved for clinical use.")
        self.resize(layout.sizeHint())


    def show_license_window(self):
        self.license_window = LicenseWindow()
        self.license_window.show()

    def show_about_window(self):
        self.about_window = AboutWindow()
        self.about_window.show()

    def show_shortcut_window(self):
        self.shortcut_window = ShortcutWindow()
        self.shortcut_window.show()

    def update_window_title(self):
        proj_dirname = os.path.basename(self.proj_location)
        self.setWindowTitle(f"RootPainter3D {proj_dirname}"
                            f" {os.path.basename(self.image_path)}"
                            " - Not approved for clinical use")

    def closeEvent(self, _):
        if hasattr(self, 'proj_location'):
            delete_lock_files_for_current_user(self.proj_location)
        if hasattr(self, 'contrast_slider'):
            self.contrast_slider.close()
        if hasattr(self, 'sagittal_viewer'):
            self.sagittal_viewer.close()
        if hasattr(self, 'coronal_viewer'):
            self.coronal_viewer.close()


    def contrast_updated(self):
        self.update_viewer_image_slice()
        self.update_viewer_guide()


    def update_viewer_image_slice(self):
        for v in self.viewers:
            if v.isVisible():
                v.update_image_slice()

    def update_viewer_annot_slice(self):
        for v in self.viewers:
            if v.isVisible():
                v.update_annot_slice()

    def update_viewer_outline(self):
        for v in self.viewers:
            if v.isVisible():
                v.update_outline()

    def update_viewer_guide(self):
        for v in self.viewers:
            if v.isVisible():
                v.update_guide_image_slice()

    def before_nav_change(self):
        """
        I'm trying to make sure the user doesn't forget to remove
        disconnected regions.
        """
        if self.seg_data is None:
            button_reply = QtWidgets.QMessageBox.question(
                self,
                'Confirm',
                f"The segmentation has not yet loaded for this image. "
                "Are you sure you want to proceed to the next image?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
                QtWidgets.QMessageBox.No)
            return button_reply == QtWidgets.QMessageBox.Yes

        # return False to block nav change
        num_regions = im_utils.get_num_regions(self.seg_data,
                                               self.annot_data)
        if num_regions == 1:
            return True
        button_reply = QtWidgets.QMessageBox.question(
            self,
            'Confirm',
            f"There are {num_regions} regions in this image. "
            "Are you sure you want to proceed to the next image?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
            QtWidgets.QMessageBox.No)

        return button_reply == QtWidgets.QMessageBox.Yes

    def init_active_project_ui(self):
        # container for both nav and im_viewer.
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.container = container
        self.container_layout = container_layout
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        self.viewers_container = QtWidgets.QWidget()
        self.viewers_layout = QtWidgets.QHBoxLayout()
        self.viewers_container.setLayout(self.viewers_layout)

        self.axial_viewer = ImViewer(self, 'axial')
        self.sagittal_viewer = ImViewerWindow(self, 'sagittal')
        self.sagittal_viewer.show()
        
        #self.coronal_viewer = ImViewerWindow(self, 'coronal')
        self.viewers = [self.axial_viewer, self.sagittal_viewer] #, self.coronal_viewer]
        self.viewers_layout.addWidget(self.axial_viewer)

        container_layout.addWidget(self.viewers_container)
        self.contrast_slider = ContrastSlider(self.contrast_presets)
        self.contrast_slider.changed.connect(self.contrast_updated)

        self.nav = NavWidget(self.image_fnames,
                             self.classes,
                             self.before_nav_change)

        # bottom bar right
        bottom_bar_r = QtWidgets.QWidget()
        bottom_bar_r_layout = QtWidgets.QVBoxLayout()
        bottom_bar_r.setLayout(bottom_bar_r_layout)
        self.axial_viewer.bottom_bar_layout.addWidget(bottom_bar_r)
        self.hu_label = QtWidgets.QLabel()
        self.hu_label.setAlignment(Qt.AlignRight)
        bottom_bar_r_layout.addWidget(self.hu_label)

        
        # Nav
        self.nav.file_change.connect(self.update_file)
        self.nav.class_change.connect(self.update_class)
        self.nav.image_path = self.image_path
        self.nav.update_nav_label()

        # info label
        info_container = QtWidgets.QWidget()
        info_container_layout = QtWidgets.QHBoxLayout()
        info_container_layout.setAlignment(Qt.AlignCenter)
        info_label = QtWidgets.QLabel()
        info_label.setText("")
        info_container_layout.addWidget(info_label)
        # left, top, right, bottom
        info_container_layout.setContentsMargins(0, 0, 0, 0)
        info_container.setLayout(info_container_layout)
        self.info_label = info_label
        # add nav and info label to the axial viewer.
        bottom_bar_r_layout.addWidget(info_container)
        bottom_bar_r_layout.addWidget(self.nav)

        self.add_menu()
        self.resize(container_layout.sizeHint())
        self.axial_viewer.update_cursor()
        self.update_file(self.image_path)

        def view_fix():
            """ started as hack for linux bug.
                now used for setting defaults """
            self.axial_viewer.update_cursor()
            # These are causing issues on windows so commented out until I find a better solution
            # self.set_to_left_half_screen()
            # self.sagittal_viewer.set_to_right_half_screen()
            self.set_default_view_size()

        QtCore.QTimer.singleShot(100, view_fix)


    def set_default_view_size(self):
        # sensible defaults for CT scans
        self.axial_viewer.graphics_view.zoom = 2.4
        self.sagittal_viewer.graphics_view.zoom = 2.0
        self.axial_viewer.graphics_view.update_zoom()
        self.sagittal_viewer.graphics_view.update_zoom()

    def set_to_left_half_screen(self): 
        screen_shape = QtWidgets.QDesktopWidget().screenGeometry()
        w = screen_shape.width() // 2
        h = screen_shape.height()
        x = 0
        y = 0
        self.setGeometry(x, y, w, h)

    def track_changes(self):
        if self.tracking:
            return
        self.tracking = True
        def check():
            # check for any messages
            messages = os.listdir(str(self.message_dir))

            for m in messages:
                if hasattr(self, 'info_label'):
                    self.info_label.setText(m)
                try:
                    # Added try catch because this error happened (very rarely)
                    # PermissionError: [WinError 32]
                    # The process cannot access the file because `it is
                    # being used by another process
                    os.remove(os.path.join(self.message_dir, m))
                except Exception as e:
                    print('Caught exception when trying to detele msg', e)
                                
            # if a segmentation exists (on disk)
            if self.fname and os.path.isfile(self.get_seg_path()):
                try:
                    # seg mtime is not actually used any more.
                    new_mtime = os.path.getmtime(self.get_seg_path())

                    # seg_mtime is None before the seg is loaded.
                    if self.seg_mtime is None or new_mtime != self.seg_mtime:
                        print('load new seg now')
                        self.log(f'load_seg,fname:{os.path.basename(self.get_seg_path())}')
                        self.seg_data = im_utils.load_seg(self.get_seg_path())
                        self.axial_viewer.update_seg_slice()
                        # Change to annotating state.                        
                        self.view_state = ViewState.ANNOTATING
                        for v in self.viewers:
                            v.update_cursor()
                            # for some reason cursor doesn't update straight away sometimes.
                            # trigger again half a second later to make sure 
                            # the correct cursor is shown.
                            QtCore.QTimer.singleShot(500, v.update_cursor)
                            if v.isVisible():
                                v.update_seg_slice()

                        self.seg_mtime = new_mtime
                        self.nav.next_image_button.setText('Save && Next >')
                        self.nav.next_image_button.setEnabled(True)
                    else:
                        pass
           
                except Exception as e:
                    print(f'Exception loading segmentation,{e},{traceback.format_exc()}')
                    # sometimes problems reading file.
                    # don't worry about this exception
            else:
                pass
            QtCore.QTimer.singleShot(500, check)
        QtCore.QTimer.singleShot(500, check)

    def close_project_window(self):
        self.close()
        self.closed.emit()

    def add_menu(self):
        menu_bar = self.menuBar()
        menu_bar.clear()

        self.project_menu = menu_bar.addMenu("Project")
        # Open project
        self.close_project_action = QtWidgets.QAction(QtGui.QIcon(""), "Close project", self)
        self.project_menu.addAction(self.close_project_action)
        self.close_project_action.triggered.connect(self.close_project_window)
        menus.add_edit_menu(self, self.axial_viewer, menu_bar)


        #options_menu = menu_bar.addMenu("Options")

        self.menu_bar = menu_bar

        # add brushes menu for axial slice navigation
        menus.add_brush_menu(self.axial_viewer, self.menu_bar)

        # add view menu for axial slice navigation.
        view_menu = menus.add_view_menu(self, self.axial_viewer, self.menu_bar)
        self.add_contrast_setting_options(view_menu)

        menus.add_network_menu(self, self.menu_bar)
        menus.add_windows_menu(self)
        if len(self.classes) > 1:
            menus.add_class_menu(self, self.menu_bar)
        menus.add_help_menu(self, self.menu_bar)
        menus.add_extras_menu(self, menu_bar, project_open=True)


    def add_contrast_setting_options(self, view_menu):
        preset_count = 0
        for preset in self.contrast_presets:
            def add_preset_option(new_preset, preset_count):
                preset = new_preset
                preset_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                               f'{preset} contrast settings', self)
                preset_btn.setShortcut(QtGui.QKeySequence(f"Alt+{preset_count}"))
                preset_btn.setStatusTip(f'Use {preset} contrast settings')
                def on_select():
                    self.contrast_slider.preset_selected(preset)
                preset_btn.triggered.connect(on_select)
                view_menu.addAction(preset_btn)
            preset_count += 1
            add_preset_option(preset, preset_count)

    def stop_training(self):
        self.info_label.setText("Stopping training...")
        content = {"message_dir": self.message_dir}
        self.send_instruction('stop_training', content)

    def start_training(self):
        self.info_label.setText("Starting training...")
        # 3D just uses the name of the first class
        content = {
            "model_dir": self.model_dir,
            "dataset_dir": self.dataset_dir,
            "train_annot_dirs": self.train_annot_dirs,
            "train_seg_dirs": self.train_seg_dirs,
            "val_annot_dirs": self.val_annot_dirs,
            "seg_dir": self.seg_dir,
            "log_dir": self.log_dir,
            "message_dir": self.message_dir,
            "classes": self.classes
        }
        self.send_instruction('start_training', content)

    def save_annotation(self):
        if self.annot_data is not None:
            for v in self.viewers:
                v.store_annot_slice()
            fname = os.path.basename(self.get_seg_path())
            self.annot_path = maybe_save_annotation_3d(self.img_data.shape,
                                                       self.annot_data,
                                                       self.annot_path,
                                                       fname,
                                                       self.get_train_annot_dir(),
                                                       self.get_val_annot_dir(),
                                                       self.log)
            if self.annot_path:
                #if self.auto_complete_enabled:
                # also save the segmentation, as this updated due to patch updates (potencially).
                img = nib.Nifti1Image(self.seg_data.astype(np.int8), np.eye(4))
                img.to_filename(self.get_seg_path())
                # if annotation was saved to train 
                if str(self.get_train_annot_dir()) in self.annot_path:
                    # if there are at least 4 train annotations (happens after
                    # saving annotaiton for image 5) then save segmentation for
                    # training. The first 4 segmentations are not fully
                    # corrected per the new protocol from image 5 and onwards
                    # all segmentations will be saved for training as they are
                    # fully corrected.
                    if len(os.listdir(self.get_train_annot_dir())) >= 4:
                        img.to_filename(self.get_train_seg_path())
                else:
                    # otherwise if it was saved to validation then start training
                    # as we now believe there is training and validation data.
                    self.start_training()
