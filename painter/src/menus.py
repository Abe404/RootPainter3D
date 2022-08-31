from functools import partial

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from segment_folder import SegmentFolderWidget
from segment import segment_full_image

def add_network_menu(window, menu_bar):
    """ Not in use right now as training happens automatically when the 
        annotation is saved
    """
    network_menu = menu_bar.addMenu('Network')

    # start training
    start_training_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Start training', window)
    start_training_btn.triggered.connect(window.start_training)
    network_menu.addAction(start_training_btn)

    # stop training
    stop_training_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Stop training', window)
    stop_training_btn.triggered.connect(window.stop_training)
    network_menu.addAction(stop_training_btn)

    # segment folder
    segment_folder_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Segment folder', window)

    # Alt+S
    segment_action = QtWidgets.QAction(QtGui.QIcon(""), "Segment full image", window)
    segment_action.setShortcut("Alt+S")
    network_menu.addAction(segment_action)
    def seg_im(window):
        segment_full_image(window, overwrite=True)
    segment_action.triggered.connect(partial(seg_im, window))

    def show_segment_folder():
        window.segment_folder_widget = SegmentFolderWidget(window.sync_dir,
                                                         window.instruction_dir,
                                                         window.classes)
        window.segment_folder_widget.show()
    segment_folder_btn.triggered.connect(show_segment_folder)
    network_menu.addAction(segment_folder_btn)


def add_edit_menu(window, im_viewer, menu_bar, skip_fill=True):
    edit_menu = menu_bar.addMenu("Edit")

    # Undo
    undo_action = QtWidgets.QAction(QtGui.QIcon(""), "Undo", window)
    undo_action.setShortcut("Z")
    edit_menu.addAction(undo_action)
    undo_action.triggered.connect(im_viewer.scene.undo)

    # Redo
    redo_action = QtWidgets.QAction(QtGui.QIcon(""), "Redo", window)
    redo_action.setShortcut("Ctrl+Shift+Z")
    edit_menu.addAction(redo_action)
    redo_action.triggered.connect(im_viewer.scene.redo)

    # Save annotation
    save_annotation_action = QtWidgets.QAction(QtGui.QIcon(""), "Save annotation", window)
    save_annotation_action.setShortcut("Ctrl+Shift+Z")
    edit_menu.addAction(save_annotation_action)
    save_annotation_action.triggered.connect(im_viewer.parent.save_annotation)

    if not skip_fill:
        raise Exception('disabled')
        # skip the fill with the sagittal view, it's too annoying when this gets pressed by accident

        # Using alt key and clicking is slow when wanting to fill a 
        # closed foreground annotation with foreground and background with background.
        fill_slice_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                    'Fill slice', window)
        fill_slice_btn.setShortcut('G')
        fill_slice_btn.setStatusTip('Fill slice')
        fill_slice_btn.triggered.connect(im_viewer.fill_slice)
        edit_menu.addAction(fill_slice_btn)

    return edit_menu


def add_windows_menu(main_window):
    # contrast slider
    menu = main_window.menu_bar.addMenu("Windows")
    contrast_settings_action = QtWidgets.QAction(QtGui.QIcon(""), "Contrast settings", main_window)
    contrast_settings_action.setStatusTip('Show contrast settings')
    contrast_settings_action.triggered.connect(main_window.contrast_slider.show)
    menu.addAction(contrast_settings_action)

    show_sagittal_view_action = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Sagittal View', main_window)
    show_sagittal_view_action.setStatusTip('Show sagittal view')
    show_sagittal_view_action.triggered.connect(main_window.sagittal_viewer.show)
    menu.addAction(show_sagittal_view_action)

    #show_coronal_view_action = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Coronal View', main_window)
    #show_coronal_view_action.setStatusTip('Show coronal view')
    #show_coronal_view_action.triggered.connect(main_window.coronal_viewer.show)
    #menu.addAction(show_coronal_view_action)


def add_brush_menu(im_viewer, menu_bar):
    brush_menu = menu_bar.addMenu("Brushes")

    def add_brush(name, color_val, shortcut=None):
        color_action = QtWidgets.QAction(QtGui.QIcon(""), name, im_viewer)
        if shortcut:
            color_action.setShortcut(shortcut)
        brush_menu.addAction(color_action)
        color_action.triggered.connect(partial(im_viewer.set_color,
                                               color=QtGui.QColor(*color_val)))
        if im_viewer.brush_color is None:
            im_viewer.brush_color = QtGui.QColor(*color_val)
    # These don't need to be modified. Each class has background and foreground
    brushes = [
        ('Background', (0, 255, 0, 180), 'W'),
        ('Foreground', (255, 0, 0, 180), 'Q'),
        ('Eraser', (255, 205, 180, 0), 'E')
    ]
    for name, rgba, shortcut in brushes:
        add_brush(name, rgba, shortcut)


def add_class_menu(self, menu_bar):
    class_menu = menu_bar.addMenu('Classes')
    for i, class_name in enumerate(self.classes):
        class_action = QtWidgets.QAction(QtGui.QIcon('missing.png'), 
                                            class_name, self)
        # update class via nav to keep nav up to date whilst avoiding a double update.
        class_action.triggered.connect(partial(self.nav.cb.setCurrentIndex,
                                            self.classes.index(class_name)))
        class_action.setShortcut(str(i+1))
        class_menu.addAction(class_action)


def add_help_menu(self,  menu_bar):
    help_menu = menu_bar.addMenu('Help')
    license_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'License', self)
    license_btn.triggered.connect(self.show_license_window)
    help_menu.addAction(license_btn)

    about_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'About', self)
    about_btn.triggered.connect(self.show_about_window)
    help_menu.addAction(about_btn)
    
    shortcut_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Keyboard Shortcuts', self)
    shortcut_btn.triggered.connect(self.show_shortcut_window)
    help_menu.addAction(shortcut_btn)
    

def add_extras_menu(main_window, menu_bar, project_open=False):
    extras_menu = menu_bar.addMenu('Extras')

    if project_open:
        extend_dataset_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Extend dataset', main_window)
        def update_dataset_after_check():
            was_extended, file_names = check_extend_dataset(main_window,
                                                            main_window.dataset_dir,
                                                            main_window.image_fnames,
                                                            main_window.proj_file_path)
            if was_extended:
                main_window.image_fnames = file_names
                main_window.nav.all_fnames = file_names
                main_window.nav.update_nav_label()
        extend_dataset_btn.triggered.connect(update_dataset_after_check)
        extras_menu.addAction(extend_dataset_btn)


def check_extend_dataset(main_window, dataset_dir, prev_fnames, proj_file_path):

    all_image_names = [f for f in os.listdir(dataset_dir) if is_image(f)]

    new_image_names = [f for f in all_image_names if f not in prev_fnames]

    button_reply = QtWidgets.QMessageBox.question(main_window,
        'Confirm',
        f"There are {len(new_image_names)} new images in the dataset."
        " Are you sure you want to extend the project to include these new images?",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
        QtWidgets.QMessageBox.No)

    if button_reply == QtWidgets.QMessageBox.Yes:
        # shuffle the new file names
        shuffle(new_image_names)
        # load the project json for reading and writing
        settings = json.load(open(proj_file_path, 'r'))
        # read the file_names
        all_file_names = settings['file_names'] + new_image_names
        settings['file_names'] = all_file_names

        # Add the new_files to the list
        # then save the json again
        json.dump(settings, open(proj_file_path, 'w'), indent=4)
        return True, all_file_names
    else:
        return False, all_image_names


def add_view_menu(window, im_viewer, menu_bar):
    """ Create view menu with options for
        * fit to view
        * actual size
        * toggle segmentation visibility
        * toggle annotation visibility
        * toggle image visibility
    """
    view_menu = menu_bar.addMenu('View')

    # Fit to view
    fit_to_view_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Fit to View', window)
    fit_to_view_btn.setShortcut('Ctrl+F')
    fit_to_view_btn.setStatusTip('Fit image to view')
    fit_to_view_btn.triggered.connect(im_viewer.graphics_view.fit_to_view)
    view_menu.addAction(fit_to_view_btn)

    # Actual size
    actual_size_view_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Actual size', window)
    actual_size_view_btn.setShortcut('Ctrl+A')
    actual_size_view_btn.setStatusTip('Show image at actual size')
    actual_size_view_btn.triggered.connect(im_viewer.graphics_view.show_actual_size)
    actual_size_view_btn.triggered.connect(im_viewer.graphics_view.show_actual_size)
    view_menu.addAction(actual_size_view_btn)

    # toggle segmentation visibility
    toggle_seg_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                  'Toggle segmentation visibility', window)
    toggle_seg_visibility_btn.setShortcut('S')
    toggle_seg_visibility_btn.setStatusTip('Show or hide segmentation')
    toggle_seg_visibility_btn.triggered.connect(im_viewer.show_hide_seg)
    view_menu.addAction(toggle_seg_visibility_btn)

    # toggle annotation visibility
    toggle_annot_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                    'Toggle annotation visibility', window)
    toggle_annot_visibility_btn.setShortcut('A')
    toggle_annot_visibility_btn.setStatusTip('Show or hide annotation')
    toggle_annot_visibility_btn.triggered.connect(im_viewer.show_hide_annot)
    view_menu.addAction(toggle_annot_visibility_btn)

    # toggle image visibility
    toggle_image_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                    'Toggle image visibility', window)
    toggle_image_visibility_btn.setShortcut('I')
    toggle_image_visibility_btn.setStatusTip('Show or hide image')
    toggle_image_visibility_btn.triggered.connect(im_viewer.show_hide_image)
    view_menu.addAction(toggle_image_visibility_btn)


    toggle_guide_image_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                    'Toggle guide image visibility', window)
    toggle_guide_image_visibility_btn.setShortcut('H')
    toggle_guide_image_visibility_btn.setStatusTip('Show or hide image')
    toggle_guide_image_visibility_btn.triggered.connect(im_viewer.show_hide_guide_image)
    view_menu.addAction(toggle_guide_image_visibility_btn)


    # toggle outline visibility
    toggle_outline_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                      'Toggle outline visibility', window)
    toggle_outline_visibility_btn.setShortcut('T')
    toggle_outline_visibility_btn.setStatusTip('Show or hide outline')
    toggle_outline_visibility_btn.triggered.connect(im_viewer.show_hide_outline)
    view_menu.addAction(toggle_outline_visibility_btn)

    up_slice_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                 'Move slice up', window)
    up_slice_btn.setShortcut('R')
    up_slice_btn.setStatusTip('move slice up')
    up_slice_btn.triggered.connect(im_viewer.slice_nav.up_slice)
    view_menu.addAction(up_slice_btn)

    down_slice_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                   'Move slice down', window)
    down_slice_btn.setShortcut('F')
    down_slice_btn.setStatusTip('move slice down')
    down_slice_btn.triggered.connect(im_viewer.slice_nav.down_slice)
    view_menu.addAction(down_slice_btn)

    # zoom in (V)
    zoom_in_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                'Zoom in', window)
    zoom_in_btn.setShortcut('V')
    zoom_in_btn.setStatusTip('Zoom in')
    zoom_in_btn.triggered.connect(im_viewer.zoom_in)
    view_menu.addAction(zoom_in_btn)

    # zoom in (V)
    zoom_out_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                'Zoom out', window)
    zoom_out_btn.setShortcut('C')
    zoom_out_btn.setStatusTip('Zoom out')
    zoom_out_btn.triggered.connect(im_viewer.zoom_out)
    view_menu.addAction(zoom_out_btn)

    return view_menu
