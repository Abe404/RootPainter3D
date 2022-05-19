"""
Show image in either sagittal or axial view

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

# Too many instance attributes
# pylint: disable=R0902
import sys

import qimage2ndarray
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

from view_state import ViewState
from graphics_scene import GraphicsScene
from graphics_view import CustomGraphicsView
from slice_nav import SliceNav
from visibility_widget import VisibilityWidget

import im_utils
import menus


class ImViewer(QtWidgets.QWidget):
    """ Show image, annotation and segmentation data and allow interaction
        Contains graphics scene which is where the image will be drawn
    """

    def __init__(self, parent, view_mode):
        super().__init__()
        self.annot_visible = True
        self.seg_visible = False
        self.image_visible = True
        self.guide_image_visible = False
        self.outline_visible = False
        self.image_pixmap_holder = None
        self.guide_image_pixmap_holder = None
        self.seg_pixmap_holder = None
        self.blank_pixmap = None
        self.black_pixmap = None
        self.seg_pixmap = None
        self.annot_pixmap_holder = None
        self.cur_slice_idx = None
        self.im_width = None
        self.im_height = None
        self.brush_color = None
        self.mode = view_mode # axial or sagittal
        self.parent = parent # lazy event handling.
        self.setMouseTracking(True) # track mouse over to remove cursor from pixmap
        self.init_ui()

    def activate(self):
        self.activateWindow()
        # important so pan works on first click
        self.graphics_view.setFocus()

    def init_ui(self):
        """ Create and assign settings for graphical components """

        scene = GraphicsScene(self)
        self.scene = scene
        
        self.graphics_view = CustomGraphicsView()
        self.graphics_view.zoom_change.connect(self.update_cursor)
        self.graphics_view.leave.connect(self.scene.clear_cursor)
        self.graphics_view.enter.connect(self.activate)
        self.outer_layout = QtWidgets.QVBoxLayout()
        if self.mode == 'axial':
            self.setLayout(self.outer_layout)

        self.inner_container = QtWidgets.QWidget()
        self.inner_layout = QtWidgets.QHBoxLayout()
        self.inner_container.setLayout(self.inner_layout)
        self.outer_layout.addWidget(self.inner_container)

        self.graphics_view.setScene(scene)
        self.graphics_view.mouse_scroll_event.connect(self.mouse_scroll)

        # Required so graphics scene can track mouse up when mouse is not pressed
        self.graphics_view.setMouseTracking(True)

        # bottom bar
        bottom_bar = QtWidgets.QWidget()
        self.bottom_bar_layout = QtWidgets.QHBoxLayout()
        self.bottom_bar_layout.setSpacing(0)
        bottom_bar.setLayout(self.bottom_bar_layout)
        self.outer_layout.addWidget(bottom_bar)

        # slice nav
        self.slice_nav = SliceNav()
        self.inner_layout.addWidget(self.slice_nav)
        self.slice_nav.changed.connect(self.update_slice_index)
        self.inner_layout.addWidget(self.graphics_view)
        if self.mode == 'axial':
            self.vis_widget = VisibilityWidget(QtWidgets.QVBoxLayout, self,
                                               show_guide=hasattr(self.parent, 'guide_image_dir'))
            self.vis_widget.setMaximumWidth(200)
            # left, top, right, bottom
            self.bottom_bar_layout.setContentsMargins(20, 0, 20, 0)
        else:
            self.vis_widget = VisibilityWidget(QtWidgets.QHBoxLayout, self,
                                               show_guide=hasattr(self.parent, 'guide_image_dir'))

            self.vis_widget.setMaximumWidth(500)
            # left, top, right, bottom
            self.bottom_bar_layout.setContentsMargins(120, 0, 0, 10)
        self.vis_widget.setMinimumWidth(200)
      
        self.bottom_bar_layout.addWidget(self.vis_widget)

    def set_color(self, _event, color=None):
        """ change the brush to a new color """
        self.brush_color = color
        self.update_cursor()

    def seg_checkbox_change(self, state):
        """ set checkbox to specified state and update visibility if required """
        checked = (state == QtCore.Qt.Checked)
        if checked is not self.seg_visible:
            self.show_hide_seg()

    def annot_checkbox_change(self, state):
        """ set checkbox to specified state and update visibility if required """
        checked = (state == QtCore.Qt.Checked)
        if checked is not self.annot_visible:
            self.show_hide_annot()

    def im_checkbox_change(self, state):
        """ set checkbox to specified state and update visibility if required """
        checked = (state == QtCore.Qt.Checked)
        if checked is not self.image_visible:
            self.show_hide_image()

    def outline_checkbox_change(self, state):
        """ set checkbox to specified state and update visibility if required """
        checked = (state == QtCore.Qt.Checked)
        if checked is not self.outline_visible:
            self.show_hide_outline()

    def guide_checkbox_change(self, state):
        """ set checkbox to specified state and update visibility if required """
        checked = (state == QtCore.Qt.Checked)
        if checked is not self.guide_image_visible:
            self.show_hide_guide_image()

    def store_annot_slice(self):
        if self.scene.annot_pixmap:
            im_utils.store_annot_slice(self.scene.annot_pixmap,
                                       self.parent.annot_data,
                                       self.cur_slice_idx,
                                       self.mode)

    def update_slice_index(self):
        """ Render the new slice as the slice index may have changed """
        self.parent.log(f'update_slice,mode:{self.mode},'
                        f'view_state:{self.parent.view_state},'
                        f'new_slice:{self.slice_nav.slice_idx},'
                        f'zoom:{self.graphics_view.zoom}')

        # As we have likley moved to a new slice. The history needs erasing.
        # to avoid undo taking us back to prevous states of another slice.
        if len(self.scene.history) > 0:
            # before updating the slice idx, store current annotation information
            if hasattr(self.scene, 'annot_pixmap'):
                self.parent.annot_data = im_utils.store_annot_slice(self.scene.annot_pixmap,
                                                                    self.parent.annot_data,
                                                                    self.cur_slice_idx,
                                                                    self.mode)
            self.scene.history = []
            self.scene.redo_list = []

        self.cur_slice_idx = self.slice_nav.slice_idx
        
        # update image, seg and annot at current slice.
        self.update_image_slice()
        self.update_guide_image_slice()

        self.update_seg_slice()
        self.update_annot_slice()
        self.update_outline()

        self.scene.update_axial_slice_pos_indicator()

    def keyPressEvent(self, event):
        # if control key is pressed then hide the paint brush as the user wants to pan now.
        if event.key() == 16777249:
            self.scene.setFocus()
            self.graphics_view.setFocus()
            # need to manually set ctrl_press to true to tell the cursor how to look
            self.update_cursor(ctrl_press=True)

    def mouse_scroll(self, event):
        """ handle mouse scroll event. zoom, change slice or change brush size """
        scroll_up = event.angleDelta().y() > 0
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        alt_down = (modifiers & QtCore.Qt.AltModifier)
        shift_down = (modifiers & QtCore.Qt.ShiftModifier)
        ctrl_down = (modifiers & QtCore.Qt.ControlModifier)

        if alt_down or shift_down:
            # change by 10% (nearest int) or 1 (min)
            increment = max(1, int(round(self.scene.brush_size / 10)))
            if scroll_up:
                self.scene.brush_size += increment
            else:
                self.scene.brush_size -= increment
            self.scene.brush_size = max(1, self.scene.brush_size)
            self.parent.log(f'update_brush_size,brush_size:{self.scene.brush_size}')
            self.update_cursor()

        elif ctrl_down:
            if scroll_up:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            if scroll_up:
                self.slice_nav.slider.setValue(self.slice_nav.slice_idx + 1)
            else:
                self.slice_nav.slider.setValue(self.slice_nav.slice_idx - 1)

    def zoom_in(self):
        self.graphics_view.zoom *= 1.1
        self.parent.log(f'update_zoom_scroll,zoom:{self.graphics_view.zoom}')
        self.graphics_view.update_zoom()

    def zoom_out(self):
        self.graphics_view.zoom /= 1.1
        self.parent.log(f'update_zoom_scroll,zoom:{self.graphics_view.zoom}')
        self.graphics_view.update_zoom()

    def show_hide_image(self):
        """ show or hide the current image.
            Could be useful to help inspect the
            segmentation or annotation """
        if self.image_visible:
            self.image_pixmap_holder.setPixmap(self.black_pixmap)
            self.image_visible = False
        else:
            self.image_pixmap_holder.setPixmap(self.graphics_view.image)
            self.image_visible = True
        self.vis_widget.im_checkbox.setChecked(self.image_visible)

    def show_hide_guide_image(self):
        """ show or hide the guide image.
            guide image can be used to help the annotator by providing another modality to inspect etc """
        if hasattr(self.graphics_view, 'guide_image'):
            if self.guide_image_visible:
                self.guide_image_pixmap_holder.setPixmap(self.blank_pixmap)
                self.guide_image_visible = False
            else:
                self.guide_image_pixmap_holder.setPixmap(self.graphics_view.guide_image)
                self.guide_image_visible = True
            self.vis_widget.guide_image_checkbox.setChecked(self.guide_image_visible)

    def show_hide_annot(self):
        """ show or hide the current annotations.
            Could be useful to help inspect the background image
        """
        if self.annot_visible:
            self.annot_pixmap_holder.setPixmap(self.blank_pixmap)
            self.annot_visible = False
        else:
            self.scene.annot_pixmap_holder.setPixmap(self.scene.annot_pixmap)
            self.annot_visible = True
        self.vis_widget.annot_checkbox.setChecked(self.annot_visible)

    def show_hide_outline(self):
        """ show or hide the current outline.
            Could be useful to help inspect the
            segmentation or annotation """
        if self.outline_visible:
            self.scene.outline_pixmap_holder.setPixmap(self.blank_pixmap)
            self.outline_visible = False
        else:
            self.scene.outline_pixmap_holder.setPixmap(self.scene.outline_pixmap)
            self.outline_visible = True
        self.vis_widget.outline_checkbox.setChecked(self.outline_visible)

    def show_hide_seg(self):
        """ show or hide the current segmentation. """
        if self.seg_visible:
            self.seg_pixmap_holder.setPixmap(self.blank_pixmap)
            self.seg_visible = False
        else:
            self.seg_pixmap_holder.setPixmap(self.seg_pixmap)
            self.seg_visible = True
        self.vis_widget.seg_checkbox.setChecked(self.seg_visible)

    def update_image(self):
        """ show a new image, annotation and segmentation """
        self.slice_nav.update_range(self.parent.img_data, self.mode)
        # used for saving the edited annotation information
        # before changing slice
        self.cur_slice_idx = self.slice_nav.slice_idx

        # clear history when moving to a new image
        self.scene.history = []
        self.scene.redo_list = []

        # render image, seg and annot at current slice.
        self.update_image_slice()
        self.update_guide_image_slice()

        self.update_seg_slice()
        self.update_annot_slice()
        self.update_outline()

        # draw cursor showing what will be painted
        if self.scene.cursor_pixmap == None: 
            self.scene.cursor_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
            self.scene.cursor_pixmap.fill(Qt.transparent)
            self.scene.cursor_pixmap_holder = self.scene.addPixmap(self.scene.cursor_pixmap)

    def fill_slice(self):
        self.scene.annot_pixmap = im_utils.fill_annot(self.scene.annot_pixmap)
        self.scene.annot_pixmap_holder.setPixmap(self.scene.annot_pixmap)
        # Store updated slice
        self.parent.annot_data = im_utils.store_annot_slice(self.scene.annot_pixmap,
                                                            self.parent.annot_data,
                                                            self.cur_slice_idx,
                                                            self.mode)
        self.scene.history.append(self.scene.annot_pixmap.copy())
        # update the other views
        self.parent.update_viewer_annot_slice()
        self.parent.update_viewer_outline()
    
    def update_image_slice(self):
        """ show the already loaded image we are looking at, at the specific
            slice the user has specified and using the specific
            user specified contrast settings.
        """
        img_slice = im_utils.get_slice(self.parent.img_data, self.slice_nav.slice_idx, self.mode)
        img_slice = im_utils.norm_slice(img_slice,
                                        self.parent.contrast_slider.min_value,
                                        self.parent.contrast_slider.max_value,
                                        self.parent.contrast_slider.brightness_value)
        q_image = qimage2ndarray.array2qimage(img_slice)
        image_pixmap = QtGui.QPixmap.fromImage(q_image)
        im_size = image_pixmap.size()
        im_width, im_height = im_size.width(), im_size.height()
        assert im_width > 0
        assert im_height > 0
        self.graphics_view.image = image_pixmap # for resize (and re-assignment after re-showing) later
        self.im_width = im_width
        self.im_height = im_height
        self.scene.setSceneRect(-15, -15, im_width+30, im_height+30)

        # Used to replace the segmentation or annotation when they are not visible.
        self.blank_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
        self.blank_pixmap.fill(Qt.transparent)

        self.black_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
        self.black_pixmap.fill(Qt.black)
        
        if self.image_pixmap_holder:
            self.image_pixmap_holder.setPixmap(image_pixmap)
        else:
            self.image_pixmap_holder = self.scene.addPixmap(image_pixmap)
            # interpolate the image pixels to make it look less pixelated
            self.image_pixmap_holder.setTransformationMode(Qt.SmoothTransformation)
        if not self.image_visible:
            self.image_pixmap_holder.setPixmap(self.black_pixmap)

    def mouseMoveEvent(self, event):
        if self.scene.cursor_shown:
            self.scene.cursor_pixmap.fill(Qt.transparent)
            self.scene.cursor_pixmap_holder.setPixmap(self.scene.cursor_pixmap)
            self.scene.cursor_shown = False

    def update_seg_slice(self):
        """ render the slice for the current index
            or show a loading message if it doesn't exist """
        if self.parent.seg_data is not None:
            seg_slice = im_utils.get_slice(self.parent.seg_data,
                                           self.slice_nav.slice_idx,
                                           self.mode)
            self.seg_pixmap = im_utils.seg_slice_to_pixmap(seg_slice)
            self.scene.seg_pixmap = self.seg_pixmap

        else:
            # otherwise use blank
            self.seg_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
            self.seg_pixmap.fill(Qt.transparent)
            painter = QtGui.QPainter()
            painter.begin(self.seg_pixmap)
            font = QtGui.QFont()
            font.setPointSize(48)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255), Qt.SolidPattern))
            if sys.platform == 'win32':
                # For some reason the text has a different size
                # and position on windows
                # so change the background rectangle also.
                painter.drawRect(0, 0, 657, 75)
            else:
                painter.drawRect(10, 10, 465, 55)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 150)))
            painter.drawText(16, 51, 'Loading segmentation')
            painter.end()

        if self.seg_pixmap_holder:
            self.seg_pixmap_holder.setPixmap(self.seg_pixmap)
            self.scene.seg_pixmap = self.seg_pixmap
        else:
            self.seg_pixmap_holder = self.scene.addPixmap(self.seg_pixmap)
        if not self.seg_visible:
            self.seg_pixmap_holder.setPixmap(self.blank_pixmap)


    def update_cursor(self, ctrl_press=False):
        """ render cursor based on zoom level,
            brush color and brush size """
        self.scene.update_cursor(ctrl_press)

        if self.parent.view_state == ViewState.ANNOTATING:
            brush_w = self.scene.brush_size * self.graphics_view.zoom * 0.93
            brush_w = round(max(brush_w, 3))
            canvas_w = round(max(brush_w, 30))
            pixel_map = QtGui.QPixmap(canvas_w, canvas_w)
            pixel_map.fill(Qt.transparent)
            painter = QtGui.QPainter(pixel_map)
            painter.drawPixmap(canvas_w, canvas_w, pixel_map)
            brush_rgb = self.brush_color.toRgb()
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 2,
                                      Qt.SolidLine, Qt.FlatCap))
            # Draw black to show where cursor is even when brush is small
            painter.drawLine(0, round(canvas_w/2), canvas_w*2, round(canvas_w/2))
            painter.drawLine(round(canvas_w/2), 0, round(canvas_w/2), canvas_w*2)
            painter.end()
            cursor = QtGui.QCursor(pixel_map)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.setCursor(cursor)

    def update_annot_slice(self):
        """ Update the annotation the user views """
        # if it's 3d then get the slice from the loaded numpy array
        annot_slice = im_utils.get_slice(self.parent.annot_data,
                                         self.slice_nav.slice_idx, self.mode)
        self.scene.annot_pixmap = im_utils.annot_slice_to_pixmap(annot_slice)


        # if annot_pixmap_holder is ready then setPixmap
        if self.annot_pixmap_holder:
            self.annot_pixmap_holder.setPixmap(self.scene.annot_pixmap)
        else:
            # or create the holder with the loaded pixmap
            self.annot_pixmap_holder = self.scene.addPixmap(self.scene.annot_pixmap)
        self.scene.annot_pixmap_holder = self.annot_pixmap_holder

        # so we can go back to the first stage of history
        if not self.scene.history:
            self.scene.history.append(self.scene.annot_pixmap.copy())
        if not self.annot_visible:
            self.annot_pixmap_holder.setPixmap(self.blank_pixmap)

    def update_outline(self):
        """ outline includes the segmentation corrected by the annotation
            and then just a line around the outside """
        if self.parent.seg_data is not None:
            seg_slice = im_utils.get_slice(self.parent.seg_data,
                                           self.slice_nav.slice_idx,
                                           self.mode)
            annot_slice = im_utils.get_slice(self.parent.annot_data,
                                             self.slice_nav.slice_idx, self.mode)
            self.scene.outline_pixmap = im_utils.get_outline_pixmap(seg_slice, annot_slice)
        else:
            self.scene.outline_pixmap = self.blank_pixmap
        

        # if outline_pixmap_holder is ready then setPixmap
        if self.scene.outline_pixmap_holder:
            self.scene.outline_pixmap_holder.setPixmap(self.scene.outline_pixmap)
        else:
            # or create the holder with the loaded pixmap
            self.scene.outline_pixmap_holder = self.scene.addPixmap(self.scene.outline_pixmap)
            self.scene.outline_pixmap_holder.setTransformationMode(Qt.SmoothTransformation)

        if not self.outline_visible:
            self.scene.outline_pixmap_holder.setPixmap(self.blank_pixmap)

    def update_guide_image_slice(self):
        """ guide shows another image on top of the main image """
        if hasattr(self.parent, 'guide_img_data'):
            img_slice = im_utils.get_slice(self.parent.guide_img_data, self.slice_nav.slice_idx, self.mode)
            img_slice = im_utils.norm_slice(img_slice,
                                            self.parent.contrast_slider.min_value,
                                            self.parent.contrast_slider.max_value,
                                            self.parent.contrast_slider.brightness_value)
            q_image = qimage2ndarray.array2qimage(img_slice)
            guide_image_pixmap = QtGui.QPixmap.fromImage(q_image)
            self.graphics_view.guide_image = guide_image_pixmap # for re-assignment after re-showing

            if self.guide_image_pixmap_holder:
                self.guide_image_pixmap_holder.setPixmap(guide_image_pixmap)
            else:
                self.guide_image_pixmap_holder = self.scene.addPixmap(guide_image_pixmap)
                # interpolate the image pixels to make it look less pixelated
                self.guide_image_pixmap_holder.setTransformationMode(Qt.SmoothTransformation)
            if not self.guide_image_visible:
                self.guide_image_pixmap_holder.setPixmap(self.blank_pixmap)


class ImViewerWindow(QtWidgets.QMainWindow, ImViewer):
    """ seperate viewer that has a menu in its own window """

    def init_ui(self):
        """ Create and assign settings for graphical components """
        super().init_ui()
        menus.add_view_menu(self, self, self.menuBar())
        menus.add_edit_menu(self, self, self.menuBar(), skip_fill=self.mode == 'sagittal')
        menus.add_brush_menu(self, self.menuBar())

        # This widget is designed to be a standalone window
        # so a central widget should be assigned.
        self.container = QtWidgets.QWidget()
        self.setCentralWidget(self.container)
        self.container.setLayout(self.outer_layout)
        self.setWindowTitle("Not Approved for clinical use. RootPainter3D - " + self.mode + 'view')

    def set_to_right_half_screen(self): 
        screenShape = QtWidgets.QDesktopWidget().screenGeometry()
        w = screenShape.width() // 2
        h = screenShape.height()
        x = screenShape.width() // 2
        y = 0
        self.setGeometry(x, y, w, h)
