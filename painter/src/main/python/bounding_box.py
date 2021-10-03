"""
Copyright (C) 2020 Abraham George Smith

BoundingBox provides a way to specify a subregion of an image.

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
import math
import json

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import nibabel as nib
import numpy as np

from view_state import ViewState

def dimension_offsets(box_start, box_size, patch_input_size, patch_output_size):
    """
    Input:
        box_start - position relative to original image
                    e.g x, y or z of the bounding box
        
        box_size - the depth, height or width of the specified bounding box

        patch_input_size, patch_output_size - 
            Used to calculate the difference between the
            input and output of the cnn for this particular dimension.
            This determines the minimum padding required to obtain a particular
            output. The input size is also used to determine the padding
            if the padded image is smaller than cnn_input_size

    Returns:
        start, end - 
            what to take from the image (including the padding)
        pad_start, pad_end - 
            how much padding at either side of the bounding box for this
            dimension. padding may be larger than than the minimum required by
            the difference in input and output size if the resultant image
            is less than cnn_input_size. This is because we always require
            the input to be at least as big as cnn_input_size.
            We need to keep track of this information to strip the
            excess padding from the output.
    """
    # should be even to ensure equal padding each side
    input_size_diff = patch_input_size - patch_output_size
    assert input_size_diff % 2 == 0, input_size_diff 
    pad_end = pad_start = input_size_diff // 2

    # add 1 to box_size to include both top and bottom slice of 
    # bounding box in output
    box_size += 1

    # -1 as I want to top slice also
    start = (box_start - pad_start)
    end = box_start + box_size + pad_end
    input_size = end - start
    
    if input_size < patch_input_size:
        pad_extra = patch_input_size - input_size

        # split the required extra between start and end 
        pad_start += (pad_extra // 2)
        pad_end += pad_extra - (pad_extra // 2)
        
        # recompute input size 
        start = box_start - pad_start
        end = box_start + box_size + pad_end
        input_depth = end - start
        
        assert input_depth >= patch_input_size, f"{input_depth}, {patch_input_size}"

    return start, end, pad_start, pad_end
    

def define_bounding_box(root_painter):
    """ This will delete the existing annotation and segmentation
        set view state to bounding box
        and then call update_file again so that 
        the bounding box is displayed.

        A problem with deleating and readding annotations
        is that you can potentially end up with an image move
        from train to validation or vice-versa.
    """
    message_box = QtWidgets.QMessageBox
    ret = message_box.question(root_painter,'',
                               "Redrawing the bounding box will delete any existing "
                               "annotation or segmentation files for this image. "
                               "Are you sure you want to do redraw the bounding box?",
                               message_box.Yes | message_box.No)
    if ret == message_box.Yes:
        if root_painter.annot_path:
            if os.path.exists(root_painter.annot_path):
                os.remove(root_painter.annot_path)
            root_painter.annot_path = None
        if root_painter.seg_path:
            if os.path.exists(root_painter.seg_path):
                os.remove(root_painter.seg_path)
            root_painter.seg_path = None
        if root_painter.bounded_fname:
            bounded_fpath = os.path.join(root_painter.proj_location,
                                        'bounded_images',
                                         root_painter.bounded_fname)
            if os.path.exists(bounded_fpath):
                os.remove(bounded_fpath)
            root_painter.bounded_fname = None
        root_painter.view_state = ViewState.BOUNDING_BOX
        root_painter.update_file(root_painter.image_path)
    

def resegment_current_image(root_painter):
    return
    message_box = QtWidgets.QMessageBox
    ret = message_box.question(root_painter,'',
                               "Applying the bounding box again will delete the existing "
                               "segmentation and segment the image again with the latest model. "
                               "Are you sure you want to do apply the bounding box?",
                               message_box.Yes | message_box.No)
    if ret == message_box.Yes:

        spaths = root_painter.get_all_seg_paths()
        # delete all existing segmentations.
        for spath in spaths:
            if os.path.isfile(spath):
                print('deleting existing ', spath)
                os.remove(spath)

        bounded_im_dir = os.path.join(root_painter.proj_location, 'bounded_images')
        root_painter.set_seg_loading()
        root_painter.log(f'resegment,bounded_fname:{root_painter.bounded_fname}')
        # send instruction to segment the new image.
        root_painter.send_instruction('segment', {
            "dataset_dir": bounded_im_dir,
            "seg_dir": root_painter.seg_dir,
            "file_names": [root_painter.bounded_fname],
            "message_dir": root_painter.message_dir,
            "model_dir": root_painter.model_dir,
            "classes": root_painter.classes,
            "overwrite": True
        })
        root_painter.track_changes()


def apply_bounding_box(root_painter, full_size):
    """ Save the bounded image and show loading icon
        as the segmentation will now be loading """


    box = root_painter.box

    # if the segmentation already exists then give the user the option toresegment the current image.
    if root_painter.bounded_fname and os.path.isfile(root_painter.get_seg_path()):
        print('resegment current image')
        resegment_current_image(root_painter)
        return

    im_shape = root_painter.img_data.shape 

    if full_size:
        x = 0
        y = 0
        z = 0
        width = im_shape[2] - 1
        height = im_shape[1] - 1
        depth = im_shape[0] - 1

    else:

        if not box['visible']:
            return
        root_painter.log(f'apply_box,box:{json.dumps(box)}')
        x = box['x']
        y = box['y']
        z = box['z']
        depth = box['depth']
        height = box['height']
        width = box['width']

        # if it's a bit outside the image then just adjust to only include the bit
        # covered by the image

        if x < 0:
            width += x # substract 
            x += -x # and shift accross
        if y < 0:
            height += y # substract 
            y += -y # and shift accross
        if z < 0:
            depth += x # substract 
            z += -z # and shift accross

        # make sure depth height and width don't go outside the box
        depth -= max(0, (z + depth) - (im_shape[0] - 1))
        height -= max(0, (y + height) - (im_shape[1] - 1))
        width -= max(0, (x + width) - (im_shape[2] - 1))

        assert z+depth < im_shape[0]
        assert y+height < im_shape[1]
        assert x+width < im_shape[2]

        assert z >= 0
        assert y >= 0
        assert x >= 0
 

    (z_start, z_end,
     z_pad_start, z_pad_end) = dimension_offsets(z, depth,
                                                 root_painter.input_shape[0],
                                                 root_painter.output_shape[0])
    (y_start, y_end,
     y_pad_start, y_pad_end) = dimension_offsets(y, height,
                                                 root_painter.input_shape[1],
                                                 root_painter.output_shape[1])
    (x_start, x_end,
     x_pad_start, x_pad_end) = dimension_offsets(x, width,
                                                 root_painter.input_shape[2],
                                                 root_painter.output_shape[2])

    bounded_im = get_bounded_im_from_img(root_painter.img_data,
                                         z_start, z_end, y_start,
                                         y_end, x_start, x_end)
    im_fpath = root_painter.image_path
    im_fname = os.path.basename(im_fpath)
    proj_location = root_painter.proj_location
    bounded_im_dir = os.path.join(proj_location, 'bounded_images')
    bounded_im_name = im_fname.replace('.nii.gz', '').replace('.nrrd', '')

    # add coordinates and pad size to file name so we know where to show the segmentation.
    bounded_im_name += (f"_x_{x}_y_{y}_z_{z}_pad_"
                        f"x_{x_pad_start}_{x_pad_end}_"
                        f"y_{y_pad_start}_{y_pad_end}_"
                        f"z_{z_pad_start}_{z_pad_end}.nii.gz")

    bounded_im_fpath = os.path.join(bounded_im_dir, bounded_im_name)
    img = nib.Nifti1Image(bounded_im, np.eye(4))
    img.to_filename(bounded_im_fpath)
    QtWidgets.QApplication.instance().setOverrideCursor(Qt.BusyCursor)
    root_painter.set_seg_loading()

    # send instruction to segment the new image.
    root_painter.send_instruction('segment', {
        "dataset_dir": bounded_im_dir,
        "seg_dir": root_painter.seg_dir,
        "file_names": [bounded_im_name],
        "message_dir": root_painter.message_dir,
        "model_dir": root_painter.model_dir,
        "classes": root_painter.classes # used for saving segmentation output to correct directories
    })
    root_painter.bounded_fname = bounded_im_name
    root_painter.track_changes()


def get_bounded_im_from_img(img_data, z_start, z_end, y_start, y_end, x_start, x_end):    
    """ Ensure there is enough data to include our padded selection
        Pad with zeros if required """

    # if image has depth of 10 and z_end is 12
    # that means z_pad_end is 2, so we need to pad the image by 2
    z_pad_start = max(0, -z_start)
    z_pad_end = max(0, z_end - img_data.shape[0])
    y_pad_start = max(0, -y_start)
    y_pad_end = max(0, y_end - img_data.shape[1])
    x_pad_start = max(0, -x_start)
    x_pad_end = max(0, x_end - img_data.shape[2])

    # The padding will be applied to the bounded image.
    # so first adjust the start and end coordinates so we get what is available in the img_data    
    z_start2 = z_start + z_pad_start
    y_start2 = y_start + y_pad_start
    x_start2 = x_start + x_pad_start

    z_end2 = z_end - z_pad_end
    y_end2 = y_end - y_pad_end
    x_end2 = x_end - x_pad_end

    assert z_end2 > z_start2 >= 0
    assert y_end2 > y_start2 >= 0
    assert x_end2 > x_start2 >= 0
    
    bounded_im = img_data[z_start2:z_end2, y_start2:y_end2, x_start2:x_end2]

    bounded_im = np.pad(bounded_im,
                        ((z_pad_start, z_pad_end),
                         (y_pad_start, y_pad_end),
                         (x_pad_start, x_pad_end)))

    assert bounded_im.shape[0] == (z_end - z_start), f"{bounded_im.shape[0]}, {z_end}, {z_start}"
    assert bounded_im.shape[1] == (y_end - y_start), f"{bounded_im.shape[1]}, {y_end}, {y_start}"
    assert bounded_im.shape[2] == (x_end - x_start), f"{bounded_im.shape[2]}, {x_end}, {x_start}"
    return bounded_im


class Handle(QtWidgets.QGraphicsEllipseItem):

    def __init__(self, x, y, parent, cursor):
        self.circle_diam = 3
        super().__init__(x, y, self.circle_diam, self.circle_diam)
        self.setParentItem(parent)
        self.cursor = cursor
        self.parent = parent
        self.setPen(QtGui.QPen(QtGui.QColor(250, 250, 250), 0.5, QtCore.Qt.DashLine))
        self.setBrush(QtGui.QBrush(QtGui.QColor(40, 120, 250), style = QtCore.Qt.SolidPattern))
        self.setAcceptHoverEvents(True)
        self.drag_start_x = None
        self.drag_start_y = None

    def hoverEnterEvent(self, event):
        super().hoverEnterEvent(event)
        self.parent.over_corner = True
        QtWidgets.QApplication.instance().setOverrideCursor(self.cursor)

    def mousePressEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            # if control key is pressed then don't do anything 
            return

        self.drag_start_x = event.scenePos().x()
        self.drag_start_y = event.scenePos().y()

    def mouseReleaseEvent(self, event):
        if hasattr(self, 'on_release'):
            self.on_release() 

    def hoverLeaveEvent(self, event):
        self.parent.over_corner = False
        QtWidgets.QApplication.restoreOverrideCursor()
        
    def mouseMoveEvent(self, event): 
        new_x = event.scenePos().x()
        new_y = event.scenePos().y()
        diff_x = new_x - self.drag_start_x
        diff_y = new_y - self.drag_start_y
        self.setPos(self.pos().x() + diff_x, self.pos().y() + diff_y)
        self.drag_start_x = new_x
        self.drag_start_y = new_y

        #super().mouseMoveEvent(event)
        if self.on_move is not None:
            self.on_move(event, diff_x, diff_y)

    def setPosR(self, x, y):
        # use x and y as center of position
        super().setPos(x-(self.circle_diam/2), y-(self.circle_diam/2))


class BoundingBox(QtWidgets.QGraphicsRectItem):

    def __init__(self, x, y, parent):
        start_rect = QtCore.QRectF(0, 0, 20, 20)
        super().__init__(start_rect)
        self.parent = parent
        self.first_resize = True
        self.x_start = x
        self.y_start = y
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)
        self.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 0.2, QtCore.Qt.DashLine))
        self.setBrush(QtGui.QBrush(QtGui.QColor(40, 120, 200, 70),
                                   style=QtCore.Qt.SolidPattern))
        self.left_cursor = Qt.SizeHorCursor
        self.right_cursor = Qt.SizeHorCursor
        self.top_cursor = Qt.SizeVerCursor
        self.bottom_cursor = Qt.SizeVerCursor
        self.tl_circle = Handle(x, y, self,  Qt.SizeFDiagCursor)
        self.tl_circle.on_move = self.tl_handle_moved
        self.tl_circle.on_release = self.release_handle
        self.bl_circle = Handle(x, y, self, Qt.SizeBDiagCursor)
        self.bl_circle.on_move = self.bl_handle_moved
        self.bl_circle.on_release = self.release_handle
        self.tr_circle = Handle(x, y, self, Qt.SizeBDiagCursor)
        self.tr_circle.on_release = self.release_handle
        self.tr_circle.on_move = self.tr_handle_moved
        self.br_circle = Handle(x, y, self, Qt.SizeFDiagCursor)
        self.br_circle.on_move = self.br_handle_moved
        self.br_circle.on_release = self.release_handle
        self.resize_drag(x, y)
        self.mouse_over = True
        self.over_corner = False # mouse over corner
        self.edge = None # for resizing with edges
        
    def print_rect(self):
        point = QtCore.QPoint(0, 0)
        scenePos = self.mapToScene(point)
        x = self.rect().x() + scenePos.x()
        y = self.rect().y() + scenePos.y()
        print('x:', x, 'y:', y, 'width:', self.rect().width(), 'height:', self.rect().height())

    def left_edge_resize(self, new_x):
        old_x = self.rect().x()
        width = self.rect().width() - (new_x - old_x)
        self.setRect(new_x, self.rect().y(), width, self.rect().height())
        x_diff = new_x - old_x
        self.update_corners(x_diff, 0)

    def top_edge_resize(self, new_y):
        old_y = self.rect().y()
        height = self.rect().height() - (new_y - old_y)
        self.setRect(self.rect().x(), new_y, self.rect().width(), height)
        y_diff = new_y - old_y
        self.update_corners(0, y_diff)

    def right_edge_resize(self, new_x):
        old_x = self.rect().x() + self.rect().width()
        width = self.rect().width() + (new_x - old_x)
        self.setRect(self.rect().x(), self.rect().y(), width, self.rect().height())
        self.update_corners(0, 0)

    def bottom_edge_resize(self, new_y):
        old_y = self.rect().y() + self.rect().height()
        height = self.rect().height() + (new_y - old_y)
        self.setRect(self.rect().x(), self.rect().y(), self.rect().width(), height)
        self.update_corners(0, 0)

    def update_corners(self, x_diff, y_diff):
        width = self.rect().width()
        height = self.rect().height()
        self.tl_circle.setPos(self.tl_circle.pos().x() + x_diff,
                              self.tl_circle.pos().y() + y_diff)
        self.bl_circle.setPos(self.tl_circle.pos().x() + x_diff,
                              self.tl_circle.pos().y() + y_diff + height)
        self.tr_circle.setPos(self.tl_circle.pos().x() + width + x_diff,
                              self.tl_circle.pos().y() + y_diff)
        self.br_circle.setPos(self.tl_circle.pos().x() + width + x_diff,
                              self.tl_circle.pos().y() + y_diff + height)

    def mouseMoveEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            # if control key is pressed then don't do anything 
            return
        
        """ Triggers on mouse down (drag events) """
        mouse_x = event.scenePos().x()
        mouse_y = event.scenePos().y()
        scene_rect = self.scene_rect()
        cur_cursor = QtWidgets.QApplication.instance().overrideCursor()
        
        point = QtCore.QPoint(0, 0)
        scenePos = self.mapToScene(point)
        new_x = mouse_x - scenePos.x()
        new_y = mouse_y - scenePos.y()

        if self.edge == 'left':
            self.left_edge_resize(new_x)
        elif self.edge == 'right':
            self.right_edge_resize(new_x)
        elif self.edge == 'top':
            self.top_edge_resize(new_y)
        elif self.edge == 'bottom':
            self.bottom_edge_resize(new_y)
        else:
            # handle built in drag behaviour
            super().mouseMoveEvent(event)
        self.parent.update_bounding_boxes()
    
    def assignCursor(self, new_cursor):
        cur_cursor = QtWidgets.QApplication.instance().overrideCursor()
        if cur_cursor != new_cursor:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QApplication.instance().setOverrideCursor(new_cursor)

    def mouseMove(self, event):

        """ could be mouse down or mouse up. """
        if not self.over_corner and self.parent.mouse_down:
            self.mouseMoveEvent(event)
        else: 
            mouse_x = event.scenePos().x()
            mouse_y = event.scenePos().y()
            scene_rect = self.scene_rect()
            x, y, w, h = scene_rect
            tol = 3 # tolerance
            is_left_edge = abs(mouse_x - x) < tol
            is_right_edge = abs(mouse_x - (x + w)) < tol
            is_top_edge = abs(mouse_y - y) < tol
            is_bottom_edge = abs(mouse_y - (y + h)) < tol 
            if self.over_corner:
                self.edge = None
            elif is_left_edge:
                self.edge = 'left'
                self.assignCursor(self.left_cursor)
            elif is_right_edge:
                self.edge = 'right'
                self.assignCursor(self.right_cursor)
            elif is_top_edge:
                self.edge = 'top'
                self.assignCursor(self.top_cursor)
            elif is_bottom_edge:
                self.edge = 'bottom'
                self.assignCursor(self.bottom_cursor)
            elif self.mouse_over:
                self.edge = None
                cur_cursor = QtWidgets.QApplication.instance().overrideCursor()
                if cur_cursor != Qt.OpenHandCursor:
                    QtWidgets.QApplication.restoreOverrideCursor()
                    QtWidgets.QApplication.instance().setOverrideCursor(Qt.OpenHandCursor)
            elif not self.over_corner:
                self.edge = None
                QtWidgets.QApplication.restoreOverrideCursor()

    def scene_rect(self):
        point = QtCore.QPoint(0, 0)
        scenePos = self.mapToScene(point);
        x = self.rect().x() + scenePos.x()
        y = self.rect().y() + scenePos.y()
        return x, y, self.rect().width(), self.rect().height()

    def update_from_box_info(self, info):
        mode = self.parent.parent.mode
        if mode == 'sagittal':
            self.update(info['y'], info['z'], info['height'], info['depth'])
        elif mode == 'axial':
            self.update(info['x'], info['y'], info['width'], info['height'])
        elif mode == 'coronal':
            self.update(info['x'], info['z'], info['width'], info['depth'])
        else:
            raise Exception(f'Unhandled view mode {view.mode}')
        
    def update(self, x, y, width, height):
        point = QtCore.QPoint(0, 0)
        scenePos = self.mapToScene(point)
        x = x - scenePos.x()
        y = y - scenePos.y()
        x_diff = x - self.rect().x()
        y_diff = y - self.rect().y()
        self.setRect(x, y, width, height)
        self.update_corners(x_diff, y_diff)

    def tl_handle_moved(self, event, diff_x, diff_y):
        new_x = self.rect().x() + diff_x
        old_x = self.rect().x()
        width_increase = old_x - new_x
        new_width = self.rect().width() + width_increase
        new_y = self.rect().y() + diff_y
        old_y = self.rect().y()
        height_increase = old_y - new_y
        new_height = self.rect().height() + height_increase
        self.setRect(new_x, new_y, new_width, new_height)
        self.update_corners(0, 0)
        self.parent.update_bounding_boxes()

    def bl_handle_moved(self, event, diff_x, diff_y):
        old_x = self.rect().x()
        new_x = old_x + diff_x
        width_increase = old_x - new_x
        new_width = self.rect().width() + width_increase
        old_y = self.rect().y() 
        new_height = self.rect().height() + diff_y
        self.setRect(new_x, old_y, new_width, new_height)
        self.tl_circle.setPos(self.bl_circle.pos().x(), 
                              self.bl_circle.pos().y() - new_height)
        self.update_corners(0, 0)
        self.parent.update_bounding_boxes()

    def tr_handle_moved(self, event, diff_x, diff_y):
        old_x = self.rect().x()
        new_width = self.rect().width() + diff_x
        new_y = self.rect().y() + diff_y
        new_height = self.rect().height() - diff_y
        self.setRect(old_x, new_y, new_width, new_height)
        self.tl_circle.setPos(self.tr_circle.pos().x() - new_width, 
                              self.tr_circle.pos().y())
        self.update_corners(0, 0)
        self.parent.update_bounding_boxes()

    def br_handle_moved(self, event, diff_x, diff_y):
        old_x = self.rect().x()
        old_y = self.rect().y() 
        new_width = self.rect().width() + diff_x
        new_height = self.rect().height() + diff_y
        self.setRect(old_x, old_y, new_width, new_height)
        self.tl_circle.setPos(self.br_circle.pos().x() - new_width, 
                              self.br_circle.pos().y() - new_height)
        self.update_corners(0, 0)
        self.parent.update_bounding_boxes()

    def release_handle(self):
        """ User could have flipped the rect. sort it out 
            otherwise there are side effects with mouse rollover events
        """
        r = self.rect()
        x = r.x()
        y = r.y()
        width = r.width()
        height = r.height()
        if width < 0:
            x = x + width
            width = -width
            # switch left and right locations
            left = self.br_circle.pos().x()
            right = self.tl_circle.pos().x()
            self.tl_circle.setX(left)
            self.bl_circle.setX(left)
            self.tr_circle.setX(right)
            self.br_circle.setX(right)
        
        if height < 0:
            y = y + height
            height = -height
            # switch top and bottom locations
            top = self.bl_circle.pos().y()
            bottom = self.tl_circle.pos().y()
            self.tl_circle.setY(top)
            self.tr_circle.setY(top)
            self.bl_circle.setY(bottom)
            self.br_circle.setY(bottom)
        self.setRect(x, y, width, height)

    def inside_slice(self):
        mode = self.parent.parent.mode
        box = self.parent.parent.parent.box
        if mode == 'axial': 
            idx = self.parent.parent.slice_nav.max_slice_idx - self.parent.parent.cur_slice_idx
            inside = ((idx >= box['z']) and idx <= (box['z'] + box['depth']))
        elif mode == 'sagittal':
            idx = self.parent.parent.cur_slice_idx 
            inside = ((idx >= box['x']) and idx <= (box['x'] + box['width']))
        elif mode == 'coronal':
            raise Exception("Not yet implemented")
            inside = ((idx >= box['y']) and idx <= (box['y'] + box['height']))
        return inside

    def resize_drag(self, x, y):
        width = abs(x - self.x_start)
        height = abs(y - self.y_start)
        x = min(x, self.x_start)
        y = min(y, self.y_start)
        self.setRect(x, y, width, height)
        self.tl_circle.setPosR(x-self.x_start, y-self.y_start)
        self.bl_circle.setPosR(x-self.x_start, (y-self.y_start) + height)
        self.tr_circle.setPosR((x-self.x_start) + width, y-self.y_start)
        self.br_circle.setPosR((x-self.x_start) + width, (y-self.y_start) + height)

    def hoverEnterEvent(self, event):
        QtWidgets.QApplication.instance().setOverrideCursor(Qt.OpenHandCursor)
        self.mouse_over = True
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        super().hoverLeaveEvent(event)
        self.mouse_over = False
        QtWidgets.QApplication.restoreOverrideCursor()
