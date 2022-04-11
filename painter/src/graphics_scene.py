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

# pylint: disable=I1101, C0111, E0611, R0902
""" Canvas where image and annotations can be drawn """
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import nibabel as nib
import time
import qimage2ndarray
from skimage.segmentation import flood
import im_utils
from view_state import ViewState
from patch_seg import SegmentPatchThread, PatchSegmentor

class GraphicsScene(QtWidgets.QGraphicsScene):
    """
    Canvas where image and lines will be drawn
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        if self.parent.parent.auto_complete_enabled:
            self.patch_segmentor = PatchSegmentor(self.parent.parent)
        self.regions_to_restrict_to = []
        self.cursor_shown = False
        self.brush_size = 25
        # history is a list of pixmaps
        self.history = []
        self.redo_list = []
        self.last_x = None
        self.last_y = None
        self.annot_pixmap = None
        self.outline_pixmap = None
        self.cursor_pixmap_holder = None
        self.outline_pixmap_holder = None
        self.cursor_pixmap = None
        self.mouse_down = False
 
    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Alt:
            # then remove everything except for the clicked regions
            n = len(self.regions_to_restrict_to)
            # dont do anything unless some regions were selected.
            if n > 0:
                button_reply = QtWidgets.QMessageBox.question(self.parent,
                    'Confirm',
                    "Are you sure you want to assign corrections to restrict" 
                    f" to the {n} clicked 3D object(s)? This action cannot be undone.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, 
                    QtWidgets.QMessageBox.Cancel)

                if button_reply == QtWidgets.QMessageBox.Yes:
                    self.parent.parent.info_label.setText("Removing disconnected regions")
                    idx = self.parent.slice_nav.max_slice_idx - self.parent.cur_slice_idx 
                    # correct to restrict to only the selected region
                    new_annot, removed_count, holes, error = im_utils.restrict_to_regions_containing_points(
                        self.parent.parent.seg_data,
                        self.parent.parent.annot_data,
                        self.regions_to_restrict_to)
                    if error:
                        self.parent.parent.info_label.setText(error)
                    else:
                        message = f"{removed_count} foreground regions and {holes} holes were removed"
                        self.parent.parent.info_label.setText(message)
                    self.parent.parent.annot_data = new_annot
                    self.parent.parent.update_viewer_annot_slice()
                    self.parent.parent.update_viewer_outline()

                    self.regions_to_restrict_to = []
        super().keyReleaseEvent(event)

    def undo(self):
        if len(self.history) > 1:
            self.redo_list.append(self.history.pop().copy())
            # remove top item from history.
            new_state = self.history[-1].copy()
            self.annot_pixmap_holder.setPixmap(new_state)
            self.annot_pixmap = new_state
            self.parent.store_annot_slice()
            self.parent.parent.update_viewer_annot_slice()
            self.parent.parent.update_viewer_outline()

    def redo(self):
        if self.redo_list:
            new_state = self.redo_list.pop()
            self.history.append(new_state.copy())
            self.annot_pixmap_holder.setPixmap(new_state)
            self.annot_pixmap = new_state
            self.parent.store_annot_slice()
            # Update all views with new state.
            self.parent.parent.update_viewer_annot_slice()

    
    def update_axial_slice_pos_indicator(self):
        """ update the position of the axial slice indicator in the sagittal view """
        for v in self.parent.parent.viewers:
            if v.mode == 'sagittal':
                slice_nav = self.parent.parent.axial_viewer.slice_nav
                x1 = 0.0
                x2 = self.parent.parent.annot_data.shape[2]
                y1 = slice_nav.max_slice_idx - slice_nav.slice_idx
                y1 += 0.5
                y2 = y1
                if hasattr(v.scene, 'line'):
                   v.scene.line.setLine(x1, y1, x2, y2)
                else:
                    # add line if it doesn't exist
                    v.scene.line = QtWidgets.QGraphicsLineItem(x1, y1, x2, y2)
                    v.scene.line.setPen(QtGui.QPen(QtGui.QColor(255, 60, 60), 0.2,
                                        QtCore.Qt.DashLine))
                    v.scene.addItem(v.scene.line)
                    v.scene.line.setVisible(True)


    def flood_fill(self, x, y):
        raise Exception('disabled')
        x = round(x)
        y = round(y)
        # all rgb channels must match for the flood region to expand
        image =  self.annot_pixmap.toImage()
        rgb_np = np.array(qimage2ndarray.rgb_view(image))
        alpha_np = np.array(qimage2ndarray.alpha_view(image))
        mask = np.ones((rgb_np.shape[0], rgb_np.shape[1]), dtype=np.int)
        for i in range(3):
            mask *= flood(rgb_np[:, :, i], (y, x), connectivity=1)
        mask *= flood(alpha_np, (y, x), connectivity=1)
        
        # Assign previous labels to the new annotation before adding flooded region.        
        np_rgba = np.zeros((rgb_np.shape[0], rgb_np.shape[1], 4))
        fg = rgb_np[:, :, 0] > 0
        bg = rgb_np[:, :, 1] > 0
        np_rgba[:, :, 1] = bg * 255 # green is bg
        np_rgba[:, :, 0] = fg * 255 # red is fg
        np_rgba[:, :, 3] = (bg + fg) * 180 # alpha

        # flood fill should be with current brush color
        np_rgba[:, :, :][mask > 0] = [
            self.parent.brush_color.red(),
            self.parent.brush_color.green(),
            self.parent.brush_color.blue(),
            self.parent.brush_color.alpha()
        ]
        q_image = qimage2ndarray.array2qimage(np_rgba)
        return QtGui.QPixmap.fromImage(q_image)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            # if control key is pressed then don't do anything 
            return
        self.mouse_down = True
        if (not modifiers & QtCore.Qt.ControlModifier
              and (self.parent.annot_visible or self.parent.outline_visible)
              and self.parent.parent.view_state == ViewState.ANNOTATING):
            pos = event.scenePos()
            x, y = pos.x(), pos.y()
            self.parent.parent.log(f'mouse_press,drawing,x:{x},y:{y}'
                                   f',brush_size:{self.brush_size}'
                                   f',brush_color:{self.parent.brush_color.name()}'
                                   f',mode:{self.parent.mode}')

            if modifiers == (QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier):
                idx = self.parent.slice_nav.max_slice_idx - self.parent.cur_slice_idx 
                self.regions_to_restrict_to.append((round(x), round(y), idx))
                self.mouse_down = False
            #elif modifiers == QtCore.Qt.AltModifier:
                # if alt key is pressed then we want to flood fill
                #  from the clicked region.
            #     self.annot_pixmap = self.flood_fill(x, y)
            #     self.annot_pixmap_holder.setPixmap(self.annot_pixmap)
            else:
                # draw the circle
                if self.brush_size == 1:
                    circle_x = x
                    circle_y = y
                else:
                    circle_x = x - (self.brush_size / 2) + 0.5
                    circle_y = y - (self.brush_size / 2) + 0.5

                painter = QtGui.QPainter(self.annot_pixmap)
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
                painter.drawPixmap(0, 0, self.annot_pixmap)
                painter.setPen(QtGui.QPen(self.parent.brush_color, 0, Qt.SolidLine,
                                          Qt.RoundCap, Qt.RoundJoin))
                painter.setBrush(QtGui.QBrush(self.parent.brush_color, Qt.SolidPattern))            
                if self.brush_size == 1:
                    painter.drawPoint(round(circle_x), round(circle_y))
                else:
                    painter.drawEllipse(round(circle_x), round(circle_y), round(self.brush_size-1), round(self.brush_size-1))
                self.annot_pixmap_holder.setPixmap(self.annot_pixmap)
                painter.end()
            self.last_x = x
            self.last_y = y


    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            # if control key is pressed then don't do anything with the graphics scene
            return

        if self.parent.parent.view_state == ViewState.ANNOTATING:
            self.parent.parent.log(f'mouse_release,drawing,mode:{self.parent.mode}')
            # has to be some limit to history or RAM will run out
            if len(self.history) > 50:
                self.history = self.history[-50:]
            self.history.append(self.annot_pixmap.copy())
            self.redo_list = []
            self.parent.store_annot_slice()
            # update all views with new state.
            self.parent.parent.update_viewer_annot_slice()
            self.parent.parent.update_viewer_outline()

            pos = event.scenePos()
            x, y = pos.x(), pos.y()
            idx = self.parent.slice_nav.max_slice_idx - self.parent.cur_slice_idx            
            
            if self.parent.parent.auto_complete_enabled:
                prev_annot = np.array(qimage2ndarray.rgb_view(self.history[-2].toImage()))
                recent_annot = np.array(qimage2ndarray.rgb_view(self.history[-1].toImage()))
                diff = np.absolute(np.array(prev_annot) - np.array(recent_annot))
                if np.any(diff):
                    yy, xx, __ = np.where(diff > 0)
                    centroid_y = int(round(np.mean(yy)))
                    centroid_x = int(round(np.mean(xx)))
                    self.patch_segmentor.segment_patch(round(centroid_x), round(centroid_y), idx)
         
        self.mouse_down = False


    def clear_cursor(self):
        if self.cursor_shown:
            self.cursor_pixmap.fill(Qt.transparent) 
            self.cursor_pixmap_holder.setPixmap(self.cursor_pixmap)
            self.cursor_shown = False

    def update_cursor(self, ctrl_press=None):
        # implemented for changing brush color of static cursor
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier or ctrl_press:
            self.clear_cursor()
        elif self.cursor_shown and hasattr(self, 'cursor_x'):
            self.drawPaintCursorXY(self.cursor_x, self.cursor_y)

    def drawPaintCursor(self, event):
        # code from the mousedown draw function
        pos = event.scenePos()
        x, y = pos.x(), pos.y()
        self.drawPaintCursorXY(x, y)

    def drawPaintCursorXY(self, x, y):
        self.cursor_x = x
        self.cursor_y = y
        self.clear_cursor()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if (not modifiers == QtCore.Qt.ControlModifier and
            self.parent.parent.view_state == ViewState.ANNOTATING):
            if self.brush_size == 1:
                circle_x = x
                circle_y = y
            else:
                circle_x = x - (self.brush_size / 2) + 0.5
                circle_y = y - (self.brush_size / 2) + 0.5
            painter = QtGui.QPainter(self.cursor_pixmap)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
            painter.drawPixmap(0, 0, self.cursor_pixmap)
            color = QtGui.QColor.fromRgbF(self.parent.brush_color.red(),
                                          self.parent.brush_color.green(),
                                          self.parent.brush_color.blue(), 0.4)
            painter.setPen(QtGui.QPen(color, 0, Qt.SolidLine,
                                      Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(QtGui.QBrush(color, Qt.SolidPattern))

            if self.brush_size == 1:
                painter.drawPoint(round(circle_x), round(circle_y))
            else:
                painter.drawEllipse(round(circle_x), round(circle_y), round(self.brush_size-1), round(self.brush_size-1))
            painter.end()
            self.cursor_shown = True
            self.cursor_pixmap_holder.setPixmap(self.cursor_pixmap)

    def update_info_label(self, x, y):
        try:
            idx = (self.parent.slice_nav.max_slice_idx - self.parent.cur_slice_idx) - 1
            x = round(x)
            y = round(y)
            x = max(0, x)
            y = max(0, y)
            x = min(self.parent.parent.img_data.shape[2]-1, x)
            y = min(self.parent.parent.img_data.shape[1]-1, y)
            hu = self.parent.parent.img_data[idx, y, x]
            self.parent.parent.hu_label.setText(f'HU: {hu} ') # | Position: (x:{round(x)}, y:{round(y)}')
        except Exception as _:
            pass

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.drawPaintCursor(event)
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        shift_down = (modifiers & QtCore.Qt.ShiftModifier)

        pos = event.scenePos()
        x, y = pos.x(), pos.y()

        if modifiers == QtCore.Qt.ControlModifier:
            # if control key is pressed then don't do anything with the graphics scene
            return

        elif shift_down:
            dist = self.last_y - y
            self.brush_size += dist
            self.brush_size = max(1, self.brush_size)
            # Warning: very tight coupling.
            self.parent.update_cursor()

        elif (self.parent.parent.view_state == ViewState.ANNOTATING
              and self.mouse_down and (self.parent.annot_visible or self.parent.outline_visible)):

            painter = QtGui.QPainter(self.annot_pixmap)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
            painter.drawPixmap(0, 0, self.annot_pixmap)
            pen = QtGui.QPen(self.parent.brush_color, self.brush_size, Qt.SolidLine,
                             Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            pos = event.scenePos()
            x, y = pos.x(), pos.y()

            # Based on empirical observation
            if self.brush_size % 2 == 0:
                painter.drawLine(self.last_x+0.5, self.last_y+0.5, x+0.5, y+0.5)
            else:
                painter.drawLine(round(self.last_x), round(self.last_y), round(x), round(y))

            self.annot_pixmap_holder.setPixmap(self.annot_pixmap)
            painter.end()
        self.update_info_label(x, y) 

        self.last_x = x
        self.last_y = y
