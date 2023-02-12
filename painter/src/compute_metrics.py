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

#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914
import os


from PyQt5 import QtWidgets
from PyQt5 import QtCore
import numpy as np
from progress_widget import BaseProgressWidget
import im_utils
from medpy.metric.binary import dc 
import csv

metrics_headers = ['fname', 'dice', 'tp', 'tn', 'fp', 'fn']



# FIXME this function only works with binary masks as gt. It would be helpful if it
#       also worked with RootPainter two channel annotations.
def get_seg_metrics(seg_dir, gt_dir, fname, headers):
    # warning some images might have different axis order or might need flipping
    # we ignore that here.
    seg = im_utils.load_seg(os.path.join(seg_dir, fname))
    seg = seg.astype(bool).astype(int)

    gt = im_utils.load_seg(os.path.join(gt_dir, fname))
    gt = gt.astype(bool).astype(int)

    # if they dont match in shape assume the gt needs the depth moving from last to first
    if gt.shape != seg.shape:
        gt = np.moveaxis(gt, -1, 0)

    dice = dc(seg, gt)

    tp = np.sum(np.logical_and(seg == 1, gt == 1))
    tn = np.sum(np.logical_and(seg == 0, gt == 0))
    fp = np.sum(np.logical_and(seg == 1, gt == 0))
    fn = np.sum(np.logical_and(seg == 0, gt == 1))

    return [fname, dice, tp, tn, fp, fn]

class Thread(QtCore.QThread):
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal()


    def __init__(self, segment_dir, gt_dir, fnames, csv_path, headers):
        super().__init__()
        self.segment_dir = segment_dir
        self.gt_dir = gt_dir
        self.csv_path = csv_path
        self.headers = headers
        self.fnames = fnames

    def run(self):
        # if the file already exists then delete it.
        if os.path.isfile(self.csv_path):
            os.remove(self.csv_path)
        try: 
            all_props = []
            with open(self.csv_path, 'w+')  as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                # Write the column headers
                writer.writerow(self.headers)
                for i, fname in enumerate(self.fnames):
                    self.progress_change.emit(i+1, len(self.fnames))
                    # headers allow the output options to be detected.
                    props = get_seg_metrics(self.segment_dir, self.gt_dir, fname, self.headers)
                    print('props= ', props)
                    all_props.append(props)
                    writer.writerow(props)

            # write a summary file with .summary on the end.
            with open(os.path.splitext(self.csv_path)[0] + '.summary.csv', 'w+')  as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                summary_headers = []
                results = []
                for i, h in enumerate(self.headers):
                    if h != 'fname':
                        summary_headers.append(f'{h}_mean')
                        results.append(np.mean([p[i] for p in all_props]))
                        summary_headers.append(f'{h}_min')
                        results.append(np.min([p[i] for p in all_props]))
                        summary_headers.append(f'{h}_max')
                        results.append(np.max([p[i] for p in all_props]))
                        summary_headers.append(f'{h}_std')
                        results.append(np.std([p[i] for p in all_props]))
                # Write the column headers
                writer.writerow(summary_headers)
                writer.writerow(results)
                self.done.emit()
        except Exception as e:
            print(e)


class ExtractProgressWidget(BaseProgressWidget):

    def __init__(self, feature):
        super().__init__(f'Extracting {feature}')

    def run(self, seg_dir, gt_dir, csv_path, headers):
        seg_fnames = os.listdir(seg_dir)
        gt_fnames = os.listdir(gt_dir)
        fnames = list(set(seg_fnames).intersection(set(gt_fnames)))
        fnames = [f for f in fnames if im_utils.is_image(f)]
        self.progress_bar.setMaximum(len(fnames))
        self.thread = Thread(seg_dir, gt_dir, fnames, csv_path, headers)
        self.thread.progress_change.connect(self.onCountChanged)
        self.thread.done.connect(self.done)
        self.thread.start()


class ExtractSegMetricsWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.input_dir = None
        self.gt_dir = None
        self.output_csv = None
        self.feature = 'Segmentation Metrics'
        self.headers = metrics_headers
        self.initUI()
    
    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.setWindowTitle(f"Extract {self.feature}")
        self.add_seg_dir_btn()
        self.add_gt_dir_btn()
        self.add_output_csv_btn()
        self.add_info_label()

    def add_output_csv_btn(self):
        out_csv_label = QtWidgets.QLabel()
        out_csv_label.setText("Output CSV: Not yet specified")
        self.layout.addWidget(out_csv_label)
        self.out_csv_label = out_csv_label

        specify_output_csv_btn = QtWidgets.QPushButton('Specify output CSV')
        specify_output_csv_btn.clicked.connect(self.select_output_csv)
        self.layout.addWidget(specify_output_csv_btn)

    def add_info_label(self):
        info_label = QtWidgets.QLabel()
        info_label.setText("Segmentation directory, Ground truth directory and output CSV"
                           " must be specified.")
        self.layout.addWidget(info_label)
        self.info_label = info_label

        submit_btn = QtWidgets.QPushButton('Extract')
        submit_btn.clicked.connect(self.extract)
        self.layout.addWidget(submit_btn)
        submit_btn.setEnabled(False)
        self.submit_btn = submit_btn


    def add_seg_dir_btn(self):
        # Add specify image directory button
        in_dir_label = QtWidgets.QLabel()
        in_dir_label.setText("Segmentation directory: Not yet specified")
        self.layout.addWidget(in_dir_label)
        self.in_dir_label = in_dir_label

        specify_input_dir_btn = QtWidgets.QPushButton('Specify segmentation directory')
        specify_input_dir_btn.clicked.connect(self.select_input_dir)
        self.layout.addWidget(specify_input_dir_btn)

    def select_gt_dir(self):
        self.gt_dialog = QtWidgets.QFileDialog(self)
        self.gt_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def gt_selected():
            self.gt_dir = self.gt_dialog.selectedFiles()[0]
            self.gt_dir_label.setText('Ground truth directory: ' + self.gt_dir)
            self.validate()

        self.gt_dialog.fileSelected.connect(gt_selected)
        self.gt_dialog.open()


    def validate(self):
        if not self.gt_dir:
            self.info_label.setText("Ground truth directory must be specified "
                                    f"to extract {self.feature.lower()}")
            self.submit_btn.setEnabled(False)
            return

        if not self.input_dir:
            self.info_label.setText("Segmentation directory must be specified "
                                    f"to extract {self.feature.lower()}")
            self.submit_btn.setEnabled(False)
            return

        if not self.output_csv:
            self.info_label.setText("Output CSV must be specified to extract "
                                    "region propertie.")
            self.submit_btn.setEnabled(False)
            return

        self.info_label.setText("")
        self.submit_btn.setEnabled(True)

    def extract(self):
        self.progress_widget = ExtractProgressWidget(self.feature)
        self.progress_widget.run(self.input_dir, self.gt_dir, self.output_csv,
                                 self.headers)
        self.progress_widget.show()
        self.close()

    def add_gt_dir_btn(self):
        # Add specify image directory button
        gt_dir_label = QtWidgets.QLabel()
        gt_dir_label.setText("Ground truth directory: Not yet specified")
        self.layout.addWidget(gt_dir_label)
        self.gt_dir_label = gt_dir_label

        specify_gt_dir_btn = QtWidgets.QPushButton('Ground truth directory')
        specify_gt_dir_btn.clicked.connect(self.select_gt_dir)
        self.layout.addWidget(specify_gt_dir_btn)

    def select_input_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def input_selected():
            self.input_dir = self.input_dialog.selectedFiles()[0]
            self.in_dir_label.setText('Segmentation directory: ' + self.input_dir)
            self.validate()
        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()


    def select_output_csv(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Output CSV')
        if file_name:
            file_name = os.path.splitext(file_name)[0] + '.csv'
            self.output_csv = file_name
            self.out_csv_label.setText('Output CSV: ' + self.output_csv)
            self.validate()
