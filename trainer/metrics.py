"""
Computing metrics on output segmentations for root images

Copyright (C) 2019, 2020 Abraham George Smith

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

# pylint: disable=C0111,R0913
from datetime import datetime
from collections import namedtuple
import numpy as np

def get_metrics_str(all_metrics, to_use=None):
    out_str = ""
    for name, val in all_metrics.items():
        if to_use is None or name in to_use:
            out_str += f" {name} {val:.4g}"
    return out_str

def get_metric_csv_row(metrics):
    now_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    parts = [now_str, metrics['tp'], metrics['fp'], metrics['tn'],
             metrics['fn'], round(metrics['precision'], 4),
             round(metrics['recall'], 4), round(metrics['dice'], 4)]
    return ','.join([str(p) for p in parts]) + '\n'


def get_metrics_from_arrays(y_pred, y_true):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    assert len(y_true) == len(y_pred)
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    m = get_metrics(tp, fp, tn, fn)
    return m

def get_metrics(tp:int, fp:int, tn:int, fn:int) -> dict:
    # for the mtrics function in model utils
    assert not np.isnan(tp)
    assert not np.isnan(fp)
    assert not np.isnan(tn)
    assert not np.isnan(fn)
    total = (tp + tn + fp + fn)
    accuracy = (tp + tn) / total

    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
    else: 
        precision = recall = f1 = float('NaN')
    return {
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "dice": f1,
        "true_mean": (tp + fn) / total,
        "true": (tp + fn),
        "pred": (fp + tp)
    }



def metrics_from_val_tile_refs(val_tile_refs):
    tps = 0 
    fps = 0 
    tns = 0
    fns = 0
    for ref in val_tile_refs:
        assert ref[3] is not None, ref
        (tp, fp, tn, fn) = ref[3]
        tps += tp
        fps += fp
        tns += tn
        fns += fn
    return get_metrics(tps, fps, tns, fns)
