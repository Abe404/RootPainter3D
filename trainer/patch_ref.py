""" 
Dataclass to help with caching validation information

Copyright (C) 2023 Abraham George Smith

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
from dataclasses import dataclass
from typing import Optional # supporting python3.8

import numpy as np

from metrics import Metrics


@dataclass
class PatchRef:
    annot_dir: str
    annot_fname: str
    # patch origin position relative to annotation
    # for addressing the location within the padded image
    x: int
    y: int
    z: int
    """
    The image annotation may get updated by the user at any time.
    We can use the mtime to check for this.
    If the annotation has changed then we need to retrieve patch
    coords for this image again. The reason for this is that we
    only want patch coords with annotations in. The user may have added or removed
    annotation in part of an image. This could mean a different set of coords (or
    not) should be returned for this image.
    """
    mtime: int # when was the corresponding annotation modified

    # ignore_mask (regions to ignore because they overlap with another patch)
    # numpy array saying which voxels should be ignored when computing metrics
    # because these voxels exist in another overlapping patch.
    ignore_mask: np.ndarray 
    # FIXME: Could ignore_mask be a list of coordinates describing a cuboid rather than 
    #        a likely memory intensive exaustive list of voxels?
 
    # These metrics are the cached performance for this patch 
    # with previous (current best) model.
    metrics: Optional[Metrics] = None

    def annot_fpath(self):
        return os.path.join(self.annot_dir, self.annot_fname)
    
    def has_metrics(self):
        return self.metrics is not None
 
    def is_same_region_as(self, other):
        return (self.annot_fname == other.annot_fname and 
                self.x == other.x and
                self.y == other.y and
                self.z == other.z)
