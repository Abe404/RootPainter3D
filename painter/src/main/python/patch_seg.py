
import os
import time
import numpy as np
import nibabel as nib
from PyQt5 import QtCore

from instructions import send_instruction

def segment_patch(x, y, z, root_painter):
    """
    Create an instruction to segment the patch including:

        - filename (of bounded/padded image)

        - time (to avoid conflicts with multiple instructions)

        - dataset_dir (where to get the file from to segment)

        - model_dir (where to get the latest model from to segment with)
                    (this will probably already be in memory on the server,
                     if the latest model has been recently used)

        - seg_dir (where to put the output segmentation)

        - x,y,z coordinates of patch centroid (which bit of the image to sement)
                actually we will supply z_start,z_end,y_start etc to avoid recalculation on server.
                these coordinates are without padding added.

        - patch_annot_dir (for now this is just where to find the annotation for this patch
                     for the current class, but we may want to consider how 
                     other class information is utilised in a multiclass scenario).
                     Perhaps the full annotations could be used for the
                     other classes (which the server already knows about)
                     And then we don't need to sync them as quickly or specify
                     info in the instruction.
                     
        - classes (just one for now but lets come back to this)
                  (we may want to have some sort of priority class that
                   gets updated first because the user is currently viewing it)
    
    """
    z = z + 1 # we dont segment the current slice. It gets confusing.
    
    # keep track of the region the user wanted to update. We dont allow changes outside of this region.
    z_valid_min = z
    z_valid_max = z_valid_min + root_painter.output_shape[0]

    # Move the centroid of the predicted region down by half the output depth
    # That way the whole output has a chance to be in the valid region.
    z = z + (root_painter.output_shape[0] // 2)

    # clip so it's at least output//2. This should allow the edge of the image to 
    # be resegmented.
    z = max((root_painter.output_shape[0] // 2), z)
    y = max((root_painter.output_shape[1] // 2), y)
    x = max((root_painter.output_shape[2] // 2), x)

    # image dimensions
    _, d, h, w = root_painter.annot_data.shape

    # also make sure we don't try to segment too far out on the other side
    z = min(d - (root_painter.output_shape[0] // 2), z)
    y = min(h - (root_painter.output_shape[1] // 2), y)
    x = min(w - (root_painter.output_shape[2] // 2), x)

    # and not more than the image shape
    # as only centroid is supplied, we must calculate left and right.
    z_start =  z - (root_painter.input_shape[0] // 2)
    z_end = z_start + root_painter.input_shape[0]
    y_start = y - (root_painter.input_shape[1] // 2)
    y_end = y_start + root_painter.input_shape[1]
    x_start = x - (root_painter.input_shape[2] // 2)
    x_end = x_start + root_painter.input_shape[2]

    content = {
        "file_name": root_painter.bounded_fname,
        "patch_annot_fname": str(time.time()) + '.npy',
        "dataset_dir": os.path.join(root_painter.proj_location, 'bounded_images'),
        "model_dir": root_painter.model_dir,
        "seg_dir": os.path.join(root_painter.proj_location, 'patch', 'segmentation'),
        "z_start": z_start, "z_end": z_end,
        "y_start": y_start, "y_end": y_end,
        "x_start": x_start, "x_end": x_end,
        "patch_annot_dir": os.path.join(root_painter.proj_location, 'patch', 'annotation'),
        "classes": root_painter.classes,
        "scp_in_dir": os.path.join(root_painter.proj_location, 'scp_in')
    }

    # And then create an annotation for the patch input region
    # NOTE: annot_data shape is [2, d, h, w]
    #       First dimension is for bg (0) and fg (1)

    # if annotation goes outside the image, then we must pad it.
    pad_z_start = min(0, z_start) * -1
    pad_y_start = min(0, y_start) * -1
    pad_x_start = min(0, x_start) * -1 
    
    annot_patch = root_painter.annot_data[:,
                                          max(0, z_start):z_end,
                                          max(0, y_start):y_end,
                                          max(0, x_start):x_end]
    annot_patch = annot_patch.astype(bool)
    pad_z_end = root_painter.input_shape[0] - (annot_patch.shape[1] + pad_z_start)
    pad_y_end = root_painter.input_shape[1] - (pad_y_start + annot_patch.shape[2])
    pad_x_end = root_painter.input_shape[2] - (pad_x_start + annot_patch.shape[3])

    if sum([pad_z_start, pad_z_end, pad_y_start, pad_y_end, pad_x_start, pad_x_end]):
        annot_patch = np.pad(annot_patch, 
                             [(0, 0), # dont add channels
                              (pad_z_start, pad_z_end),
                              (pad_y_start, pad_y_end),
                              (pad_x_start, pad_x_end)],
                              mode='constant')

    # and save it to patch_annot_dir.
    # time is used as name to connect the patch to the instruction.
    annot_patch_path = os.path.join(content['patch_annot_dir'],
                                    content['patch_annot_fname'])

    # To use numpy or nifty? Does it make a difference? I don't think so.
    # nib.Nifti1Image(annot_patch, np.eye(4)).to_filename(annot_patch_path)
    np.save(annot_patch_path, annot_patch)
    # And then save the segment_patch instruction in the instructions folder.
    # NOTE: the server will delete the instruction after the patch is segmented.
    send_instruction(
        'segment_patch', content,
        root_painter.instruction_dir,
        root_painter.sync_dir
    )
    def check():
        scp_in = content['scp_in_dir']
        seg_fpath = os.path.join(scp_in,  content['patch_annot_fname'].replace('.npy', '.npz'))
        if os.path.isfile(seg_fpath):
            time_str = content['patch_annot_fname'].replace('.npz', '').replace('.npy', '')
            patch_seg = None
            retries = 0
            latest_ex = None
            while patch_seg is None:
                try:
                    patch_seg = np.load(seg_fpath)['seg']
                    os.remove(seg_fpath)
                except Exception as ex:
                    print('exception loading', seg_fpath, ex)
                    retries += 1
                    latest_ex = ex
                    QtCore.QTimer.singleShot(50, check) 
                    return
            print('loading patch seg: retries', retries, 'with latest ex', latest_ex)
            # TODO: I know 17 from memory. Compute '17' by 
            #       comparing input and output size
            seg_start_x = x_start + 17
            seg_start_y = y_start + 17
            seg_start_z = z_start + 17
            # first copy the seg data.
            new_seg_data = np.array(root_painter.seg_data)
            # update based on new patch data
            new_seg_data[
                seg_start_z:seg_start_z+patch_seg.shape[0],
                seg_start_y:seg_start_y+patch_seg.shape[1],
                seg_start_x:seg_start_x+patch_seg.shape[2]
            ] = patch_seg
            # now only update the displayed seg data based on the region the user intended to update
            root_painter.seg_data[z_valid_min:z_valid_max] = new_seg_data[z_valid_min:z_valid_max]
            for v in root_painter.viewers:
                if v.isVisible():
                    v.update_seg_slice()
                    v.update_outline()
            print('patch seg duration = ', time.time() - float(time_str))
        else:
            QtCore.QTimer.singleShot(50, check)

    QtCore.QTimer.singleShot(100, check)