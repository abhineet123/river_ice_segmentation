import copy
import os
import cv2
import sys
import numpy as np
import random
import math

import pandas as pd
from tqdm import tqdm
from datetime import datetime
import skvideo.io
from PIL import Image

p2s_path = os.path.join(os.path.expanduser("~"), "pix2seq")
sys.path.append(p2s_path)

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")
sys.path.append(dproc_path)

from densenet.utils import linux_path, sort_key, col_bgr, resize_ar
from tasks import task_utils
from tasks.visualization import vis_utils
from eval_utils import mask_pts_to_img, draw_box, mask_img_to_rle_coco, map_class_ids, unmap_class_ids, \
    get_class_ids_map

import paramparse


class Params(paramparse.CFG):
    """
    :ivar min_frg: minimum ratio of foreground pixels in a patch for it to be saved;
    can be used to exclude rotated patches with too much out-of-image portion
    """

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='sub_patch')

        self.enable_labels = 1
        self.class_names_path = ''
        self.proc_labels = 0
        self.subseq_mode = 0
        self.patch_mode = 0

        self.enable_flip = 0

        self.enable_rot = 0
        self.max_rot = 0
        self.min_rot = 0

        self.max_bkg_ratio = 0.33

        self.min_stride = 0
        self.max_stride = 0

        self.patch_height = 0
        self.patch_width = 0

        self.resize = 0
        self.resize_x = 0
        self.resize_y = 0

        self.resize_check = 0
        self.max_diff_rate = 5
        self.rot_check = 0

        self.seq_name = ''
        self.start_id = 0
        self.end_id = -1
        self.n_frames = 0

        self.sample = 0
        self.shuffle = 0

        self.img_ext = 'tif'
        self.labels_ext = 'jpg'

        self.out_suffix = ''
        self.out_seq_name = ''
        self.out_img_ext = 'jpg'
        self.out_labels_ext = 'png'
        self.out_vid_ext = 'mp4'
        self.vis_ext = 'jpg'

        self.vid_fps = 5

        self.out_root_dir = ''
        self.out_img_dir = ''
        self.out_labels_dir = ''
        self.out_instance_dir = ''
        self.out_vis_path = ''

        self.db_root_dir = ''
        self.src_path = ''
        self.labels_path = ''

        self.enable_instance = 0
        self.instance_coco_rle = 0
        self.ignore_invalid = 0
        self.clamp_scores = 0
        self.percent_scores = 0

        self.save_img = 0
        self.save_mapped = 0
        self.save_palette = 1
        self.save_vid = 0
        self.save_stacked = 0
        self.save_vis = 0
        self.vis = 0
        self.vis_portrait = 0
        """show subsampled mask"""
        self.subsample = 0

        self.rle = Params.RLE()

    class RLE:
        def __init__(self):
            self.enable = 0
            self.json = 0
            self.check = 0
            self.label = 1
            self.max_len = 0
            self.starts_2d = 0
            self.starts_offset = 1000
            self.lengths_offset = 100
            self.subsample = 0


def check_instance_match(gt, proc, name1, name2):
    if len(gt) == len(proc):
        assert np.all(np.equal(gt, proc)), \
            f"{name1} masks {name2} mismatch:\n\t{gt}\n\t{proc}"
    elif len(gt) == len(proc) + 1:
        assert 0 in gt and 0 not in proc, \
            f"{name1} masks {name2} mismatch:\n\t{gt}\n\t{proc}"
        gt.remove(0)
        assert np.all(np.equal(gt, proc)), \
            f"{name1} masks {name2} mismatch:\n\t{gt}\n\t{proc}"
    elif len(gt) == len(proc) - 1:
        assert 0 in proc and 0 not in gt, \
            f"{name1} masks {name2} mismatch:\n\t{gt}\n\t{proc}"
        proc = list(proc)
        proc.remove(0)
        assert np.all(np.equal(proc, gt)), \
            f"{name1} masks {name2} mismatch:\n\t{gt}\n\t{proc}"
    else:
        raise AssertionError(
            f"{name1} masks {name2} mismatch:\n\t{gt}\n\t{proc}"
        )


def resize_mask_without_interpolation(mask, params, n_classes):
    h, w = mask.shape[:2]
    if w >= params.resize and h >= params.resize:
        mask = task_utils.subsample_mask(
            mask, n_classes=n_classes, factor=None, shape=(params.resize, params.resize),
            is_vis=0, check=params.resize_check, max_diff_rate=params.max_diff_rate)
    elif w <= params.resize and h <= params.resize:
        mask = task_utils.supersample_mask(
            mask, n_classes=n_classes, factor=None, shape=(params.resize, params.resize),
            is_vis=0, check=params.resize_check, max_diff_rate=params.max_diff_rate)
    else:
        mask = task_utils.sub_and_super_sample_mask(
            mask, n_classes=n_classes, factor=None, shape=(params.resize, params.resize),
            is_vis=0, check=params.resize_check, max_diff_rate=params.max_diff_rate)

        # labels_img_2 = task_utils.resize_mask(
        #     labels_img, shape=(params.resize, params.resize), n_classes=n_classes,
        #     is_vis=0, check=params.resize_check, max_diff_rate=params.max_diff_rate)

        #
        # labels_img_vis = task_utils.mask_id_to_vis(labels_img, n_classes=n_classes, copy=True)
        # labels_img_2_vis = task_utils.mask_id_to_vis(labels_img_2, n_classes=n_classes, copy=True)
        #
        # cv2.imshow('labels_img_vis', labels_img_vis)
        # cv2.imshow('labels_img_2_vis', labels_img_2_vis)
        # cv2.waitKey(0)

    return mask


def rotate_mask_without_interpolation(img, rotation_amount_degree):
    rotation_amount_rad = rotation_amount_degree * np.pi / 180.0

    rotated_image = np.zeros_like(img)
    height, width = img.shape[:2]
    rotated_height, rotated_width = rotated_image.shape

    mid_row = int((rotated_height + 1) / 2)
    mid_col = int((rotated_width + 1) / 2)

    #  for each pixel in output image, find which pixel
    # it corresponds to in the input image
    for r in range(rotated_height):
        for c in range(rotated_width):
            #  apply rotation matrix, the other way
            y = (r - mid_col) * math.cos(rotation_amount_rad) + (c - mid_row) * math.sin(rotation_amount_rad)
            x = -(r - mid_col) * math.sin(rotation_amount_rad) + (c - mid_row) * math.cos(rotation_amount_rad)

            #  add offset
            y += mid_col
            x += mid_row

            #  get nearest index
            # a better way is linear interpolation
            x = round(x)
            y = round(y)

            # print(r, " ", c, " corresponds to-> " , y, " ", x)

            #  check if x/y corresponds to a valid pixel in input image
            if x >= 0 and y >= 0 and x < width and y < height:
                rotated_image[r][c] = img[y][x]
    return rotated_image


def get_instance_mask(params: Params, src_img, img_fname, df, grouped_predictions, class_name_to_id):
    row_ids = grouped_predictions.groups[img_fname]
    img_df = df.loc[row_ids]
    # objs = []
    img_h, img_w = src_img.shape[:2]
    instance_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    instance_to_target_id = {0: (0, 0)}
    instance_id = 1
    for _, row in img_df.iterrows():

        try:
            confidence = row['confidence']
        except KeyError:
            confidence = 1.0

        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])

        if xmin >= xmax or ymin >= ymax:
            msg = f'Invalid box {[xmin, ymin, xmax, ymax]}\n for file {img_fname}\n'
            if params.ignore_invalid:
                print(msg)
            else:
                raise AssertionError(msg)

        if confidence == 0:
            """annoying meaningless unexplained crappy boxes that exist for no apparent reason at all"""
            continue

        label = str(row['class'])

        try:
            class_id = class_name_to_id[label]
        except KeyError:
            print(f'ignoring invalid class {label}')
            continue

        try:
            target_id = int(row['target_id'])
        except KeyError:
            raise AssertionError('target_id not found')

        bbox = [xmin, ymin, xmax, ymax]
        try:
            mask_str = str(row['mask']).strip('"')
        except KeyError:
            raise AssertionError('mask not found')
        obj_mask_pts = [[float(k) for k in point_string.split(",")]
                        for point_string in mask_str.split(";") if point_string]

        # obj_h, obj_w = int(ymax - ymin), int(xmax - xmin)
        obj_mask_img = mask_pts_to_img(obj_mask_pts, img_h, img_w, to_rle=False)

        # obj_instance_patch = instance_mask[ymin:ymax, xmin:xmax]

        # assert np.all(instance_mask[obj_mask_img] == 0), "overlapping instance masks found"

        instance_mask[obj_mask_img] = instance_id
        instance_to_target_id[instance_id] = (target_id, class_id)
        instance_id += 1

        # mask_img_vis = obj_mask_img.astype(np.uint8) * 255
        # mask_img_vis = resize_ar(mask_img_vis, 640, 360)
        # src_img_vis = np.copy(src_img)
        # draw_box(src_img_vis, bbox, xywh=False, thickness=2)
        # src_img_vis = resize_ar(src_img_vis, 640, 360)
        # cv2.imshow('mask_img_vis', mask_img_vis)
        # cv2.imshow('src_img_vis', src_img_vis)
        # cv2.waitKey(0)

    return instance_mask, instance_to_target_id


def subsample_mask_id(mask, subsample):
    orig_size = (mask.shape[1], mask.shape[0])
    sub_size = (
        int(orig_size[0] // subsample),
        int(orig_size[1] // subsample))
    mask_sub = cv2.resize(mask, sub_size, interpolation=cv2.INTER_NEAREST)
    mask_sub = cv2.resize(mask_sub, orig_size, interpolation=cv2.INTER_NEAREST)

    return mask_sub


def subsample_mask(mask, subsample):
    orig_size = (mask.shape[1], mask.shape[0])
    sub_size = (
        int(orig_size[0] // subsample),
        int(orig_size[1] // subsample))
    mask_sub = cv2.resize(mask, sub_size)
    mask_sub = cv2.resize(mask_sub, orig_size)

    return mask_sub


def get_vis_image(src_img, labels_img, n_classes, out_fname):
    labels_patch_ud_vis = labels_img * (255 / n_classes)
    if len(labels_patch_ud_vis.shape) == 1:
        labels_patch_ud_vis = cv2.cvtColor(labels_patch_ud_vis, cv2.COLOR_GRAY2BGR)
    labels_patch_ud_vis = np.concatenate((src_img, labels_patch_ud_vis), axis=1)
    cv2.imwrite(out_fname, labels_patch_ud_vis)

    return labels_patch_ud_vis


def get_padded_patch(img, labels_img, instance_img, patch_size, ul):
    n_rows, n_cols = img.shape[:2]
    patch_height, patch_width = patch_size
    min_row, min_col = ul
    max_row = min_row + patch_height
    if max_row > n_rows:
        diff = max_row - n_rows
        min_row -= diff
        max_row -= diff

    max_col = min_col + patch_width
    if max_col > n_cols:
        diff = max_col - n_cols
        min_col -= diff
        max_col -= diff
    act_h, act_w = max_row - min_row, max_col - min_col

    src_patch = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)
    src_patch[:act_h, :act_w, :] = img[min_row:max_row, min_col:max_col, :]

    labels_patch = None
    if labels_img is not None:
        labels_patch = np.zeros((patch_height, patch_width), dtype=np.uint8)
        labels_patch[:act_h, :act_w] = labels_img[min_row:max_row, min_col:max_col]

    instance_patch = None
    if instance_img is not None:
        instance_patch = np.zeros((patch_height, patch_width), dtype=np.uint8)
        instance_patch[:act_h, :act_w] = instance_img[min_row:max_row, min_col:max_col]

    return src_patch, labels_patch, instance_patch, (min_row, min_col), (max_row, max_col)


def save_image_and_label(
        params: Params,
        src_patch,
        labels_patch,
        instance_patch,
        out_img_fname,
        out_img_dir,
        out_labels_dir,
        out_instance_dir,
        out_root_dir,
        out_seq_name, frame_id,
        n_classes,
        instance_to_target_id,
        out_vid_writer,
        out_labels_writer,
        out_instances_writer,
        out_csv_rows,
        class_id_to_name,
        image_infos, rle_lens,
        patch_info,
        palette_flat,
        class_ids_map,
):
    out_img_path = linux_path(out_img_dir, f'{out_img_fname}.{params.out_img_ext}')
    out_labels_path = linux_path(out_labels_dir, f'{out_img_fname}.{params.out_labels_ext}')
    out_instance_path = linux_path(out_instance_dir, f'{out_img_fname}.{params.out_labels_ext}')

    rel_path = os.path.relpath(out_img_path, out_root_dir).rstrip('.' + os.sep).replace(os.sep, '/')
    patch_height, patch_width = src_patch.shape[:2]
    image_info = {
        'file_name': rel_path,
        'height': patch_height,
        'width': patch_width,
        'seq': f'{out_seq_name}',
        'img_id': f'{out_img_fname}',
        'frame_id': f'{frame_id}',
    }
    if patch_info is not None:
        image_info['patch_info'] = copy.copy(patch_info)

    out_frames = [src_patch, ]

    if params.enable_instance:
        n_img_instances = len(instance_to_target_id)
        patch_instances = np.unique(instance_patch).tolist()
        n_patch_instances = len(patch_instances)
        patch_instance_to_target_id = {
            instance_id: instance_to_target_id[instance_id] for instance_id in patch_instances if instance_id != 0
        }
        image_info['instance_to_target_id'] = patch_instance_to_target_id
        if params.instance_coco_rle:
            instance_to_rle_coco = {}
            for instance_id in patch_instances:
                if instance_id == 0:
                    continue
                instance_binary_mask = instance_patch == instance_id
                instance_rle_coco = mask_img_to_rle_coco(instance_binary_mask)
                instance_to_rle_coco[instance_id] = instance_rle_coco

                target_id, class_id_ = instance_to_target_id[instance_id]
                height, width = instance_patch.shape[:2]

                inst_ys, inst_xs = np.nonzero(instance_binary_mask)
                class_ids = labels_patch[inst_ys, inst_xs]
                unique_class_ids = np.unique(class_ids)
                assert len(unique_class_ids) == 1, "multiple class IDs found for the same instance"
                xmin, xmax = np.amin(inst_xs), np.amax(inst_xs)
                ymin, ymax = np.amin(inst_xs), np.amax(inst_xs)
                class_id = unique_class_ids[0]
                assert class_id_ == class_id, "class_id mismatch"
                label = class_id_to_name[class_id]

                out_csv_row = {
                    'target_id': int(target_id),
                    'filename': out_img_fname,
                    'width': width,
                    'height': height,
                    'class': label,
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax)
                }
                out_csv_rows.append(out_csv_row)

            image_info['instance_to_rle_coco'] = instance_to_rle_coco
        else:
            if params.save_stacked:
                # instance_patch_vis = np.copy(instance_patch)
                instance_patch_vis = task_utils.mask_id_to_vis(
                    instance_patch, n_classes=n_img_instances, to_rgb=True)
                out_frames.append(instance_patch_vis)
            else:
                # instance_patch_vis = np.copy(instance_patch)
                instance_patch_vis, instance_id_to_col = task_utils.mask_id_to_vis(
                    instance_patch, n_classes=n_img_instances, to_rgb=False, copy=True, return_class_id_to_col=True)
                instance_id_to_col = {
                    instance_id: instance_id_to_col[instance_id] for instance_id in patch_instances
                }
                unique_cols = np.unique(instance_patch_vis).tolist()
                n_unique_cols = len(unique_cols)
                assert n_unique_cols == n_patch_instances, "n_unique_cols mismatch"

                instance_rel_path = os.path.relpath(out_instance_path, out_root_dir).rstrip('.' + os.sep).replace(
                    os.sep, '/')

                image_info['instance_file_name'] = instance_rel_path
                image_info['instance_id_to_col'] = instance_id_to_col

                if params.save_vid:
                    out_instances_writer.writeFrame(instance_patch_vis)

                if params.save_img:
                    cv2.imwrite(out_instance_path, instance_patch_vis)
                    # instance_patch_vis_read = cv2.imread(out_instance_path)
                    # unique_cols_read = np.unique(instance_patch_vis_read).tolist()
                    # assert unique_cols_read == unique_cols, "unique_cols_read mismatch"

    if params.enable_labels:
        if params.save_stacked:
            labels_patch_vis = task_utils.mask_id_to_vis(
                labels_patch, n_classes=n_classes, to_rgb=1)
            out_frames.append(labels_patch_vis)
        else:
            labels_vis_needed = params.save_vid or (params.save_img and not params.save_palette)

            if labels_vis_needed:
                labels_patch_vis = task_utils.mask_id_to_vis(
                    labels_patch, n_classes=n_classes, to_rgb=0, copy=1)

            mask_rel_path = os.path.relpath(out_labels_path, out_root_dir).rstrip('.' + os.sep).replace(
                os.sep, '/')
            image_info['mask_file_name'] = mask_rel_path

            if params.save_vid:
                out_labels_writer.writeFrame(labels_patch_vis)

            if params.save_img:
                if params.save_mapped:
                    assert class_ids_map is not None
                    labels_patch = unmap_class_ids(labels_patch, class_ids_map)

                mask_img_pil = Image.fromarray(labels_patch)
                if params.save_palette:
                    assert params.out_labels_ext == 'png', "save_palette is only supported with png masks"
                    mask_img_pil = mask_img_pil.convert('P')
                    mask_img_pil.putpalette(palette_flat)
                mask_img_pil.save(out_labels_path)

        if params.rle.enable:
            mask = np.copy(labels_patch)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if params.rle.label:
                mask[mask > 0] = params.rle.label

            rle, rle_norm = task_utils.mask_to_rle(
                mask, params.rle.max_len)

            if params.rle.check:
                mask_rec = task_utils.rle_to_mask(
                    rle,
                    mask.shape,
                    params.rle.max_len,
                    params.rle.starts_offset,
                    params.rle.lengths_offset,
                    params.rle.starts_2d,
                )

                mask_mismatch = np.nonzero(mask != mask_rec)
                assert mask_mismatch[0].size == 0, "mask_rec mismatch"

            if params.rle.json:
                image_info['rle'] = rle

            rle_len = len(rle)
            rle_lens.append(str(rle_len))

    if len(out_frames) > 1:
        out_frame = np.concatenate(out_frames, axis=1)
    else:
        out_frame = out_frames[0]
    if params.save_vid:
        out_vid_writer.writeFrame(out_frame[:, :, ::-1])
    if params.save_img:
        cv2.imwrite(out_img_path, out_frame)

    image_infos.append(image_info)


def rotate_bound(image, angle, border_val=-1, interp='linear'):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]

    if len(image.shape) == 3:
        border_val = [border_val, border_val, border_val]

    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    image_int = image.astype(np.int16)
    if interp == 'nn':
        interp_flag = cv2.INTER_NEAREST
    elif interp == 'linear':
        interp_flag = cv2.INTER_LINEAR
    elif interp == 'cubic':
        interp_flag = cv2.INTER_CUBIC
    elif interp == 'lancz':
        interp_flag = cv2.INTER_LANCZOS4
    else:
        raise AssertionError(f'invalid interp: {interp}')

    rot_image = cv2.warpAffine(
        image_int, M, (nW, nH),
        borderValue=border_val,
        flags=interp_flag)

    rot_mask = rot_image == border_val
    rot_image[rot_mask] = 0
    rot_image = rot_image.astype(np.uint8)

    rot_mask_gs = rot_mask[..., 0].astype(np.uint8) * 255
    # cv2.imshow('rot_mask_gs', rot_mask_gs)
    # cv2.imshow('rot_image', rot_image)
    # cv2.waitKey(0)

    return rot_image, rot_mask_gs


def run(image_infos, params: Params):
    """

    :param list image_infos:
    :param Params params:
    :return:
    """
    out_root_dir = params.out_root_dir
    out_img_dir = params.out_img_dir
    out_labels_dir = params.out_labels_dir
    out_instance_dir = params.out_instance_dir
    out_vis_dir = params.out_vis_path

    db_root_dir = params.db_root_dir
    seq_name = params.seq_name
    out_seq_name = params.out_seq_name
    vis = params.vis
    vis_portrait = params.vis_portrait
    if vis == 2:
        params.save_vis = 1
    save_vis = params.save_vis
    subsample = params.subsample
    _patch_height = params.patch_height
    _patch_width = params.patch_width
    min_stride = params.min_stride
    max_stride = params.max_stride
    enable_flip = params.enable_flip
    enable_rot = params.enable_rot
    min_rot = params.min_rot
    max_rot = params.max_rot
    n_frames = params.n_frames
    start_id = params.start_id
    end_id = params.end_id
    sample = params.sample
    shuffle = params.shuffle
    enable_labels = params.enable_labels
    enable_instance = params.enable_instance
    proc_labels = params.proc_labels

    src_path = params.src_path
    labels_path = params.labels_path

    assert params.class_names_path, "class_names_path must be provided"

    class_id_to_col, class_id_to_name = task_utils.read_class_info(
        params.class_names_path)

    n_classes = len(class_id_to_col)

    class_id_to_col_orig = class_id_to_col
    class_id_to_name_orig = class_id_to_name

    class_ids = sorted(list(class_id_to_name.keys()))
    class_ids_map = None

    # class_ids_map_func = np.vectorize(lambda x: class_ids_map[x])

    if params.batch.dataset == 'cityscapes':
        class_ids_map = get_class_ids_map(class_ids)
        """map discontinuous class IDs to continuous ones"""
        class_id_to_col = {
            class_ids_map[class_id]: class_col for class_id, class_col in class_id_to_col.items()
        }
        class_id_to_name = {
            class_ids_map[class_id]: class_name for class_id, class_name in class_id_to_name.items()
        }

    class_name_to_id = {
        val: key for key, val in class_id_to_name.items()
    }

    palette = []
    for class_id in range(n_classes):
        col = class_id_to_col[class_id]
        try:
            col_rgb = col_bgr[col][::-1]
        except KeyError:
            b, g, r = map(int, col.split('_'))
            col_rgb = (r, g, b)
        palette.append(col_rgb)

    palette_flat = [value for color in palette for value in color]

    if not src_path:
        src_path = linux_path(db_root_dir, seq_name, 'images')

    if not labels_path:
        labels_path = linux_path(db_root_dir, seq_name, 'labels')

    if enable_labels:
        assert os.path.isdir(labels_path), f'Labels folder does not exist: {labels_path}'

    if enable_rot:
        assert enable_labels, 'Rotation cannot be enabled without labels'
        print('rot: {:d}, {:d}'.format(min_rot, max_rot))

    print('Reading source images from: {}'.format(src_path))

    src_files = [k for k in os.listdir(src_path) if k.endswith('.{:s}'.format(params.img_ext))]
    assert src_files, SystemError('No input frames found')
    total_frames = len(src_files)

    print('total_frames: {}'.format(total_frames))
    print('n_frames: {}'.format(n_frames))

    src_files.sort(key=sort_key)

    # src_file_list = src_file_list.sort()

    if n_frames <= 0:
        n_frames = total_frames

    if end_id < start_id:
        end_id = n_frames - 1

    patch_width, patch_height = _patch_width, _patch_height

    if patch_width <= 0:
        patch_width = patch_height

    if min_stride <= 0:
        min_stride = patch_height

    if max_stride <= min_stride:
        max_stride = min_stride

    image_as_patch = 0
    if _patch_width <= 0 and _patch_height <= 0:
        image_as_patch = 1
        print('Using entire image as the patch')

    rle_suffix = f'{patch_width}_{patch_height}'
    if params.rle.max_len <= 0:
        params.rle.max_len = params.patch_width
    else:
        rle_suffix = f'{rle_suffix}_max_{params.rle.max_len}'

    if params.rle.starts_2d:
        rle_suffix = f'{rle_suffix}_2d'

    if not out_seq_name:
        """
        probably obsolete now since out_seq_name is set within sub_patch_multi.get_spd_params
        check out_suffixes and spd_params.out_suffix for the protocol
        """
        if image_as_patch:
            out_seq_name = '{:s}_{:d}_{:d}'.format(seq_name, start_id, end_id)
        else:
            out_seq_name = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                seq_name, start_id, end_id, patch_height, patch_width, min_stride, max_stride)

        if shuffle:
            out_seq_name = '{}_rnd'.format(out_seq_name)

        if sample:
            out_seq_name = '{}_smp_{}'.format(out_seq_name, sample)

        if enable_rot:
            out_seq_name = '{}_rot_{:d}_{:d}'.format(out_seq_name, min_rot, max_rot)

        if enable_flip:
            out_seq_name = '{}_flip'.format(out_seq_name)

    src_path_root_dir = os.path.dirname(src_path)
    labels_path_root_dir = os.path.dirname(labels_path)

    if not out_root_dir:
        if db_root_dir:
            out_root_dir = db_root_dir
        else:
            out_root_dir = src_path_root_dir

    if not out_img_dir:
        if db_root_dir:
            out_img_dir = linux_path(db_root_dir, out_seq_name, 'images')
        else:
            out_img_dir = linux_path(src_path_root_dir, out_seq_name)

    if not out_instance_dir:
        if db_root_dir:
            out_instance_dir = linux_path(db_root_dir, out_seq_name, 'instances')
        else:
            out_instance_dir = linux_path(out_img_dir, 'instances')

    if not out_labels_dir:
        if db_root_dir:
            out_labels_dir = linux_path(db_root_dir, out_seq_name, 'labels')
            out_vis_dir = linux_path(db_root_dir, out_seq_name, 'vis_labels')
        else:
            out_labels_dir = linux_path(labels_path_root_dir, out_seq_name)
            out_vis_dir = linux_path(labels_path_root_dir, 'vis', out_seq_name)
    else:
        if not out_vis_dir:
            out_vis_dir = linux_path(f'{out_labels_dir}_vis')

    if save_vis:
        assert out_vis_dir, "out_vis_dir must be provided"

    out_vid_writer = None
    out_labels_writer = None
    out_instance_writer = None

    ffmpeg_params = dict(
        inputdict={
            '-r': str(params.vid_fps),
        },
        outputdict={
            '-r': str(params.vid_fps),
            '-vcodec': 'libx264',  # use the h.264 codec
            '-crf': '0',  # set the constant rate factor to 0, which is lossless
            '-preset': 'medium'  # the slower the better compression, in princple, try
            # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        }
    )

    if params.save_vid:
        out_vid_path = f'{out_img_dir}.{params.out_vid_ext}'
        print('Writing output video to: {}'.format(out_vid_path))
        os.makedirs(os.path.dirname(out_vid_path), exist_ok=True)
        ffmpeg_params_ = copy.deepcopy(ffmpeg_params)
        ffmpeg_params_['outputdict']['-crf'] = '17'
        out_vid_writer = skvideo.io.FFmpegWriter(
            out_vid_path, **ffmpeg_params_)

    if params.save_img:
        print('Writing output images to: {}'.format(out_img_dir))
        os.makedirs(out_img_dir, exist_ok=True)

    if params.resize:
        params.resize_x = params.resize_y = params.resize

    enable_resize = 0
    if params.resize_x or params.resize_y:
        assert params.resize_x and params.resize_y, "either both or none of resize_x and resize_y must be provided"
        print(f'resizing input images to {params.resize_x}x{params.resize_y} before patch extraction')
        enable_resize = 1

    if enable_labels:
        if not params.save_stacked:
            if params.save_vid:
                out_labels_vid = f'{out_labels_dir}.{params.out_vid_ext}'
                print('Writing output label video to: {}'.format(out_labels_vid))

                os.makedirs(os.path.dirname(out_labels_vid), exist_ok=True)
                ffmpeg_params_ = copy.deepcopy(ffmpeg_params)
                out_labels_writer = skvideo.io.FFmpegWriter(
                    out_labels_vid, **ffmpeg_params_)

            if params.save_img:
                print('Writing output label images to: {}'.format(out_labels_dir))
                os.makedirs(out_labels_dir, exist_ok=True)

        if save_vis:
            print('Writing output visualization images to: {}'.format(out_vis_dir))
            os.makedirs(out_vis_dir, exist_ok=True)

    if enable_instance:
        if not params.instance_coco_rle and not params.save_stacked:
            if params.save_vid:
                out_instance_vid = f'{out_instance_dir}.{params.out_vid_ext}'
                print('Writing output instance video to: {}'.format(out_instance_vid))

                os.makedirs(os.path.dirname(out_instance_vid), exist_ok=True)
                ffmpeg_params_ = copy.deepcopy(ffmpeg_params)
                out_instance_writer = skvideo.io.FFmpegWriter(
                    out_instance_vid, **ffmpeg_params_)

            if params.save_img:
                print('Writing output instance images to: {}'.format(out_instance_dir))
                os.makedirs(out_instance_dir, exist_ok=True)

    rot_angle = 0
    frame_id = 0

    pause_after_frame = 1

    # n_frames = end_id - start_id + 1
    rle_lens = []

    img_ids = list(range(start_id, end_id + 1))

    if shuffle:
        random.shuffle(img_ids)

    if sample > 0:
        img_ids = img_ids[::sample]

    img_ids_pbar = tqdm(img_ids, position=0, leave=True)

    labels_diff_rates = []
    labels_img = None
    labels_diff_rate = 0

    if enable_instance:
        csv_path = linux_path(src_path, "annotations.csv")
        assert os.path.isfile(csv_path), f"csv_path does not exist: {csv_path}"
        df = pd.read_csv(csv_path)
        df['filename'] = df['filename'].astype(str)

        grouped_predictions = df.groupby("filename")

    for img_id in img_ids_pbar:
        # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, params.img_ext)
        img_fname = src_files[img_id]
        img_fname_no_ext, _ = os.path.splitext(img_fname)

        src_img_fname = linux_path(src_path, img_fname)
        src_img = cv2.imread(src_img_fname)
        assert src_img is not None, f"invalid src_img_fname: {src_img_fname}"

        instance_mask = None
        instance_to_target_id = None
        out_csv_rows = []

        if enable_instance:
            instance_mask, instance_to_target_id = get_instance_mask(
                params, src_img, img_fname, df, grouped_predictions, class_name_to_id)
            unique_instances_orig = list(np.unique(instance_mask))

            if enable_resize:
                n_instances = len(unique_instances_orig)
                if 0 not in unique_instances_orig:
                    n_instances += 1

                # instance_mask = resize_mask_without_interpolation(instance_mask, params, n_instances)
                instance_mask = cv2.resize(instance_mask, (params.resize_x, params.resize_y),
                                           interpolation=cv2.INTER_NEAREST)

                unique_instances_resized = list(np.unique(instance_mask))
                if params.resize_check:
                    check_instance_match(unique_instances_orig, unique_instances_resized, "instance", "resizing")

        if enable_resize:
            src_img = cv2.resize(src_img, (params.resize_x, params.resize_y))

        src_height, src_width, _ = src_img.shape
        src_height, src_width, _ = src_img.shape
        src_height, src_width, _ = src_img.shape

        if image_as_patch:
            patch_width, patch_height = src_width, src_height

        if src_height < patch_height or src_width < patch_width:
            print('\nImage {} is too small {}x{} for the given patch size {}x{}\n'.format(
                src_img_fname, src_width, src_height, patch_width, patch_height))
            continue

        assert src_img is not None, 'Source image could not be read: {}'.format(src_img_fname)

        n_rows, n_cols, n_channels = src_img.shape

        if enable_labels:
            labels_img_fname = linux_path(labels_path, img_fname_no_ext + '.' + params.labels_ext)
            from PIL import Image

            im = Image.open(labels_img_fname)
            labels_img = np.array(im)

            if len(labels_img.shape) == 3:
                labels_img = task_utils.mask_to_gs(labels_img)

            if class_ids_map is not None:
                labels_img = map_class_ids(labels_img, class_ids_map)

            if params.batch.check_labels:
                """GT mask sanity/consistency check"""
                col_labels_img_fname = labels_img_fname.replace(params.batch.labels_dir, 'gtFine_color')
                col_labels_img = np.array(Image.open(col_labels_img_fname))
                col_labels_img_sem = col_labels_img[..., :3]
                col_labels_img_rec = task_utils.mask_id_to_vis_bgr(labels_img, class_id_to_col)

                cv2.imshow('col_labels_img_sem', col_labels_img_sem)
                cv2.imshow('col_labels_img_rec', col_labels_img_rec)

                # col_labels_img_ins = col_labels_img[..., 3]
                # cv2.imshow('col_labels_img_ins', col_labels_img_ins)

                cv2.waitKey(1)

                if not np.array_equal(col_labels_img_sem, col_labels_img_rec):
                    raise AssertionError("col_labels_img_rec mismatch")

            # labels_orig = np.unique(labels_img)
            if proc_labels:
                spurious_mids = proc_labels == 2
                if spurious_mids:
                    precise = False
                    max_diff_rate = 5
                else:
                    precise = True
                    max_diff_rate = 0

                labels_img, labels_diff_rate = task_utils.mask_vis_to_id(
                    labels_img, n_classes, check=True, max_diff_rate=max_diff_rate,
                    precise=precise, spurious_mids=spurious_mids)

                labels_diff_rates.append(labels_diff_rate)

            if params.resize_check or params.rot_check:
                unique_labels_orig = np.unique(labels_img)

            if enable_resize:
                # labels_img_orig = np.copy(labels_img)

                # labels_img = resize_mask_without_interpolation(labels_img, params, n_classes)
                labels_img = cv2.resize(labels_img, (params.resize_x, params.resize_y),
                                        interpolation=cv2.INTER_NEAREST)

                if params.resize_check:
                    unique_labels = np.unique(labels_img)
                    check_instance_match(unique_labels_orig, unique_labels, "semantic", "resizing")
                    labels_img, labels_diff_rate = labels_img
                    labels_diff_rates.append(labels_diff_rate)

            _n_rows, _ncols = labels_img.shape

            assert n_rows == _n_rows and n_cols == _ncols, (
                'Dimension mismatch between image and label for file: {}'.format(img_fname))

            assert np.all(labels_img <= n_classes - 1), f"labels_img cannot have values exceeding {n_classes - 1}"

            # labels_img_cv = cv2.imread(labels_img_fname)
            # if params.resize:
            #     labels_img_cv = cv2.resize(labels_img_cv, (params.resize, params.resize))
            # mask_1 = (labels_img == 1).astype(np.uint8) * 255
            # mask_2 = (labels_img == 2).astype(np.uint8) * 255
            # if labels_img_cv is None:
            #     msg = 'Labels image could not be read from: {}'.format(labels_img_fname)
            #     if allow_missing_labels:
            #         print('\n' + msg + '\n')
            #         continue
            #     raise AssertionError(msg)

            # cv2.imshow('mask_1', mask_1)
            # cv2.imshow('mask_2', mask_2)
            # cv2.imshow('labels_img', labels_img)
            # cv2.imshow('labels_img_cv', labels_img_cv)
            # cv2.waitKey(0)

            # if params.labels_rgb:
            #     assert class_to_col is not None, "class_to_col must be provided to map RGB mask to class IDs"
            #     for class_id, class_col in class_to_col.items():
            #         labels_img[labels_img == class_col] = class_id

            # cv2.imshow('src_img', src_img)
            # cv2.imshow('labels_img', labels_img)
            # cv2.waitKey(100)

        rot_mask = None
        if enable_rot:
            rot_angle = random.randint(min_rot, max_rot)
            src_img, rot_mask = rotate_bound(src_img, rot_angle)

            n_rows, n_cols = src_img.shape[:2]

            if enable_instance:
                # instance_mask_vis = task_utils.mask_id_to_vis(instance_mask, n_instances, copy=True, to_rgb=True)
                # instance_mask = rotate_mask_without_interpolation(instance_mask, rot_angle)
                instance_mask, _ = rotate_bound(instance_mask, rot_angle, interp='nn')

                unique_instances_rot = np.unique(instance_mask)
                if params.rot_check:
                    check_instance_match(unique_instances_orig, unique_instances_rot, "instance", "rotation")

                # instance_mask_rot_vis = task_utils.mask_id_to_vis(instance_mask, n_instances, copy=True, to_rgb=True)
                # cv2.imshow('instance_mask_vis', instance_mask_vis)
                # cv2.imshow('instance_mask_rot_vis', instance_mask_rot_vis)
                # cv2.waitKey(100)

            if enable_labels:
                """increase pixel value gap between the classes to prevent them getting confounded by 
                interpolation required for rotation"""
                # task_utils.mask_id_to_vis(labels_img, n_classes)
                labels_img, _ = rotate_bound(labels_img, rot_angle, interp='nn')
                unique_labels_rot = np.unique(labels_img)
                if params.rot_check:
                    check_instance_match(unique_labels_orig, unique_labels_rot, "semantic", "rotation")

                """restore the pixel values to class IDs"""
                # task_utils.mask_vis_to_id(labels_img, n_classes)

                _n_rows, _ncols = labels_img.shape

                assert n_rows == _n_rows and n_cols == _ncols, (
                    'Dimension mismatch between rotated image and label for file: {}'.format(img_fname))

        # if enable_instance:
        #     for instance_id, target_id in instance_to_target_id.items():
        #         if instance_id == 0:
        #             continue
        #         instance_mask[instance_mask == instance_id] = target_id

        out_id = 0
        # skip_id = 0
        min_row = 0
        skipped_patches = 0
        while True:
            min_col = 0
            while True:
                src_patch, labels_patch, instance_patch, (min_row, min_col), (max_row, max_col) = get_padded_patch(
                    src_img, labels_img, instance_mask, (patch_height, patch_width), (min_row, min_col)
                )
                patch_info = dict(
                    window=[min_row, max_row, min_col, max_col],
                    rot=0,
                    flip='none',
                )

                skip_patch = False
                if enable_rot:
                    rot_mask_patch = rot_mask[min_row:max_row, min_col:max_col]
                    bkg_ratio = np.count_nonzero(rot_mask_patch) / rot_mask_patch.size

                    # print(f'bkg_ratio: {bkg_ratio}')
                    # cv2.imshow('src_patch', src_patch)
                    # cv2.imshow('rot_mask_patch', rot_mask_patch)
                    # cv2.waitKey(100)

                    if bkg_ratio > params.max_bkg_ratio:
                        skip_patch = True
                    patch_info['rot'] = rot_angle

                if skip_patch:
                    skipped_patches += 1
                    fmt = f'5d'
                    img_ids_pbar.set_description(
                        f'img: {img_id:{fmt}} valid: {frame_id} skip: {skipped_patches}')
                else:
                    if image_as_patch:
                        out_img_fname = img_fname_no_ext
                    else:
                        out_img_fname = '{:s}_{:d}'.format(img_fname_no_ext, out_id + 1)

                    if vis:
                        labels_patches = []
                        if enable_labels:
                            # labels = np.unique(labels_patch)
                            # label_masks = []
                            # for label in labels:
                            #     if label == 0:
                            #         continue
                            #     label_mask = (labels_patch == label).astype(np.uint8) * 255
                            #
                            #     cv2.imshow(f'label_mask {label}', label_mask)
                            #     cv2.waitKey(0)
                            #
                            #     label_masks.append(label_mask)

                            # labels_patch_id = task_utils.mask_vis_to_id(labels_patch, n_classes, copy=True)
                            labels_patch_rgb = task_utils.mask_id_to_vis_bgr(labels_patch, class_id_to_col)
                            labels_patches.append(labels_patch_rgb)
                            # if enable_rot and (labels_patch == -1).any():
                            #     skip_ patch = True

                    if enable_rot:
                        out_img_fname = '{:s}_rot_{:d}'.format(out_img_fname, rot_angle)

                    out_id += 1
                    frame_id += 1
                    fmt = f'5d'
                    desc = f'img_id: {img_id:{fmt}} valid: {frame_id} skip: {skipped_patches}'

                    if params.resize_check:
                        desc = f'{desc} diff_rate: {labels_diff_rate:.2f}, {np.mean(labels_diff_rates):.2f}'

                    img_ids_pbar.set_description(desc)

                    save_image_and_label(
                        params,
                        src_patch, labels_patch, instance_patch,
                        out_img_fname,
                        out_img_dir,
                        out_labels_dir,
                        out_instance_dir,
                        out_root_dir,
                        out_seq_name, frame_id,
                        n_classes,
                        instance_to_target_id,
                        out_vid_writer,
                        out_labels_writer,
                        out_instance_writer,
                        out_csv_rows,
                        class_id_to_name,
                        image_infos, rle_lens,
                        patch_info,
                        palette_flat,
                        class_ids_map,
                    )
                    src_patches = [src_patch]

                    if enable_flip:
                        """
                        LR flip
                        """
                        src_patch_lr = np.fliplr(src_patch)
                        labels_patch_lr = None
                        instance_patch_lr = None

                        if enable_instance:
                            instance_patch_lr = np.fliplr(instance_patch)

                        if enable_labels:
                            labels_patch_lr = np.fliplr(labels_patch)
                            if vis:
                                labels_patch_lr_id = task_utils.mask_vis_to_id(labels_patch_lr, n_classes, copy=True)
                                labels_patch_lr_rgb = task_utils.mask_id_to_vis_bgr(labels_patch_lr_id, class_id_to_col)
                                labels_patches.append(labels_patch_lr_rgb)

                        frame_id += 1
                        # img_ids_pbar.set_description(f'frame_id: {frame_id}')

                        patch_info['flip'] = 'lr'
                        save_image_and_label(
                            params,
                            src_patch_lr,
                            labels_patch_lr,
                            instance_patch_lr,
                            f'{out_img_fname}_lr',
                            out_img_dir,
                            out_labels_dir,
                            out_instance_dir,
                            out_root_dir,
                            out_seq_name, frame_id,
                            n_classes,
                            instance_to_target_id,
                            out_vid_writer,
                            out_labels_writer,
                            out_instance_writer,
                            out_csv_rows,
                            class_id_to_name,
                            image_infos, rle_lens,
                            patch_info,
                            palette_flat,
                            class_ids_map,
                        )
                        src_patches.append(src_patch_lr)
                        """
                        UD flip
                        """
                        src_patch_ud = np.flipud(src_patch)
                        labels_patch_ud = None
                        instance_patch_ud = None
                        frame_id += 1
                        # img_ids_pbar.set_description(f'frame_id: {frame_id}')
                        if enable_instance:
                            instance_patch_ud = np.flipud(instance_patch)

                        if enable_labels:
                            labels_patch_ud = np.flipud(labels_patch)

                            if vis:
                                labels_patch_ud_id = task_utils.mask_vis_to_id(labels_patch_ud, n_classes, copy=True)
                                labels_patch_ud_rgb = task_utils.mask_id_to_vis_bgr(labels_patch_ud_id, class_id_to_col)
                                labels_patches.append(labels_patch_ud_rgb)

                        patch_info['flip'] = 'ud'
                        save_image_and_label(
                            params,
                            src_patch_ud,
                            labels_patch_ud,
                            instance_patch_ud,
                            f'{out_img_fname}_ud',
                            out_img_dir,
                            out_labels_dir,
                            out_instance_dir,
                            out_root_dir,
                            out_seq_name, frame_id,
                            n_classes,
                            instance_to_target_id,
                            out_vid_writer,
                            out_labels_writer,
                            out_instance_writer,
                            out_csv_rows,
                            class_id_to_name,
                            image_infos, rle_lens,
                            patch_info,
                            palette_flat,
                            class_ids_map,
                        )
                        src_patches.append(src_patch_ud)

                    if vis:
                        src_img_vis = src_img.copy()
                        cv2.rectangle(src_img_vis, (min_col, min_row), (max_col, max_row), (255, 255, 0), 3)
                        src_img_vis = cv2.resize(src_img_vis, (640, 640))
                        src_img_vis, _, _ = vis_utils.write_text(src_img_vis, img_fname_no_ext, 5, 5, (255, 255, 255))

                        # disp_labels_img = labels_img.copy()
                        # cv2.rectangle(disp_labels_img, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)
                        src_patch_vis = np.concatenate(src_patches, axis=0 if vis_portrait else 1)
                        src_patch_vis, _, _ = vis_utils.write_text(src_patch_vis, out_img_fname, 5, 5, (255, 255, 255))

                        if enable_labels:
                            # labels_img_vis = task_utils.mask_id_to_vis(labels_img, n_classes=n_classes, to_rgb=1)
                            labels_img_rgb = task_utils.mask_id_to_vis_bgr(labels_img, class_id_to_col)
                            cv2.rectangle(labels_img_rgb, (min_col, min_row), (max_col, max_row), (255, 255, 0), 3)
                            labels_img_vis = cv2.resize(labels_img_rgb, (640, 640))

                            if subsample:
                                labels_img_sub = subsample_mask(labels_img_rgb, subsample)
                                cv2.rectangle(
                                    labels_img_sub, (min_col, min_row), (max_col, max_row), (255, 255, 0), 3)
                                labels_img_sub_vis = cv2.resize(labels_img_sub, (640, 640))
                                labels_img_vis = np.concatenate((labels_img_vis, labels_img_sub_vis),
                                                                axis=0 if vis_portrait else 1)

                            src_img_vis = np.concatenate((src_img_vis, labels_img_vis),
                                                         axis=0 if vis_portrait else 1)
                            labels_patch_vis = np.concatenate(labels_patches,
                                                              axis=0 if vis_portrait else 1)

                            labels_patch_vis = cv2.resize(labels_patch_vis, (640, 640))
                            src_patch_vis = cv2.resize(src_patch_vis, (640, 640))

                            if subsample:
                                """resize ID mask with strict NN interpolation and then convert to RGB 
                                so it corresponds to the masks that models are actually trained on"""
                                labels_patch_sub = subsample_mask_id(labels_patch, subsample)
                                labels_patch_sub = task_utils.mask_id_to_vis_bgr(labels_patch_sub, class_id_to_col)

                                """resize RGB mask with bilinear interpolation so it appears 
                                to be much higher quality than the stuff that models that are actually trained on"""
                                # labels_patch_sub = subsample_mask(labels_patch_rgb, subsample)

                                labels_patch_sub_vis = cv2.resize(labels_patch_sub, (640, 640))
                                labels_patch_vis = np.concatenate((labels_patch_vis, labels_patch_sub_vis),
                                                                  axis=0 if vis_portrait else 1)

                            if enable_flip:
                                vis_axis = 1 if vis_portrait else 0
                            else:
                                vis_axis = 0 if vis_portrait else 1

                            # cv2.imshow('src_patch_vis', src_patch_vis)
                            # cv2.imshow('labels_patch_vis', labels_patch_vis)
                            # cv2.imshow('src_img_vis', src_img_vis)
                            # k = cv2.waitKey(0)

                            src_patch_vis = np.concatenate((src_patch_vis, labels_patch_vis),
                                                           axis=vis_axis)

                        vis_img_cat = np.concatenate((src_img_vis, src_patch_vis),
                                                     axis=1 if vis_portrait else 0)

                        disp_img = resize_ar(vis_img_cat, 1200, 1400)
                        cv2.imshow('disp_img', disp_img)
                        # cv2.imshow('src_img', src_img_vis)
                        # cv2.imshow('patch', src_patch_vis)

                        if save_vis:
                            out_vis_path = linux_path(out_vis_dir,
                                                      f'{out_img_fname:s}.{params.vis_ext:s}')
                            cv2.imwrite(out_vis_path, vis_img_cat)

                        # cv2.imshow('disp_labels_img', disp_labels_img)
                        k = cv2.waitKey(1 - pause_after_frame)
                        if k == 27:
                            sys.exit(0)
                        elif k == 32:
                            pause_after_frame = 1 - pause_after_frame

                min_col += random.randint(min_stride, max_stride)
                if image_as_patch or max_col >= n_cols:
                    break

            if image_as_patch or max_row >= n_rows:
                break
            min_row += random.randint(min_stride, max_stride)

    # sys.stdout.write('\n')
    # sys.stdout.flush()

    assert frame_id > 0, "no valid frames found"

    sys.stdout.write('Total frames generated: {}\n'.format(frame_id))

    if out_vid_writer is not None:
        out_vid_writer.close()
    if out_labels_writer is not None:
        out_labels_writer.close()
    if out_instance_writer is not None:
        out_instance_writer.close()

    if params.rle.enable:
        out_rle_dir = linux_path(out_root_dir, 'rle')
        os.makedirs(out_rle_dir, exist_ok=True)

        rle_lens_str = '\n'.join(rle_lens)
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        rle_len_path = linux_path(out_rle_dir, f'{rle_suffix}_{seq_name}_{time_stamp}.txt')
        with open(rle_len_path, 'w') as fid:
            fid.write(f'{rle_lens_str}\n')

        cmb_rle_len_path = linux_path(out_rle_dir, f'{rle_suffix}.txt')
        with open(cmb_rle_len_path, 'a') as fid:
            fid.write(f'{rle_lens_str}\n')

    return image_infos


def main():
    _params = Params()
    paramparse.process(_params)

    image_infos = []
    run(image_infos, _params)


if __name__ == '__main__':
    main()
