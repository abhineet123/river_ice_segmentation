import copy
import os
import shutil
import cv2
import sys
import multiprocessing
from tqdm import tqdm
from collections import defaultdict, OrderedDict

from PIL import Image

p2s_path = os.path.join(os.path.expanduser("~"), "pix2seq")
sys.path.append(p2s_path)

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")
sys.path.append(dproc_path)

import numpy as np
from scipy import stats

from datetime import datetime

import paramparse
# from _paramparse_ import paramparse

from densenet.utils import linux_path, sort_key, resize_ar, read_data, print_and_write, col_bgr
from densenet.evaluation.eval_segm import Metrics, metrics_to_csv

from new_deeplab.datasets.build_utils import read_class_info

import eval_utils
from tasks import task_utils
from tasks.visualization import vis_utils


class Params(paramparse.CFG):
    class Video:
        def __init__(self):
            self.length = 2
            self.stride = 1
            self.sample = 0
            self.frame_gap = 0

    class RLE:
        def __init__(self):
            self.subsample = 0
            self.starts_2d = 0
            self.no_starts = 0
            self.diff_mask = 0
            self.shared_coord = 0
            self.time_as_class = 0
            self.length_as_class = 0
            self.flat_order = 'C'

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='stitch')
        self.sleep = 30

        self.codec = 'H264'
        self.del_patch_seq = 0
        self.disp_resize_factor = 1.0
        self.ignore_eval_flag = 0
        self.mp = 1
        self.debug = 0

        self.end_id = -1
        self.fname_templ = 'img'
        self.fps = 30
        self.height = 720

        self.subsample_gt = 0

        self.images_ext = 'tif'
        self.labels_ext = 'png'

        self.db_root_dir = '/data'
        self.patch_seq_name = ''
        self.patch_seq_type = 'images'
        self.labels_dir = 'Labels'
        self.images_dir = 'Images'

        self.out_dir = '/data/seg'
        self.out_name = ''

        self.class_info_path = ''

        self.model_type = ""
        self.logits = 0

        self.in_root_dir = ''
        self.in_name = ''
        self.in_iter = ''

        self.allow_missing = 0

        self.frg_metrics = 1

        self.method = 0
        self.n_classes = 3
        self.n_frames = 0
        self.normalize_patches = 0
        self.out_ext = 'png'
        self.patch_ext = 'png'

        self.patch_height = 0
        self.patch_width = 0
        self.min_stride = 0
        self.max_stride = 0

        self.resize = 0
        self.resize_x = 0
        self.resize_y = 0

        self.n_rot = 0
        self.min_rot = 0
        self.max_rot = 0

        self.sample = 0
        self.shuffle = 0
        self.enable_flip = 0

        self.batch_size = 0

        """precedes the batch size suffix in the saved mask folder name"""
        self.save_suffixes = []

        self.patch_suffixes = []
        self.stitched_suffixes = []

        self.seq_id = -1
        self.seq_start_id = -1
        self.seq_end_id = -1

        """first stitch mask patches, then super sample the stitched mask"""
        self.post_sample = 1

        self.enable_instance = 0

        self.eval_db_prefix = ''
        self.split_suffix = ''
        self.patch_mode = 0
        self.subseq_mode = 0

        self.is_video = 0
        self.strides = []
        # self.video_ids = []

        self.ovl_cmb = ''
        self.mask_cmb = 'vote'
        self.mask_cmbs = []
        self.vid = Params.Video()
        self.rle = Params.RLE()

        self.resize_factor = 1.0
        self.seq_name = 'Training'
        self.show_img = 0

        self.stacked = 1
        self.start_id = 0

        self.src_path = ''
        self.patch_seq_path = ''
        self.load_stitched = 0
        self.save_stitched = 1
        self.save_mapped = 0
        self.save_vis = 0
        self.stitched_raw_path = ''
        self.stitched_mapped_path = ''
        self.stitched_vis_path = ''

        self.width = 1280

        self.train_info = paramparse.MultiPath()
        self.train_split = paramparse.MultiPath()

        self.multi_sequence_db = 1
        self.seg_on_subset = 0
        self.split = paramparse.MultiPath()

        self.images_path = ''
        self.labels_path = ''

        self.image_dir = ''
        self.labels_dir = ''

        self.dataset = ''

        self.seq_patch_info = None
        self.frame_to_eval_info = None

    def process(self):
        assert self.split or not self.multi_sequence_db, "split must be provided for multi_sequence_db"
        dataset = self.dataset.lower()

        """some repeated code here to allow better IntelliSense"""
        if dataset in ['617', 'river_ice']:
            from db_info import RiverIceInfo as DBInfo
        elif dataset == 'ctc':
            from db_info import CTCInfo as DBInfo
        elif dataset == 'ipsc':
            from db_info import IPSCInfo as DBInfo
        elif dataset == 'coco':
            from db_info import COCOInfo as DBInfo
        elif dataset in ('ipsc_dev', 'ipsc_2_class', 'ipsc_5_class'):
            from db_info import IPSCDevInfo as DBInfo
        elif dataset == 'ipsc_patches':
            from db_info import IPSCPatchesInfo as DBInfo
        elif dataset in ['cityscapes', 'ctscp']:
            from db_info import CityscapesInfo as DBInfo
        else:
            raise AssertionError('unsupported multi_sequence_db: {}'.format(self.dataset))

        db_splits = DBInfo.DBSplits().__dict__
        sequences = DBInfo.sequences
        seq_ids = db_splits[self.split]
        n_seq = len(seq_ids)

        if self.seq_id >= 0:
            self.seq_start_id = self.seq_end_id = self.seq_id
        if self.seq_start_id < 0:
            self.seq_start_id = 0
        if self.seq_end_id < 0:
            self.seq_end_id = n_seq - 1

        assert self.seq_end_id >= self.seq_start_id, "seq_end_id must be >= start_seq_id"

        return seq_ids, sequences


def get_p2s_eval_suffixes(params: Params, n_all_seq, n_classes):
    start_id = params.start_id
    end_id = params.end_id
    patch_width = params.patch_width
    patch_height = params.patch_height
    min_stride = params.min_stride
    max_stride = params.max_stride
    n_rot = params.n_rot
    min_rot = params.min_rot
    max_rot = params.max_rot
    enable_flip = params.enable_flip

    seq_start_id = params.seq_start_id
    seq_end_id = params.seq_end_id
    is_video = params.is_video

    subsample = params.rle.subsample
    time_as_class = params.rle.time_as_class
    length_as_class = params.rle.length_as_class
    starts_2d = params.rle.starts_2d
    no_starts = params.rle.no_starts
    diff_mask = params.rle.diff_mask
    shared_coord = params.rle.shared_coord
    flat_order = params.rle.flat_order
    enable_instance = params.enable_instance

    # assert end_id >= start_id, f"invalid end_id: {end_id}"

    patch_mode = params.patch_mode
    subseq_mode = params.subseq_mode or start_id > 0

    if params.end_id >= params.start_id:
        subseq_mode = 1

    if patch_width <= 0:
        patch_width = patch_height
    else:
        patch_mode = 1

    if min_stride <= 0:
        min_stride = patch_height
    else:
        patch_mode = 1

    if max_stride <= min_stride:
        max_stride = min_stride
    else:
        patch_mode = 1

    assert max_stride >= min_stride, "min_stride must be <= max_stride"

    enable_resize = params.resize_x or params.resize_y

    if not patch_mode and enable_resize:
        if (patch_height > 0 and patch_height != params.resize_y) or (
                patch_width > 0 and patch_width != params.resize_x):
            patch_mode = 1

    db_suffixes = []

    if params.split_suffix:
        db_suffixes.append(params.split_suffix)

    if enable_resize:
        if params.resize:
            db_suffixes.append(f'resize_{params.resize}')
        else:
            db_suffixes.append(f'resize_{params.resize_x}x{params.resize_y}')

    if subseq_mode:
        db_suffixes += [
            f'{start_id:d}_{end_id:d}',
        ]

    if patch_mode:
        db_suffixes += [
            f'{patch_height:d}_{patch_width:d}',
            f'{min_stride:d}_{max_stride:d}',
        ]

    if params.shuffle:
        db_suffixes.append('rnd')

    if params.sample:
        db_suffixes.append('smp_{}'.format(params.sample))

    if n_rot > 0:
        db_suffixes.append(f'rot_{min_rot:d}_{max_rot:d}_{n_rot:d}')

    if enable_flip:
        db_suffixes.append('flip')

    if enable_instance:
        db_suffixes.append('inst')

    db_suffix = '-'.join(db_suffixes)

    seq_suffix = ''
    if seq_start_id > 0 or seq_end_id < n_all_seq - 1:
        seq_suffix = f'seq_{seq_start_id}_{seq_end_id}'

    vid_suffixes = []
    if is_video:
        length = params.vid.length
        stride = params.vid.stride
        sample = params.vid.sample

        """video specific suffixes"""
        assert length > 1, "video length must be > 1"

        vid_suffixes.append(f'length-{length}')
        if stride:
            vid_suffixes.append(f'stride-{stride}')
        if sample:
            vid_suffixes.append(f'sample-{sample}')

    vid_suffix = '-'.join(vid_suffixes)

    """RLE specific suffixes"""
    rle_suffixes = []
    if subsample > 1:
        rle_suffixes.append(f'sub_{subsample}')

    if diff_mask == 1:
        rle_suffixes.append(f'dm')
    elif diff_mask == 2:
        rle_suffixes.append(f'dm2')

    if no_starts:
        assert not starts_2d, "no_starts and starts_2d cannot both be enabled"
        rle_suffixes.append(f'nos')

    if starts_2d:
        rle_suffixes.append(f'2d')

    if shared_coord:
        assert starts_2d, "shared_coord should only be used with shared_coord"
        assert not length_as_class, "shared_coord cannot be used with length_as_class"
        rle_suffixes.append(f'sco')

    multi_class = n_classes > 2

    if is_video:
        if time_as_class:
            if length_as_class:
                rle_suffixes.append(f'ltac')
            else:
                rle_suffixes.append(f'tac')

        elif length_as_class:
            assert multi_class, "multi_class must be enabled for length_as_class"
            rle_suffixes.append(f'lac')
        if multi_class:
            rle_suffixes.append(f'mc')
    else:
        if length_as_class:
            assert multi_class, "multi_class must be enabled for length_as_class"
            rle_suffixes.append(f'lac')
        elif multi_class:
            rle_suffixes.append(f'mc')

    if flat_order != 'C':
        rle_suffixes.append(f'flat_{flat_order}')

    if flat_order != 'C':
        rle_suffixes.append(f'flat_{flat_order}')

    rle_suffix = '-'.join(rle_suffixes)

    patch_suffixes = []
    if params.save_suffixes:
        patch_suffixes += params.save_suffixes

    if params.batch_size:
        patch_suffixes.append(f'batch_{params.batch_size}')

    if params.patch_suffixes:
        patch_suffixes += params.patch_suffixes

    patch_suffix = '-'.join(patch_suffixes)

    stitched_suffix = patch_suffix
    if params.stitched_suffixes:
        stitched_suffix = f'{stitched_suffix}-{"-".join(params.stitched_suffixes)}'

    return db_suffix, seq_suffix, vid_suffix, rle_suffix, patch_suffix, stitched_suffix


def eval_stitched(stitched_img, gt_labels_orig, class_id_to_name, class_id_to_col):
    if len(gt_labels_orig.shape) == 3:
        gt_labels = task_utils.mask_to_gs(gt_labels_orig)
    else:
        gt_labels = gt_labels_orig

    if len(stitched_img.shape) == 3:
        seg_labels = task_utils.mask_to_gs(stitched_img)
    else:
        seg_labels = stitched_img

    metrics = Metrics(class_id_to_name, class_id_to_col)
    metrics.update(seg_labels, gt_labels)

    return metrics

    # pix_acc = eval_segm.pixel_accuracy(seg_labels, gt_labels, class_ids)
    # _acc, mean_acc = eval_segm.mean_accuracy(seg_labels, gt_labels, class_ids, return_acc=1)
    # _IU, mean_IU = eval_segm.iu(seg_labels, gt_labels, class_ids, return_iu=1)
    # fw_IU = eval_segm.frequency_weighted_IU(seg_labels, gt_labels, class_ids, return_freq=0)


def stitch_p2s_patches(
        src_img, img_eval_infos, img_patch_infos, patch_vid_reader, patch_seq_path,
        n_classes, subsample, is_video, ovl_cmb, mask_cmb,
        class_id_to_col, show_img, img_id, post_sample):
    n_patches = len(img_patch_infos)
    assert len(img_eval_infos) == n_patches, "img_eval_infos len mismatch"

    multi_class = n_classes > 2

    img_h, img_w = src_img.shape[:2]
    n_rows, n_cols = src_img.shape[:2]

    patch_iter = zip(img_patch_infos, img_eval_infos)
    if n_patches > 100:
        patch_iter = tqdm(patch_iter, ncols=100, total=n_patches, desc=img_id)

    if subsample > 1 and post_sample:
        n_rows, n_cols = n_rows // subsample, n_cols // subsample

    if ovl_cmb:
        stitched_img = np.full((n_patches, n_rows, n_cols), np.nan, dtype=np.float16)
    else:
        stitched_img = np.zeros((n_rows, n_cols), dtype=np.uint8)

    for patch_id, (img_patch_info, eval_infos) in enumerate(patch_iter):
        seq = img_patch_info['seq']
        img_id = img_patch_info['img_id']
        frame_id = int(img_patch_info['frame_id'])

        if not is_video:
            assert len(eval_infos) == 1, "eval_info must be unique for static segmentation"

        patch_imgs = []
        for eval_info in eval_infos:
            eval_image_id = eval_info['image_id']
            eval_src_frame_id = int(eval_info['src_frame_id'])

            eval_image_id_split = eval_image_id.split('/')
            eval_seq = '/'.join(eval_image_id_split[:-1])
            eval_image_id = eval_image_id_split[-1]

            assert eval_image_id == img_id, "eval_image_id mismatch"
            assert eval_seq == seq, "eval_seq mismatch"
            assert eval_src_frame_id == frame_id, "eval_src_frame_id mismatch"

            if patch_vid_reader is not None:
                if is_video:
                    vid_frame_id = eval_info["out_frame_id"]
                else:
                    vid_frame_id = frame_id
                patch_img_ = task_utils.read_frame(patch_vid_reader, vid_frame_id - 1)
            else:
                patch_file_name = f'{eval_image_id}'
                # if is_video:
                #     vid_frame_id = eval_info["out_frame_id"]
                #     patch_file_name = f'{patch_file_name}_{vid_frame_id}'

                patch_seq_dir, patch_ext = os.path.splitext(patch_seq_path)
                patch_img_path = linux_path(patch_seq_dir, f'{patch_file_name}{patch_ext}')
                assert os.path.isfile(patch_img_path), f"nonexistent patch_img_path:\n{patch_img_path}"

                im = Image.open(patch_img_path)
                patch_img_ = np.array(im)

            patch_img_ = task_utils.mask_to_gs(patch_img_)

            patch_imgs.append(patch_img_)

        if len(eval_infos) == 1:
            patch_img = patch_imgs[0]
        else:
            if multi_class:
                patch_imgs_all = np.stack(patch_imgs, axis=0)
                if mask_cmb == 'max':
                    patch_img = np.amax(patch_imgs_all, axis=0)
                elif mask_cmb == 'min':
                    patch_img = np.amin(patch_imgs_all, axis=0)
                elif mask_cmb == 'mode' or mask_cmb == 'vote':
                    mode_info = stats.mode(patch_imgs_all, axis=0)
                    patch_img = mode_info.mode
                    # values, counts = np.unique(patch_imgs_all, return_counts=True, axis=2)
                else:
                    raise AssertionError(f'invalid multi class mask_cmb: {mask_cmb}')
                # print()
            else:
                if mask_cmb == 'or':
                    patch_img = np.logical_or.reduce(patch_imgs)
                elif mask_cmb == 'and':
                    patch_img = np.logical_and.reduce(patch_imgs)
                else:
                    patch_imgs_all = np.stack(patch_imgs, axis=0)
                    if mask_cmb == 'max':
                        patch_img = np.amax(patch_imgs_all, axis=0)
                    elif mask_cmb == 'min':
                        patch_img = np.amin(patch_imgs_all, axis=0)
                    elif mask_cmb == 'mode' or mask_cmb == 'vote':
                        mode_info = stats.mode(patch_imgs_all, axis=0)
                        patch_img = mode_info.mode
                    else:
                        raise AssertionError(f'invalid mask_cmb: {mask_cmb}')

        patch_info = img_patch_info['patch_info']
        win_orig = patch_info['window']

        min_row, max_row, min_col, max_col = win_orig

        assert 0 <= min_row <= img_h, f"invalid min_row: {min_row} for img_h: {img_h}"
        assert min_row < max_row <= img_h, f"invalid max_row: {max_row} for  min_row: {min_row} and img_h: {img_h}"
        assert 0 <= min_col <= img_w, f"invalid min_col: {min_col} for img_w: {img_w}"
        assert min_col < max_col <= img_w, f"invalid max_col: {max_col} for min_col: {min_col} and img_w: {img_w}"

        win_subsampled = win_orig[:]
        if subsample > 1:
            if post_sample:
                win_subsampled = [k // subsample for k in win_orig]
                win_rec = [k * subsample for k in win_subsampled]
                assert win_rec == win_orig, "win_rec mismatch"
            else:
                patch_img = cv2.resize(patch_img, (0, 0), fx=subsample, fy=subsample,
                                       interpolation=cv2.INTER_NEAREST_EXACT)
                # patch_img = task_utils.supersample_mask(patch_img, subsample, n_classes, is_vis=0)

        min_row, max_row, min_col, max_col = win_subsampled

        rot = patch_info['rot']
        flip = patch_info['flip']

        assert rot == 0, "rot must be 0"
        assert flip == "none", "flip must be none"

        if ovl_cmb:
            stitched_img[patch_id, min_row:max_row, min_col:max_col] = patch_img
        else:
            stitched_img[min_row:max_row, min_col:max_col] = patch_img
            if show_img == 2:
                patch_img_vis = task_utils.mask_id_to_vis_bgr(patch_img, class_id_to_col)
                stitched_img_vis = task_utils.mask_id_to_vis_bgr(stitched_img, class_id_to_col)
                cv2.rectangle(stitched_img_vis, (min_col, min_row), (max_col, max_row), (255, 255, 255), 1)
                patch_img_vis = cv2.resize(patch_img_vis, (640, 640))
                stitched_img_vis = cv2.resize(stitched_img_vis, (640, 640))
                stitched_img_vis = vis_utils.annotate(stitched_img_vis, f"{img_id}")
                patch_img_vis = vis_utils.annotate(patch_img_vis, f"{img_id}")
                cv2.imshow('patch_img_vis', patch_img_vis)
                cv2.imshow('stitched_img_vis', stitched_img_vis)
                cv2.waitKey(250)

    if ovl_cmb:
        if ovl_cmb == 'max':
            stitched_img = np.nanmax(stitched_img, axis=0)
        elif ovl_cmb == 'min':
            stitched_img = np.nanmin(stitched_img, axis=0)
        elif ovl_cmb == 'mode':
            mode_info = stats.mode(stitched_img, axis=0, nan_policy='omit')
            stitched_img = mode_info.mode
        else:
            raise AssertionError(f'invalid ovl_cmb: {ovl_cmb}')

    # if params.resize:
    #     stitched_img = task_utils.resize_mask(stitched_img, src_img.shape, n_classes, is_vis=0)

    if subsample > 1 and post_sample:
        # stitched_img_unique_orig = np.unique(stitched_img)
        stitched_img = cv2.resize(stitched_img, (0, 0), fx=subsample, fy=subsample,
                                  interpolation=cv2.INTER_NEAREST_EXACT)
        # stitched_img = task_utils.supersample_mask(stitched_img, subsample, n_classes, is_vis=0)
        # stitched_img_unique_res = np.unique(stitched_img)

    return stitched_img, n_patches


def stitch_patches(src_img, img_fname_no_ext, label_diff, params):
    """
    
    :param src_img:
    :param Params params: 
    :return: 
    """
    pause_after_frame = 1

    n_rows, n_cols, n_channels = src_img.shape

    if params.method == 0:
        stitched_img = None
    else:
        """stacked"""
        stitched_img = np.zeros((n_rows, n_cols, n_channels), dtype=np.uint8)

    out_id = 0
    # skip_id = 0
    min_row = 0

    while True:
        max_row = min_row + params.patch_height
        if max_row > n_rows:
            diff = max_row - n_rows
            min_row -= diff
            max_row -= diff

        curr_row = None
        min_col = 0
        while True:
            max_col = min_col + params.patch_width
            if max_col > n_cols:
                diff = max_col - n_cols
                min_col -= diff
                max_col -= diff

            patch_img_fname = '{:s}_{:d}'.format(img_fname_no_ext, out_id + 1)
            patch_src_img_fname = linux_path(params.patch_seq_path,
                                             '{:s}.{:s}'.format(patch_img_fname, params.patch_ext))

            assert os.path.exists(patch_src_img_fname), f'Patch image does not exist: {patch_src_img_fname}'

            src_patch = cv2.imread(patch_src_img_fname)
            seg_height, seg_width, _ = src_patch.shape

            if seg_width == 2 * params.patch_width or seg_width == 3 * params.patch_width:
                _start_id = seg_width - params.patch_width
                src_patch = src_patch[:, _start_id:]

            if params.normalize_patches:
                src_patch = (src_patch * label_diff).astype(np.uint8)
            out_id += 1

            if params.method == 0:
                if curr_row is None:
                    curr_row = src_patch
                else:
                    curr_row = np.concatenate((curr_row, src_patch), axis=1)
            else:
                stitched_img[min_row:max_row, min_col:max_col, :] = src_patch

            if params.show_img:
                disp_img = src_img.copy()
                cv2.rectangle(disp_img, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)

                stitched_img_disp = stitched_img
                if params.disp_resize_factor != 1:
                    disp_img = cv2.resize(disp_img, (0, 0), fx=params.disp_resize_factor,
                                          fy=params.disp_resize_factor)
                    if stitched_img_disp is not None:
                        stitched_img_disp = cv2.resize(stitched_img_disp, (0, 0), fx=params.disp_resize_factor,
                                                       fy=params.disp_resize_factor)

                cv2.imshow('disp_img', disp_img)
                cv2.imshow('src_patch', src_patch)

                if stitched_img_disp is not None:
                    cv2.imshow('stacked_img', stitched_img_disp)
                if curr_row is not None:
                    cv2.imshow('curr_row', curr_row)

                k = cv2.waitKey(1 - pause_after_frame)
                if k == 27:
                    sys.exit(0)
                elif k == 32:
                    pause_after_frame = 1 - pause_after_frame
            if max_col >= n_cols:
                break

            min_col = max_col

        if params.method == 0:
            if stitched_img is None:
                stitched_img = curr_row
            else:
                stitched_img = np.concatenate((stitched_img, curr_row), axis=0)

        if max_row >= n_rows:
            break

        min_row = max_row

    return stitched_img, out_id


def run(params: Params, class_info):
    video_exts = ['mp4', 'mkv', 'avi', 'mpg', 'mpeg', 'mjpg']

    # if not params.src_path:
    #     assert params.db_root_dir, "either params.src_path or params.db_root_dir must be provided"
    #
    #     params.src_path = linux_path(params.db_root_dir, params.seq_name, 'images')

    print('Reading source images from: {}'.format(params.src_path))

    enable_resize = params.resize_x or params.resize_y

    write_to_video = params.out_ext in video_exts
    src_from_video = params.images_ext in video_exts
    labels_from_video = params.labels_ext in video_exts
    patch_from_video = params.patch_ext in video_exts
    eval_mode = params.labels_path and params.labels_ext

    src_files = src_vid_reader = src_vid_width = src_vid_height = None
    labels_files = labels_vid_reader = None
    patch_vid_reader = None

    src_vid_path = f'{params.src_path}.{params.images_ext}'
    if src_from_video:
        assert not params.shuffle, "src_from_video isd incompatible with shuffled source IDs"
        src_vid_reader, src_vid_width, src_vid_height, total_frames = task_utils.load_video(src_vid_path)
    else:
        src_files = [k for k in os.listdir(params.src_path) if k.endswith('.{:s}'.format(params.images_ext))]
        total_frames = len(src_files)
        src_files.sort(key=sort_key)
        assert total_frames > 0, f'No input frames of type {params.images_ext} found in {params.src_path}'

    if eval_mode:
        if labels_from_video:
            labels_path = f'{params.labels_path}.{params.labels_ext}'
            labels_vid_reader, labels_vid_width, labels_vid_height, labels_total_frames = task_utils.load_video(
                labels_path)
            if src_from_video:
                assert src_vid_width == labels_vid_width, "labels_vid_width mismatch"
                assert src_vid_height == labels_vid_height, "labels_vid_height mismatch"
        else:
            _, labels_files, labels_total_frames = read_data(labels_path=params.labels_path,
                                                             labels_ext=params.labels_ext)

        assert labels_total_frames == total_frames, 'Mismatch between no. of frames in GT and labels'

    patch_seq_path = f'{params.patch_seq_path}.{params.patch_ext}'
    if patch_from_video:
        patch_vid_reader, patch_vid_width, patch_vid_height, patch_total_frames = task_utils.load_video(
            patch_seq_path)

        if src_from_video:
            assert src_vid_width == patch_vid_width, "patch_vid_width mismatch"
            assert src_vid_height == patch_vid_height, "patch_vid_height mismatch"
            assert total_frames == patch_total_frames, "patch_total_frames mismatch"

    print('total_frames: {}'.format(total_frames))

    classes, composite_classes, n_classes = class_info[:3]
    if len(class_info) > 3:
        class_ids_map = class_info[3]
    else:
        class_ids_map = None

    multi_class = n_classes > 2

    class_id_to_col = {
        class_id: col for class_id, (class_name, col) in enumerate(classes)
    }
    palette = []
    for class_id, col in class_id_to_col.items():
        col_rgb = class_id_to_col[class_id][::-1]
        palette.append(col_rgb)

    palette_flat = [value for color in palette for value in color]
    class_id_to_name = {
        class_id: class_name for class_id, (class_name, col) in enumerate(classes)
        if class_id > 0 or not params.frg_metrics
    }

    for name_, col_, class_ids_ in composite_classes:
        class_id_to_name[class_ids_] = name_

    class_ids = list(range(n_classes))

    if params.n_frames <= 0:
        params.n_frames = total_frames

    if params.end_id < params.start_id:
        params.end_id = params.n_frames - 1

    if params.patch_width <= 0:
        params.patch_width = params.patch_height

    if not params.patch_seq_path:
        if not params.patch_seq_name:
            params.patch_seq_name = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                params.seq_name, params.start_id, params.end_id, params.patch_height, params.patch_width,
                params.patch_height, params.patch_width)
        params.patch_seq_path = linux_path(params.db_root_dir, params.patch_seq_name, params.patch_seq_type)
        assert os.path.isdir(params.patch_seq_path), f'patch_seq_path does not exist: {params.patch_seq_path}'
    else:
        if not params.patch_seq_name:
            params.patch_seq_name = os.path.basename(params.patch_seq_path)

    if not params.stitched_raw_path:
        stitched_seq_name = '{}_stitched_{}'.format(params.patch_seq_name, params.method)
        if params.stacked:
            stitched_seq_name = '{}_stacked'.format(stitched_seq_name)
            params.method = 1
        stitched_seq_name = '{}_{}_{}'.format(stitched_seq_name, params.start_id, params.end_id)
        params.stitched_raw_path = linux_path(params.db_root_dir, stitched_seq_name, params.patch_seq_type)

    gt_labels_orig = gt_labels = stitched_vid = None

    if not params.save_stitched:
        print('not saving stitched masks')

    if write_to_video:
        stitched_vis_dir = os.path.dirname(params.stitched_vis_path)
        stitched_vid_path = params.stitched_vis_path
        if not stitched_vid_path.endswith(f'.{params.out_ext}'):
            stitched_vid_path = f'{stitched_vid_path}.{params.out_ext}'
        if params.save_stitched and params.save_vis:
            # is_windows = os.name == 'nt'
            stitched_vid = vis_utils.get_video_writer(
                stitched_vid_path,
                # cv=is_windows
            )
    else:
        stitched_vis_dir = params.stitched_vis_path

    stitched_raw_dir = params.stitched_raw_path
    stitched_mapped_dir = params.stitched_mapped_path

    if params.save_stitched:
        print(f'Writing raw masks to: {stitched_raw_dir}')
        os.makedirs(stitched_raw_dir, exist_ok=True)

        if params.save_mapped:
            assert class_ids_map is not None, "class_ids_map must be provided to save_mapped"
            print(f'Writing mapped masks to: {stitched_mapped_dir}')
            os.makedirs(stitched_mapped_dir, exist_ok=True)

        if params.save_vis:
            print(f'Writing vis masks to: {params.stitched_vis_path}')
            os.makedirs(stitched_vis_dir, exist_ok=True)

    log_fname = None

    # log_fname = linux_path(stitched_raw_dir, f'log_{timestamp:s}.txt')
    # print_and_write(f'Saving log to: {log_fname}', log_fname)

    print_and_write(f'Reading patch images from: {params.patch_seq_path}', log_fname)

    n_patches_all = 0
    label_diff = int(255.0 / (params.n_classes - 1))

    avg_metrics = Metrics(class_id_to_name, class_id_to_col)
    img_to_metrics = dict()

    img_ids = list(range(params.start_id, params.end_id + 1))

    if params.sample > 0:
        img_ids = img_ids[::params.sample]

    _n_frames = len(img_ids)

    img_iter = img_ids
    # img_iter = range(params.start_id, params.start_id + 2)
    if not params.mp or not eval_mode:
        from tqdm import tqdm
        img_iter = tqdm(img_iter, ncols=150)

    if params.model_type == 'p2s':
        src_ids = list(params.seq_patch_info.keys())

    for id_, img_id in enumerate(img_iter):
        img_fname_no_ext = None
        if src_from_video:
            src_img = task_utils.read_frame(src_vid_reader, img_id, src_vid_path)
        else:

            img_fname = src_files[img_id]
            img_fname_no_ext, img_fname_ext = os.path.splitext(img_fname)

            if params.model_type == 'p2s':
                src_id = src_ids[id_]

                assert params.shuffle or img_fname_no_ext == src_id, "src_id mismatch"

                img_fname = f'{src_id}{img_fname_ext}'
                img_fname_no_ext = src_id

            src_img_fname = linux_path(params.src_path, img_fname)
            src_img_pil = Image.open(src_img_fname)

            src_img = np.array(src_img_pil)

            if len(src_img.shape) == 2:
                """COCO has some amazingly annoying single-channel greyscale images"""
                src_img = np.stack((src_img, src_img, src_img), axis=2)

            # src_img = cv2.imread(src_img_fname)
            # assert src_img is not None, f'Labels image could not be read: {src_img_fname}'

        img_h, img_w = src_img.shape[:2]

        if eval_mode:
            if labels_from_video:
                gt_labels_orig = task_utils.read_frame(labels_vid_reader, img_id)
                if not multi_class:
                    gt_labels_orig[gt_labels_orig > 0] = 255
                gt_labels_orig = task_utils.mask_vis_to_id(gt_labels_orig, n_classes)
                gt_labels_orig = task_utils.mask_to_gs(gt_labels_orig)
            else:
                labels_img_fname = linux_path(params.labels_path, img_fname_no_ext + '.{}'.format(params.labels_ext))
                gt_labels_pil = Image.open(labels_img_fname)

                mask_w, mask_h = gt_labels_pil.size[:2]

                assert img_h == mask_h, "img_h mismatch"
                assert img_w == mask_w, "img_w mismatch"

                # if params.resize:
                #     gt_labels_pil = gt_labels_pil.resize((params.resize, params.resize))

                gt_labels_orig = gt_labels_np = np.array(gt_labels_pil)
                # gt_labels_unique = np.unique(gt_labels_orig)

                if enable_resize:
                    gt_labels_orig = cv2.resize(gt_labels_orig, (params.resize_x, params.resize_y),
                                                interpolation=cv2.INTER_NEAREST_EXACT)

                    # gt_labels_unique_res = np.unique(gt_labels_orig)

                if params.subsample_gt:
                    gt_labels_orig = cv2.resize(gt_labels_orig, (0, 0),
                                                fx=1.0 / params.rle.subsample,
                                                fy=1.0 / params.rle.subsample,
                                                interpolation=cv2.INTER_NEAREST_EXACT)

                    gt_labels_orig = cv2.resize(gt_labels_orig, (0, 0),
                                                fx=params.rle.subsample,
                                                fy=params.rle.subsample,
                                                interpolation=cv2.INTER_NEAREST_EXACT)

                if not multi_class:
                    gt_labels_orig[gt_labels_orig > 0] = 1

            if class_ids_map is not None:
                gt_labels_orig = eval_utils.map_class_ids(gt_labels_orig, class_ids_map)

        # if params.debug:
        #     gt_labels_orig_vis = task_utils.mask_id_to_vis_bgr(gt_labels_orig, class_id_to_col)
        #     cv2.imshow('gt_labels_orig_vis', gt_labels_orig_vis)

        if enable_resize:
            src_img = cv2.resize(src_img, (params.resize_x, params.resize_y))

        raw_img_path = linux_path(stitched_raw_dir, f'{img_fname_no_ext}.png')

        if params.load_stitched:
            im = Image.open(raw_img_path)
            stitched_img = np.array(im)
            n_patches = 0
        else:
            if params.model_type == 'p2s':
                img_fname_no_ext = src_id

                img_patch_infos = params.seq_patch_info[src_id]
                img_eval_infos = [params.frame_to_eval_info[int(img_patch_info['frame_id'])]
                                  for img_patch_info in img_patch_infos]
                stitched_img, n_patches = stitch_p2s_patches(
                    src_img, img_eval_infos, img_patch_infos, patch_vid_reader, patch_seq_path,
                    n_classes, params.rle.subsample, params.is_video, params.ovl_cmb, params.mask_cmb,
                    class_id_to_col, params.show_img, src_id, params.post_sample)
                # if params.debug:
                #     stitched_img_vis = task_utils.mask_id_to_vis_bgr(stitched_img, class_id_to_col)
                #     cv2.imshow('stitched_img_vis', stitched_img_vis)
                #     cv2.waitKey(0)
                stitched_img = task_utils.mask_to_gs(stitched_img)
            else:
                if params.method == -1:
                    """no stitching since patch size = image size"""
                    patch_src_img_fname = linux_path(params.patch_seq_path, f'{img_fname_no_ext}.{params.patch_ext}')
                    patch_src_img_fname = os.path.abspath(patch_src_img_fname)
                    if not os.path.exists(patch_src_img_fname):
                        msg = f'Patch image does not exist: {patch_src_img_fname}'
                        if not params.allow_missing:
                            raise AssertionError(msg)

                        print('\n' + msg + '\n')

                        stitched_img = np.zeros((img_h, img_w), dtype=np.uint8)
                    else:
                        im = Image.open(patch_src_img_fname)
                        stitched_img = np.array(im)
                    n_patches = 1
                else:
                    stitched_img, n_patches = stitch_patches(src_img, img_fname_no_ext, label_diff, params)

        if eval_mode:
            # cv2.imshow('gt_labels_orig', gt_labels_orig)
            # cv2.waitKey(0)

            # if not params.labels_bgr:
            #     gt_labels_orig = task_utils.mask_vis_to_id(gt_labels_orig, n_classes)

            metrics = eval_stitched(stitched_img, gt_labels_orig, class_id_to_name, class_id_to_col)
            img_to_metrics[f'{params.patch_seq_name}/{img_fname_no_ext}'] = metrics.to_dict()
            avg_metrics.update_average(metrics, id_)

            if n_classes <= 3:
                log_txt = metrics.to_str('dice')
            else:
                """too many classes for printing class-specific metrics"""
                avg_miou = avg_metrics.miou * 100
                img_miou = metrics.miou * 100

                avg_camiou = avg_metrics.ca_miou * 100
                img_camiou = metrics.ca_miou * 100

                avg_mean_dice = avg_metrics.dice['mean'] * 100
                img_mean_dice = metrics.dice['mean'] * 100

                avg_mean_iou = avg_metrics.iu['mean'] * 100
                img_mean_iou = metrics.iu['mean'] * 100

                avg_mean_acc = avg_metrics.acc['mean'] * 100
                img_mean_acc = metrics.acc['mean'] * 100

                log_txt = f'miou: {img_miou:4.1f} {avg_miou:4.1f} '
                log_txt += f' camiou: {img_camiou:4.1f} {avg_camiou:4.1f} '
                log_txt += f' iou: {img_mean_iou:4.1f} {avg_mean_iou:4.1f} '
                log_txt += f' dice: {img_mean_dice:4.1f} {avg_mean_dice:4.1f} '
                log_txt += f' acc: {img_mean_acc:4.1f} {avg_mean_acc:4.1f} '

            log_txt = f"frame {id_ + 1:d}/{_n_frames:d}: {img_fname_no_ext} :: {log_txt}"
            if not params.mp or not eval_mode:
                img_iter.set_description(log_txt)

            # print_and_write(log_txt, log_fname)

            # avg_log_txt = avg_metrics.to_str('dice')
            # avg_log_txt = f"average :: {avg_log_txt}"
            # print_and_write(avg_log_txt, log_fname)

        if not params.normalize_patches:
            seg_img = task_utils.mask_id_to_vis_bgr(stitched_img, class_id_to_col)
            # seg_img = task_utils.mask_id_to_vis(stitched_img, n_classes)
            # seg_img = (stitched_img * label_diff).astype(np.uint8)
        else:
            seg_img = stitched_img

        if params.stacked:
            if eval_mode:
                labels_img = task_utils.mask_id_to_vis_bgr(gt_labels_orig, class_id_to_col)
                # labels_img = task_utils.mask_id_to_vis(gt_labels_orig, n_classes)
                # labels_img = (gt_labels_orig * label_diff).astype(np.uint8)
                stitched = np.concatenate((src_img, labels_img), axis=1)
            else:
                stitched = src_img
            vis_img = np.concatenate((stitched, seg_img), axis=1)
        else:
            vis_img = seg_img

        vis_img = resize_ar(vis_img, params.width, params.height)

        if vis_img is not None:
            vis_img = vis_utils.annotate(vis_img, img_fname_no_ext)

        if params.show_img:
            cv2.imshow('vis_img', vis_img)
            k = cv2.waitKey(0)
            if k == 27:
                exit(0)

        if params.save_stitched:
            mask_img_pil = Image.fromarray(stitched_img)
            mask_img_pil = mask_img_pil.convert('P')
            mask_img_pil.putpalette(palette_flat)
            mask_img_pil.save(raw_img_path)

            if params.save_mapped:
                mapped_img_path = linux_path(stitched_mapped_dir, f'{img_fname_no_ext}.png')
                stitched_img_mapped = eval_utils.unmap_class_ids(stitched_img, class_ids_map)
                Image.fromarray(stitched_img_mapped).save(mapped_img_path)

            if params.save_vis:
                if write_to_video:
                    vis_utils.write_frames_to_videos(stitched_vid, vis_img)
                else:
                    # if params.resize_factor != 1:
                    #     vis_img = cv2.resize(vis_img, (0, 0), fx=params.resize_factor, fy=params.resize_factor)
                    stacked_img_path = linux_path(params.stitched_vis_path, f'{img_fname_no_ext}.{params.out_ext}')
                    cv2.imwrite(stacked_img_path, vis_img)

        n_patches_all += n_patches

    print('\nTotal patches processed: {}\n'.format(n_patches_all))

    if params.show_img:
        cv2.destroyAllWindows()

    if stitched_vid is not None:
        vis_utils.close_video_writers(stitched_vid)

    if params.del_patch_seq:
        print('Removing patch folder {}'.format(params.patch_seq_path))
        shutil.rmtree(params.patch_seq_path)

    if eval_mode:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        img_to_metrics_path = linux_path(stitched_raw_dir, f'{params.patch_seq_name}_{time_stamp}.csv')

        # log_txt = avg_metrics.to_str()
        # img_iter.set_description(log_txt)

        # print_and_write(log_txt, log_fname)

        metrics_to_csv(img_to_metrics, img_to_metrics_path)

        # print_and_write('Saved log to: {}'.format(log_fname), log_fname)
        # print_and_write('Read patch images from: {}'.format(params.patch_seq_path), log_fname)

    return img_to_metrics


def run_sws(params: Params, seq_ids, seq_info, class_info):
    params.patch_seq_name = f'{params.in_name}'

    eval_root_dir = linux_path(params.in_root_dir, params.in_name)

    out_name = params.out_name
    in_iter_name = 'vis'

    if params.in_iter:
        in_iter_name = f'{in_iter_name}-iter_{params.in_iter}'
        out_name = f'{out_name}-{params.in_iter}'

    patch_seq_dir = linux_path(eval_root_dir, in_iter_name, f'masks')

    metrics_root_dir = linux_path(params.out_dir, f'#metrics')
    os.makedirs(metrics_root_dir, exist_ok=True)

    out_root_dir = linux_path(params.out_dir, f'{out_name}')

    img_to_metrics_path = linux_path(metrics_root_dir, f'{out_name}.csv')
    img_to_metrics_path_dup = linux_path(out_root_dir, 'metrics.csv')

    stitched_seq_dir = linux_path(out_root_dir, f'raw')
    stitched_mapped_dir = linux_path(out_root_dir, f'mapped')
    stitched_vis_dir = linux_path(out_root_dir, 'vis')

    os.makedirs(stitched_seq_dir, exist_ok=True)
    if params.save_mapped:
        os.makedirs(stitched_mapped_dir, exist_ok=True)
    if params.save_vis:
        os.makedirs(stitched_vis_dir, exist_ok=True)

    all_img_to_metrics = dict()

    n_seq = len(seq_ids)

    for _id, seq_id in enumerate(seq_ids):
        seq_params: Params = copy.deepcopy(params)

        seq_name, n_frames = seq_info[seq_id]
        print(f'seq {_id + 1} / {n_seq}: {seq_name}')

        seq_params.patch_seq_name = seq_name

        params.src_path = linux_path(params.images_path, f'{seq_name}')
        params.labels_path = linux_path(params.images_path, f'{seq_name}', 'masks')

        seq_params.src_path = linux_path(params.images_path, f'{seq_name}')
        seq_params.labels_path = linux_path(params.images_path, f'{seq_name}', 'masks')

        seq_params.patch_seq_path = linux_path(patch_seq_dir, seq_name)
        seq_params.stitched_raw_path = linux_path(stitched_seq_dir, seq_name)
        seq_params.stitched_mapped_path = linux_path(stitched_mapped_dir, seq_name)
        seq_params.stitched_vis_path = linux_path(stitched_vis_dir, seq_name)

        seq_img_to_metrics = run(seq_params, class_info)
        all_img_to_metrics.update(seq_img_to_metrics)

    if img_to_metrics_path is not None:
        metrics_to_csv(all_img_to_metrics, img_to_metrics_path, img_to_metrics_path_dup)


def run_p2s(params: Params, seq_ids, seq_info, seq_to_patch_info, eval_root_dir, eval_info, class_info,
            patch_suffix, out_suffixes=None, video_ids=None):
    patch_seq_dir = linux_path(eval_root_dir, f'masks-{patch_suffix}')
    logits_patch_seq_dir = linux_path(eval_root_dir, f'masks_logits-{patch_suffix}')

    out_name = params.out_name
    if out_suffixes:
        out_suffix = '-'.join(out_suffixes)
        out_name = f'{out_name}-{out_suffix}'

    metrics_root_dir = linux_path(params.out_dir, f'#metrics')
    os.makedirs(metrics_root_dir, exist_ok=True)

    out_root_dir = linux_path(params.out_dir, f'{out_name}')

    # time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    img_to_metrics_path = linux_path(metrics_root_dir, f'{out_name}.csv')
    img_to_metrics_path_dup = linux_path(out_root_dir, 'metrics.csv')

    img_to_metrics_path_logits = linux_path(metrics_root_dir, f'{out_name}-logits.csv')
    img_to_metrics_path_logits_dup = linux_path(out_root_dir, 'metrics-logits.csv')

    stitched_seq_dir = linux_path(out_root_dir, f'raw')
    stitched_mapped_dir = linux_path(out_root_dir, f'mapped')
    stitched_vis_dir = linux_path(out_root_dir, f'vis')
    if params.save_mapped:
        os.makedirs(stitched_mapped_dir, exist_ok=True)
    if params.save_vis:
        os.makedirs(stitched_vis_dir, exist_ok=True)

    os.makedirs(stitched_seq_dir, exist_ok=True)

    if params.logits:
        logits_stitched_seq_dir = linux_path(out_root_dir, f'raw_logits')
        logits_stitched_mapped_dir = linux_path(out_root_dir, f'mapped_logits')
        logits_stitched_vis_dir = linux_path(out_root_dir, f'vis_logits')
        os.makedirs(logits_stitched_seq_dir, exist_ok=True)
        if params.save_mapped:
            os.makedirs(logits_stitched_mapped_dir, exist_ok=True)
        if params.save_vis:
            os.makedirs(logits_stitched_vis_dir, exist_ok=True)

    all_img_to_metrics = dict()
    all_img_to_metrics_logits = dict()

    n_seq = len(seq_ids)

    for _id, seq_id in enumerate(seq_ids):
        seq_params: Params = copy.deepcopy(params)

        seq_name, n_frames = seq_info[seq_id]
        print(f'seq {_id + 1} / {n_seq}: {seq_name}')

        seq_params.patch_seq_name = seq_name
        seq_eval_info = eval_info[seq_name]

        frame_to_eval_info = defaultdict(list)
        for k in seq_eval_info:
            if video_ids is not None and k['vid_id'] not in video_ids:
                continue
            frame_to_eval_info[k['src_frame_id']].append(k)

        frame_to_eval_info = OrderedDict(sorted(frame_to_eval_info.items()))

        seq_params.frame_to_eval_info = frame_to_eval_info
        seq_params.seq_patch_info = seq_to_patch_info[seq_name]

        if not seq_params.src_path:
            if params.image_dir:
                seq_params.src_path = linux_path(params.images_path, params.image_dir, seq_name)
            else:
                seq_params.src_path = linux_path(params.images_path, seq_name)

        if not seq_params.labels_path:
            if params.labels_dir:
                seq_params.labels_path = linux_path(params.images_path, params.labels_dir, seq_name)
            else:
                seq_params.labels_path = linux_path(params.images_path, f'{seq_name}', 'masks')

        seq_params.patch_seq_path = linux_path(patch_seq_dir, seq_name)
        seq_params.stitched_raw_path = linux_path(stitched_seq_dir, seq_name)
        seq_params.stitched_mapped_path = linux_path(stitched_mapped_dir, seq_name)
        seq_params.stitched_vis_path = linux_path(stitched_vis_dir, seq_name)

        # return
        """run the actual stitching and evaluation operation"""
        seq_img_to_metrics = run(seq_params, class_info)
        all_img_to_metrics.update(seq_img_to_metrics)

        if params.logits:
            seq_params.patch_seq_path = linux_path(logits_patch_seq_dir, seq_name)
            seq_params.stitched_raw_path = linux_path(logits_stitched_seq_dir, seq_name)
            seq_params.stitched_mapped_path = linux_path(logits_stitched_mapped_dir, seq_name)
            seq_params.stitched_vis_path = linux_path(logits_stitched_vis_dir, seq_name)

            seq_img_to_metrics = run(seq_params, class_info)
            all_img_to_metrics_logits.update(seq_img_to_metrics)

    if img_to_metrics_path is not None:
        metrics_to_csv(all_img_to_metrics, img_to_metrics_path, img_to_metrics_path_dup)
        if params.logits:
            metrics_to_csv(all_img_to_metrics_logits, img_to_metrics_path_logits, img_to_metrics_path_logits_dup)


def run_single_ckpt(params: Params, eval_name, patch_images_path, patch_info_json_name,
                    seq_ids, seq_info, class_info, patch_suffix, success=None):
    eval_root_dir = linux_path(params.in_root_dir, eval_name)

    eval_info_json_path = linux_path(eval_root_dir, f'vid_info.json.gz')
    eval_info = task_utils.load_json(eval_info_json_path)

    patch_info_json_path = linux_path(patch_images_path, f'{patch_info_json_name}.json.gz')

    print(f'loading patch_info from : {patch_info_json_path}')

    patch_json_info = task_utils.load_json(patch_info_json_path)
    seq_to_patch_info = defaultdict(lambda: OrderedDict())

    patch_info_key = 'images'
    patch_img_infos = patch_json_info[patch_info_key]
    for patch_img_info in patch_img_infos:
        # try:
        #     src_id = patch_img_info['src_id']
        # except KeyError:

        img_id = patch_img_info['img_id']
        temp = img_id.split('_')
        patch_id = temp[-1]
        src_id = '_'.join(temp[:-1])

        patch_img_info['src_id'] = src_id
        patch_img_info['patch_id'] = patch_id

        seq = patch_img_info['seq']
        seq_patch_info = seq_to_patch_info[seq]
        try:
            src_patch_info = seq_patch_info[src_id]
        except KeyError:
            src_patch_info = []
        src_patch_info.append(patch_img_info)
        seq_patch_info[src_id] = src_patch_info

    if params.is_video and params.strides:
        stride_to_video_ids = dict(
            (int(key), list(map(int, val.split(','))))
            for key, val in eval_info['stride_to_video_ids'].items())

        if len(params.strides) == 1 and params.strides[0] == 0:
            params.strides = list(range(1, params.vid.length + 1))
        for stride in params.strides:
            video_ids = stride_to_video_ids[stride]
            out_suffixes = [f'vstrd-{stride:d}', ]
            # params.video_ids = video_ids
            run_p2s(params, seq_ids, seq_info, seq_to_patch_info, eval_root_dir, eval_info, class_info,
                    patch_suffix, out_suffixes, video_ids)
            if params.enable_instance:
                run_p2s(params, seq_ids, seq_info, seq_to_patch_info, eval_root_dir, eval_info, class_info,
                        patch_suffix, out_suffixes, video_ids)
    else:
        run_p2s(params, seq_ids, seq_info, seq_to_patch_info, eval_root_dir, eval_info, class_info, patch_suffix)

    if success is not None:
        success.value = 1

    return True


def main():
    params = Params()
    paramparse.process(params, cfg_cache=0)

    if params.resize:
        params.resize_x = params.resize_y = params.resize

    if params.resize_x or params.resize_y:
        assert params.resize_x and params.resize_y, \
            "either both or neither of resize_x and resize_y can be specified"

    class_ids_map = None

    if params.dataset in ['cityscapes', 'ctscp']:
        classes, composite_classes, class_ids = read_class_info(params.class_info_path)
        class_ids_map = eval_utils.get_class_ids_map(class_ids)
    else:
        classes, composite_classes = read_class_info(params.class_info_path)

    n_classes = len(classes)

    if params.load_stitched:
        print('loading stitched images')
        params.save_stitched = 0

    class_info = (classes, composite_classes, n_classes)

    if class_ids_map is not None:
        class_info += (class_ids_map,)

    if not params.multi_sequence_db:
        run(params, class_info)
        return

    all_seq_ids, seq_info = params.process()
    n_all_seq = len(all_seq_ids)
    seq_ids = all_seq_ids[params.seq_start_id:params.seq_end_id + 1]

    if params.model_type == 'sws':
        run_sws(params, seq_ids, seq_info, class_info)
        return

    if params.model_type != 'p2s':
        """old-style evaluation from the pre-pix2seq era"""
        for _id, seq_id in enumerate(seq_ids):
            seq_params: Params = copy.deepcopy(params)
            seq_name, n_frames = seq_info[seq_id]
            if params.image_dir:
                seq_params.src_path = linux_path(params.images_path, params.image_dir, seq_name)
            else:
                seq_params.src_path = linux_path(params.images_path, seq_name)

            run(seq_params, class_info)
        return

    assert params.in_root_dir, "in_root_dir must be provided"
    assert params.in_name, "in_name must be provided"
    assert params.out_name, "out_name must be provided"
    assert params.batch_size > 0, "batch_size must be > 0"

    db_suffix, seq_suffix, vid_suffix, rle_suffix, patch_suffix, stitched_suffix = get_p2s_eval_suffixes(
        params, n_all_seq, n_classes)

    eval_name = f'{params.in_name}'

    patch_info_json_name = f'{db_suffix}'
    patch_images_path = f'{params.images_path}-{db_suffix}'

    if seq_suffix:
        patch_info_json_name = f'{patch_info_json_name}-{seq_suffix}'

    if not eval_name.endswith('/'):
        """hack to allow providing the actual absolute path to the results without needlessly and 
        most annoyingly adding random suffixes to it"""
        if params.eval_db_prefix:
            eval_name = f'{eval_name}-{params.eval_db_prefix}-{db_suffix}'
        else:
            eval_name = f'{eval_name}-{db_suffix}'
        if seq_suffix:
            eval_name = f'{eval_name}-{seq_suffix}'
        if vid_suffix:
            eval_name = f'{eval_name}-{vid_suffix}'
        if rle_suffix:
            eval_name = f'{eval_name}-{rle_suffix}'

    wc = '__var__'
    if wc not in eval_name:
        """evaluate results from a single checkpoint"""
        run_single_ckpt(params, eval_name, patch_images_path, patch_info_json_name,
                        seq_ids, seq_info, class_info, patch_suffix)
        return

    seg_dir_templ = linux_path(params.in_root_dir, eval_name, f'masks-{patch_suffix}')

    wc_start_idx = seg_dir_templ.find(wc)
    wc_end_idx = wc_start_idx + len(wc)
    pre_wc = seg_dir_templ[:wc_start_idx]
    post_wc = seg_dir_templ[wc_end_idx:]
    rep1, rep2 = (pre_wc, post_wc) if len(pre_wc) > len(post_wc) else (post_wc, pre_wc)

    seg_dir_parent = seg_dir_templ
    while wc in seg_dir_parent:
        seg_dir_parent = os.path.dirname(seg_dir_parent)
    assert os.path.isdir(seg_dir_parent), f"Nonexistent seg_dir_parent: {seg_dir_parent}"

    seg_dir_templ = seg_dir_templ.replace(wc, '*')

    eval_flag_id = '__eval'
    if params.out_name:
        eval_flag_id = f'{eval_flag_id}-{params.out_name}'

    print(f'seg_dir_templ: {seg_dir_templ}')
    print(f'eval_flag_id: {eval_flag_id}')

    if params.ignore_eval_flag:
        print('ignoring the existence of eval_flag')

    proc_seg_paths = []

    import glob

    while True:

        all_matching_paths = glob.glob(seg_dir_templ)
        # all_matching_dirs = [os.path.dirname(k) for k in all_matching_paths]

        all_matching_paths = [linux_path(_path) for _path in all_matching_paths]
        new_matching_paths = [_path for _path in all_matching_paths
                              if os.path.isfile(linux_path(_path, '__inference'))]

        if not params.ignore_eval_flag:
            new_matching_paths = [_path for _path in new_matching_paths
                                  if not os.path.isfile(linux_path(_path, eval_flag_id))]

        new_seg_paths = [k for k in new_matching_paths if k not in proc_seg_paths]
        new_seg_paths.sort(reverse=True)

        if not new_seg_paths:
            # print('no new_seg_paths found')
            # print(f'all_matching_paths:\n{utils.list_to_str(all_matching_paths)}\n')
            # print(f'new_matching_paths:\n{utils.list_to_str(new_matching_paths)}\n')

            if not eval_utils.sleep_with_pbar(params.sleep):
                break
            continue

        seg_path_ = new_seg_paths.pop()

        params_ = copy.deepcopy(params)

        print(f'evaluating {seg_path_}')

        match_substr1 = seg_path_.replace(rep1, '')
        match_substr2 = match_substr1.replace(rep2, '')

        print(f'\nckpt {match_substr2}\n')

        eval_name_1_ = os.path.relpath(seg_path_, params.in_root_dir)
        eval_name_ = os.path.dirname(eval_name_1_)

        if params_.out_name:
            params_.out_name = f'{params_.out_name}/{match_substr2}'
        else:
            params_.out_name = match_substr2

        args = [params_, eval_name_, patch_images_path, patch_info_json_name,
                seq_ids, seq_info, class_info, patch_suffix]

        if params.mp:
            success = multiprocessing.Value('b', 0)
            args.append(success)
            p = multiprocessing.Process(target=run_single_ckpt, args=args)
            p.start()
            p.join()
            if p.is_alive():
                p.terminate()
            success = success.value
        else:
            success = run_single_ckpt(*args)

        if success:
            eval_flag_path = linux_path(seg_path_, eval_flag_id)
            print(f'\neval_flag_path {eval_flag_path}\n')
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
            with open(eval_flag_path, "w") as fid:
                fid.write(timestamp)
            proc_seg_paths.append(seg_path_)
        else:
            raise AssertionError('run failed')


if __name__ == '__main__':
    main()
