import copy
import os
import cv2
import sys
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import skvideo.io

p2s_path = os.path.join(os.path.expanduser("~"), "pix2seq")
sys.path.append(p2s_path)

from densenet.utils import linux_path, sort_key, col_bgr
from tasks import task_utils

import paramparse


class Params(paramparse.CFG):
    """
    :ivar min_frg: minimum ratio of or ground pixels in a patch for it to be saved
    """

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='sub_patch')

        self.enable_labels = 1
        self.class_names_path = ''
        self.n_classes = 3
        self.proc_labels = 0
        self.allow_missing_labels = 1

        self.enable_flip = 0

        self.enable_rot = 0
        self.max_rot = 0
        self.min_rot = 0

        self.max_bkg_ratio = 0

        self.min_stride = 0
        self.max_stride = 0

        self.patch_height = 32
        self.patch_width = 0

        self.resize = 0

        self.seq_name = ''
        self.start_id = 0
        self.end_id = -1
        self.n_frames = 0

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
        self.out_vis_path = ''

        self.db_root_dir = ''
        self.src_path = ''
        self.labels_path = ''

        self.save_img = 0
        self.save_vid = 0
        self.save_stacked = 0
        self.save_vis = 0
        self.show_img = 0

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


def read_class_info(class_names_path):
    class_info = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]
    class_names, class_cols = zip(*[k.split('\t') for k in class_info])

    n_classes = len(class_cols)
    """background is class ID 0 with color black"""
    palette = [[0, 0, 0], ]
    for class_id in range(n_classes):
        col = class_cols[class_id]

        col_rgb = col_bgr[col][::-1]

        palette.append(col_rgb)

    palette_flat = [value for color in palette for value in color]

    class_dict = {x.strip(): i + 1 for (i, x) in enumerate(class_names)}

    return class_dict, palette_flat


def get_vis_image(src_img, labels_img, n_classes, out_fname):
    labels_patch_ud_vis = labels_img * (255 / n_classes)
    if len(labels_patch_ud_vis.shape) == 1:
        labels_patch_ud_vis = cv2.cvtColor(labels_patch_ud_vis, cv2.COLOR_GRAY2BGR)
    labels_patch_ud_vis = np.concatenate((src_img, labels_patch_ud_vis), axis=1)
    cv2.imwrite(out_fname, labels_patch_ud_vis)

    return labels_patch_ud_vis


def save_image_and_label(params, src_patch, labels_patch,
                         out_img_fname,
                         out_img_dir, out_labels_dir, out_vis_dir, out_root_dir,
                         out_seq_name, frame_id, n_classes,
                         out_vid_writer,
                         out_labels_writer,
                         image_infos, rle_lens,
                         ):
    """
        :param Params params:
    """
    out_img_path = linux_path(out_img_dir, f'{out_img_fname}.{params.img_ext}')
    out_labels_path = linux_path(out_labels_dir, f'{out_img_fname}.{params.labels_ext}')

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

    out_frames = [src_patch, ]

    if params.enable_labels:
        if params.save_stacked:
            labels_patch = task_utils.mask_id_to_vis(labels_patch, n_classes=n_classes, to_rgb=1)
            out_frames.append(labels_patch)
        else:

            labels_patch = task_utils.mask_id_to_vis(labels_patch, n_classes=n_classes, to_rgb=0)

            mask_rel_path = os.path.relpath(out_labels_path, out_root_dir).rstrip('.' + os.sep).replace(
                os.sep,
                '/')
            image_info['mask_file_name'] = mask_rel_path

            if params.save_vid:
                out_labels_writer.writeFrame(labels_patch)

            if params.save_img:
                cv2.imwrite(out_labels_path, labels_patch)

        if params.rle.enable:
            mask = np.copy(labels_patch)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if params.rle.label:
                mask[mask > 0] = params.rle.label

            rle, rle_norm = task_utils.mask_to_rle(
                mask, params.rle.max_len, params.rle.starts_2d,
                params.rle.starts_offset, params.rle.lengths_offset,
                params.rle.subsample)

            if params.rle.check:
                mask_rec = task_utils.rle_to_mask(
                    rle,
                    mask.shape,
                    params.rle.max_len,
                    params.rle.starts_offset,
                    params.rle.lengths_offset,
                    params.rle.starts_2d,
                    subsample=0,
                )

                mask_mismatch = np.nonzero(mask != mask_rec)
                assert mask_mismatch[0].size == 0, "mask_rec mismatch"

            if params.rle.json:
                image_info['rle'] = rle

            rle_len = len(rle)
            rle_lens.append(str(rle_len))

        # if params.save_vis:
        #     out_vis_labels_img_fname = linux_path(out_vis_dir,
        #                                           '{:s}_lr.{:s}'.format(out_img_fname, params.vis_ext))
        #     get_vis_image(src_patch, labels_patch, n_classes, out_vis_labels_img_fname)

    if len(out_frames) > 1:
        out_frame = np.concatenate(out_frames, axis=1)
    else:
        out_frame = out_frames[0]
    if params.save_vid:
        out_vid_writer.writeFrame(out_frame[:, :, ::-1])
    if params.save_img:
        cv2.imwrite(out_img_path, out_frame)

    image_infos.append(image_info)


def rotate_bound(image, angle, border_val=-1):
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
    rot_image = cv2.warpAffine(image_int, M, (nW, nH), borderValue=border_val)
    rot_mask = rot_image == border_val
    rot_image[rot_mask] = 0
    rot_image = rot_image.astype(np.uint8)

    rot_mask_gs = rot_mask[..., 0].astype(np.uint8) * 255
    # cv2.imshow('rot_mask_gs', rot_mask_gs)
    # cv2.imshow('rot_image', rot_image)
    # cv2.waitKey(0)

    return rot_image, rot_mask_gs


def run(image_infos, params):
    """

    :param Params params:
    :return:
    """
    out_root_dir = params.out_root_dir
    out_img_dir = params.out_img_dir
    out_labels_dir = params.out_labels_dir
    out_vis_dir = params.out_vis_path

    db_root_dir = params.db_root_dir
    seq_name = params.seq_name
    out_seq_name = params.out_seq_name
    show_img = params.show_img
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
    enable_labels = params.enable_labels
    allow_missing_labels = params.allow_missing_labels
    n_classes = params.n_classes
    proc_labels = params.proc_labels

    src_path = params.src_path
    labels_path = params.labels_path

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
        if image_as_patch:
            out_seq_name = '{:s}_{:d}_{:d}'.format(seq_name, start_id, end_id)
        else:
            out_seq_name = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                seq_name, start_id, end_id, patch_height, patch_width, min_stride, max_stride)

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

    if not out_labels_dir:
        if db_root_dir:
            out_labels_dir = linux_path(db_root_dir, out_seq_name, 'labels')
            out_vis_dir = linux_path(db_root_dir, out_seq_name, 'vis_labels')
        else:
            out_labels_dir = linux_path(labels_path_root_dir, out_seq_name)
            out_vis_dir = linux_path(labels_path_root_dir, 'vis', out_seq_name)
    else:
        if params.save_vis:
            assert out_vis_dir, "out_vis_dir must be provided"

    out_vid_writer = None
    out_labels_writer = None

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
        print('Writing output images to video: {}'.format(out_vid_path))
        os.makedirs(os.path.dirname(out_vid_path), exist_ok=True)
        ffmpeg_params_ = copy.deepcopy(ffmpeg_params)
        ffmpeg_params_['outputdict']['-crf'] = '17'
        out_vid_writer = skvideo.io.FFmpegWriter(
            out_vid_path, **ffmpeg_params_)

    if params.save_img:
        print('Writing output images to: {}'.format(out_img_dir))
        os.makedirs(out_img_dir, exist_ok=True)

    if params.resize:
        print(f'resizing input images to {params.resize}x{params.resize} before patch extraction')

    if enable_labels:
        if not params.save_stacked:
            if params.save_vid:
                out_labels_vid = f'{out_labels_dir}.{params.out_vid_ext}'
                print('Writing output labels to video: {}'.format(out_labels_vid))

                os.makedirs(os.path.dirname(out_labels_vid), exist_ok=True)
                ffmpeg_params_ = copy.deepcopy(ffmpeg_params)
                out_labels_writer = skvideo.io.FFmpegWriter(
                    out_labels_vid, **ffmpeg_params_)

            if params.save_img:
                print('Writing output labels to: {}'.format(out_labels_dir))
                os.makedirs(out_labels_dir, exist_ok=True)

        if params.save_vis:
            print('Writing output visualization labels to: {}'.format(out_vis_dir))
            os.makedirs(out_vis_dir, exist_ok=True)

    rot_angle = 0
    frame_id = 0

    pause_after_frame = 1

    n_frames = end_id - start_id + 1
    rle_lens = []
    pbar = tqdm(range(start_id, end_id + 1), position=0, leave=True)

    for img_id in pbar:
        # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, params.img_ext)
        img_fname = src_files[img_id]
        img_fname_no_ext, _ = os.path.splitext(img_fname)

        src_img_fname = linux_path(src_path, img_fname)
        src_img = cv2.imread(src_img_fname)
        assert src_img is not None, f"invalid src_img_fname: {src_img_fname}"

        if params.resize:
            src_img = cv2.resize(src_img, (params.resize, params.resize))

        src_height, src_width, _ = src_img.shape

        if image_as_patch:
            patch_width, patch_height = src_width, src_height

        if src_height < patch_height or src_width < patch_width:
            print('\nImage {} is too small {}x{} for the given patch size {}x{}\n'.format(
                src_img_fname, src_width, src_height, patch_width, patch_height))
            continue

        assert src_img is not None, 'Source image could not be read: {}'.format(src_img_fname)

        n_rows, ncols, n_channels = src_img.shape

        if enable_labels:
            labels_img_fname = linux_path(labels_path, img_fname_no_ext + '.' + params.labels_ext)
            from PIL import Image

            im = Image.open(labels_img_fname)
            if params.resize:
                im = im.resize((params.resize, params.resize))
            labels_img = np.array(im)

            if proc_labels:
                task_utils.mask_vis_to_id(labels_img, n_classes)

            _n_rows, _ncols = labels_img.shape

            if n_rows != _n_rows or ncols != _ncols:
                raise SystemError('Dimension mismatch between image and label for file: {}'.format(img_fname))

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

            if enable_labels:
                """increase pixel value gap between the classes to prevent them getting confounded by 
                interpolation required for rotation"""
                task_utils.mask_id_to_vis(labels_img, n_classes)
                labels_img, _ = rotate_bound(labels_img, rot_angle)
                """restore the pixel values to class IDs"""
                task_utils.mask_vis_to_id(labels_img, n_classes)

                # if params.save_vis:
                #     out_labels_path = 'labels_{:d}_rot_{:d}.{:s}'.format(
                #         img_id + 1, rot_angle, params.vis_ext)
                #
                #     if db_root_dir:
                #         out_labels_img_dir = linux_path(db_root_dir, out_seq_name)
                #     else:
                #         out_labels_img_dir = linux_path(labels_path_root_dir, 'rot', out_seq_name)
                #
                #     os.makedirs(out_labels_img_dir, exist_ok=True)
                #
                #     out_labels_img_path = linux_path(out_labels_img_dir, out_labels_path)
                #     labels_img_vis = np.concatenate((src_img, labels_img), axis=1)
                #     cv2.imwrite(out_labels_img_path, labels_img_vis)

            # else:
            #     if params.save_vis:
            #         out_img_name = 'img_{:d}_rot_{:d}.{:s}'.format(
            #             img_id + 1, rot_angle, params.vis_ext)
            #
            #         if db_root_dir:
            #             out_src_img_dir = linux_path(db_root_dir, out_seq_name)
            #         else:
            #             out_src_img_dir = linux_path(src_path_root_dir, 'rot', out_seq_name)
            #
            #         os.makedirs(out_src_img_dir, exist_ok=True)
            #
            #         out_src_img_path = linux_path(out_src_img_dir, out_img_name)
            #         cv2.imwrite(out_src_img_path, src_img)

        out_id = 0
        # skip_id = 0
        min_row = 0
        while True:
            max_row = min_row + patch_height
            if max_row > n_rows:
                diff = max_row - n_rows
                min_row -= diff
                max_row -= diff

            min_col = max_col = 0
            labels_patch = None
            while True:
                max_col = min_col + patch_width
                if max_col > ncols:
                    diff = max_col - ncols
                    min_col -= diff
                    max_col -= diff

                src_patch = src_img[min_row:max_row, min_col:max_col, :]

                skip_patch = False
                if enable_rot:
                    rot_mask_patch = rot_mask[min_row:max_row, min_col:max_col]
                    bkg_ratio = np.count_nonzero(rot_mask_patch) / rot_mask_patch.size

                    # print(f'bkg_ratio: {bkg_ratio}')
                    # cv2.imshow('src_patch', src_patch)
                    # cv2.imshow('rot_mask_patch', rot_mask_patch)
                    # cv2.waitKey(0)

                    if bkg_ratio > params.max_bkg_ratio:
                        skip_patch = True

                if not skip_patch:
                    if image_as_patch:
                        out_img_fname = img_fname_no_ext
                    else:
                        out_img_fname = '{:s}_{:d}'.format(img_fname_no_ext, out_id + 1)

                    labels_patches = []
                    if enable_labels:
                        labels_patch = labels_img[min_row:max_row, min_col:max_col]
                        labels_patch = labels_patch.astype(np.uint8)
                        labels_patches.append(labels_patch)
                        # if enable_rot and (labels_patch == -1).any():
                        #     skip_ patch = True

                    if enable_rot:
                        out_img_fname = '{:s}_rot_{:d}'.format(out_img_fname, rot_angle)

                    out_id += 1
                    frame_id += 1
                    pbar.set_description(f'frame_id: {frame_id}')

                    save_image_and_label(
                        params,
                        src_patch, labels_patch, out_img_fname,
                        out_img_dir, out_labels_dir, out_vis_dir, out_root_dir,
                        out_seq_name, frame_id, n_classes,
                        out_vid_writer,
                        out_labels_writer,
                        image_infos, rle_lens,
                    )
                    src_patches = [src_patch]

                    if enable_flip:
                        """
                        LR flip
                        """
                        src_patch_lr = np.fliplr(src_patch)
                        labels_patch_lr = None
                        if enable_labels:
                            labels_patch_lr = np.fliplr(labels_patch)
                            labels_patches.append(labels_patch_lr)

                        frame_id += 1
                        pbar.set_description(f'frame_id: {frame_id}')

                        save_image_and_label(
                            params,
                            src_patch_lr, labels_patch_lr, f'{out_img_fname}_lr',
                            out_img_dir, out_labels_dir, out_vis_dir, out_root_dir,
                            out_seq_name, frame_id, n_classes,
                            out_vid_writer,
                            out_labels_writer,
                            image_infos, rle_lens,
                        )
                        src_patches.append(src_patch_lr)
                        """
                        UD flip
                        """
                        src_patch_ud = np.flipud(src_patch)
                        labels_patch_ud = None
                        frame_id += 1
                        pbar.set_description(f'frame_id: {frame_id}')
                        if enable_labels:
                            labels_patch_ud = np.flipud(labels_patch)
                            labels_patches.append(labels_patch_ud)

                        save_image_and_label(
                            params,
                            src_patch_ud, labels_patch_ud, f'{out_img_fname}_ud',
                            out_img_dir, out_labels_dir, out_vis_dir, out_root_dir,
                            out_seq_name, frame_id, n_classes,
                            out_vid_writer,
                            out_labels_writer,
                            image_infos, rle_lens,
                        )
                        src_patches.append(src_patch_ud)

                    if show_img:
                        src_img_vis = src_img.copy()
                        cv2.rectangle(src_img_vis, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)
                        src_img_vis = cv2.resize(src_img_vis, (640, 640))

                        # disp_labels_img = labels_img.copy()
                        # cv2.rectangle(disp_labels_img, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)
                        src_patch_vis = np.concatenate(src_patches, axis=1)

                        if enable_labels:
                            labels_img_vis = labels_img.copy()
                            labels_img_vis = task_utils.mask_id_to_vis(labels_img_vis, n_classes=n_classes, to_rgb=1)
                            cv2.rectangle(labels_img_vis, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)
                            labels_img_vis = cv2.resize(labels_img_vis, (640, 640))

                            src_img_vis = np.concatenate((src_img_vis, labels_img_vis), axis=1)
                            labels_patch_vis = np.concatenate(labels_patches, axis=1)
                            labels_patch_vis = task_utils.mask_id_to_vis(labels_patch_vis, n_classes=n_classes,
                                                                         to_rgb=1)

                            src_patch_vis = np.concatenate((src_patch_vis, labels_patch_vis),
                                                           axis=0 if enable_flip else 1)

                        cv2.imshow('src_img', src_img_vis)
                        cv2.imshow('patch', src_patch_vis)

                        # cv2.imshow('disp_labels_img', disp_labels_img)
                        k = cv2.waitKey(1 - pause_after_frame)
                        if k == 27:
                            sys.exit(0)
                        elif k == 32:
                            pause_after_frame = 1 - pause_after_frame

                min_col += random.randint(min_stride, max_stride)
                if image_as_patch or max_col >= ncols:
                    break

            if image_as_patch or max_row >= n_rows:
                break
            min_row += random.randint(min_stride, max_stride)

    # sys.stdout.write('\n')
    # sys.stdout.flush()
    sys.stdout.write('Total frames generated: {}\n'.format(frame_id))

    if out_vid_writer is not None:
        out_vid_writer.close()
    if out_labels_writer is not None:
        out_labels_writer.close()

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
