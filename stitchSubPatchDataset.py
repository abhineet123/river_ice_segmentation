import copy
import os, shutil
import cv2
import sys
import imageio

import numpy as np

import paramparse
from paramparse import MultiPath

from densenet.utils import linux_path, sort_key, resize_ar, read_data, print_and_write, getDateTime
from datasets.build_utils import read_class_info


class Params(paramparse.CFG):

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='stitch')
        self.codec = 'H264'
        self.del_patch_seq = 0
        self.disp_resize_factor = 1.0
        self.end_id = -1
        self.fname_templ = 'img'
        self.fps = 30
        self.height = 720

        self.images_ext = 'tif'
        self.labels_ext = 'png'

        self.method = 0
        self.n_classes = 3
        self.n_frames = 0
        self.normalize_patches = 0
        self.out_ext = 'png'
        self.patch_ext = 'png'
        self.patch_height = 32

        self.patch_width = 0
        self.resize_factor = 1.0
        self.seq_name = 'Training'
        self.show_img = 0

        self.stacked = 0
        self.start_id = 0

        self.src_path = ''
        self.patch_seq_path = ''
        self.stitched_seq_path = ''

        self.db_root_dir = '/data'
        self.patch_seq_name = ''
        self.patch_seq_type = 'images'
        self.labels_dir = 'Labels'
        self.images_dir = 'Images'

        self.class_info_path = 'data/classes_ice.txt'

        self.width = 1280

        self.model_info = MultiPath()
        self.train_info = MultiPath()
        self.train_split = MultiPath()

        self.multi_sequence_db = 0
        self.seg_on_subset = 0
        self.split = MultiPath()
        self.images_path = ''
        self.labels_path = ''

        self.dataset = ''

    def process(self):
        if not self.src_path:
            self.src_path = linux_path(self.db_root_dir, self.dataset, self.images_dir)


class Metrics:
    avg_pix_acc = avg_mean_acc = avg_mean_IU = avg_fw_IU = 0
    avg_mean_acc_ice = avg_mean_acc_ice_1 = avg_mean_acc_ice_2 = 0
    avg_mean_IU_ice = avg_mean_IU_ice_1 = avg_mean_IU_ice_2 = 0

    skip_mean_acc_ice_1 = skip_mean_acc_ice_2 = 0
    skip_mean_IU_ice_1 = skip_mean_IU_ice_2 = 0


def eval_stitched(stitched_img, gt_labels_orig, class_ids, img_id, metrics):
    import densenet.evaluation.eval_segm as eval_segm

    if len(gt_labels_orig.shape) == 3:
        gt_labels = np.squeeze(gt_labels_orig[:, :, 0])
    else:
        gt_labels = gt_labels_orig

    seg_labels = np.squeeze(stitched_img[:, :, 0])

    pix_acc = eval_segm.pixel_accuracy(seg_labels, gt_labels, class_ids)
    _acc, mean_acc = eval_segm.mean_accuracy(seg_labels, gt_labels, class_ids, return_acc=1)
    _IU, mean_IU = eval_segm.mean_IU(seg_labels, gt_labels, class_ids, return_iu=1)
    fw_IU = eval_segm.frequency_weighted_IU(seg_labels, gt_labels, class_ids)

    metrics.avg_pix_acc += (pix_acc - metrics.avg_pix_acc) / (img_id + 1)
    metrics.avg_mean_acc += (mean_acc - metrics.avg_mean_acc) / (img_id + 1)
    metrics.avg_mean_IU += (mean_IU - metrics.avg_mean_IU) / (img_id + 1)
    metrics.avg_fw_IU += (fw_IU - metrics.avg_fw_IU) / (img_id + 1)

    # print('_acc: {}'.format(_acc))
    # print('_IU: {}'.format(_IU))

    mean_acc_ice = np.mean(list(_acc.values())[1:])
    metrics.avg_mean_acc_ice += (mean_acc_ice - metrics.avg_mean_acc_ice) / (img_id + 1)
    try:
        mean_acc_ice_1 = _acc[1]
        metrics.avg_mean_acc_ice_1 += (mean_acc_ice_1 - metrics.avg_mean_acc_ice_1) / (
                img_id - metrics.skip_mean_acc_ice_1 + 1)
    except KeyError:
        print('\nskip_mean_acc_ice_1: {}'.format(img_id))
        metrics.skip_mean_acc_ice_1 += 1
    try:
        mean_acc_ice_2 = _acc[2]
        metrics.avg_mean_acc_ice_2 += (mean_acc_ice_2 - metrics.avg_mean_acc_ice_2) / (
                img_id - metrics.skip_mean_acc_ice_2 + 1)
    except KeyError:
        print('\nskip_mean_acc_ice_2: {}'.format(img_id))
        metrics.skip_mean_acc_ice_2 += 1

    mean_IU_ice = np.mean(list(_IU.values())[1:])
    metrics.avg_mean_IU_ice += (mean_IU_ice - metrics.avg_mean_IU_ice) / (img_id + 1)
    try:
        mean_IU_ice_1 = _IU[1]
        metrics.avg_mean_IU_ice_1 += (mean_IU_ice_1 - metrics.avg_mean_IU_ice_1) / (
                img_id - metrics.skip_mean_IU_ice_1 + 1)
    except KeyError:
        print('\nskip_mean_IU_ice_1: {}'.format(img_id))
        metrics.skip_mean_IU_ice_1 += 1
    try:
        mean_IU_ice_2 = _IU[2]
        metrics.avg_mean_IU_ice_2 += (mean_IU_ice_2 - metrics.avg_mean_IU_ice_2) / (
                img_id - metrics.skip_mean_IU_ice_2 + 1)
    except KeyError:
        print('\nskip_mean_IU_ice_2: {}'.format(img_id))
        metrics.skip_mean_IU_ice_2 += 1


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
            patch_src_img_fname = os.path.join(params.patch_seq_path,
                                               '{:s}.{:s}'.format(patch_img_fname, params.patch_ext))

            if not os.path.exists(patch_src_img_fname):
                raise SystemError('Patch image does not exist: {}'.format(patch_src_img_fname))

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


def run(params):
    """

    :param Params params:
    :return:
    """
    video_exts = ['mp4', 'mkv', 'avi', 'mpg', 'mpeg', 'mjpg']

    # if not params.src_path:
    #     assert params.db_root_dir, "either params.src_path or params.db_root_dir must be provided"
    #
    #     params.src_path = os.path.join(params.db_root_dir, params.seq_name, 'images')

    print('Reading source images from: {}'.format(params.src_path))

    src_files = [k for k in os.listdir(params.src_path) if k.endswith('.{:s}'.format(params.images_ext))]
    total_frames = len(src_files)

    assert total_frames > 0, f'No input frames of type {params.images_ext} found'

    print('total_frames: {}'.format(total_frames))

    classes, composite_classes = read_class_info(params.class_info_path)
    n_classes = len(classes)
    class_ids = list(range(n_classes))

    src_files.sort(key=sort_key)

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
        params.patch_seq_path = os.path.join(params.db_root_dir, params.patch_seq_name, params.patch_seq_type)
        assert os.path.isdir(params.patch_seq_path), f'patch_seq_path does not exist: {params.patch_seq_path}'
    else:
        params.patch_seq_name = os.path.basename(params.patch_seq_path)

    if not params.stitched_seq_path:
        stitched_seq_name = '{}_stitched_{}'.format(params.patch_seq_name, params.method)
        if params.stacked:
            stitched_seq_name = '{}_stacked'.format(stitched_seq_name)
            params.method = 1
        stitched_seq_name = '{}_{}_{}'.format(stitched_seq_name, params.start_id, params.end_id)
        params.stitched_seq_path = os.path.join(params.db_root_dir, stitched_seq_name, params.patch_seq_type)

    gt_labels_orig = gt_labels = video_out = None
    write_to_video = params.out_ext in video_exts
    if write_to_video:
        if not params.stitched_seq_path.endswith('.{}'.format(params.out_ext)):
            params.stitched_seq_path = '{}.{}'.format(params.stitched_seq_path, params.out_ext)
        print('Writing {}x{} output video to: {}'.format(params.width, params.height, params.stitched_seq_path))
        save_dir = os.path.dirname(params.stitched_seq_path)

        fourcc = cv2.VideoWriter_fourcc(*params.codec)
        video_out = cv2.VideoWriter(params.stitched_seq_path, fourcc, params.fps, (params.width, params.height))
    else:
        print('Writing output images to: {}'.format(params.stitched_seq_path))
        save_dir = params.stitched_seq_path

    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    log_fname = os.path.join(save_dir, 'log_{:s}.txt'.format(getDateTime()))
    print_and_write('Saving log to: {}'.format(log_fname), log_fname)
    print_and_write('Reading patch images from: {}'.format(params.patch_seq_path), log_fname)

    n_patches_all = 0
    label_diff = int(255.0 / (params.n_classes - 1))

    metrics = Metrics()

    eval_mode = False
    if params.labels_path and params.labels_ext:
        _, labels_list, labels_total_frames = read_data(labels_path=params.labels_path, labels_ext=params.labels_ext)

        assert labels_total_frames == total_frames, 'Mismatch between no. of frames in GT and labels'

        eval_mode = True

    _n_frames = params.end_id - params.start_id + 1

    img_iter = range(params.start_id, params.end_id + 1)
    if not eval_mode:
        from tqdm import tqdm
        img_iter = tqdm(img_iter)

    for img_id in img_iter:
        img_fname = src_files[img_id]
        img_fname_no_ext = os.path.splitext(img_fname)[0]

        src_img_fname = os.path.join(params.src_path, img_fname)
        src_img = cv2.imread(src_img_fname)
        assert src_img is not None, f'Labels image could not be read: {src_img_fname}'

        if params.method == -1:
            """no stitching / patch size = image size"""
            patch_src_img_fname = os.path.join(params.patch_seq_path,
                                               '{:s}.{:s}'.format(img_fname_no_ext, params.patch_ext))
            if not os.path.exists(patch_src_img_fname):
                raise SystemError('Patch image does not exist: {}'.format(patch_src_img_fname))
            stitched_img = cv2.imread(patch_src_img_fname)
            n_patches = 1
        else:
            stitched_img, n_patches = stitch_patches(src_img, img_fname_no_ext, label_diff, params)

        if eval_mode:
            labels_img_fname = os.path.join(params.labels_path, img_fname_no_ext + '.{}'.format(params.labels_ext))
            gt_labels_orig = imageio.imread(labels_img_fname)

            assert gt_labels_orig is not None, f'Labels image could not be read: {labels_img_fname}'

            eval_stitched(stitched_img, gt_labels_orig, class_ids, img_id, metrics)

            log_txt = "frame {:d}/{:d} :: pix_acc: {:.5f} mean_acc: {:.5f} mean_IU: {:.5f} fw_IU: {:.5f}" \
                      " avg_acc_ice: {:.5f} avg_acc_ice_1: {:.5f} avg_acc_ice_2: {:.5f}" \
                      " avg_IU_ice: {:.5f} avg_IU_ice_1: {:.5f} avg_IU_ice_2: {:.5f}".format(
                img_id - params.start_id + 1, _n_frames,
                metrics.avg_pix_acc, metrics.avg_mean_acc, metrics.avg_mean_IU, metrics.avg_fw_IU,
                metrics.avg_mean_acc_ice, metrics.avg_mean_acc_ice_1, metrics.avg_mean_acc_ice_2,
                metrics.avg_mean_IU_ice, metrics.avg_mean_IU_ice_1, metrics.avg_mean_IU_ice_2,
            )
            print_and_write(log_txt, log_fname)

        if not params.normalize_patches:
            seg_img = (stitched_img * label_diff).astype(np.uint8)
        else:
            seg_img = stitched_img

        if params.stacked:
            if eval_mode:
                labels_img = (gt_labels_orig * label_diff).astype(np.uint8)
                stitched = np.concatenate((src_img, labels_img), axis=1)
            else:
                stitched = src_img
            out_img = np.concatenate((stitched, seg_img), axis=1)
        else:
            out_img = seg_img

        if write_to_video:
            out_img = resize_ar(out_img, params.width, params.height)
            video_out.write(out_img)
        else:
            if params.resize_factor != 1:
                out_img = cv2.resize(out_img, (0, 0), fx=params.resize_factor, fy=params.resize_factor)
            stacked_img_path = os.path.join(params.stitched_seq_path, '{}.{}'.format(img_fname_no_ext, params.out_ext))
            cv2.imwrite(stacked_img_path, out_img)

        n_patches_all += n_patches

    print('\nTotal patches processed: {}\n'.format(n_patches_all))

    if params.show_img:
        cv2.destroyAllWindows()
    if write_to_video:
        video_out.release()

    if params.del_patch_seq:
        print('Removing patch folder {}'.format(params.patch_seq_path))
        shutil.rmtree(params.patch_seq_path)

    if eval_mode:
        log_txt = "pix_acc\t mean_acc\t mean_IU\t fw_IU\n{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
            metrics.avg_pix_acc, metrics.avg_mean_acc, metrics.avg_mean_IU, metrics.avg_fw_IU)
        log_txt += "mean_acc_ice\t mean_acc_ice_1\t mean_acc_ice_2\n{:.5f}\t{:.5f}\t{:.5f}\n".format(
            metrics.avg_mean_acc_ice, metrics.avg_mean_acc_ice_1, metrics.avg_mean_acc_ice_2)
        log_txt += "mean_IU_ice\t mean_IU_ice_1\t mean_IU_ice_2\n{:.5f}\t{:.5f}\t{:.5f}\n".format(
            metrics.avg_mean_IU_ice, metrics.avg_mean_IU_ice_1, metrics.avg_mean_IU_ice_2)
        print_and_write(log_txt, log_fname)

        print_and_write('Saved log to: {}'.format(log_fname), log_fname)
        print_and_write('Read patch images from: {}'.format(params.patch_seq_path), log_fname)


def main():
    params = Params()
    paramparse.process(params)
    params.process()

    if not params.multi_sequence_db:
        run(params)
        return

    assert params.split, "split must be provided for multi_sequence_db"

    """some repeated code here to allow better IntelliSense"""
    if params.dataset.lower() == 'ctc':
        from ctc_info import CTCInfo as DBInfo
    elif params.dataset.lower() == 'ipsc':
        from ipsc_info import IPSCInfo as DBInfo
    elif params.dataset.lower() in ('ipsc_dev', 'ipsc_2_class', 'ipsc_5_class'):
        from ipsc_info import IPSCDevInfo as DBInfo
    elif params.dataset.lower() == 'ipsc_patches':
        from ipsc_info import IPSCPatchesInfo as DBInfo
    else:
        raise AssertionError('unsupported multi_sequence_db: {}'.format(params.dataset))

    db_splits = DBInfo.DBSplits().__dict__
    sequences = DBInfo.sequences

    seq_ids = db_splits[params.split]
    n_seq = len(seq_ids)

    for _id, seq_id in enumerate(seq_ids):
        seq_params = copy.deepcopy(params)

        seq_name, n_frames = sequences[seq_id]

        print(f'seq {_id} / {n_seq}: {seq_name}')

        seq_params.src_path = os.path.join(params.images_path, seq_name)

        run(seq_params)


if __name__ == '__main__':
    main()
