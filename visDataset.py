import os, sys
import numpy as np

import paramparse
from paramparse import MultiPath

import densenet.evaluation.eval_segm as eval
from densenet.utils import read_data, getDateTime, print_and_write, linux_path
from datasets.build_utils import remove_fuzziness_in_mask, raw_seg_to_rgb, read_class_info

import cv2


class Params(paramparse.CFG):
    class Patch:
        def __init__(self):
            self.enable = 0

            self.enable_flip = 0

            self.enable_rot = 0
            self.max_rot = 0
            self.min_rot = 10

            self.max_stride = 0
            self.min_stride = 10

            self.patch_height = 32
            self.patch_width = 0

            self.resize = 0

            self.suffix = 0

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='vis_dataset')
        self.end_id = -1
        self.images_ext = 'png'
        self.labels_ext = 'png'
        self.log_dir = ''
        self.normalize_labels = 1
        self.out_ext = 'jpg'
        self.save_path = ''
        self.save_stitched = 1
        self.seg_ext = 'png'
        self.seg_path = ''
        self.selective_mode = 0
        self.show_img = 0
        self.start_id = 0
        self.stitch = 0
        self.stitch_seg = 1
        self.no_labels = 1
        self.class_info_path = 'data/classes_ice.txt'

        self.multi_sequence_db = 0
        self.seg_on_subset = 0
        self.add_border = 0
        self.blended = 1

        self.log_root_dir = 'log'
        self.db_root_dir = '/data'

        self.images_path = ''
        self.labels_path = ''
        self.labels_dir = 'labels'
        self.images_dir = 'images'

        self.dataset = ''

        self.model_info = MultiPath()
        self.train_split = MultiPath()
        self.vis_split = MultiPath()
        self.train_info = MultiPath()
        self.vis_info = MultiPath()

        self.patch = Params.Patch()

    def process(self):
        if not self.images_path:
            self.images_path = os.path.join(self.db_root_dir, self.dataset, self.images_dir)

        if not self.labels_path:
            self.labels_path = os.path.join(self.db_root_dir, self.dataset, self.labels_dir)

        log_dir = linux_path(self.log_root_dir, self.train_info, self.model_info)

        if not self.seg_path:
            self.seg_path = linux_path(log_dir, self.vis_info, 'raw')

        if not self.save_path:
            self.save_path = linux_path(log_dir, self.vis_info, 'vis')


def run(params):
    """

    :param Params params:
    :return:
    """
    eval_mode = False

    if params.multi_sequence_db:
        assert params.vis_split, "vis_split must be provided for multi_sequence_db"

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
            raise AssertionError('multi_sequence_db {} is not supported yet'.format(params.dataset))

        db_splits = DBInfo.DBSplits().__dict__
        sequences = DBInfo.sequences

        seq_ids = db_splits[params.vis_split]

        src_files = []
        seg_labels_list = []
        if params.no_labels:
            src_labels_list = None
        else:
            src_labels_list = []

        total_frames = 0
        seg_total_frames = 0

        for seq_id in seq_ids:
            seq_name, n_frames = sequences[seq_id]

            images_path = os.path.join(params.images_path, seq_name)

            if params.no_labels:
                labels_path = ''
            else:
                labels_path = os.path.join(params.labels_path, seq_name)

            _src_files, _src_labels_list, _total_frames = read_data(images_path, params.images_ext,
                                                                    labels_path,
                                                                    params.labels_ext)

            _src_filenames = [os.path.splitext(os.path.basename(k))[0] for k in _src_files]

            if not params.no_labels:
                _src_labels_filenames = [os.path.splitext(os.path.basename(k))[0] for k in _src_labels_list]

                assert _src_labels_filenames == _src_filenames, "mismatch between image and label filenames"

            eval_mode = False
            if params.seg_path and params.seg_ext:
                seg_path = os.path.join(params.seg_path, seq_name)

                _, _seg_labels_list, _seg_total_frames = read_data(labels_path=seg_path, labels_ext=params.seg_ext,
                                                                   labels_type='seg')

                _seg_labels__filenames = [os.path.splitext(os.path.basename(k))[0] for k in _seg_labels_list]

                if _seg_total_frames != _total_frames:

                    if params.seg_on_subset and _seg_total_frames < _total_frames:
                        matching_ids = [_src_filenames.index(k) for k in _seg_labels__filenames]

                        _src_files = [_src_files[i] for i in matching_ids]
                        if not params.no_labels:
                            _src_labels_list = [_src_labels_list[i] for i in matching_ids]

                        _total_frames = _seg_total_frames

                    else:
                        raise AssertionError('Mismatch between no. of frames in GT and seg labels: {} and {}'.format(
                            _total_frames, _seg_total_frames))

                seg_labels_list += _seg_labels_list

                seg_total_frames += _seg_total_frames
                eval_mode = True

            src_files += _src_files
            if not params.no_labels:
                src_labels_list += _src_labels_list
            else:
                params.stitch = params.save_stitched = 1

            total_frames += _total_frames
    else:
        src_files, src_labels_list, total_frames = read_data(params.images_path, params.images_ext, params.labels_path,
                                                             params.labels_ext)

        eval_mode = False
        if params.labels_path and params.seg_path and params.seg_ext:
            _, seg_labels_list, seg_total_frames = read_data(labels_path=params.seg_path, labels_ext=params.seg_ext,
                                                             labels_type='seg')
            if seg_total_frames != total_frames:
                raise SystemError('Mismatch between no. of frames in GT and seg labels: {} and {}'.format(
                    total_frames, seg_total_frames))
            eval_mode = True
        else:
            params.stitch = params.save_stitched = 1

    if params.end_id < params.start_id:
        params.end_id = total_frames - 1

    classes, composite_classes = read_class_info(params.class_info_path)
    n_classes = len(classes)
    class_ids = list(range(n_classes))
    class_id_to_color = {i: k[1] for i, k in enumerate(classes)}

    all_classes = [k[0] for k in classes + composite_classes]

    if not params.save_path:
        if eval_mode:
            params.save_path = os.path.join(os.path.dirname(params.seg_path), 'vis')
        else:
            params.save_path = os.path.join(os.path.dirname(params.images_path), 'vis')

    if not os.path.isdir(params.save_path):
        os.makedirs(params.save_path)

    if params.stitch and params.save_stitched:
        print('Saving visualization images to: {}'.format(params.save_path))

    log_fname = os.path.join(params.save_path, 'vis_log_{:s}.txt'.format(getDateTime()))
    print('Saving log to: {}'.format(log_fname))

    save_path_parent = os.path.dirname(params.save_path)
    templ_1 = os.path.basename(save_path_parent)
    templ_2 = os.path.basename(os.path.dirname(save_path_parent))

    templ = '{}_{}'.format(templ_1, templ_2)

    # if params.selective_mode:
    #     label_diff = int(255.0 / n_classes)
    # else:
    #     label_diff = int(255.0 / (n_classes - 1))

    print('templ: {}'.format(templ))
    # print('label_diff: {}'.format(label_diff))

    n_frames = params.end_id - params.start_id + 1

    pix_acc = np.zeros((n_frames,))

    mean_acc = np.zeros((n_frames,))
    # mean_acc_ice = np.zeros((n_frames,))
    # mean_acc_ice_1 = np.zeros((n_frames,))
    # mean_acc_ice_2 = np.zeros((n_frames,))
    #
    mean_IU = np.zeros((n_frames,))
    # mean_IU_ice = np.zeros((n_frames,))
    # mean_IU_ice_1 = np.zeros((n_frames,))
    # mean_IU_ice_2 = np.zeros((n_frames,))

    fw_IU = np.zeros((n_frames,))
    fw_sum = np.zeros((n_classes,))

    print_diff = max(1, int(n_frames * 0.01))

    avg_mean_acc = {c: 0 for c in all_classes}
    avg_mean_IU = {c: 0 for c in all_classes}
    skip_mean_acc = {c: 0 for c in all_classes}
    skip_mean_IU = {c: 0 for c in all_classes}

    _pause = 1
    labels_img = None

    for img_id in range(params.start_id, params.end_id + 1):

        stitched = []

        # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
        src_img_fname = src_files[img_id]
        img_dir = os.path.dirname(src_img_fname)
        seq_name = os.path.basename(img_dir)
        img_fname = os.path.basename(src_img_fname)

        img_fname_no_ext = os.path.splitext(img_fname)[0]

        src_img = None
        border_img = None

        if params.stitch or params.show_img:
            # src_img_fname = os.path.join(params.images_path, img_fname)
            src_img = cv2.imread(src_img_fname)
            if src_img is None:
                raise SystemError('Source image could not be read from: {}'.format(src_img_fname))

            try:
                src_height, src_width, _ = src_img.shape
            except ValueError as e:
                print('src_img_fname: {}'.format(src_img_fname))
                print('src_img: {}'.format(src_img))
                print('src_img.shape: {}'.format(src_img.shape))
                print('error: {}'.format(e))
                sys.exit(1)

            if not params.blended:
                stitched.append(src_img)

            border_img = np.full_like(src_img, 255)
            border_img = border_img[:, :5, ...]

        if not params.no_labels:
            # labels_img_fname = os.path.join(params.labels_path, img_fname_no_ext + '.{}'.format(params.labels_ext))
            labels_img_fname = src_labels_list[img_id]

            labels_img_orig = cv2.imread(labels_img_fname)
            if labels_img_orig is None:
                raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))

            _, src_width = labels_img_orig.shape[:2]

            # if len(labels_img_orig.shape) == 3:
            #     labels_img_orig = np.squeeze(labels_img_orig[:, :, 0])

            if params.show_img:
                cv2.imshow('labels_img_orig', labels_img_orig)

            labels_img_orig, label_img_raw, class_to_ids = remove_fuzziness_in_mask(
                labels_img_orig, n_classes, class_id_to_color, fuzziness=5, check_equality=0)
            labels_img = np.copy(labels_img_orig)
            # if params.normalize_labels:
            #     if params.selective_mode:
            #         selective_idx = (labels_img_orig == 255)
            #         print('labels_img_orig.shape: {}'.format(labels_img_orig.shape))
            #         print('selective_idx count: {}'.format(np.count_nonzero(selective_idx)))
            #         labels_img_orig[selective_idx] = n_classes
            #         if params.show_img:
            #             cv2.imshow('labels_img_orig norm', labels_img_orig)
            #     labels_img = (labels_img_orig.astype(np.float64) * label_diff).astype(np.uint8)
            # else:
            #     labels_img = np.copy(labels_img_orig)

            # if len(labels_img.shape) != 3:
            #     labels_img = np.stack((labels_img, labels_img, labels_img), axis=2)

            if params.stitch:
                if params.blended:
                    full_mask_gs = cv2.cvtColor(labels_img, cv2.COLOR_BGR2GRAY)
                    mask_binary = full_mask_gs == 0
                    labels_img_vis = (0.5 * src_img + 0.5 * labels_img).astype(np.uint8)
                    labels_img_vis[mask_binary] = src_img[mask_binary]
                else:
                    labels_img_vis = labels_img

                if params.add_border:
                    stitched.append(border_img)
                stitched.append(labels_img_vis)

        if eval_mode:
            # seg_img_fname = os.path.join(params.seg_path, img_fname_no_ext + '.{}'.format(params.seg_ext))
            seg_img_fname = seg_labels_list[img_id]

            seg_img = cv2.imread(seg_img_fname)
            if seg_img is None:
                raise SystemError('Segmentation image could not be read from: {}'.format(seg_img_fname))

            # seg_img = convert_to_raw_mask(seg_img, n_classes, seg_img_fname)

            if len(seg_img.shape) == 3:
                seg_img = np.squeeze(seg_img[:, :, 0])

            seg_height, seg_width = seg_img.shape

            if seg_width == 2 * src_width or seg_width == 3 * src_width:
                _start_id = seg_width - src_width
                seg_img = seg_img[:, _start_id:]

            if not params.no_labels:
                eval_cl, _ = eval.extract_classes(seg_img)
                gt_cl, _ = eval.extract_classes(label_img_raw)

                pix_acc[img_id] = eval.pixel_accuracy(seg_img, label_img_raw, class_ids)
                _acc, mean_acc[img_id] = eval.mean_accuracy(seg_img, label_img_raw, class_ids, return_acc=1)
                _IU, mean_IU[img_id] = eval.mean_IU(seg_img, label_img_raw, class_ids, return_iu=1)
                fw_IU[img_id], _fw = eval.frequency_weighted_IU(seg_img, label_img_raw, class_ids, return_freq=1)

                for _class_name, _, base_ids in composite_classes:
                    _acc_list = np.asarray(list(_acc.values()))
                    _mean_acc = np.mean(_acc_list[base_ids])
                    avg_mean_acc[_class_name] += (_mean_acc - avg_mean_acc[_class_name]) / (img_id + 1)

                    _IU_list = np.asarray(list(_IU.values()))
                    _mean_IU = np.mean(_IU_list[base_ids])
                    avg_mean_IU[_class_name] += (_mean_IU - avg_mean_IU[_class_name]) / (img_id + 1)

                for _class_id, _class_data in enumerate(classes):
                    _class_name = _class_data[0]
                    try:
                        _mean_acc = _acc[_class_id]
                        avg_mean_acc[_class_name] += (_mean_acc - avg_mean_acc[_class_name]) / (
                                img_id - skip_mean_acc[_class_name] + 1)
                    except KeyError:
                        print('\nskip_mean_acc {}: {}'.format(_class_name, img_id))
                        skip_mean_acc[_class_name] += 1
                    try:
                        _mean_IU = _IU[_class_id]
                        avg_mean_IU[_class_name] += (_mean_IU - avg_mean_IU[_class_name]) / (
                                img_id - skip_mean_IU[_class_name] + 1)
                    except KeyError:
                        print('\nskip_mean_IU {}: {}'.format(_class_name, img_id))
                        skip_mean_IU[_class_name] += 1

                # seg_img = (seg_img * label_diff).astype(np.uint8)

        seg_img_vis = raw_seg_to_rgb(seg_img, class_id_to_color)

        if params.stitch and params.stitch_seg:
            if params.blended:
                full_mask_gs = cv2.cvtColor(seg_img_vis, cv2.COLOR_BGR2GRAY)
                mask_binary = full_mask_gs == 0
                seg_img_vis = (0.5 * src_img + 0.5 * seg_img_vis).astype(np.uint8)
                seg_img_vis[mask_binary] = src_img[mask_binary]

            if params.add_border:
                stitched.append(border_img)
            stitched.append(seg_img_vis)

        if not params.stitch and params.show_img:
            cv2.imshow('seg_img', seg_img_vis)

        if params.stitch:
            stitched = np.concatenate(stitched, axis=1)
            if params.save_stitched:
                seg_save_path = os.path.join(params.save_path,
                                             '{}_{}.{}'.format(seq_name, img_fname_no_ext, params.out_ext))
                cv2.imwrite(seg_save_path, stitched)

            if params.show_img:
                cv2.imshow('stitched', stitched)
        else:
            if params.show_img:
                cv2.imshow('src_img', src_img)
                if params.labels_path:
                    cv2.imshow('labels_img', labels_img)

        if params.show_img:
            k = cv2.waitKey(1 - _pause)
            if k == 27:
                sys.exit(0)
            elif k == 32:
                _pause = 1 - _pause
        img_done = img_id - params.start_id + 1
        if img_done % print_diff == 0:
            log_txt = 'Done {:5d}/{:5d} frames'.format(img_done, n_frames)
            if eval_mode:
                log_txt = '{:s} pix_acc: {:.5f} mean_acc: {:.5f} mean_IU: {:.5f} fw_IU: {:.5f}'.format(
                    log_txt, pix_acc[img_id], mean_acc[img_id], mean_IU[img_id], fw_IU[img_id])
                for _class in all_classes:
                    log_txt += ' acc {}: {:.5f} '.format(_class, avg_mean_acc[_class])

                for _class in all_classes:
                    log_txt += ' IU {}: {:.5f} '.format(_class, avg_mean_acc[_class])

            print_and_write(log_txt, log_fname)

    sys.stdout.write('\n')
    sys.stdout.flush()

    if eval_mode:
        log_txt = "pix_acc\t mean_acc\t mean_IU\t fw_IU\n{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
            np.mean(pix_acc), np.mean(mean_acc), np.mean(mean_IU), np.mean(fw_IU))

        for _class in all_classes:
            log_txt += ' mean_acc {}\t'.format(_class)
        log_txt += '\n'

        for _class in all_classes:
            log_txt += ' {:.5f}\t'.format(avg_mean_acc[_class])
        log_txt += '\n'

        for _class in all_classes:
            log_txt += ' mean_IU {}\t'.format(_class)
        log_txt += '\n'

        for _class in all_classes:
            log_txt += ' {:.5f}\t'.format(avg_mean_IU[_class])
        log_txt += '\n'

        print_and_write(log_txt, log_fname)

        log_txt = templ + '\n\t'
        for _class in classes:
            log_txt += '{}\t '.format(_class[0])
        log_txt += 'all_classes\t all_classes(fw)\n'

        log_txt += 'recall\t'
        for _class in classes:
            log_txt += '{:.5f}\t'.format(avg_mean_acc[_class[0]])
        log_txt += '{:.5f}\t{:.5f}\n'.format(np.mean(mean_acc), np.mean(pix_acc))

        log_txt += 'precision\t'
        for _class in classes:
            log_txt += '{:.5f}\t'.format(avg_mean_IU[_class[0]])
        log_txt += '{:.5f}\t{:.5f}\n'.format(np.mean(mean_IU), np.mean(fw_IU))

        print_and_write(log_txt, log_fname)

    # fw_sum_total = np.sum(fw_sum)
    # fw_sum_frac = fw_sum / float(fw_sum_total)

    # print('fw_sum_total: {}'.format(fw_sum_total))
    # print('fw_sum_frac: {}'.format(fw_sum_frac))

    print('Wrote log to: {}'.format(log_fname))


if __name__ == '__main__':
    import paramparse

    _params = Params()
    paramparse.process(_params)
    _params.process()
    run(_params)
