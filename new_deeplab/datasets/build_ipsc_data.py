"""Converts IPSC data to TFRecord file format with Example protos."""

import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import cv2
import numpy as np

import tensorflow as tf

if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

from tqdm import tqdm

import paramparse

from build_data import ImageReader, image_seg_to_tfexample
from build_utils import resize_ar, read_class_info, remove_fuzziness_in_mask, col_bgr
from db_info import IPSCInfo, IPSCPatchesInfo, IPSCDevInfo


class Params:

    def __init__(self):
        self.db_split = 'all'

        self.dev = 1
        self.patches = 0

        self.cfg = ()
        self.ignore_missing_gt = 1
        self.ignore_missing_seg = 1
        self.ignored_region_only = 0

        self.resize = 0
        self.root_dir = '/data/ipsc'
        self.output_root_dir = '/data/tfrecord'
        self.output_dir = ''

        self.start_id = 0
        self.end_id = -1

        self.start_frame_id = -1
        self.end_frame_id = -1

        self.write_gt = 0
        self.write_img = 1
        self.raad_gt = 0
        self.tra_only = 0

        self.preprocess = 0
        self.n_classes = 2
        self.class_info_path = '../data/classes_ipsc_2_class.txt'

        """uncertainty in mask pixel values for each class"""
        self.fuzziness = 5

        self.shuffle = 1
        self.train_ratio = 0

        self.show_img = 0
        self.save_img = 0
        self.save_vid = 0

        self.disable_seg = 0
        self.disable_tqdm = 0
        self.codec = 'H264'

        self.out_size = (513, 513)

        self.vis_height = 1080
        self.vis_width = 1920
        self.num_shards = 4

        self.db_splits = IPSCInfo.DBSplits().__dict__


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def create_tfrecords(img_src_files, seg_src_files, n_shards, db_split, output_dir):
    image_reader = ImageReader('jpeg', channels=1)

    label_reader = ImageReader('png', channels=1)

    n_images = len(img_src_files)
    n_per_shard = int(math.ceil(n_images / float(n_shards)))

    os.makedirs(output_dir, exist_ok=True)

    print('Creating {} shards with {} images ({} per shard)'.format(n_shards, n_images, n_per_shard))

    for shard_id in range(n_shards):

        output_file_path = os.path.join(
            output_dir,
            '{:s}-{:05d}-of-{:05d}.tfrecord'.format(db_split, shard_id, n_shards))

        with tf.python_io.TFRecordWriter(output_file_path) as tfrecord_writer:
            start_idx = shard_id * n_per_shard
            end_idx = min((shard_id + 1) * n_per_shard, n_images)

            for img_id in tqdm(range(start_idx, end_idx), ncols=50):

                img_src_path = img_src_files[img_id]

                image_data = tf.gfile.FastGFile(img_src_path, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)

                if seg_src_files is not None and seg_src_files:
                    seg_src_path = seg_src_files[img_id]
                    seg_data = tf.gfile.FastGFile(seg_src_path, 'rb').read()
                    seg_height, seg_width = label_reader.read_image_dims(seg_data)
                    assert height == seg_height and width == seg_width, 'Shape mismatch found between image and label'
                else:
                    seg_data = None

                # Convert to tf example.
                example = image_seg_to_tfexample(
                    image_data, img_src_path, height, width, seg_data)

                tfrecord_writer.write(example.SerializeToString())


def _convert_dataset(params):
    """

    :param Params params:
    :return:
    """

    seq_ids = params.db_splits[params.db_split]

    output_dir = params.output_dir
    if params.n_classes > 2:
        print('using {} class version'.format(params.n_classes - 1))

    if params.patches:
        print('using patch version')

    if not output_dir:
        output_dir = 'ipsc'
        if params.n_classes > 2:
            print('using {} class version'.format(params.n_classes - 1))
            output_dir += '_{}_class'.format(params.n_classes - 1)

        if params.patches:
            print('using patch version')
            output_dir += '_patches'

    print('root_dir: {}'.format(params.root_dir))
    print('db_split: {}'.format(params.db_split))
    print('seq_ids: {}'.format(seq_ids))

    if params.output_root_dir:
        output_dir = linux_path(params.output_root_dir, output_dir)

    print('output_dir: {}'.format(output_dir))

    os.makedirs(output_dir, exist_ok=True)

    if params.disable_seg:
        print('\nSegmentations are disabled\n')

    if params.start_id > 0:
        seq_ids = seq_ids[params.start_id:]

    n_seq = len(seq_ids)
    img_exts = ('.jpg',)
    seg_exts = ('.png',)

    n_train_files = 0
    n_test_files = 0

    train_img_files = []
    train_seg_files = []

    test_img_files = []
    test_seg_files = []

    vis_seg_files = []

    class_to_color = {
        0: (0, 0, 0)
    }

    if params.class_info_path:
        print('reading class info from: {}'.format(params.class_info_path))
        classes, _ = read_class_info(params.class_info_path)
        class_to_color = {i: k[1] for i, k in enumerate(classes)}
    else:
        col_diff = 255.0 / params.n_classes
        class_id_to_col_gs = {
            _id + 1: int(col_diff * (_id + 1)) for _id in range(params.n_classes)
        }
        class_to_color.update(
            {
                _id: (col, col, col) for _id, col in class_id_to_col_gs.items()
            }
        )

    print('using class colors:\n{}'.format(class_to_color))

    for __id, seq_id in enumerate(seq_ids):

        if params.patches:
            seq_name, n_frames = IPSCPatchesInfo.sequences[seq_id]
        else:
            if params.dev:
                seq_name, n_frames = IPSCDevInfo.sequences[seq_id]
            else:
                seq_name, n_frames = IPSCInfo.sequences[seq_id]

        print('\tseq {} / {}\t{}\t{}\t{} frames'.format(__id + 1, n_seq, seq_id, seq_name, n_frames))

        if params.dev:
            img_dir_path = linux_path(params.root_dir, 'images', seq_name)
        else:
            img_dir_path = linux_path(params.root_dir, seq_name)

        print('reading images from: {}'.format(img_dir_path))

        _img_src_files = [linux_path(img_dir_path, k) for k in os.listdir(img_dir_path) if
                          os.path.splitext(k.lower())[1] in img_exts]

        n_img_files = len(_img_src_files)
        assert n_img_files == n_frames, \
            "Mismatch between number of the specified frames and number of actual images in folder"

        _img_src_files.sort()

        img_indices = list(range(n_img_files))

        if params.start_frame_id >= 0 or params.end_frame_id >= 0:
            if params.start_frame_id < 0:
                params.start_frame_id = 0
            if params.end_frame_id < 0:
                params.end_frame_id = n_frames - 1
            img_indices = img_indices[params.start_frame_id:params.end_frame_id + 1]

        if params.shuffle:
            random.shuffle(img_indices)

        _img_src_files = [_img_src_files[i] for i in img_indices]
        n_img_files = len(_img_src_files)

        if params.train_ratio:
            _n_train_files = int(n_img_files * params.train_ratio)
        else:
            _n_train_files = n_img_files

        train_img_files += _img_src_files[:_n_train_files]
        test_img_files += _img_src_files[_n_train_files:]

        n_train_files += _n_train_files
        n_test_files += n_img_files - _n_train_files

        if not params.disable_seg:
            if params.dev:
                vis_seg_path = linux_path(params.root_dir, 'vis_labels', seq_name)
                raw_seg_path = linux_path(params.root_dir, 'raw_labels', seq_name)
                seg_path = linux_path(params.root_dir, 'labels', seq_name)

            else:
                vis_seg_path = linux_path(params.root_dir, seq_name, 'vis_masks')
                raw_seg_path = linux_path(params.root_dir, seq_name, 'raw_masks')
                seg_path = linux_path(params.root_dir, seq_name, 'masks')

            os.makedirs(vis_seg_path, exist_ok=True)

            if not params.preprocess:
                raw_seg_path = seg_path
            else:
                assert os.path.exists(seg_path), f"invalid seg_path: {seg_path}"
                print(f'reading segmentations from: {seg_path}')

                _seg_src_fnames = [k for k in os.listdir(seg_path) if
                                   os.path.splitext(k.lower())[1] in seg_exts]
                _seg_src_files = [linux_path(seg_path, k) for k in _seg_src_fnames]

                print('writing raw segmentation images to {}'.format(raw_seg_path))

                os.makedirs(raw_seg_path, exist_ok=True)

                raw_seg_src_files = [linux_path(raw_seg_path, k) for k in _seg_src_fnames]

                for img_src_file_id, img_src_file in enumerate(tqdm(_img_src_files)):
                    seg_src_file = _seg_src_files[img_src_file_id]
                    raw_seg_src_file = raw_seg_src_files[img_src_file_id]
                    seg_img_orig = cv2.imread(seg_src_file)

                    seg_img, raw_seg_img, class_to_ids = remove_fuzziness_in_mask(seg_img_orig, params.n_classes,
                                                                                  class_to_color, params.fuzziness)

                    raw_seg_vals = np.unique(raw_seg_img, return_index=0)
                    raw_seg_vals = list(raw_seg_vals)
                    n_raw_seg_vals = len(raw_seg_vals)

                    seg_img_flat = seg_img.reshape((-1, 3))
                    seg_vals = np.unique(seg_img_flat, return_index=0, axis=0)
                    seg_vals = list(seg_vals)
                    n_seg_vals = len(seg_vals)

                    if n_seg_vals != n_raw_seg_vals:
                        print("number of classes is original and raw  segmentation masks do not match")
                        print('seg_vals: {}'.format(seg_vals))
                        print('raw_seg_vals: {}'.format(raw_seg_vals))

                        _seg_img_orig = resize_ar(seg_img_orig, 900, 900)[0]
                        _seg_img = resize_ar(seg_img, 900, 900)[0]

                        cv2.imshow('seg_img_orig', _seg_img_orig)
                        cv2.imshow('seg_img', _seg_img)

                        cv2.waitKey(0)

                    if n_seg_vals > params.n_classes or n_raw_seg_vals > params.n_classes:
                        print("number of classes is less than the number of unique pixel values in {}".format(
                            seg_src_file))

                        seg_img_raw_min = np.amin(raw_seg_img)
                        seg_img_raw_max = np.amax(raw_seg_img)
                        print('seg_img_raw_min: {}'.format(seg_img_raw_min))
                        print('seg_img_raw_max: {}'.format(seg_img_raw_max))

                        _seg_img_orig = resize_ar(seg_img_orig, 900, 900)[0]
                        _seg_img = resize_ar(seg_img, 900, 900)[0]

                        cv2.imshow('seg_img_orig', _seg_img_orig)
                        cv2.imshow('seg_img', _seg_img)

                        cv2.waitKey(0)

                    # raw_seg_img, raw_seg_vals = convert_to_raw_mask(seg_img, params.n_classes, seg_src_file,
                    # class_to_color,
                    #                                             class_to_ids)

                    # raw_seg_img_min = np.amin(raw_seg_img)
                    # raw_seg_img_max = np.amax(raw_seg_img)

                    # if params.out_size:
                    #     seg_img, _, _ = resize_ar(seg_img, width=out_w, height=out_h, bkg_col=(255, 255, 255))

                    cv2.imwrite(raw_seg_src_file, raw_seg_img)

            assert os.path.isdir(raw_seg_path), "raw_seg_path does not exist"
            print('reading raw segmentations from: {}'.format(raw_seg_path))

            raw_seg_src_fnames = [k for k in os.listdir(raw_seg_path) if
                                  os.path.splitext(k.lower())[1] in seg_exts]

            n_seg_src_files = len(raw_seg_src_fnames)

            assert n_seg_src_files == n_frames, "mismatch between number of source and segmentation images"

            raw_seg_src_fnames.sort()

            raw_seg_src_fnames = [raw_seg_src_fnames[i] for i in img_indices]

            raw_seg_src_files = [linux_path(raw_seg_path, k) for k in raw_seg_src_fnames]

            train_seg_files += raw_seg_src_files[:_n_train_files]
            test_seg_files += raw_seg_src_files[_n_train_files:]

    n_total_files = n_train_files + n_test_files
    print('Found {} files with {} training and {} testing files corresponding to a training ratio of {}'.format(
        n_total_files, n_train_files, n_test_files, params.train_ratio
    ))

    if train_seg_files:
        train_output_dir = linux_path(output_dir, 'train')
        print('saving train tfrecords to {}'.format(train_output_dir))
        os.makedirs(train_output_dir, exist_ok=True)
        create_tfrecords(train_img_files, train_seg_files, params.num_shards,
                         params.db_split, train_output_dir)

    if test_img_files:
        test_output_dir = linux_path(output_dir, 'test')
        os.makedirs(test_output_dir, exist_ok=True)
        print('saving test tfrecords to {}'.format(test_output_dir))
        create_tfrecords(test_img_files, test_seg_files, params.num_shards,
                         params.db_split, test_output_dir)


def main():
    params = Params()
    paramparse.process(params)

    _convert_dataset(params)


if __name__ == '__main__':
    main()
