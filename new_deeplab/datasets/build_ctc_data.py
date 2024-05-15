# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts CTC data to TFRecord file format with Example protos."""

import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import paramparse

from ctc_info import CTCInfo
import build_data


def irange(a, b):
    return list(range(a, b + 1))


class Params:

    def __init__(self):
        self.db_split = 'huh'

        self.cfg = ()
        self.ignore_missing_gt = 1
        self.ignore_missing_seg = 1
        self.ignored_region_only = 0

        self.resize = 0
        self.root_dir = '/data'
        self.output_dir = '/data'

        self.start_id = 0
        self.end_id = -1
        # self.seq_ids = [6, 7, 14, 15]

        self.write_gt = 0
        self.write_img = 1
        self.raad_gt = 0
        self.tra_only = 0

        self.two_classes = 1

        self.show_img = 0
        self.save_img = 0
        self.save_vid = 0

        self.disable_seg = 0
        self.disable_tqdm = 0
        self.codec = 'H264'
        self.use_tif = 0

        self.vis_height = 1080
        self.vis_width = 1920
        self.db_splits = CTCInfo.DBSplits().__dict__
        self.num_shards = 4


def seg_to_png(gold_seg_src_file_ids, silver_seg_src_file_ids, img_src_file_id,
               silver_seg_path, gold_seg_path, png_seg_src_path, img_src_file, two_classes):
    gold_seg_img = silver_seg_img = None

    try:
        gold_seg_src_file = gold_seg_src_file_ids[img_src_file_id]
    except KeyError:
        n_gold_seg_objs = 0
    else:
        gold_seg_src_path = os.path.join(gold_seg_path, gold_seg_src_file)
        gold_seg_img = cv2.imread(gold_seg_src_path, cv2.IMREAD_UNCHANGED)
        gold_seg_obj_ids = list(np.unique(gold_seg_img, return_counts=False))
        gold_seg_obj_ids.remove(0)

        n_gold_seg_objs = len(gold_seg_obj_ids)

    try:
        silver_seg_src_file = silver_seg_src_file_ids[img_src_file_id]
    except KeyError:
        n_silver_seg_objs = 0
    else:
        silver_seg_src_path = os.path.join(silver_seg_path, silver_seg_src_file)
        silver_seg_img = cv2.imread(silver_seg_src_path, cv2.IMREAD_UNCHANGED)

        silver_seg_obj_ids = list(np.unique(silver_seg_img, return_counts=False))
        silver_seg_obj_ids.remove(0)

        n_silver_seg_objs = len(silver_seg_obj_ids)

    if n_silver_seg_objs == 0 and n_gold_seg_objs == 0:
        # print('\nno segmentations found for {}\n'.format(img_src_file))
        return 0

    if n_silver_seg_objs > n_gold_seg_objs:
        seg_img = silver_seg_img
    else:
        seg_img = gold_seg_img

    if two_classes:
        seg_img[seg_img > 0] = 1
        seg_img = seg_img.astype(np.uint8)

    cv2.imwrite(png_seg_src_path, seg_img)

    return 1


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def _convert_dataset(params):
    """

    :param Params params:
    :return:
    """

    seq_ids = params.db_splits[params.db_split]

    print('root_dir: {}'.format(params.root_dir))
    print('sub_seq: {}'.format(params.db_split))
    print('seq_ids: {}'.format(seq_ids))

    jpg_img_root_path = linux_path(params.root_dir, 'CTC', 'Images')
    tif_img_root_path = linux_path(params.root_dir, 'CTC', 'Images_TIF')
    png_img_root_path = linux_path(params.root_dir, 'CTC', 'Images_PNG')
    os.makedirs(png_img_root_path, exist_ok=True)

    output_root_dir = linux_path(params.root_dir, 'CTC', 'tfrecord')
    os.makedirs(output_root_dir, exist_ok=True)

    if not params.disable_seg:
        tif_labels_root_path = linux_path(params.root_dir, 'CTC', 'tif')
        png_labels_root_path = linux_path(params.root_dir, 'CTC', 'Labels_PNG')
        os.makedirs(png_labels_root_path, exist_ok=True)
    else:
        print('\nSegmentations are disabled\n')

    if params.start_id > 0:
        seq_ids = seq_ids[params.start_id:]

    n_seq = len(seq_ids)

    if params.use_tif:
        img_root_path = tif_img_root_path
        img_exts = ('.tif',)
    else:
        img_root_path = jpg_img_root_path
        img_exts = ('.jpg',)

    if not params.disable_seg:
        gold_seg_src_file_ids = {}
        silver_seg_src_file_ids = {}
    img_src_file_ids = {}

    n_total_src_files = 0

    img_src_files = []
    for __id, seq_id in enumerate(seq_ids):

        seq_name, n_frames = CTCInfo.sequences[seq_id]

        print('\tseq {} / {}\t{}\t{}\t{} frames'.format(__id + 1, n_seq, seq_id, seq_name, n_frames))

        if not params.disable_seg:

            silver_seg_path = linux_path(tif_labels_root_path, seq_name + '_ST', 'SEG')
            gold_seg_path = linux_path(tif_labels_root_path, seq_name + '_GT', 'SEG')

            assert os.path.exists(silver_seg_path) or os.path.exists(gold_seg_path), \
                "Neither silver nor gold segmentations found for sequence: {}".format(seq_name)

            if os.path.exists(gold_seg_path):
                _gold_seg_src_files = [linux_path(gold_seg_path, k) for k in os.listdir(gold_seg_path) if
                                       os.path.splitext(k.lower())[1] in ('.tif',)]
                _gold_seg_src_files.sort()

                _gold_seg_src_file_ids = {
                    seq_name + '::' + ''.join(k for k in os.path.basename(src_file) if k.isdigit()): src_file
                    for src_file in _gold_seg_src_files
                }
                gold_seg_src_file_ids.update(_gold_seg_src_file_ids)
            else:
                print("\ngold  segmentations not found for sequence: {}\n".format(seq_name))

            if os.path.exists(silver_seg_path):
                _silver_seg_src_files = [linux_path(silver_seg_path, k) for k in os.listdir(silver_seg_path) if
                                         os.path.splitext(k.lower())[1] in ('.tif',)]
                _silver_seg_src_files.sort()

                _silver_seg_src_file_ids = {
                    seq_name + '::' + ''.join(k for k in os.path.basename(src_file) if k.isdigit()): src_file
                    for src_file in _silver_seg_src_files
                }
                silver_seg_src_file_ids.update(_silver_seg_src_file_ids)
            else:
                print("\nsilver  segmentations not found for sequence: {}\n".format(seq_name))

        # unique_gold_seg_src_files = [v for k,v in gold_seg_src_file_ids.items() if k not in
        # silver_seg_src_file_ids.keys()]

        img_dir_path = linux_path(img_root_path, seq_name)
        png_img_dir_path = linux_path(png_img_root_path, seq_name)

        if params.use_tif:
            os.makedirs(png_img_dir_path, exist_ok=True)

        _img_src_files = [linux_path(img_dir_path, k) for k in os.listdir(img_dir_path) if
                          os.path.splitext(k.lower())[1] in img_exts]
        _img_src_files.sort()

        n_total_src_files += len(_img_src_files)

        for img_src_file in _img_src_files:
            img_src_file_no_ext = os.path.splitext(os.path.basename(img_src_file))[0]
            img_src_file_id = seq_name + '::' + ''.join(k for k in os.path.basename(img_src_file) if k.isdigit())
            if not params.disable_seg:
                silver_seg_path = linux_path(tif_labels_root_path, seq_name + '_ST', 'SEG')
                gold_seg_path = linux_path(tif_labels_root_path, seq_name + '_GT', 'SEG')
                png_seg_path = linux_path(png_labels_root_path, seq_name)
                os.makedirs(png_seg_path, exist_ok=True)

                png_seg_src_path = os.path.join(png_seg_path, img_src_file_no_ext + '.png')
                if not os.path.exists(png_seg_src_path):
                    segmentation_found = seg_to_png(gold_seg_src_file_ids, silver_seg_src_file_ids, img_src_file_id,
                                                    silver_seg_path, gold_seg_path, png_seg_src_path, img_src_file,
                                                    params.two_classes)
                    if not segmentation_found:
                        continue
            else:
                png_seg_src_path = None

            png_img_dir_path = linux_path(png_img_root_path, seq_name)
            jpg_img_dir_path = linux_path(jpg_img_root_path, seq_name)
            tif_img_dir_path = linux_path(tif_img_root_path, seq_name)

            if params.use_tif:
                png_img_src_path = os.path.join(png_img_dir_path, img_src_file)
                if not os.path.exists(png_img_src_path):
                    tif_img_src_path = os.path.join(tif_img_dir_path, img_src_file)
                    img = cv2.imread(tif_img_src_path, cv2.IMREAD_UNCHANGED)
                    cv2.imwrite(png_img_src_path, img)
                img_src_path = png_img_src_path
            else:
                jpg_img_src_path = os.path.join(jpg_img_dir_path, img_src_file)
                img_src_path = jpg_img_src_path

            img_src_files.append(img_src_file)
            img_src_file_ids[img_src_file] = (img_src_file_id, seq_name, png_seg_src_path, img_src_path)

    n_src_files = len(img_src_files)
    print('\n\n{}: {} / {}\n\n'.format(params.db_split, n_src_files, n_total_src_files))

    # return

    # output_dir = linux_path(output_root_dir, seq_name)
    output_dir = output_root_dir

    create_tfrecords(img_src_files, img_src_file_ids, params.num_shards, params.db_split, params.use_tif, output_dir)


def create_tfrecords(src_files, file_ids, n_shards, sub_seq, use_tif, output_dir):
    if use_tif:
        image_reader = build_data.ImageReader('png', channels=1)
    else:
        image_reader = build_data.ImageReader('jpeg', channels=1)

    label_reader = build_data.ImageReader('png', channels=1)

    n_images = len(src_files)
    n_per_shard = int(math.ceil(n_images / float(n_shards)))

    os.makedirs(output_dir, exist_ok=True)

    print('Creating {} shards with {} images ({} per shard)'.format(n_shards, n_images, n_per_shard))

    for shard_id in range(n_shards):

        output_file_path = os.path.join(
            output_dir,
            '{:s}-{:05d}-of-{:05d}.tfrecord'.format(sub_seq, shard_id, n_shards))

        with tf.python_io.TFRecordWriter(output_file_path) as tfrecord_writer:
            start_idx = shard_id * n_per_shard
            end_idx = min((shard_id + 1) * n_per_shard, n_images)

            for img_id in tqdm(range(start_idx, end_idx), ncols=50):

                img_src_file = src_files[img_id]
                img_src_file_id, seq_name, seg_src_path, img_src_path = file_ids[img_src_file]
                image_data = tf.gfile.FastGFile(img_src_path, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)

                if seg_src_path is not None:
                    seg_data = tf.gfile.FastGFile(seg_src_path, 'rb').read()
                    seg_height, seg_width = label_reader.read_image_dims(seg_data)

                    if height != seg_height or width != seg_width:
                        raise RuntimeError('Shape mismatch found between image and label')
                else:
                    seg_data = None

                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, img_src_path, height, width, seg_data)

                tfrecord_writer.write(example.SerializeToString())


def main():
    params = Params()
    paramparse.process(params)

    # for _sub_seq in params.sub_seq_dict:
    #     params.sub_seq = _sub_seq
    #     _convert_dataset(params)

    _convert_dataset(params)


if __name__ == '__main__':
    main()
