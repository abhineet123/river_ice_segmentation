# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import glob
import math
import os.path
import sys
import build_data
import tensorflow as tf
# from scipy.misc import imread, imsave
from scipy import misc
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('db_root_dir',
                           '/data/617/images/',
                           'Folder containing the 617 datasets')

tf.app.flags.DEFINE_string('db_dir',
                           'training_256_256_25_100_rot_15_90_flip',
                           'folder containing the dataset files')

tf.app.flags.DEFINE_string('image_format',
                           'png',
                           'image format')

tf.app.flags.DEFINE_string('label_format',
                           'png',
                           'label format')

tf.app.flags.DEFINE_string('output_dir',
                           '',
                           'Path to save converted SSTable of TensorFlow examples.')

tf.app.flags.DEFINE_integer('create_dummy_labels',
                           0,
                           'Flag to specify that no labels are available and so '
                           'dummy ones should be created.')

tf.app.flags.DEFINE_integer('selective_loss',
                           0,
                           'Use only the specified number of pixels per class per image')
_NUM_SHARDS = 4


def _convert_dataset(db_name):
    """Converts the specified dataset split to TFRecord format.

    Args:
      db_name: The dataset split (e.g., train, test).

    Raises:
      RuntimeError: If loaded image and label have different shape.
    """

    output_dir = os.path.join(FLAGS.db_root_dir, FLAGS.output_dir, 'tfrecord')
    sys.stdout.write('Processing {}\n\n'.format(db_name))
    images = os.path.join(FLAGS.db_root_dir, db_name, 'images', '*.{}'.format(FLAGS.image_format))
    print('Reading images from: {}'.format(images))

    image_filenames = glob.glob(images)
    if image_filenames is None:
        raise SystemError('No images found at {}'.format(images))

    if FLAGS.create_dummy_labels:
        labels_path = os.path.join(FLAGS.db_root_dir, db_name, 'labels')
        if not os.path.isdir(labels_path):
            os.makedirs(labels_path)
        print('Creating dummy labels at: {}'.format(labels_path))
        for image_filename in image_filenames:
            image = misc.imread(image_filename)
            height, width, _ = image.shape
            dummy_label = np.zeros((height, width), dtype=np.uint8)
            out_fname =  os.path.splitext(os.path.basename(image_filename))[0] + '.{}'.format(FLAGS.label_format)
            misc.imsave(os.path.join(labels_path,out_fname), dummy_label)
        print('Done')

    labels = os.path.join(FLAGS.db_root_dir, db_name, 'labels', '*.{}'.format(FLAGS.label_format))
    print('Reading labels from: {}'.format(labels))
    
    seg_filenames = glob.glob(labels)
    if seg_filenames is None:
        raise SystemError('No labels found at {}'.format(labels))

    # filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = len(image_filenames)
    num_labels = len(seg_filenames)

    if num_images != num_labels:
        raise SystemError('Mismatch between image and label file counts: {}, {}'.format(
            num_images, num_labels))

    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader('png', channels=3)
    label_reader = build_data.ImageReader('png', channels=3)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print('Writing tfrecords to: {}'.format(output_dir))

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            output_dir,
            '%s-%05d-of-%05d.tfrecord' % (db_name, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()
                image_filename = image_filenames[i]
                f1 = os.path.basename(image_filename)[:-4]
                seg_filename = seg_filenames[i]
                f2 = os.path.basename(image_filename)[:-4]
                if f1 != f2:
                    raise SystemError('Mismatch between image and label filenames: {}, {}'.format(
                        f1, f2))

                # Read the image.
                image_data = tf.gfile.FastGFile(image_filename, 'r').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_data = tf.gfile.FastGFile(seg_filename, 'r').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, image_filename, height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(unused_argv):
    _convert_dataset(FLAGS.db_dir)


if __name__ == '__main__':
    tf.app.run()
