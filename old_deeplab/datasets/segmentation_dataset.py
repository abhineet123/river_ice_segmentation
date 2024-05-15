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

"""Provides data from semantic segmentation datasets.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow. Currently, we
support the following datasets:

1. PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

PASCAL VOC 2012 semantic segmentation dataset annotates 20 foreground objects
(e.g., bike, person, and so on) and leaves all the other semantic classes as
one background class. The dataset contains 1464, 1449, and 1456 annotated
images for the training, validation and test respectively.

2. Cityscapes dataset (https://www.cityscapes-dataset.com)

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.
"""
import collections
import os.path
import tensorflow as tf

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['splits_to_sizes',  # Splits of the dataset into training, val, and test.
     'num_classes',  # Number of semantic classes.
     'ignore_label',  # Ignore label value.
     ]
)

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 2975,
        'val': 500,
    },
    num_classes=19,
    ignore_label=255,
)

_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'trainval': 2913,
        'val': 1449,
    },
    num_classes=21,
    ignore_label=255,
)
_617_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'training_0_49_256_256_25_100_rot_15_345_4_flip': 159336,
        'training_0_31_256_256_25_100_rot_15_125_235_345_flip': 84075,
        'training_32_49_256_256_25_100_rot_15_125_235_345_flip': 44013,
        'validation_0_563_256_256_256_256': 16920,
        'validation_0_20_256_256_256_256': 630,
        'YUN00001_0_239_256_256_256_256': 32400,
    },
    num_classes=3,
    ignore_label=255,
)
_617_384_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'training_0_49_384_384_25_100_rot_15_345_4_flip': 106371,
        'training_0_31_384_384_25_100_rot_15_345_4_flip': 69375,
        'training_32_49_384_384_25_100_rot_15_345_4_flip': 36969,
        'validation_0_563_384_384_384_384': 6768,
        'validation_0_20_384_384_384_384': 252,
        'YUN00001_0_239_384_384_384_384': 14400,
    },
    num_classes=3,
    ignore_label=255,
)
_617_512_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'training_0_49_512_512_25_100_rot_15_345_4_flip': 67683,
        'training_0_31_512_512_25_100_rot_15_345_4_flip': 43971,
        'training_32_49_512_512_25_100_rot_15_345_4_flip': 22467,
        'validation_0_563_512_512_512_512': 5076,
        'validation_0_20_512_512_512_512': 189,
        'YUN00001_0_239_512_512_512_512': 9600,
    },
    num_classes=3,
    ignore_label=255,
)

_617_640_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'training_0_49_640_640_25_100_rot_15_345_4_flip': 7347,
        'training_0_31_640_640_25_100_rot_15_345_4_flip': 4860,
        'training_32_49_640_640_25_100_rot_15_345_4_flip': 2358,
        'training_0_3_640_640_64_256_rot_15_345_4_flip': 726,
        'training_0_3_640_640_640_640': 25,
        'training_0_3_640_640_640_640_sel_2': 25,
        'training_0_3_640_640_640_640_sel_10': 25,
        'training_0_3_640_640_640_640_sel_100': 25,
        'training_0_3_640_640_640_640_sel_1000': 25,
        'training_0_7_640_640_64_256_rot_15_345_4_flip': 1290,
        'training_0_15_640_640_64_256_rot_15_345_4_flip': 2694,
        'training_0_23_640_640_64_256_rot_15_345_4_flip': 3678,
        'training_0_31_640_640_64_256_rot_15_345_4_flip': 4857,
        'training_0_49_640_640_64_256_rot_15_345_4_flip': 7347,
        'training_32_49_640_640_64_256_rot_15_345_4_flip': 2358,
        'training_32_49_640_640_640_640': 108,
        'training_4_49_640_640_640_640': 276,
        'validation_0_563_640_640_640_640': 3384,
        'validation_0_20_640_640_640_640': 126,
        'YUN00001_0_239_640_640_640_640': 5760,
        'YUN00001_3600_0_3599_640_640_640_640': 86400,
        'YUN00001_0_8999_640_640_640_640': 216000,
        '20160122_YUN00002_700_2500_0_1799_640_640_640_640': 43200,
        '20160122_YUN00020_2000_3800_0_1799_640_640_640_640': 43200,
        '20161203_Deployment_1_YUN00001_900_2700_0_1799_640_640_640_640': 43200,
        '20161203_Deployment_1_YUN00002_1800_0_1799_640_640_640_640': 43200,
    },
    num_classes=3,
    ignore_label=255,
)

_617_800_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'training_0_3_800_800_80_320_rot_15_345_4_flip': 321,
        'training_0_49_800_800_80_320_rot_15_345_4_flip': 2292,
        'training_0_31_800_800_80_320_rot_15_345_4_flip': 1587,
        'training_32_49_800_800_80_320_rot_15_345_4_flip': 831,
    },
    num_classes=2,
    ignore_label=255,
)

_ACAMP_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'bear_1_1_masks_448x448_ar_1p0': 100,
        'bear_1_1_500x500_10': 100,
        'bear_1_1_to_1_6_500x500_10_test': 1536,
    },
    num_classes=2,
    ignore_label=128,
)

_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'training_0_31_49_256_256_25_100_rot_15_125_235_345_flip': _617_256_INFORMATION,
    'training_0_31_49_384_384_25_100_rot_15_345_4_flip': _617_384_INFORMATION,
    'training_0_31_49_512_512_25_100_rot_15_345_4_flip': _617_512_INFORMATION,
    'training_0_31_49_640_640_64_256_rot_15_345_4_flip': _617_640_INFORMATION,
    'training_0_31_49_800_800_80_320_rot_15_345_4_flip': _617_800_INFORMATION,
    'deeplab_acamp': _ACAMP_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_cityscapes_dataset_name():
    return 'cityscapes'


def get_dataset(dataset_name, split_name, dataset_dir):
    """Gets an instance of slim Dataset.

    Args:
      dataset_name: Dataset name.
      split_name: A train/val Split name.
      dataset_dir: The directory of the dataset sources.

    Returns:
      An instance of slim Dataset.

    Raises:
      ValueError: if the dataset_name or split_name is not recognized.
    """
    if dataset_name not in _DATASETS_INFORMATION:
        raise ValueError('The specified dataset is not supported yet.')

    splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

    if split_name not in splits_to_sizes:
        raise ValueError('data split name %s not recognized' % split_name)

    # Prepare the variables for different datasets.
    num_classes = _DATASETS_INFORMATION[dataset_name].num_classes
    ignore_label = _DATASETS_INFORMATION[dataset_name].ignore_label

    file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Specify how the TF-Examples are decoded.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/segmentation/class/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/segmentation/class/format': tf.FixedLenFeature(
            (), tf.string, default_value='png'),
    }
    items_to_handlers = {
        'image': tfexample_decoder.Image(
            image_key='image/encoded',
            format_key='image/format',
            channels=3),
        'image_name': tfexample_decoder.Tensor('image/filename'),
        'height': tfexample_decoder.Tensor('image/height'),
        'width': tfexample_decoder.Tensor('image/width'),
        'labels_class': tfexample_decoder.Image(
            image_key='image/segmentation/class/encoded',
            format_key='image/segmentation/class/format',
            channels=1),
    }

    decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=splits_to_sizes[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        ignore_label=ignore_label,
        num_classes=num_classes,
        name=dataset_name,
        multi_label=True)
